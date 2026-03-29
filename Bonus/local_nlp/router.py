"""本地 NLP Bonus：Ollama Fast-Path + 云端双引擎路由。"""

from __future__ import annotations

import asyncio
import inspect
import os
import platform
import subprocess
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable

from openai import APIConnectionError, APITimeoutError, OpenAIError

try:
    from .client import CLOUD_MODEL, LOCAL_SLM_MODEL, local_slm_client
except ImportError:
    from client import CLOUD_MODEL, LOCAL_SLM_MODEL, local_slm_client  # type: ignore

try:
    from src.orchestrator.intent_classifier import (
        _detect_temporal_query,
        _infer_fallback_risk,
        _looks_like_realtime_time_query,
        apply_intent_overrides,
        apply_confidence_policy,
        handle_intent,
        parse_llm_json,
    )
except ImportError:
    from orchestrator.intent_classifier import (  # type: ignore
        _detect_temporal_query,
        _infer_fallback_risk,
        _looks_like_realtime_time_query,
        apply_intent_overrides,
        apply_confidence_policy,
        handle_intent,
        parse_llm_json,
    )


StatusCallback = Callable[[str], None | Awaitable[None]]
LOCAL_PASS_CONFIDENCE = 0.8
LOCAL_PASS_INTENTS = {"shell_agent", "direct_answer", "memory_agent"}
LOCAL_SUBTASK_SCHEMA = {
    "type": "object",
    "required": [
        "agent",
        "task_description",
        "context_passed",
        "risk_level",
        "memory_action",
        "memory_content",
    ],
    "additionalProperties": False,
    "properties": {
        "agent": {"type": "string", "enum": ["shell_agent", "tool_agent", "memory_agent"]},
        "task_description": {"type": "string"},
        "context_passed": {"type": "array", "items": {"type": "string"}},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "memory_action": {
            "anyOf": [
                {"type": "string", "enum": ["save", "search", "delete", "list"]},
                {"type": "null"},
            ]
        },
        "memory_content": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
    },
}
LOCAL_FASTPATH_JSON_SCHEMA_NAME = "local_fastpath_routing_result"
LOCAL_FASTPATH_JSON_SCHEMA = {
    "type": "object",
    "required": [
        "reasoning",
        "confidence",
        "intent",
        "task_description",
        "reply",
        "question",
        "options",
        "subtasks",
    ],
    "additionalProperties": False,
    "properties": {
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "intent": {
            "type": "string",
            "enum": ["shell_agent", "tool_agent", "memory_agent", "multi_agent", "direct_answer", "clarification"],
        },
        "task_description": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
        "reply": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
        "question": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
        "options": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ]
        },
        "subtasks": {
            "anyOf": [
                {"type": "array", "items": LOCAL_SUBTASK_SCHEMA},
                {"type": "null"},
            ]
        },
    },
}


@dataclass
class LocalAttempt:
    accepted: bool
    result: dict[str, Any] | None
    latency_ms: float | None
    reason: str
    raw_text: str = ""


def get_local_context(*, max_files: int = 10) -> dict[str, str]:
    """收集精简版本地上下文：隐藏文件过滤 + 最多 10 项。"""
    cwd = os.getcwd()
    files: list[str] = []
    try:
        with os.scandir(cwd) as iterator:
            entries = sorted(iterator, key=lambda entry: (not entry.is_dir(), entry.name.lower()))
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                suffix = "/" if entry.is_dir() else ""
                files.append(f"- {entry.name}{suffix}")
                if len(files) >= max_files:
                    break
    except OSError:
        files = ["- 无法获取目录概览"]

    try:
        git_output = subprocess.check_output(
            ["git", "status", "-s"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_output = "当前不是 Git 仓库"

    return {
        "os": platform.system(),
        "shell": os.environ.get("SHELL", "unknown"),
        "pwd": cwd,
        "files": "\n".join(files) if files else "- （当前目录为空）",
        "git_status": git_output or "干净的工作区",
    }


def _format_extra_context(extra_context: dict[str, str] | None) -> str:
    if not extra_context:
        return ""

    sections: list[str] = []
    short_term = str(extra_context.get("short_term_memory") or "").strip()
    long_term = str(extra_context.get("long_term_memory") or "").strip()

    if short_term:
        sections.append(f"""短期会话上下文（RAM）：
{short_term}""")

    if long_term:
        sections.append(f"""长期记忆注入（ROM）：
{long_term}""")

    last_file_path = str(extra_context.get("last_file_path") or "").strip()
    last_document_summary = str(extra_context.get("last_document_summary") or "").strip()
    last_document_content = str(extra_context.get("last_document_content") or "").strip()
    if last_file_path or last_document_summary or last_document_content:
        lines = ["最近文档 Artifact（若用户说“前文所述/那个 README/该文档”，优先指向它）："]
        if last_file_path:
            lines.append(f"- 路径: {last_file_path}")
        if last_document_summary:
            lines.append(f"- 摘要: {last_document_summary}")
        if last_document_content:
            lines.append(f"- 内容片段:\n{last_document_content}")
        sections.append("\n".join(lines))

    return ("\n" + "\n\n".join(sections)) if sections else ""


def get_local_system_prompt(context: dict[str, str], extra_context: dict[str, str] | None = None) -> str:
    """给 4B 本地小模型使用的精简版 Prompt。"""
    return f"""你是多 Agent 命令行系统的本地快速路由器，只负责快速判断用户意图。

你必须且只能输出一个 JSON 对象，不要输出 Markdown、解释或代码块。
必须按字段顺序输出：reasoning -> confidence -> intent -> task_description -> reply -> question -> options

仅支持六种 intent：
1. shell_agent：明确的本地命令/文件系统任务。
2. tool_agent：需要复杂推理、读文件理解、外部工具或 MCP。
3. memory_agent：显式的长期记忆读写请求。
4. multi_agent：一个请求包含多个顺序子任务。
5. direct_answer：纯问答或闲聊。
6. clarification：信息不足，必须追问。

决策要求：
- 如果不确定，降低 confidence。
- 如果像“查看目录 / 创建文件 / 运行命令”这类明确本地任务，优先判为 shell_agent。
- tool_agent 只在你明确觉得 4B 本地模型不该继续判断时使用。
- 如果是“记住 / 你还记得 / 列出记忆 / 忘掉”这类显式记忆请求，判为 memory_agent。
- 如果请求中同时包含“执行任务 + 记住结果”等两个动作，判为 multi_agent（即使你不给 subtasks，也不要判 direct_answer）。
- 如果判为 multi_agent，subtasks 至少要有 2 项；只有 1 项时应改回对应单一 intent。
- 如果任务是“读取文件/代码/文档后再总结、分析、解释、翻译”，必须判为 tool_agent。
- 如果任务依赖“当前时间/日期/星期、天气、新闻、汇率、最新/实时数据”等时效性信息，不能判为 direct_answer，必须交给 tool_agent。
- shell_agent 只输出简短 task_description，不做深入风险分析。

当前环境：
- OS: {context['os']}
- Shell: {context['shell']}
- 当前目录: {context['pwd']}
- 目录概览（最多 10 项，已过滤隐藏文件）:
{context['files']}
- Git 状态: {context['git_status']}{_format_extra_context(extra_context)}

输出格式：
如果 intent="shell_agent"、"tool_agent" 或 "memory_agent"：
{{
  "reasoning": "简短推理",
  "confidence": 0.90,
  "intent": "shell_agent",
  "task_description": "给后续 Agent 的简短任务说明",
  "reply": null,
  "question": null,
  "options": null,
  "subtasks": null
}}

如果 intent="multi_agent"：
{{
  "reasoning": "这是复合请求",
  "confidence": 0.82,
  "intent": "multi_agent",
  "task_description": "整体任务摘要",
  "reply": null,
  "question": null,
  "options": null,
  "subtasks": [
    {{
      "agent": "shell_agent",
      "task_description": "子任务说明",
      "context_passed": [],
      "risk_level": "low",
      "memory_action": null,
      "memory_content": null
    }},
    {{
      "agent": "memory_agent",
      "task_description": "第二个子任务说明",
      "context_passed": [],
      "risk_level": "low",
      "memory_action": "save",
      "memory_content": "{{last_result}}"
    }}
  ]
}}

如果 intent="direct_answer"：
{{
  "reasoning": "简短推理",
  "confidence": 0.92,
  "intent": "direct_answer",
  "task_description": null,
  "reply": "直接回答用户",
  "question": null,
  "options": null,
  "subtasks": null
}}

如果 intent="clarification"：
{{
  "reasoning": "缺少关键信息",
  "confidence": 0.45,
  "intent": "clarification",
  "task_description": null,
  "reply": null,
  "question": "需要补充什么信息？",
  "options": ["候选1", "候选2"],
  "subtasks": null
}}""".strip()


def validate_local_result(data: dict[str, Any]) -> tuple[bool, str]:
    """校验本地路由结果。"""
    required = LOCAL_FASTPATH_JSON_SCHEMA["required"]
    for key in required:
        if key not in data:
            return False, f"缺少字段: {key}"

    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        return False, "confidence 不是合法数字"
    if not 0.0 <= confidence <= 1.0:
        return False, "confidence 超出范围"

    intent = data.get("intent")
    if intent not in {"shell_agent", "tool_agent", "memory_agent", "multi_agent", "direct_answer", "clarification"}:
        return False, "intent 非法"

    subtasks = data.get("subtasks")
    if subtasks is not None and not isinstance(subtasks, list):
        return False, "subtasks 必须为数组或 null"

    if intent in {"shell_agent", "tool_agent", "memory_agent"}:
        if subtasks is None and (not isinstance(data.get("task_description"), str) or not data["task_description"].strip()):
            return False, f"{intent} 必须提供非空 task_description"
    elif intent == "multi_agent":
        if not isinstance(subtasks, list) or not subtasks:
            return False, "multi_agent 必须提供非空 subtasks"
        if len(subtasks) < 2:
            return False, "multi_agent 至少需要 2 个子任务"
    elif intent == "direct_answer":
        if not isinstance(data.get("reply"), str) or not data["reply"].strip():
            return False, "direct_answer 必须提供非空 reply"
    elif intent == "clarification":
        if not isinstance(data.get("question"), str) or not data["question"].strip():
            return False, "clarification 必须提供非空 question"
        options = data.get("options")
        if options is not None and not isinstance(options, list):
            return False, "clarification 的 options 必须为数组或 null"

    return True, ""


def normalize_local_result(result: dict[str, Any], user_input: str) -> dict[str, Any]:
    """将本地小模型结果补全成与现有 orchestrator 兼容的结构。"""
    intent = result["intent"]
    normalized: dict[str, Any] = {
        "reasoning": str(result.get("reasoning", "")).strip(),
        "confidence": float(result.get("confidence", 0.0)),
        "intent": intent,
        "risk_level": "low",
        "task_description": None,
        "context_passed": None,
        "reply": None,
        "question": None,
        "options": None,
        "subtasks": None,
    }

    if intent in {"shell_agent", "tool_agent", "memory_agent"}:
        task_description = str(result.get("task_description", "")).strip()
        normalized["task_description"] = task_description
        normalized["context_passed"] = []
        normalized["risk_level"] = _infer_fallback_risk(f"{user_input}\n{task_description}")
    elif intent == "multi_agent":
        subtasks = result.get("subtasks")
        if isinstance(subtasks, list):
            normalized["subtasks"] = subtasks
            subtask_risks = [str(item.get("risk_level") or "low") for item in subtasks if isinstance(item, dict)]
            if "high" in subtask_risks:
                normalized["risk_level"] = "high"
            elif "medium" in subtask_risks:
                normalized["risk_level"] = "medium"
            else:
                normalized["risk_level"] = "low"
        normalized["task_description"] = str(result.get("task_description", "")).strip() or "复合任务"
        normalized["context_passed"] = []
    elif intent == "direct_answer":
        normalized["reply"] = str(result.get("reply", "")).strip()
    else:
        normalized["question"] = str(result.get("question", "")).strip()
        options = result.get("options")
        normalized["options"] = options if isinstance(options, list) else []

    normalized = apply_intent_overrides(user_input, normalized)
    return apply_confidence_policy(normalized)


async def _emit_status(callback: StatusCallback | None, message: str) -> None:
    """兼容同步/异步状态回调。"""
    if callback is None:
        return
    maybe = callback(message)
    if inspect.isawaitable(maybe):
        await maybe


async def _request_local_fastpath_json(user_input: str, extra_context: dict[str, str] | None = None) -> str:
    """向本地 Ollama 发起非流式 JSON 模式请求。"""
    context = get_local_context()
    system_prompt = get_local_system_prompt(context, extra_context=extra_context)
    response = await local_slm_client.chat.completions.create(
        model=LOCAL_SLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


async def try_local_fastpath(user_input: str, extra_context: dict[str, str] | None = None) -> LocalAttempt:
    """尝试使用本地 SLM 进行快速路由。"""
    started = perf_counter()
    try:
        raw_text = await _request_local_fastpath_json(user_input, extra_context=extra_context)
    except (APIConnectionError, APITimeoutError, ConnectionError, OSError):
        latency = (perf_counter() - started) * 1000
        return LocalAttempt(False, None, latency, "local_unavailable")
    except OpenAIError:
        latency = (perf_counter() - started) * 1000
        return LocalAttempt(False, None, latency, "local_error")
    except Exception:
        latency = (perf_counter() - started) * 1000
        return LocalAttempt(False, None, latency, "local_error")

    latency = (perf_counter() - started) * 1000
    parsed = parse_llm_json(raw_text)
    if parsed is None:
        return LocalAttempt(False, None, latency, "json_decode_error", raw_text=raw_text)

    valid, error_message = validate_local_result(parsed)
    if not valid:
        return LocalAttempt(False, None, latency, f"invalid_local_json: {error_message}", raw_text=raw_text)

    normalized = normalize_local_result(parsed, user_input)
    intent = normalized["intent"]
    confidence = float(normalized["confidence"])

    if intent == "tool_agent" and _detect_temporal_query(
        user_input,
        str(normalized.get("task_description") or ""),
    ):
        return LocalAttempt(True, normalized, latency, "accepted", raw_text=raw_text)

    if intent == "tool_agent":
        return LocalAttempt(False, normalized, latency, "tool_agent_requires_cloud", raw_text=raw_text)
    if confidence < LOCAL_PASS_CONFIDENCE:
        return LocalAttempt(False, normalized, latency, "low_confidence", raw_text=raw_text)
    if intent not in LOCAL_PASS_INTENTS:
        return LocalAttempt(False, normalized, latency, "intent_not_fastpath", raw_text=raw_text)

    return LocalAttempt(True, normalized, latency, "accepted", raw_text=raw_text)


class DualEngineRouter:
    """本地主导、云端兜底的双引擎路由器。"""

    def __init__(
        self,
        *,
        cloud_llm_client: Any | None = None,
        cloud_model: str | None = None,
        max_retries: int = 1,
        verbose: bool = False,
    ) -> None:
        self.cloud_llm_client = cloud_llm_client
        self.cloud_model = cloud_model or CLOUD_MODEL
        self.max_retries = max_retries
        self.verbose = verbose

    async def route(
        self,
        user_input: str,
        *,
        status_callback: StatusCallback | None = None,
        extra_context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """先尝试本地 Fast-Path，失败后静默降级到云端。"""
        total_started = perf_counter()
        local_attempt = await try_local_fastpath(user_input, extra_context=extra_context)

        if local_attempt.accepted and local_attempt.result is not None:
            result = dict(local_attempt.result)
            result["_dual_engine"] = {
                "engine": "local_fastpath",
                "local_model": LOCAL_SLM_MODEL,
                "cloud_model": CLOUD_MODEL,
                "local_latency_ms": round(local_attempt.latency_ms or 0.0, 2),
                "cloud_latency_ms": None,
                "total_latency_ms": round((perf_counter() - total_started) * 1000, 2),
                "fallback_reason": None,
            }
            return result

        if local_attempt.reason == "local_unavailable":
            await _emit_status(status_callback, "本地计算节点未就绪，已切换云端")
        elif local_attempt.reason == "json_decode_error" or local_attempt.reason.startswith("invalid_local_json"):
            await _emit_status(status_callback, "本地 JSON 解析失败，已切换云端")

        cloud_started = perf_counter()
        cloud_result = await asyncio.to_thread(
            handle_intent,
            user_input,
            self.cloud_llm_client,
            self.cloud_model,
            self.max_retries,
            self.verbose,
            extra_context,
        )
        cloud_latency_ms = (perf_counter() - cloud_started) * 1000
        cloud_result = dict(cloud_result)
        cloud_result["_dual_engine"] = {
            "engine": "cloud",
            "local_model": LOCAL_SLM_MODEL,
            "cloud_model": self.cloud_model,
            "local_latency_ms": round(local_attempt.latency_ms or 0.0, 2) if local_attempt.latency_ms is not None else None,
            "cloud_latency_ms": round(cloud_latency_ms, 2),
            "total_latency_ms": round((perf_counter() - total_started) * 1000, 2),
            "fallback_reason": local_attempt.reason,
            "local_raw_text": local_attempt.raw_text if self.verbose else None,
        }
        return cloud_result


async def route_user_input(
    user_input: str,
    *,
    status_callback: StatusCallback | None = None,
    extra_context: dict[str, str] | None = None,
    cloud_llm_client: Any | None = None,
    cloud_model: str | None = None,
    max_retries: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """便捷入口。"""
    router = DualEngineRouter(
        cloud_llm_client=cloud_llm_client,
        cloud_model=cloud_model,
        max_retries=max_retries,
        verbose=verbose,
    )
    return await router.route(user_input, status_callback=status_callback, extra_context=extra_context)


__all__ = [
    "DualEngineRouter",
    "LOCAL_FASTPATH_JSON_SCHEMA",
    "LOCAL_FASTPATH_JSON_SCHEMA_NAME",
    "LOCAL_PASS_CONFIDENCE",
    "get_local_context",
    "get_local_system_prompt",
    "normalize_local_result",
    "route_user_input",
    "try_local_fastpath",
]
