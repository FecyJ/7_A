"""
Task 2.3 - 总控 Agent（Orchestrator）
功能：对接前端输入，完成命令直通 / 意图分类 / 结果输出与路由，
并正式接入本地 NLP Fast-Path + 云端兜底双引擎流程。
"""

from __future__ import annotations

import asyncio
import re
import sys
from collections import deque
from typing import Any, Awaitable, Callable, Protocol

try:
    from Bonus.a2a import A2AMessage, AgentRegistry
    from Bonus.memory_agent import MemoryAgent, MemoryService
    from .intent_classifier import (
        DEFAULT_MODEL,
        INTENT_JSON_SCHEMA,
        INTENT_JSON_SCHEMA_NAME,
        _make_fallback,
        classify_intent,
        get_advanced_context,
        handle_intent,
        parse_llm_json,
        validate_intent_result,
    )
    from .llm_client import generate_text_response, get_api_mode, get_llm_client
    from ..shell_agent import ShellA2AAgent, ShellAgent
    from ..tool_agent import ToolA2AAgent, ToolAgent
    from Bonus.local_nlp import DualEngineRouter
except ImportError:
    try:
        from Bonus.a2a import A2AMessage, AgentRegistry  # type: ignore
    except ImportError:
        from src.a2a import A2AMessage, AgentRegistry  # type: ignore
    try:
        from Bonus.memory_agent import MemoryAgent, MemoryService  # type: ignore
    except ImportError:
        from src.memory_agent import MemoryAgent, MemoryService  # type: ignore
    from intent_classifier import (  # type: ignore
        DEFAULT_MODEL,
        INTENT_JSON_SCHEMA,
        INTENT_JSON_SCHEMA_NAME,
        _make_fallback,
        classify_intent,
        get_advanced_context,
        handle_intent,
        parse_llm_json,
        validate_intent_result,
    )
    from llm_client import generate_text_response, get_api_mode, get_llm_client  # type: ignore
    try:
        from src.shell_agent import ShellA2AAgent, ShellAgent  # type: ignore
    except ImportError:
        from shell_agent import ShellA2AAgent, ShellAgent  # type: ignore
    try:
        from src.tool_agent import ToolA2AAgent, ToolAgent  # type: ignore
    except ImportError:
        from tool_agent import ToolA2AAgent, ToolAgent  # type: ignore
    try:
        from Bonus.local_nlp import DualEngineRouter  # type: ignore
    except ImportError:
        DualEngineRouter = None  # type: ignore


class OrchestratorOutput(Protocol):
    """前端输出适配接口。"""

    def output_system(self, text: str, style: str = "") -> None:
        ...

    def output_workflow(self, text: str, state: str = "info") -> None:
        ...

    def output_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        ...

    async def stream_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        ...

    async def confirm_shell_command(
        self,
        *,
        command: str,
        risk_level: str,
        reason: str,
        details: list[Any] | None = None,
        confirm_label: str = "继续执行",
    ) -> bool:
        ...

    async def prompt_clarification(
        self,
        *,
        question: str,
        options: list[str],
        allow_manual: bool = True,
        manual_prompt: str = "请输入补充信息...",
    ) -> str | None:
        ...


ShellExecutor = Callable[[str, OrchestratorOutput], Awaitable[None]]
OrchestratorResult = dict[str, Any]

client = get_llm_client()


class ConsoleOutput:
    """命令行调试用输出适配器。"""

    def output_system(self, text: str, style: str = "") -> None:
        del style
        print(text)

    def output_workflow(self, text: str, state: str = "info") -> None:
        prefixes = {
            "running": "",
            "done": "",
            "info": "• ",
            "warn": "⚠ ",
            "error": "✖ ",
        }
        print(f"{prefixes.get(state, '• ')}{text}")

    def output_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        del markdown, language
        print(content)

    async def stream_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        del markdown, language
        print(content)

    async def confirm_shell_command(
        self,
        *,
        command: str,
        risk_level: str,
        reason: str,
        details: list[Any] | None = None,
        confirm_label: str = "继续执行",
    ) -> bool:
        print("\n" + "=" * 60)
        print(f"[风险确认] level={risk_level}")
        print(reason)
        print(f"命令: {command}")
        if details:
            for detail in details:
                plain = detail.plain if hasattr(detail, "plain") else str(detail)
                print(f"  {plain}")
        answer = input(f"{confirm_label}? [y/N]: ").strip().lower()
        print("=" * 60 + "\n")
        return answer in {"y", "yes"}

    async def prompt_clarification(
        self,
        *,
        question: str,
        options: list[str],
        allow_manual: bool = True,
        manual_prompt: str = "请输入补充信息...",
    ) -> str | None:
        print("\n" + "=" * 60)
        print(question)
        for index, option in enumerate(options, start=1):
            print(f"[{index}] {option}")
        manual_index = len(options) + 1 if allow_manual else None
        if manual_index is not None:
            print(f"[{manual_index}] 手动输入...")
        print("[0] 取消")

        while True:
            answer = input("请选择: ").strip()
            if answer == "0":
                print("=" * 60 + "\n")
                return None
            if answer.isdigit():
                choice = int(answer)
                if 1 <= choice <= len(options):
                    print("=" * 60 + "\n")
                    return options[choice - 1]
                if manual_index is not None and choice == manual_index:
                    manual_value = input(f"{manual_prompt}: ").strip()
                    print("=" * 60 + "\n")
                    return manual_value or None
            elif allow_manual and answer:
                print("=" * 60 + "\n")
                return answer


async def execute_command_async(command: str, ui: OrchestratorOutput | None = None) -> None:
    """异步执行 shell 命令，流式输出 stdout 和 stderr。"""

    def emit_system(text: str, style: str = "") -> None:
        if ui is not None:
            ui.output_system(text, style=style)
        else:
            print(text)

    emit_system(f"[直接执行] $ {command}", style="bold cyan")
    emit_system("-" * 40)

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def read_stream(stream, prefix: str = "", style: str = "") -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                emit_system(f"{prefix}{line.decode().rstrip()}", style=style)

        await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr, prefix="[stderr] ", style="bold red"),
        )

        await process.wait()
        emit_system("-" * 40)
        emit_system(f"[完成] 退出码: {process.returncode}", style="dim")

    except Exception as exc:
        emit_system(f"[执行错误] {exc}", style="bold red")


class OrchestratorAgent:
    """可直接对接前端的后端总控 Agent(核心逻辑)。"""

    MAX_SHORT_TERM_MESSAGES = 12
    MAX_DOCUMENT_CONTENT_CHARS = 4000

    def __init__(
        self,
        *,
        shell_executor: ShellExecutor | None = None,
        llm_client=None,
        model: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.shell_executor = shell_executor
        self.llm_client = llm_client or client
        self.model = model or DEFAULT_MODEL
        self.verbose = verbose

        self.shell_agent = ShellAgent(llm_client=self.llm_client, model=self.model)
        self.tool_agent = ToolAgent(llm_client=self.llm_client, model=self.model)
        self.memory_service = MemoryService()
        self.memory_agent = MemoryAgent(self.memory_service)
        self.chat_history: list[dict[str, str]] = []
        self.short_term_memory: deque[dict[str, str]] = deque(maxlen=self.MAX_SHORT_TERM_MESSAGES)
        self.last_file_path: str | None = None
        self.last_document_content: str | None = None
        self.last_document_summary: str | None = None

        self.agent_registry = AgentRegistry()
        self.agent_registry.register(ShellA2AAgent(self.shell_agent, shell_executor=self.shell_executor))
        self.agent_registry.register(ToolA2AAgent(self.tool_agent))
        self.agent_registry.register(self.memory_agent)

        self.dual_engine_router = (
            DualEngineRouter(
                cloud_llm_client=self.llm_client,
                cloud_model=self.model,
                max_retries=1,
                verbose=verbose,
            )
            if DualEngineRouter is not None
            else None
        )

    @staticmethod
    def _risk_label(risk_level: str) -> str:
        return {
            "low": "低风险",
            "medium": "中风险",
            "high": "高风险",
        }.get(risk_level, risk_level or "未知风险")

    async def _stream_response(
        self,
        ui: OrchestratorOutput,
        content: str,
        *,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        if not content:
            return
        await ui.stream_llm(content, markdown=markdown, language=language)

    @staticmethod
    async def _local_status_callback(ui: OrchestratorOutput, message: str) -> None:
        """本地节点不可用时的静默降级提示。"""
        ui.output_system(message, style="dim")

    def _emit_dual_engine_status(self, ui: OrchestratorOutput, result: OrchestratorResult) -> None:
        """输出双引擎路由的简洁状态提示。"""
        meta = result.get("_dual_engine")
        if not isinstance(meta, dict):
            return

        engine = meta.get("engine")
        reason = meta.get("fallback_reason")

        if engine == "local_fastpath":
            ui.output_system("已命中本地 Fast-Path", style="dim")
            return

        if engine == "cloud" and reason == "low_confidence":
            ui.output_system("本地置信度不足，已切换云端", style="dim")

    def _emit_dual_engine_metrics(self, ui: OrchestratorOutput, result: OrchestratorResult) -> None:
        """在 verbose 模式下输出双引擎路由耗时。"""
        if not self.verbose:
            return
        meta = result.get("_dual_engine")
        if not isinstance(meta, dict):
            return

        engine = meta.get("engine", "unknown")
        local_ms = meta.get("local_latency_ms")
        cloud_ms = meta.get("cloud_latency_ms")
        total_ms = meta.get("total_latency_ms")
        reason = meta.get("fallback_reason")

        parts = [f"route={engine}"]
        if local_ms is not None:
            parts.append(f"local={local_ms}ms")
        if cloud_ms is not None:
            parts.append(f"cloud={cloud_ms}ms")
        if total_ms is not None:
            parts.append(f"total={total_ms}ms")
        if reason:
            parts.append(f"fallback={reason}")
        ui.output_system("[dual-engine] " + " | ".join(parts), style="dim")

    @staticmethod
    def _build_clarification_markdown(result: OrchestratorResult) -> str:
        question = result.get("question", "请补充更多信息")
        options = result.get("options", [])
        if not options:
            return question

        option_lines = "\n".join(f"- {option}" for option in options)
        return f"{question}\n\n{option_lines}"

    @staticmethod
    def _clip_text(text: str, limit: int = 160) -> str:
        plain = " ".join(str(text or "").split())
        if len(plain) <= limit:
            return plain
        return plain[: limit - 3].rstrip() + "..."

    @staticmethod
    def _memory_category_label(category: str) -> str:
        return {
            "project": "项目",
            "preference": "偏好",
            "fact": "事实",
            "other": "其他",
        }.get(category, category or "其他")

    def _build_short_term_context(self) -> str:
        lines: list[str] = []
        for item in self.short_term_memory:
            role = "用户" if item.get("role") == "user" else "系统"
            content = self._clip_text(item.get("content", ""), limit=120)
            if content:
                lines.append(f"- {role}: {content}")
        return "\n".join(lines[-8:])

    def _build_long_term_context(self, user_input: str) -> tuple[str, list[dict[str, Any]]]:
        memories = self.memory_service.search_for_injection(user_input, top_k=3)
        if not memories:
            return "", []

        lines = []
        for memory in memories:
            if not isinstance(memory, dict):
                continue
            category = self._memory_category_label(str(memory.get("category") or "other"))
            content = str(memory.get("content") or "").strip()
            if content:
                lines.append(f"- [{category}] {content}")
        return "\n".join(lines), memories

    @staticmethod
    def _clip_block(text: str, limit: int = 800) -> str:
        value = str(text or "").strip()
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."

    def _build_document_artifact_context(self) -> dict[str, str]:
        extra: dict[str, str] = {}
        if self.last_file_path:
            extra["last_file_path"] = self.last_file_path
        if self.last_document_summary:
            extra["last_document_summary"] = self._clip_block(self.last_document_summary, 600)
        if self.last_document_content:
            extra["last_document_content"] = self._clip_block(self.last_document_content, 1800)
        return extra

    def _build_routing_extra_context(self, user_input: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
        extra_context: dict[str, str] = {}
        short_term = self._build_short_term_context()
        if short_term:
            extra_context["short_term_memory"] = short_term

        long_term_text, memory_hits = self._build_long_term_context(user_input)
        if long_term_text:
            extra_context["long_term_memory"] = long_term_text

        extra_context.update(self._build_document_artifact_context())

        return extra_context, memory_hits

    async def _dispatch_message(self, message: A2AMessage, ui: OrchestratorOutput) -> A2AMessage:
        return await self.agent_registry.dispatch(message, ui=ui)

    def _remember_exchange(self, user_input: str, assistant_text: str) -> None:
        if user_input.strip():
            self.chat_history.append({"role": "user", "content": user_input.strip()})
            self.short_term_memory.append({"role": "user", "content": self._clip_text(user_input, 160)})
        if assistant_text.strip():
            self.chat_history.append({"role": "assistant", "content": assistant_text.strip()})
            self.short_term_memory.append({"role": "assistant", "content": self._clip_text(assistant_text, 160)})

    def export_session_state(self) -> dict[str, Any]:
        """导出当前会话所需的总控 RAM / Artifact 状态。"""
        return {
            "chat_history": [dict(item) for item in self.chat_history],
            "short_term_memory": [dict(item) for item in self.short_term_memory],
            "last_file_path": self.last_file_path,
            "last_document_content": self.last_document_content,
            "last_document_summary": self.last_document_summary,
        }

    def restore_session_state(self, state: dict[str, Any] | None) -> None:
        """从持久化会话中恢复总控上下文状态。"""
        state = dict(state or {})
        self.chat_history = [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or ""),
            }
            for item in list(state.get("chat_history") or [])
            if isinstance(item, dict) and str(item.get("role") or "").strip() and str(item.get("content") or "").strip()
        ]

        restored_short_term = [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or ""),
            }
            for item in list(state.get("short_term_memory") or [])
            if isinstance(item, dict) and str(item.get("role") or "").strip() and str(item.get("content") or "").strip()
        ]
        self.short_term_memory.clear()
        if restored_short_term:
            self.short_term_memory.extend(restored_short_term[-self.MAX_SHORT_TERM_MESSAGES :])
        else:
            for item in self.chat_history[-self.MAX_SHORT_TERM_MESSAGES :]:
                self.short_term_memory.append(
                    {
                        "role": item["role"],
                        "content": self._clip_text(item["content"], 160),
                    }
                )

        self.last_file_path = str(state.get("last_file_path") or "").strip() or None
        self.last_document_content = str(state.get("last_document_content") or "").strip() or None
        self.last_document_summary = str(state.get("last_document_summary") or "").strip() or None

    def reset_session_state(self) -> None:
        """清空当前会话 RAM / Artifact。"""
        self.chat_history = []
        self.short_term_memory.clear()
        self.last_file_path = None
        self.last_document_content = None
        self.last_document_summary = None

    def _build_history_summary(self, result: OrchestratorResult) -> str:
        intent = str(result.get("intent") or "")
        if intent == "direct_answer":
            return str(result.get("reply") or "")
        if intent == "clarification":
            return str(result.get("question") or "")
        if intent == "memory_agent":
            return str(result.get("reply") or result.get("message") or "")
        if intent == "multi_agent":
            return str(result.get("reply") or result.get("task_description") or "")
        if intent == "shell_command":
            return f"执行了直接命令：{result.get('command')}"
        if intent == "shell_agent":
            shell_result = result.get("shell_agent_result") or {}
            if isinstance(shell_result, dict):
                return str(shell_result.get("summary") or shell_result.get("reason") or result.get("task_description") or "")
        if intent == "tool_agent":
            tool_result = result.get("tool_agent_result") or {}
            if isinstance(tool_result, dict):
                return str(tool_result.get("answer") or result.get("task_description") or "")
        return str(result.get("task_description") or result.get("reply") or "")

    def _extract_read_file_artifact(self, result: OrchestratorResult) -> tuple[str, str, str] | None:
        candidates: list[tuple[dict[str, Any], str]] = []

        if isinstance(result.get("tool_agent_result"), dict):
            candidates.append((result["tool_agent_result"], self._build_history_summary(result)))
        if isinstance(result.get("subtask_results"), list):
            for step in reversed(result["subtask_results"]):
                if not isinstance(step, dict):
                    continue
                step_result = step.get("result")
                summary = str(step.get("summary") or "")
                if isinstance(step_result, dict):
                    if isinstance(step_result.get("tool_agent_result"), dict):
                        candidates.append((step_result["tool_agent_result"], summary))
                    else:
                        candidates.append((step_result, summary))
        if str(result.get("intent") or "") == "tool_agent":
            candidates.append((result.get("tool_agent_result") or result, self._build_history_summary(result)))

        for tool_payload, summary in candidates:
            observations = tool_payload.get("observations")
            if not isinstance(observations, list):
                continue
            for observation in reversed(observations):
                if not isinstance(observation, dict):
                    continue
                if str(observation.get("tool_name") or "") != "read_file":
                    continue
                arguments = observation.get("arguments") or {}
                file_path = str(arguments.get("file_path") or "").strip()
                content = str(observation.get("result") or "")
                if not file_path or not content or content.startswith("读取失败"):
                    continue
                return file_path, content[: self.MAX_DOCUMENT_CONTENT_CHARS], summary or tool_payload.get("answer") or ""
        return None

    def _update_document_artifacts_from_result(self, result: OrchestratorResult) -> None:
        artifact = self._extract_read_file_artifact(result)
        if artifact is None:
            return
        file_path, content, summary = artifact
        self.last_file_path = file_path
        self.last_document_content = content
        self.last_document_summary = summary or self._clip_text(content, 200)

    def _format_memory_results(self, payload: dict[str, Any]) -> str:
        action = str(payload.get("action") or "")
        if action in {"save", "delete"}:
            return str(payload.get("message") or "记忆操作已完成。")

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return str(payload.get("message") or "没有找到相关记忆。")

        lines = [str(payload.get("message") or "找到以下记忆：")]
        for item in results[:5]:
            if not isinstance(item, dict):
                continue
            category = self._memory_category_label(str(item.get("category") or "other"))
            content = str(item.get("content") or "").strip()
            if content:
                lines.append(f"- [{category}] {content}")
        return "\n".join(lines)

    def _looks_like_document_coreference(self, user_input: str) -> bool:
        if not self.last_file_path:
            return False

        lowered = user_input.lower()
        pronoun_markers = [
            "前文",
            "上文",
            "刚才",
            "刚刚",
            "之前",
            "前面",
            "上述",
            "那个",
            "这个",
            "该",
            "其",
            "它",
            "这份",
            "这篇",
            "生成",
            "提炼",
            "重写",
            "构建",
            "基于它",
            "基于该",
            "帮我",
            "请你",
        ]
        document_markers = [
            "readme",
            "README",
            "文档",
            "文件",
            "内容",
            "大纲",
            "技术设计",
            "设计文档",
            "总结",
            "分析",
            "转成",
            "改成",
            "基于",
        ]
        basename = self.last_file_path.rsplit("/", 1)[-1].lower()
        has_pronoun = any(marker in user_input for marker in pronoun_markers)
        has_document_word = any(marker.lower() in lowered for marker in document_markers)
        return has_pronoun and (has_document_word or basename in lowered)

    def _build_document_followup_override(self, user_input: str, result: OrchestratorResult) -> OrchestratorResult | None:
        if not self._looks_like_document_coreference(user_input) or not self.last_file_path:
            return None

        if str(result.get("intent") or "") not in {"clarification", "direct_answer"}:
            return None

        summary = self._clip_block(self.last_document_summary or "", 240)
        excerpt = self._clip_block(self.last_document_content or "", 1200)
        context_passed = [f"最近文档路径：{self.last_file_path}"]
        if summary:
            context_passed.append(f"最近文档摘要：{summary}")
        if excerpt:
            context_passed.append(f"最近文档内容片段：{excerpt}")

        reasoning = str(result.get("reasoning") or "").strip()
        addon = f"；已根据最近一次成功读取的文档恢复指代，默认目标文件为 {self.last_file_path}。"
        return {
            "reasoning": (reasoning + addon).lstrip("；"),
            "confidence": max(float(result.get("confidence") or 0.0), 0.88),
            "intent": "tool_agent",
            "risk_level": "low",
            "task_description": f"读取文件 {self.last_file_path}，并基于其内容完成以下请求：{user_input}",
            "context_passed": context_passed,
            "reply": None,
            "question": None,
            "options": None,
            "subtasks": None,
        }

    @staticmethod
    def _replace_execution_placeholders(text: str, execution_context: dict[str, Any]) -> str:
        value = str(text or "")
        replacements = {
            "{{user_input}}": str(execution_context.get("user_input") or ""),
            "{{last_result}}": str(execution_context.get("last_result") or ""),
            "{{last_path}}": str(execution_context.get("last_path") or ""),
        }
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        return value.strip()

    @staticmethod
    def _extract_path_candidates(text: str) -> list[str]:
        if not text:
            return []

        candidates: list[str] = []
        patterns = [
            r">\s*([~/.\w-]+(?:/[\w.\-]+)*\.[A-Za-z0-9]+)",
            r"\b([~/.]*[\w-]+(?:/[\w.\-]+)+/?)",
            r"\b([A-Za-z0-9_.-]+\.(?:py|md|json|txt|csv|yaml|yml|toml|log|sh|js|ts|java|cpp|c|go|rs|html|css|xml))\b",
            r"\b([A-Za-z0-9_.-]+/)\b",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text):
                candidate = str(match).strip(" \t\n\r'\",;:()[]{}")
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def _extract_primary_path(self, result: OrchestratorResult) -> str | None:
        texts: list[str] = []
        texts.append(str(result.get("task_description") or ""))
        texts.append(str(result.get("reply") or ""))

        shell_result = result.get("shell_agent_result")
        if isinstance(shell_result, dict):
            for key in ("command", "summary", "reason"):
                texts.append(str(shell_result.get(key) or ""))
            original_plan = shell_result.get("original_plan")
            if isinstance(original_plan, dict):
                texts.append(str(original_plan.get("command") or ""))

        tool_result = result.get("tool_agent_result")
        if isinstance(tool_result, dict):
            texts.append(str(tool_result.get("answer") or ""))
            observations = tool_result.get("observations")
            if isinstance(observations, list):
                for observation in observations[:5]:
                    if isinstance(observation, dict):
                        texts.append(str(observation.get("result") or ""))

        for text in texts:
            candidates = self._extract_path_candidates(text)
            if candidates:
                return candidates[0]
        return None

    def _build_subtask_context(
        self,
        subtask: dict[str, Any],
        execution_context: dict[str, Any],
        global_context: list[str] | None = None,
    ) -> list[str]:
        context_passed = [str(item) for item in list(global_context or []) if str(item).strip()]
        context_passed.extend(
            [
            self._replace_execution_placeholders(str(item), execution_context)
            for item in list(subtask.get("context_passed") or [])
            if str(item).strip()
            ]
        )
        last_result = str(execution_context.get("last_result") or "").strip()
        last_path = str(execution_context.get("last_path") or "").strip()
        if last_result:
            context_passed.append(f"上一子任务结果摘要：{last_result}")
        if last_path:
            context_passed.append(f"上一子任务文件路径：{last_path}")
        return context_passed

    @staticmethod
    def _should_stop_subtasks(status: str) -> bool:
        lowered = (status or "").lower()
        stop_markers = ["error", "refused", "cancelled", "invalid", "failed", "clarification"]
        return any(marker in lowered for marker in stop_markers)

    async def _dispatch_memory_task(
        self,
        user_input: str,
        routed_task: OrchestratorResult,
        ui: OrchestratorOutput,
        *,
        execution_context: dict[str, Any] | None = None,
    ) -> OrchestratorResult:
        execution_context = execution_context or {"user_input": user_input}
        available, reason = self.memory_service.is_available()
        if not available:
            answer = f"Memory Agent 未启用：{reason}"
            ui.output_workflow("Memory Agent 不可用", state="warn")
            await ui.stream_llm(answer)
            return {"intent": "memory_agent", "status": "error", "reply": answer}

        action = str(routed_task.get("memory_action") or "").strip().lower()
        if not action:
            try:
                detected_action, detected_content = self.memory_service.detect_intent(user_input)
            except Exception as exc:
                answer = f"Memory Agent 检测失败：{exc}"
                ui.output_workflow("Memory Agent 解析失败", state="warn")
                await ui.stream_llm(answer)
                return {"intent": "memory_agent", "status": "error", "reply": answer}
            action = detected_action or "save"
            content = detected_content or str(routed_task.get("memory_content") or routed_task.get("task_description") or "")
        else:
            content = str(routed_task.get("memory_content") or routed_task.get("task_description") or "").strip()

        content = self._replace_execution_placeholders(content, execution_context)
        if not content and action != "list":
            fallback_content = str(execution_context.get("last_path") or execution_context.get("last_result") or "").strip()
            content = fallback_content

        ui.output_workflow("已将任务交给 Memory Agent", state="done")
        message = A2AMessage(
            sender="orchestrator",
            receiver="memory_agent",
            message_type=f"memory.{action}",
            payload={"action": action, "content": content},
        )
        reply = await self._dispatch_message(message, ui)
        if not reply.ok:
            answer = reply.error or "Memory Agent 执行失败。"
            ui.output_workflow("Memory Agent 执行失败", state="warn")
            await ui.stream_llm(answer)
            return {"intent": "memory_agent", "status": "error", "reply": answer}

        payload = reply.payload
        answer = self._format_memory_results(payload)
        await ui.stream_llm(answer)
        return {
            "intent": "memory_agent",
            "status": str(payload.get("status") or "handled"),
            "action": action,
            "task_description": str(routed_task.get("task_description") or ""),
            "reply": answer,
            "memory_result": payload,
        }

    async def _execute_subtasks(
        self,
        user_input: str,
        routed_task: OrchestratorResult,
        ui: OrchestratorOutput,
    ) -> OrchestratorResult:
        subtasks = [item for item in list(routed_task.get("subtasks") or []) if isinstance(item, dict)]
        if not subtasks:
            ui.output_workflow("多子任务请求缺少可执行的 subtasks", state="warn")
            await ui.stream_llm("当前请求被识别为多子任务，但没有生成有效的子任务清单。")
            return {
                **routed_task,
                "intent": "multi_agent",
                "status": "invalid_subtasks",
                "subtask_results": [],
            }

        ui.output_workflow(f"已拆分为 {len(subtasks)} 个 A2A 子任务", state="done")
        execution_context: dict[str, Any] = {
            "user_input": user_input,
            "last_result": "",
            "last_path": "",
            "step_results": [],
        }
        step_results: list[dict[str, Any]] = []

        for index, subtask in enumerate(subtasks, start=1):
            agent = str(subtask.get("agent") or "").strip()
            raw_task_description = str(subtask.get("task_description") or "").strip()
            task_description = self._replace_execution_placeholders(raw_task_description, execution_context)
            context_passed = self._build_subtask_context(
                subtask,
                execution_context,
                global_context=list(routed_task.get("context_passed") or []),
            )
            risk_level = str(subtask.get("risk_level") or "low")

            ui.output_workflow(f"子任务 {index}/{len(subtasks)}：{agent}", state="info")
            step_task: OrchestratorResult = {
                "intent": agent,
                "risk_level": risk_level,
                "task_description": task_description,
                "context_passed": context_passed,
            }
            subtask_input = task_description or user_input

            if agent == "shell_agent":
                step_result = await self._handle_shell_agent(subtask_input, step_task, ui)
            elif agent == "tool_agent":
                step_result = await self._handle_tool_agent(subtask_input, step_task, ui)
            elif agent == "memory_agent":
                step_task["memory_action"] = subtask.get("memory_action")
                step_task["memory_content"] = subtask.get("memory_content")
                step_result = await self._dispatch_memory_task(
                    subtask_input,
                    step_task,
                    ui,
                    execution_context=execution_context,
                )
            else:
                step_result = {
                    "intent": agent,
                    "status": "invalid_subtask_agent",
                    "reply": f"未知子任务 agent: {agent}",
                }
                ui.output_workflow(f"未知子任务类型：{agent}", state="warn")
                await ui.stream_llm(step_result["reply"])

            summary = self._build_history_summary(step_result)
            last_path = self._extract_primary_path(step_result) or execution_context.get("last_path") or ""
            execution_context["last_result"] = summary
            execution_context["last_path"] = last_path
            execution_context["step_results"].append(step_result)
            step_results.append(
                {
                    "index": index,
                    "agent": agent,
                    "status": step_result.get("status"),
                    "task_description": task_description,
                    "summary": summary,
                    "path": last_path,
                    "result": step_result,
                }
            )

            if self._should_stop_subtasks(str(step_result.get("status") or "")):
                ui.output_workflow(f"子任务 {index} 未成功完成，后续流程已停止", state="warn")
                return {
                    **routed_task,
                    "intent": "multi_agent",
                    "status": str(step_result.get("status") or "stopped"),
                    "subtask_results": step_results,
                    "reply": summary,
                }

        final_summary = "\n".join(
            f"{item['index']}. {item['agent']}：{self._clip_text(str(item['summary'] or ''), 120)}"
            for item in step_results
            if str(item.get("summary") or "").strip()
        )
        return {
            **routed_task,
            "intent": "multi_agent",
            "status": "completed",
            "subtask_results": step_results,
            "reply": final_summary,
        }

    async def _handle_shell_agent(
        self,
        user_input: str,
        routed_task: OrchestratorResult,
        ui: OrchestratorOutput,
    ) -> OrchestratorResult:
        """通过 A2A 注册表将 shell 任务派发给 ShellAgent。"""
        ui.output_workflow("已将任务交给 Shell Agent", state="done")
        task_description = routed_task.get("task_description")
        # if task_description:
        #     ui.output_workflow(f"任务说明：{task_description}", state="info")
        response = await self._dispatch_message(
            A2AMessage(
                sender="orchestrator",
                receiver="shell_agent",
                message_type="shell.task",
                payload={"mode": "task", "routed_task": routed_task, "user_input": user_input},
            ),
            ui,
        )
        if not response.ok:
            ui.output_workflow("Shell Agent 执行失败", state="warn")
            if response.error:
                await ui.stream_llm(response.error)
        shell_result = response.payload.get("result", {}) if response.ok else {"status": "error", "error": response.error}
        return {
            **routed_task,
            "shell_agent_result": shell_result,
            "status": shell_result.get("status", "handled"),
        }

    async def _handle_tool_agent(
        self,
        user_input: str,
        routed_task: OrchestratorResult,
        ui: OrchestratorOutput,
    ) -> OrchestratorResult:
        """通过 A2A 注册表将 tool 任务派发给 ToolAgent。"""
        ui.output_workflow("已将任务交给 Tool Agent", state="done")
        task_description = routed_task.get("task_description")
        # if task_description:
        #     ui.output_workflow(f"任务说明：{task_description}", state="info")
        response = await self._dispatch_message(
            A2AMessage(
                sender="orchestrator",
                receiver="tool_agent",
                message_type="tool.task",
                payload={"routed_task": routed_task, "user_input": user_input},
            ),
            ui,
        )
        if not response.ok:
            ui.output_workflow("Tool Agent 执行失败", state="warn")
            if response.error:
                await ui.stream_llm(response.error)
        tool_result = response.payload.get("result", {}) if response.ok else {"status": "error", "error": response.error}
        return {
            **routed_task,
            "tool_agent_result": tool_result,
            "status": tool_result.get("status", "handled"),
        }

    async def handle_input(self, user_input: str, ui: OrchestratorOutput) -> OrchestratorResult:
        """后端统一输入入口：接收用户输入并回写前端。"""
        user_input = user_input.strip()
        if not user_input:
            return {"status": "ignored", "input": user_input}

        if user_input.startswith("/"):
            command = user_input[1:].strip()
            if not command:
                ui.output_system("[提示] / 后面请输入要执行的命令", style="yellow")
                return {"status": "invalid_command", "input": user_input}

            ui.output_workflow("收到命令，准备进行审查", state="running")
            response = await self._dispatch_message(
                A2AMessage(
                    sender="orchestrator",
                    receiver="shell_agent",
                    message_type="shell.direct_command",
                    payload={"mode": "direct_command", "command": command},
                ),
                ui,
            )
            if not response.ok and response.error:
                ui.output_workflow("Shell Agent 执行失败", state="warn")
                await ui.stream_llm(response.error)
            direct_result = response.payload.get("result", {}) if response.ok else {"status": "error", "error": response.error}
            self._remember_exchange(user_input, f"执行了直接命令：{command}")
            return {
                "status": direct_result.get("status", "shell_command_handled"),
                "intent": "shell_command",
                "command": command,
                "shell_agent_result": direct_result,
            }

        extra_context, memory_hits = self._build_routing_extra_context(user_input)

        ui.output_workflow("正在分析用户意图...", state="running")
        if memory_hits:
            ui.output_system(f"已注入 {len(memory_hits)} 条长期记忆", style="dim")
        if self._looks_like_document_coreference(user_input) and self.last_file_path:
            ui.output_system(f"已恢复最近文档指代：{self.last_file_path}", style="dim")
        if self.dual_engine_router is not None:
            result = await self.dual_engine_router.route(
                user_input,
                status_callback=lambda message: self._local_status_callback(ui, message),
                extra_context=extra_context,
            )
        else:
            result = await asyncio.to_thread(
                handle_intent,
                user_input,
                self.llm_client,
                self.model,
                1,
                self.verbose,
                extra_context,
            )
        override = self._build_document_followup_override(user_input, result)
        if override is not None:
            result = override
        self._emit_dual_engine_status(ui, result)
        self._emit_dual_engine_metrics(ui, result)

        intent = result.get("intent")
        risk_level = result.get("risk_level", "low")
        if intent == "direct_answer":
            ui.output_workflow(
                f"已识别为直接回答任务（{self._risk_label(risk_level)}）",
                state="done",
            )
            ui.output_workflow("正在整理回复...", state="running")
            reply = result.get("reply", "")
            if reply:
                await self._stream_response(ui, reply)
            else:
                reply = await asyncio.to_thread(
                    generate_text_response,
                    "你是一个有用的 AI 助手，请直接回答用户的问题。",
                    user_input,
                    llm_client=self.llm_client,
                    model=self.model,
                    temperature=0.7,
                )
                await self._stream_response(ui, reply)
            self._remember_exchange(user_input, reply)
        elif intent == "clarification":
            ui.output_workflow("当前信息不足，需要进一步澄清", state="warn")
            clarification_text = self._build_clarification_markdown(result)
            await self._stream_response(
                ui,
                clarification_text,
                markdown=True,
            )
            self._remember_exchange(user_input, clarification_text)
        elif result.get("subtasks"):
            ui.output_workflow(
                f"已识别为复合任务（{self._risk_label(risk_level)}）",
                state="done",
            )
            multi_result = await self._execute_subtasks(user_input, result, ui)
            self._update_document_artifacts_from_result(multi_result)
            self._remember_exchange(user_input, self._build_history_summary(multi_result))
            return multi_result
        elif intent == "memory_agent":
            ui.output_workflow(
                f"已识别为 Memory 任务（{self._risk_label(risk_level)}）",
                state="done",
            )
            memory_result = await self._dispatch_memory_task(user_input, result, ui)
            self._remember_exchange(user_input, self._build_history_summary(memory_result))
            return memory_result
        elif intent == "shell_agent":
            ui.output_workflow(
                f"已识别为 Shell 任务（{self._risk_label(risk_level)}）",
                state="done",
            )
            shell_result = await self._handle_shell_agent(user_input, result, ui)
            self._update_document_artifacts_from_result(shell_result)
            self._remember_exchange(user_input, self._build_history_summary(shell_result))
            return shell_result
        elif intent == "tool_agent":
            ui.output_workflow(
                f"已识别为 Tool Agent 任务（{self._risk_label(risk_level)}）",
                state="done",
            )
            tool_result = await self._handle_tool_agent(user_input, result, ui)
            self._update_document_artifacts_from_result(tool_result)
            self._remember_exchange(user_input, self._build_history_summary(tool_result))
            return tool_result

        return result

# 调试用：总控Agent前端接口适配器和命令行主循环
def build_controller(
    *,
    shell_executor: ShellExecutor | None = None,
    llm_client=None,
    model: str | None = None,
    verbose: bool = False,
) -> Callable[[str, OrchestratorOutput], Awaitable[OrchestratorResult]]:
    """构造符合前端 input_handler 签名的 controller。"""
    agent = OrchestratorAgent(
        shell_executor=shell_executor,
        llm_client=llm_client,
        model=model,
        verbose=verbose,
    )

    async def controller(user_input: str, ui: OrchestratorOutput) -> OrchestratorResult:
        return await agent.handle_input(user_input, ui)

    setattr(controller, "agent", agent)
    return controller


_default_agent = OrchestratorAgent(verbose=False)


async def main_controller(user_input: str, ui: OrchestratorOutput) -> OrchestratorResult:
    """默认总控入口，可直接作为前端 command_handler 使用。"""
    return await _default_agent.handle_input(user_input, ui)


async def orchestrator_loop() -> None:
    """命令行模式下的总控 Agent 主循环。"""
    io = ConsoleOutput()
    agent = OrchestratorAgent(shell_executor=execute_command_async, verbose=True)

    print("=" * 60)
    print("  Task 2.3 - 总控 Agent (Orchestrator)")
    print("  输入自然语言指令，或用 / 开头直接执行命令")
    print("  输入 exit 或 quit 退出")
    print("=" * 60)
    print(f"[LLM 接口模式] {get_api_mode(client)}")

    ctx = get_advanced_context()
    print(f"\n[环境上下文]")
    print(f"  操作系统: {ctx['os']}")
    print(f"  Shell: {ctx['shell']}")
    print(f"  工作目录: {ctx['pwd']}")
    print(f"  文件数量: {len(ctx['files'].splitlines())} 项")
    print(f"  Git 状态: {ctx['git_status'][:50]}")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break

        await agent.handle_input(user_input, io)


def classify_without_context(user_input: str) -> OrchestratorResult:
    """不注入环境上下文的意图分类，用于报告对比。"""
    bare_prompt = """你是一个多 Agent 命令行系统的\"总控大脑\"。你的唯一任务是分析用户的自然语言输入，判断其真实意图，并输出严格的结构化 JSON 进行路由。

支持的 intent：
1. \"shell_agent\": 确定性的本地系统操作。
2. \"tool_agent\": 需要认知能力、语义分析或调用外部资源的复杂任务。
3. \"memory_agent\": 显式的长期记忆读写请求。
4. \"multi_agent\": 一个请求包含多个顺序子任务。
5. \"direct_answer\": 纯知识问答或闲聊，无需系统执行任何操作。
6. \"clarification\": 指令模糊或存在安全隐患，必须追问。

输出字段顺序：
reasoning -> confidence -> intent -> risk_level -> 专属字段

如果 intent 是 \"shell_agent\"、\"tool_agent\" 或 \"memory_agent\"：
{
    \"reasoning\": \"...\",
    \"confidence\": 0.95,
    \"intent\": \"shell_agent\" | \"tool_agent\" | \"memory_agent\",
    \"risk_level\": \"low\" | \"medium\" | \"high\",
    \"task_description\": \"给下级 Agent 的清晰任务描述\",
    \"context_passed\": [\"提取出的关键参数\"],
    \"reply\": null,
    \"question\": null,
    \"options\": null,
    \"subtasks\": null
}

如果 intent 是 \"multi_agent\"：
{
    \"reasoning\": \"...\",
    \"confidence\": 0.9,
    \"intent\": \"multi_agent\",
    \"risk_level\": \"low\" | \"medium\" | \"high\",
    \"task_description\": \"整体任务摘要\",
    \"context_passed\": [\"全局上下文\"],
    \"reply\": null,
    \"question\": null,
    \"options\": null,
    \"subtasks\": [
        {
            \"agent\": \"shell_agent\" | \"tool_agent\" | \"memory_agent\",
            \"task_description\": \"子任务描述\",
            \"context_passed\": [\"参数\"],
            \"risk_level\": \"low\" | \"medium\" | \"high\",
            \"memory_action\": null | \"save\" | \"search\" | \"delete\" | \"list\",
            \"memory_content\": null | \"支持 {{last_result}} / {{last_path}}\"
        }
    ]
}

如果 intent 是 \"direct_answer\"：
{
    \"reasoning\": \"...\",
    \"confidence\": 1.0,
    \"intent\": \"direct_answer\",
    \"risk_level\": \"low\",
    \"task_description\": null,
    \"context_passed\": null,
    \"reply\": \"完整回答\",
    \"question\": null,
    \"options\": null,
    \"subtasks\": null
}

如果 intent 是 \"clarification\"：
{
    \"reasoning\": \"...\",
    \"confidence\": 0.4,
    \"intent\": \"clarification\",
    \"risk_level\": \"low\",
    \"task_description\": null,
    \"context_passed\": null,
    \"reply\": null,
    \"question\": \"追问问题\",
    \"options\": [\"候选项1\", \"候选项2\", \"其他\"],
    \"subtasks\": null
}"""

    try:
        raw_text = generate_text_response(
            bare_prompt,
            user_input,
            llm_client=client,
            model=DEFAULT_MODEL,
            temperature=0.1,
            json_schema=INTENT_JSON_SCHEMA,
            json_schema_name=INTENT_JSON_SCHEMA_NAME,
        )
        parsed = parse_llm_json(raw_text)
        if parsed:
            valid, _ = validate_intent_result(parsed)
            if valid:
                return parsed
        return _make_fallback(user_input, raw_text)
    except Exception as exc:
        return _make_fallback(user_input, str(exc))


def compare_context_effect(user_input: str) -> tuple[OrchestratorResult, OrchestratorResult]:
    """对比有/无上下文注入时 LLM 输出的差异。"""
    print(f"\n{'=' * 60}")
    print(f"  对比测试: \"{user_input}\"")
    print(f"{'=' * 60}")

    print(f"\n--- 无上下文注入 ---")
    result_bare = classify_without_context(user_input)
    print(f"  intent: {result_bare.get('intent')}")
    print(f"  confidence: {result_bare.get('confidence')}")
    print(f"  reasoning: {result_bare.get('reasoning', 'N/A')[:120]}")
    print(f"  task_description: {result_bare.get('task_description', 'N/A')}")

    print(f"\n--- 有上下文注入 ---")
    result_ctx = handle_intent(user_input, client, DEFAULT_MODEL, 1, False)
    print(f"  intent: {result_ctx.get('intent')}")
    print(f"  confidence: {result_ctx.get('confidence')}")
    print(f"  reasoning: {result_ctx.get('reasoning', 'N/A')[:120]}")
    print(f"  task_description: {result_ctx.get('task_description', 'N/A')}")

    print(f"\n{'=' * 60}\n")
    return result_bare, result_ctx


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_cases = [
            "帮我看看这个项目有哪些 Python 文件",
            "帮我删除那个文件",
            "这个项目用了什么框架",
        ]
        for case in compare_cases:
            compare_context_effect(case)
    else:
        asyncio.run(orchestrator_loop())
