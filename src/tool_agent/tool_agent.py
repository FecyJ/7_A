"""可接入 orchestrator 的正式 Tool Agent。"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from jsonschema import ValidationError, validate
from openai import OpenAI
from rich.text import Text

try:
    from ..orchestrator.llm_client import DEFAULT_MODEL, generate_text_response, get_llm_client
    from ..orchestrator.intent_classifier import _detect_temporal_query
except ImportError:
    try:
        from src.orchestrator.llm_client import DEFAULT_MODEL, generate_text_response, get_llm_client  # type: ignore
        from src.orchestrator.intent_classifier import _detect_temporal_query  # type: ignore
    except ImportError:
        from orchestrator.llm_client import DEFAULT_MODEL, generate_text_response, get_llm_client  # type: ignore
        from orchestrator.intent_classifier import _detect_temporal_query  # type: ignore

try:
    from .mcp_agent_client import MCPClient
    from .security import ToolPermissionDecision, check_tool_permission, normalize_tool_arguments
except ImportError:
    from mcp_agent_client import MCPClient  # type: ignore
    from security import ToolPermissionDecision, check_tool_permission, normalize_tool_arguments  # type: ignore

if TYPE_CHECKING:
    from ..orchestrator.orchestrator import OrchestratorOutput


ToolPlan = dict[str, Any]

TOOL_AGENT_SCHEMA_NAME = "tool_agent_plan"
TOOL_AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["intent", "reason", "tool_name", "arguments", "question", "options", "answer"],
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["call_tool", "respond", "ask_clarification", "refuse"],
        },
        "reason": {"type": "string"},
        "tool_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "arguments": {"anyOf": [{"type": "object", "additionalProperties": True}, {"type": "null"}]},
        "question": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "options": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "answer": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
}


@dataclass
class ToolObservation:
    tool_name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool = False

    def to_prompt_dict(self) -> dict[str, Any]:
        result = self.result
        if len(result) > 2000:
            result = result[:2000].rstrip() + "..."
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "is_error": self.is_error,
            "result": result,
        }


class ToolAgent:
    """负责执行 tool_agent 路由下发的复杂工具任务。"""

    MAX_TOOL_ROUNDS = 4
    MAX_CLARIFICATION_ROUNDS = 2
    DEFAULT_TIMEZONE = "Asia/Shanghai"
    CURRENCY_ALIASES = {
        "usd": "USD",
        "美元": "USD",
        "美金": "USD",
        "us dollar": "USD",
        "cny": "CNY",
        "rmb": "CNY",
        "人民币": "CNY",
        "renminbi": "CNY",
        "eur": "EUR",
        "欧元": "EUR",
        "euro": "EUR",
        "jpy": "JPY",
        "日元": "JPY",
        "yen": "JPY",
        "gbp": "GBP",
        "英镑": "GBP",
        "pound": "GBP",
        "hkd": "HKD",
        "港币": "HKD",
        "港元": "HKD",
        "cad": "CAD",
        "加元": "CAD",
        "aud": "AUD",
        "澳元": "AUD",
        "sgd": "SGD",
        "新加坡元": "SGD",
        "krw": "KRW",
        "韩元": "KRW",
    }

    SYSTEM_PROMPT = """
你是多 Agent CLI 系统中的 Tool Agent。
你的职责是：根据总控下发的任务，在给定工具集中选择合适工具逐步完成任务；若信息不足则澄清；若已有足够证据则直接回答。

你必须且只能输出一个 JSON 对象，不要输出 Markdown、解释性前缀或代码块。
输出必须符合以下 schema：
{json_schema}

你可以输出四种 intent：
1. call_tool
   - 当你需要调用一个工具来获取信息或执行低风险工具动作时使用。
   - 一次只能选择一个工具。
   - tool_name 必须来自可用工具列表。
   - arguments 必须是 JSON 对象，并符合该工具的 input_schema。
2. respond
   - 当你已经拥有足够信息，可直接向用户输出最终结果时使用。
   - answer 必须为完整回答。
3. ask_clarification
   - 当目标文件、参数、对象不明确时使用。
   - question 说明还缺什么，options 可以给候选项。
4. refuse
   - 当任务超出当前工具能力范围，或你判断不适合继续时使用。

重要规则：
- 不要臆造文件路径、工具名或参数。
- arguments 的键名必须与 input_schema 中的字段名完全一致；例如 read_file 必须使用 file_path。
- read_file 仅允许当前工作目录内的相对路径；如果文件不明确，应先 ask_clarification。
- 对“时间/日期/星期、天气、新闻、汇率、最新/实时数据”等时效性问题，禁止凭模型记忆直接回答。
- 时间/日期优先使用 get_live_time / get_current_time；天气优先使用 get_weather；新闻优先使用 get_news；汇率优先使用 get_exchange_rate。
- 如果使用 http_request，必须提供明确的 http/https URL；禁止尝试 localhost、内网地址或虚构私有接口。
- 如果前一轮工具结果已经足够回答，优先 respond，不要无意义重复调用工具。
- 如果用户想要实时天气、联网搜索、外部 API 等，而当前工具不支持，应 refuse 并说明原因。

【总控路由结果】
- 原始用户输入: {user_input}
- task_description: {task_description}
- context_passed: {context_passed}
- orchestrator_risk_level: {orchestrator_risk_level}

【当前环境】
- cwd: {cwd}
- files_preview:
{files_preview}

【可用工具】
{available_tools}

【历史观察】
{observations}
""".strip()

    SYNTHESIS_PROMPT = """
你是一个工具结果整理助手。请根据用户问题和工具返回结果，给出简洁、直接、可靠的中文回答。
如果工具结果显示失败或信息不足，请明确说明。
""".strip()

    def __init__(
        self,
        *,
        llm_client: OpenAI | None = None,
        model: str | None = None,
        mcp_client: MCPClient | None = None,
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.model = model or DEFAULT_MODEL
        base_client = mcp_client or MCPClient()
        self._mcp_server_script_path = Path(base_client.server_script_path).resolve()
        self.mcp_client = base_client

    @staticmethod
    def _emit_workflow(ui: "OrchestratorOutput", message: str, state: str = "info") -> None:
        ui.output_workflow(message, state=state)

    @staticmethod
    def _cwd_preview(limit: int = 20) -> str:
        try:
            entries = sorted(Path.cwd().iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            return "无法获取文件列表"
        preview = []
        for entry in entries[:limit]:
            preview.append(entry.name + ("/" if entry.is_dir() else ""))
        return "\n".join(preview) if preview else "（当前目录为空）"

    @staticmethod
    def _list_cwd_entries() -> list[str]:
        try:
            return sorted(item.name + ("/" if item.is_dir() else "") for item in Path.cwd().iterdir())
        except OSError:
            return []

    @staticmethod
    def _looks_like_semantic_file_task(*texts: str) -> bool:
        combined = "\n".join(text for text in texts if text).lower()
        if not combined:
            return False
        file_markers = ["读取", "read", "文件", "文档", "代码", "readme", ".md", ".py", ".json", ".txt", ".log"]
        semantic_markers = ["总结", "概括", "分析", "解释", "翻译", "review", "summarize", "analyze", "explain"]
        return any(marker in combined for marker in file_markers) and any(
            marker in combined for marker in semantic_markers
        )

    @staticmethod
    def _detect_temporal_category(*texts: str) -> str | None:
        info = _detect_temporal_query(*texts)
        return str(info.get("category")) if info else None

    @staticmethod
    def _looks_like_realtime_time_task(*texts: str) -> bool:
        return ToolAgent._detect_temporal_category(*texts) == "time"

    @staticmethod
    def _wants_online_lookup(*texts: str) -> bool:
        combined = "\n".join(text for text in texts if text).lower()
        if not combined:
            return False
        markers = ["联网", "网络", "在线", "上网", "api", "web", "internet", "online"]
        return any(marker in combined for marker in markers)

    @staticmethod
    def _extract_timezone(*texts: str) -> str:
        combined = "\n".join(text for text in texts if text)
        lowered = combined.lower()
        if "utc" in lowered or "gmt" in lowered:
            return "UTC"
        if any(marker in combined for marker in ["北京", "上海", "中国", "北京时间"]):
            return "Asia/Shanghai"
        return ToolAgent.DEFAULT_TIMEZONE

    @staticmethod
    def _clean_extracted_fragment(value: str) -> str:
        cleaned = str(value or "").strip()
        cleaned = re.sub(
            r"^(帮我|请|查一下|查查|查询|看看|告诉我|获取|联网查找|联网搜索|搜索|检索)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(现在|当前|实时|最新|今天|明天|后天|一下|怎么样|如何|多少|是多少|情况|信息)$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.strip(" ，。,:：?？!！")
        return cleaned.strip()

    @classmethod
    def _normalize_weather_location_candidate(cls, value: str) -> str | None:
        """清洗天气查询里抽取出的地点候选，避免把整句或时间词误当地点。"""
        cleaned = str(value or "").strip()
        if not cleaned:
            return None

        prefix_pattern = (
            r"^(?:请告诉我|告诉我|帮我|请问|请你|请|麻烦你|麻烦|给出|给我|查一下|查查|查询一下|查询|"
            r"看看|看下|获取一下|获取|联网查找|联网搜索|搜索|检索|给我查下|给我查查|给我看看)\s*"
        )
        temporal_pattern = r"(?:现在|当前|今天|明天|后天|实时|最新|此刻|目前)"

        for _ in range(6):
            previous = cleaned
            cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(rf"^{temporal_pattern}\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(rf"\s*{temporal_pattern}$", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*的\s*$", "", cleaned)
            cleaned = re.sub(r"\s*(?:那边|那里|这边|这里|本地|当地)\s*$", "", cleaned)
            cleaned = cleaned.strip(" ，。,:：?？!！")
            if cleaned == previous:
                break

        cleaned = cleaned.strip()
        if not cleaned:
            return None

        normalized = cleaned.lower().replace(" ", "")
        invalid_values = {
            "现在",
            "当前",
            "今天",
            "明天",
            "后天",
            "实时",
            "最新",
            "现在的",
            "当前的",
            "今天的",
            "天气",
            "气温",
            "温度",
            "本地",
            "当地",
            "这里",
            "这边",
            "那里",
            "那边",
            "weather",
            "目标地点",
            "目标城市",
            "目标地区",
            "目标区域",
            "指定地点",
            "指定城市",
            "指定地区",
            "城市",
            "地区",
            "区域",
            "地点",
        }
        if normalized in invalid_values:
            return None

        return cleaned

    @classmethod
    def _extract_weather_location(cls, *texts: str) -> str | None:
        patterns = [
            r"(.{1,20}?)(?:今天|明天|后天|当前|现在|实时)?(?:天气|气温|温度)",
            r"(?:weather\s+in\s+)([A-Za-z][A-Za-z\s-]{1,40})",
            r"([A-Za-z][A-Za-z\s-]{1,40})\s+weather",
        ]
        for text in texts:
            if not text:
                continue
            for pattern in patterns:
                for match in re.findall(pattern, text, flags=re.IGNORECASE):
                    candidate = cls._normalize_weather_location_candidate(match)
                    if (
                        candidate
                        and len(candidate) >= 2
                        and candidate.lower() not in {"今天", "现在", "当前", "实时", "weather", "天气", "气温", "温度", "今天天气"}
                    ):
                        return candidate
        return None

    @classmethod
    def _extract_news_topic(cls, *texts: str) -> str | None:
        combined = "\n".join(text for text in texts if text)
        lowered = combined.lower()
        generic_markers = {"新闻", "最新新闻", "今日新闻", "今天新闻", "头条新闻", "新闻头条", "latest news", "top news", "headlines"}
        normalized = cls._clean_extracted_fragment(combined).lower().replace(" ", "")
        if normalized in {marker.replace(" ", "") for marker in generic_markers}:
            return None

        patterns = [
            r"(?:关于|有关)(.+?)(?:的)?(?:最新|今天|今日|实时)?新闻",
            r"(.+?)(?:的)?(?:最新|今天|今日|实时)?新闻",
            r"(?:latest|recent|today'?s)\s+news\s+about\s+(.+)",
            r"news\s+about\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, combined, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = cls._clean_extracted_fragment(match.group(1))
            if candidate and candidate.lower() not in {"新闻", "news", "头条", "headline"}:
                return candidate
        return None

    @classmethod
    def _extract_currency_pair(cls, *texts: str) -> tuple[str, str] | None:
        combined = "\n".join(text for text in texts if text)
        lowered = combined.lower()

        direct_match = re.search(r"\b([A-Z]{3})\s*(?:/|兑|对)\s*([A-Z]{3})\b", combined)
        if direct_match:
            return direct_match.group(1).upper(), direct_match.group(2).upper()

        discovered: list[str] = []
        for alias, code in cls.CURRENCY_ALIASES.items():
            if alias.lower() in lowered and code not in discovered:
                discovered.append(code)

        if len(discovered) >= 2:
            return discovered[0], discovered[1]
        if len(discovered) == 1 and discovered[0] != "CNY":
            return discovered[0], "CNY"
        return None

    @staticmethod
    def _weekday_to_chinese(value: str | None) -> str | None:
        mapping = {
            "monday": "星期一",
            "tuesday": "星期二",
            "wednesday": "星期三",
            "thursday": "星期四",
            "friday": "星期五",
            "saturday": "星期六",
            "sunday": "星期日",
        }
        if not value:
            return None
        return mapping.get(str(value).strip().lower(), value)

    @staticmethod
    def _extract_explicit_file_candidate(*texts: str) -> str | None:
        patterns = [
            r"`([^`]+\.[A-Za-z0-9]+)`",
            r'"([^"]+\.[A-Za-z0-9]+)"',
            r"'([^']+\.[A-Za-z0-9]+)'",
            r"\b([A-Za-z0-9_.-]+\.[A-Za-z0-9]+)\b",
        ]
        for text in texts:
            if not text:
                continue
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    return str(matches[0]).strip()
        return None

    @staticmethod
    def _resolve_explicit_file_candidate(candidate: str | None) -> str | None:
        if not candidate:
            return None
        cwd = Path.cwd()
        direct = cwd / candidate
        if direct.exists() and direct.is_file():
            return candidate
        lowered = candidate.lower()
        matches = [item.name for item in cwd.iterdir() if item.is_file() and item.name.lower() == lowered]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _parse_json(raw_text: str) -> ToolPlan | None:
        raw_text = raw_text.strip()
        if not raw_text:
            return None
        decoder = json.JSONDecoder()
        candidates = (
            raw_text,
            raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip(),
        )
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                try:
                    parsed, _ = decoder.raw_decode(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

        match = re.search(r"\{", raw_text)
        if not match:
            return None
        try:
            parsed, _ = decoder.raw_decode(raw_text[match.start():])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_plan(plan: ToolPlan | None) -> ToolPlan | None:
        if plan is None:
            return None
        normalized = dict(plan)
        normalized.setdefault("reason", "")
        normalized.setdefault("tool_name", None)
        normalized.setdefault("arguments", None)
        normalized.setdefault("question", None)
        normalized.setdefault("options", None)
        normalized.setdefault("answer", None)
        return normalized

    @staticmethod
    def _plan_schema(tools: list[Any]) -> dict[str, Any]:
        schema = copy.deepcopy(TOOL_AGENT_OUTPUT_SCHEMA)
        tool_names = [tool.name for tool in tools]
        schema["properties"]["tool_name"] = {
            "anyOf": [
                {"type": "string", "enum": tool_names},
                {"type": "null"},
            ]
        }
        return schema

    @staticmethod
    def _validate_plan(plan: ToolPlan, schema: dict[str, Any]) -> tuple[bool, str]:
        try:
            validate(instance=plan, schema=schema)
        except ValidationError as exc:
            return False, exc.message

        intent = plan.get("intent")
        if intent == "call_tool":
            if not isinstance(plan.get("tool_name"), str) or not plan["tool_name"].strip():
                return False, "call_tool 必须提供非空 tool_name"
            if plan.get("arguments") is not None and not isinstance(plan.get("arguments"), dict):
                return False, "call_tool 的 arguments 必须为对象或 null"
        elif intent == "respond":
            if not isinstance(plan.get("answer"), str) or not plan["answer"].strip():
                return False, "respond 必须提供非空 answer"
        elif intent == "ask_clarification":
            if not isinstance(plan.get("question"), str) or not plan["question"].strip():
                return False, "ask_clarification 必须提供非空 question"
        return True, ""

    @staticmethod
    def _fallback_plan(user_input: str, raw_text: str = "") -> ToolPlan:
        cleaned = raw_text.strip()
        return {
            "intent": "ask_clarification",
            "reason": f"Tool Agent 未能稳定生成结构化动作。原始输出: {cleaned[:160]}" if cleaned else "Tool Agent 未能稳定生成结构化动作。",
            "tool_name": None,
            "arguments": None,
            "question": f"我理解你想处理这个任务：{user_input}\n\n但我还不能稳定确定下一步工具动作，请补充更明确的目标、文件路径或参数。",
            "options": None,
            "answer": None,
        }

    @staticmethod
    def _render_tool_invocation(tool_name: str, arguments: dict[str, Any] | None) -> str:
        arguments = normalize_tool_arguments(arguments)
        if not arguments:
            return f"{tool_name}()"
        payload = ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in arguments.items())
        return f"{tool_name}({payload})"

    @staticmethod
    def _shorten_text(text: str, max_length: int = 80) -> str:
        one_line = " ".join(str(text).split())
        if len(one_line) <= max_length:
            return one_line
        return one_line[: max_length - 3].rstrip() + "..."

    def _tool_prompt_payload(self, tools: list[Any]) -> str:
        payload = [self.mcp_client.tool_to_prompt_dict(tool) for tool in tools]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _observation_payload(observations: list[ToolObservation]) -> str:
        if not observations:
            return "[]"
        payload = [observation.to_prompt_dict() for observation in observations]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _plan_action(
        self,
        user_input: str,
        routed_task: ToolPlan,
        tools: list[Any],
        observations: list[ToolObservation],
    ) -> ToolPlan:
        plan_schema = self._plan_schema(tools)
        system_prompt = self.SYSTEM_PROMPT.format(
            json_schema=json.dumps(plan_schema, ensure_ascii=False, indent=2),
            user_input=user_input,
            task_description=routed_task.get("task_description", user_input),
            context_passed=json.dumps(routed_task.get("context_passed", []), ensure_ascii=False),
            orchestrator_risk_level=routed_task.get("risk_level", "low"),
            cwd=os.getcwd(),
            files_preview=self._cwd_preview(),
            available_tools=self._tool_prompt_payload(tools),
            observations=self._observation_payload(observations),
        )
        user_prompt = routed_task.get("task_description", user_input)

        try:
            raw_text = generate_text_response(
                system_prompt,
                user_prompt,
                llm_client=self.llm_client,
                model=self.model,
                temperature=0.1,
                json_schema=plan_schema,
                json_schema_name=TOOL_AGENT_SCHEMA_NAME,
                json_schema_strict=False,
            )
        except Exception as exc:
            return {
                "intent": "refuse",
                "reason": f"调用 Tool Agent 所需 LLM 失败：{exc}",
                "tool_name": None,
                "arguments": None,
                "question": None,
                "options": None,
                "answer": None,
            }

        plan = self._normalize_plan(self._parse_json(raw_text))
        if plan is None:
            return self._fallback_plan(user_input, raw_text)
        valid, error_message = self._validate_plan(plan, plan_schema)
        if not valid:
            return self._fallback_plan(user_input, f"{raw_text}\n[SchemaError] {error_message}")
        return plan

    @staticmethod
    def _augment_user_input(user_input: str, question: str, answer: str) -> str:
        return f"{user_input}\n\n补充信息：针对“{question}”，用户回答：{answer}"

    @staticmethod
    def _augment_routed_task(routed_task: ToolPlan, answer: str) -> ToolPlan:
        enriched = dict(routed_task)
        context_passed = list(enriched.get("context_passed") or [])
        context_passed.append(answer)
        enriched["context_passed"] = context_passed
        return enriched

    @staticmethod
    def _merge_options(*option_groups: list[str] | None) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in option_groups:
            if not group:
                continue
            for option in group:
                item = str(option).strip()
                if not item or item in seen:
                    continue
                seen.add(item)
                merged.append(item)
        return merged

    async def _prompt_for_clarification(self, plan: ToolPlan, ui: "OrchestratorOutput") -> str | None:
        question = plan.get("question") or "请补充更多信息。"
        llm_options = plan.get("options") if isinstance(plan.get("options"), list) else None
        cwd_options = [entry for entry in self._list_cwd_entries() if not entry.endswith("/")][:6]
        options = self._merge_options(llm_options, cwd_options)
        return await ui.prompt_clarification(
            question=question,
            options=options,
            allow_manual=True,
            manual_prompt="请输入更明确的文件名、路径或参数...",
        )

    async def _confirm_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        decision: ToolPermissionDecision,
        ui: "OrchestratorOutput",
    ) -> bool:
        if decision.status == "allow":
            return True
        if decision.status == "deny":
            return False
        return await ui.confirm_shell_command(
            command=self._render_tool_invocation(tool_name, arguments),
            risk_level=decision.risk_level,
            reason="是否调用以下工具？",
            details=[Text(decision.reason)],
            confirm_label="Yes，调用工具",
        )

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolObservation:
        result = await self.mcp_client.call_tool(tool_name, arguments)
        text = self.mcp_client.result_to_text(result)
        return ToolObservation(
            tool_name=tool_name,
            arguments=arguments,
            result=text,
            is_error=bool(getattr(result, "isError", False)) or text.startswith(("读取失败", "PERMISSION DENIED", "Error:")),
        )

    @staticmethod
    def _resolve_tool_name(tool_name: str, tools: list[Any]) -> str | None:
        available = {tool.name for tool in tools}
        if tool_name in available:
            return tool_name

        simplified = tool_name.split(".")[-1].split("/")[-1].strip()
        if simplified in available:
            return simplified

        lowered = simplified.lower()
        for candidate in available:
            if candidate.lower() == lowered:
                return candidate
        return None

    @staticmethod
    def _tool_schema_for(tool_name: str, tools: list[Any]) -> dict[str, Any] | None:
        for tool in tools:
            if tool.name == tool_name:
                return dict(tool.inputSchema or {})
        return None

    @staticmethod
    def _coerce_arguments(arguments: dict[str, Any], input_schema: dict[str, Any] | None) -> dict[str, Any]:
        if not input_schema:
            return arguments

        properties = input_schema.get("properties") or {}
        required = list(input_schema.get("required") or [])
        if not properties or not arguments:
            return arguments

        if all(key in properties for key in arguments):
            return arguments

        alias_map = {
            "path": "file_path",
            "file": "file_path",
            "filename": "file_path",
            "content": "text",
            "message": "text",
        }
        remapped: dict[str, Any] = {}
        changed = False
        for key, value in arguments.items():
            target_key = alias_map.get(str(key).lower(), key)
            if target_key in properties:
                remapped[target_key] = value
                changed = changed or target_key != key
            else:
                remapped[key] = value

        if changed and all(key in properties for key in remapped):
            return remapped

        if len(required) == 1 and len(arguments) == 1:
            return {required[0]: next(iter(arguments.values()))}

        return arguments

    def _synthesize_answer(self, user_input: str, observations: list[ToolObservation]) -> str:
        observation_json = self._observation_payload(observations)
        return generate_text_response(
            self.SYNTHESIS_PROMPT,
            f"用户问题：{user_input}\n\n工具返回：\n{observation_json}",
            llm_client=self.llm_client,
            model=self.model,
            temperature=0.2,
        )

    @staticmethod
    def _synthesis_workflow_text(observations: list[ToolObservation]) -> str:
        """为最终整理回答阶段提供简洁的运行中提示。"""
        if not observations:
            return "正在整理结果..."

        last_tool = observations[-1].tool_name
        if last_tool == "read_file":
            return "正在归纳文档内容..."
        if last_tool in {"get_news", "http_request"}:
            return "正在整理检索结果..."
        if last_tool in {"get_weather", "get_exchange_rate", "get_current_time", "get_live_time"}:
            return "正在整理查询结果..."
        return "正在整理工具结果..."

    @staticmethod
    def _extract_json_payload(observation: ToolObservation) -> dict[str, Any] | None:
        try:
            payload = json.loads(observation.result)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _format_time_answer_from_observation(observation: ToolObservation) -> str | None:
        payload = ToolAgent._extract_json_payload(observation)
        if payload is None:
            return None

        if observation.tool_name in {"get_current_time", "get_live_time"}:
            if not payload.get("ok"):
                return None
            date = payload.get("date")
            time_text = payload.get("time")
            timezone = payload.get("timezone")
            weekday = ToolAgent._weekday_to_chinese(payload.get("weekday"))
            source = payload.get("source", "")
            if not date or not time_text or not timezone:
                return None
            answer = f"当前时间是 {date} {time_text}（{timezone}）"
            if weekday:
                answer += f"，{weekday}"
            if source == "local_system_clock":
                answer += "。该结果来自本机系统时钟。"
            elif source:
                answer += f"。该结果来自 {source}。"
            return answer

        return None

    async def _direct_time_query_answer(
        self,
        *,
        user_input: str,
        routed_task: ToolPlan,
        ui: "OrchestratorOutput",
    ) -> ToolPlan | None:
        combined_context = " ".join(str(item) for item in (routed_task.get("context_passed") or []))
        task_description = str(routed_task.get("task_description") or "")
        if not self._looks_like_realtime_time_task(user_input, task_description, combined_context):
            return None

        timezone = self._extract_timezone(user_input, task_description, combined_context)
        observations: list[ToolObservation] = []

        self._emit_workflow(ui, "检测到时效性时间查询，准备调用 REST API", state="info")
        self._emit_workflow(ui, "正在调用工具：get_live_time", state="running")
        observation = await self._call_tool("get_live_time", {"timezone": timezone})
        observations.append(observation)
        answer = self._format_time_answer_from_observation(observation)
        if answer:
            self._emit_workflow(ui, "已获取最新时间，正在输出答案...", state="done")
            await ui.stream_llm(answer)
            return {
                "intent": "respond",
                "status": "direct_time_query_live_api",
                "answer": answer,
                "observations": [item.to_prompt_dict() for item in observations],
            }
        self._emit_workflow(ui, "时间 API 获取失败，回退到本地时钟", state="warn")

        arguments = {"timezone": timezone}
        self._emit_workflow(ui, "检测到时间查询，准备读取本地时钟", state="info")
        self._emit_workflow(ui, "正在调用工具：get_current_time", state="running")
        observation = await self._call_tool("get_current_time", arguments)
        observations.append(observation)
        answer = self._format_time_answer_from_observation(observation)
        if answer is None:
            return None
        self._emit_workflow(ui, "已获取当前时间，正在输出答案...", state="done")
        await ui.stream_llm(answer)
        return {
            "intent": "respond",
            "status": "direct_time_query_local",
            "answer": answer,
            "observations": [item.to_prompt_dict() for item in observations],
        }

    @staticmethod
    def _format_weather_answer_from_observation(observation: ToolObservation) -> str | None:
        payload = ToolAgent._extract_json_payload(observation)
        if payload is None or not payload.get("ok"):
            return None
        location = payload.get("location") or payload.get("query") or "目标地点"
        description = payload.get("description") or "未知"
        temp_c = payload.get("temperature_c")
        feels_c = payload.get("feels_like_c")
        humidity = payload.get("humidity")
        wind_kmph = payload.get("wind_kmph")
        obs_time = payload.get("observation_time")

        parts = [f"{location} 当前天气：{description}"]
        if temp_c is not None:
            parts.append(f"气温 {temp_c}°C")
        if feels_c is not None:
            parts.append(f"体感 {feels_c}°C")
        if humidity is not None:
            parts.append(f"湿度 {humidity}%")
        if wind_kmph is not None:
            parts.append(f"风速 {wind_kmph} km/h")
        answer = "，".join(parts) + "。"
        if obs_time:
            answer += f" 观测时间：{obs_time}。"
        return answer

    async def _direct_weather_query_answer(
        self,
        *,
        user_input: str,
        routed_task: ToolPlan,
        ui: "OrchestratorOutput",
    ) -> ToolPlan | None:
        combined_context = " ".join(str(item) for item in (routed_task.get("context_passed") or []))
        task_description = str(routed_task.get("task_description") or "")
        if self._detect_temporal_category(user_input, task_description, combined_context) != "weather":
            return None

        location = self._extract_weather_location(user_input, combined_context)
        if not location:
            return {
                "intent": "ask_clarification",
                "status": "clarification",
                "question": "你想查询哪个城市/地区的天气？例如：北京、上海、Tokyo。",
                "options": ["北京", "上海", "广州", "深圳"],
            }

        self._emit_workflow(ui, "检测到时效性天气查询，准备调用 REST API", state="info")
        self._emit_workflow(ui, "正在调用工具：get_weather", state="running")
        observation = await self._call_tool("get_weather", {"location": location})
        answer = self._format_weather_answer_from_observation(observation)
        if answer is None:
            payload = self._extract_json_payload(observation) or {}
            message = str(payload.get("error") or "天气数据获取失败。")
            self._emit_workflow(ui, "天气数据获取失败", state="warn")
            await ui.stream_llm(message)
            return {
                "intent": "respond",
                "status": "weather_query_failed",
                "answer": message,
                "observations": [observation.to_prompt_dict()],
            }

        self._emit_workflow(ui, "已获取最新天气，正在输出答案...", state="done")
        await ui.stream_llm(answer)
        return {
            "intent": "respond",
            "status": "direct_weather_query",
            "answer": answer,
            "observations": [observation.to_prompt_dict()],
        }

    @staticmethod
    def _format_exchange_answer_from_observation(observation: ToolObservation) -> str | None:
        payload = ToolAgent._extract_json_payload(observation)
        if payload is None or not payload.get("ok"):
            return None
        amount = payload.get("amount", 1)
        base = payload.get("base")
        rate = payload.get("rate")
        target = payload.get("target")
        date = payload.get("date")
        if base is None or target is None or rate is None:
            return None
        answer = f"最新可用汇率：{amount} {base} = {rate} {target}"
        if date:
            answer += f"（报价日期：{date}）"
        return answer + "。"

    async def _direct_exchange_rate_query_answer(
        self,
        *,
        user_input: str,
        routed_task: ToolPlan,
        ui: "OrchestratorOutput",
    ) -> ToolPlan | None:
        combined_context = " ".join(str(item) for item in (routed_task.get("context_passed") or []))
        task_description = str(routed_task.get("task_description") or "")
        if self._detect_temporal_category(user_input, task_description, combined_context) != "exchange_rate":
            return None

        pair = self._extract_currency_pair(user_input, task_description, combined_context)
        if pair is None:
            return {
                "intent": "ask_clarification",
                "status": "clarification",
                "question": "你想查询哪两个币种之间的汇率？例如：USD/CNY、EUR/CNY。",
                "options": ["USD/CNY", "EUR/CNY", "JPY/CNY", "GBP/CNY"],
            }

        base, target = pair
        self._emit_workflow(ui, "检测到时效性汇率查询，准备调用 REST API", state="info")
        self._emit_workflow(ui, "正在调用工具：get_exchange_rate", state="running")
        observation = await self._call_tool("get_exchange_rate", {"base": base, "target": target})
        answer = self._format_exchange_answer_from_observation(observation)
        if answer is None:
            payload = self._extract_json_payload(observation) or {}
            message = str(payload.get("error") or "汇率数据获取失败。")
            self._emit_workflow(ui, "汇率数据获取失败", state="warn")
            await ui.stream_llm(message)
            return {
                "intent": "respond",
                "status": "exchange_rate_query_failed",
                "answer": message,
                "observations": [observation.to_prompt_dict()],
            }

        self._emit_workflow(ui, "已获取最新汇率，正在输出答案...", state="done")
        await ui.stream_llm(answer)
        return {
            "intent": "respond",
            "status": "direct_exchange_rate_query",
            "answer": answer,
            "observations": [observation.to_prompt_dict()],
        }

    @staticmethod
    def _format_news_answer_from_observation(observation: ToolObservation) -> str | None:
        payload = ToolAgent._extract_json_payload(observation)
        if payload is None or not payload.get("ok"):
            return None
        articles = payload.get("articles") or []
        if not isinstance(articles, list) or not articles:
            return None

        lines = ["为你找到以下最新新闻："]
        for index, article in enumerate(articles[:5], start=1):
            if not isinstance(article, dict):
                continue
            title = str(article.get("title") or "").strip()
            if not title:
                continue
            line = f"{index}. {title}"
            published_at = str(article.get("published_at") or "").strip()
            if published_at:
                line += f" | {published_at}"
            url = str(article.get("url") or "").strip()
            if url:
                line += f"\n   {url}"
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else None

    async def _direct_news_query_answer(
        self,
        *,
        user_input: str,
        routed_task: ToolPlan,
        ui: "OrchestratorOutput",
    ) -> ToolPlan | None:
        combined_context = " ".join(str(item) for item in (routed_task.get("context_passed") or []))
        task_description = str(routed_task.get("task_description") or "")
        if self._detect_temporal_category(user_input, task_description, combined_context) != "news":
            return None

        topic = self._extract_news_topic(user_input, task_description, combined_context)
        self._emit_workflow(ui, "检测到时效性新闻查询，准备调用 REST API", state="info")
        self._emit_workflow(ui, "正在调用工具：get_news", state="running")
        observation = await self._call_tool(
            "get_news",
            {"topic": topic, "limit": 5} if topic else {"limit": 5},
        )
        answer = self._format_news_answer_from_observation(observation)
        if answer is None:
            payload = self._extract_json_payload(observation) or {}
            message = str(payload.get("error") or "新闻数据获取失败。")
            self._emit_workflow(ui, "新闻数据获取失败", state="warn")
            await ui.stream_llm(message)
            return {
                "intent": "respond",
                "status": "news_query_failed",
                "answer": message,
                "observations": [observation.to_prompt_dict()],
            }

        self._emit_workflow(ui, "已获取最新新闻，正在输出答案...", state="done")
        await ui.stream_llm(answer)
        return {
            "intent": "respond",
            "status": "direct_news_query",
            "answer": answer,
            "observations": [observation.to_prompt_dict()],
        }

    async def _direct_temporal_query_answer(
        self,
        *,
        user_input: str,
        routed_task: ToolPlan,
        ui: "OrchestratorOutput",
    ) -> ToolPlan | None:
        combined_context = " ".join(str(item) for item in (routed_task.get("context_passed") or []))
        task_description = str(routed_task.get("task_description") or "")
        category = self._detect_temporal_category(user_input, task_description, combined_context)
        if category is None:
            return None

        if category == "time":
            return await self._direct_time_query_answer(user_input=user_input, routed_task=routed_task, ui=ui)
        if category == "weather":
            return await self._direct_weather_query_answer(user_input=user_input, routed_task=routed_task, ui=ui)
        if category == "exchange_rate":
            return await self._direct_exchange_rate_query_answer(user_input=user_input, routed_task=routed_task, ui=ui)
        if category == "news":
            return await self._direct_news_query_answer(user_input=user_input, routed_task=routed_task, ui=ui)
        return None

    async def _direct_file_read_summary(
        self,
        *,
        user_input: str,
        routed_task: ToolPlan,
        ui: "OrchestratorOutput",
    ) -> ToolPlan | None:
        candidate = self._extract_explicit_file_candidate(
            user_input,
            str(routed_task.get("task_description") or ""),
            " ".join(str(item) for item in (routed_task.get("context_passed") or [])),
        )
        resolved = self._resolve_explicit_file_candidate(candidate)
        if not resolved:
            return None

        decision = check_tool_permission("read_file", file_path=resolved)
        self._emit_workflow(ui, f"检测到明确文件任务，准备读取：{resolved}", state="info")
        if decision.status == "deny":
            return None
        if decision.status == "ask":
            confirmed = await self._confirm_tool_call("read_file", {"file_path": resolved}, decision, ui)
            if not confirmed:
                return {
                    "intent": "call_tool",
                    "status": "cancelled_by_user",
                    "tool_name": "read_file",
                    "arguments": {"file_path": resolved},
                }

        self._emit_workflow(ui, "正在调用工具：read_file", state="running")
        observation = await self._call_tool("read_file", {"file_path": resolved})
        if observation.is_error:
            self._emit_workflow(ui, f"工具返回错误：{self._shorten_text(observation.result)}", state="warn")
            return None

        # self._emit_workflow(ui, f"工具返回：{self._shorten_text(observation.result)}", state="info")
        self._emit_workflow(ui, self._synthesis_workflow_text([observation]), state="running")
        final_answer = await asyncio.to_thread(self._synthesize_answer, user_input, [observation])
        await ui.stream_llm(final_answer)
        return {
            "intent": "respond",
            "status": "direct_file_summary",
            "answer": final_answer,
            "observations": [observation.to_prompt_dict()],
        }

    async def disconnect(self) -> None:
        try:
            await self.mcp_client.disconnect()
        except Exception:
            pass

    async def run(
        self,
        routed_task: ToolPlan,
        ui: "OrchestratorOutput",
        *,
        user_input: str,
    ) -> ToolPlan:
        self.mcp_client = MCPClient(self._mcp_server_script_path)
        try:
            self._emit_workflow(ui, "正在分析任务并选择工具...", state="running")
            try:
                self._emit_workflow(ui, "正在连接工具服务...", state="running")
                await self.mcp_client.ensure_connected()
                tools = await self.mcp_client.list_tools()
                self._emit_workflow(ui, f"已连接工具服务，可用工具 {len(tools)} 个", state="done")
            except Exception as exc:
                self._emit_workflow(ui, "工具服务连接失败", state="warn")
                await ui.stream_llm(f"当前 Tool Agent 无法连接工具服务：{exc}")
                return {
                    "intent": "tool_agent",
                    "status": "mcp_connect_failed",
                    "error": str(exc),
                }

            working_input = user_input
            working_task = dict(routed_task)
            direct_time_result = await self._direct_temporal_query_answer(
                user_input=working_input,
                routed_task=working_task,
                ui=ui,
            )
            if direct_time_result is not None:
                return direct_time_result

            if self._looks_like_semantic_file_task(working_input, str(working_task.get("task_description") or "")):
                direct_result = await self._direct_file_read_summary(
                    user_input=working_input,
                    routed_task=working_task,
                    ui=ui,
                )
                if direct_result is not None:
                    return direct_result

            observations: list[ToolObservation] = []
            clarification_rounds = 0
            last_tool_signature: str | None = None
            repeated_tool_calls = 0
            summary_reason = "工具调用轮次已达上限，正在整理当前结果..."
            summary_status = "max_rounds_reached"

            for _ in range(self.MAX_TOOL_ROUNDS):
                self._emit_workflow(ui, "正在分析任务并选择工具...", state="running")
                plan = await asyncio.to_thread(self._plan_action, working_input, working_task, tools, observations)
                intent = plan.get("intent")

                if intent == "ask_clarification":
                    clarification_rounds += 1
                    if clarification_rounds > self.MAX_CLARIFICATION_ROUNDS:
                        break
                    self._emit_workflow(ui, "当前信息不足，需要补充信息", state="warn")
                    answer = await self._prompt_for_clarification(plan, ui)
                    if not answer:
                        self._emit_workflow(ui, "用户取消了澄清流程", state="warn")
                        plan["status"] = "cancelled_by_user"
                        return plan
                    self._emit_workflow(ui, f"已收到补充信息：{answer}", state="info")
                    working_input = self._augment_user_input(working_input, plan.get("question") or "补充信息", answer)
                    working_task = self._augment_routed_task(working_task, answer)
                    continue

                if intent == "refuse":
                    self._emit_workflow(ui, "当前任务超出 Tool Agent 能力范围", state="warn")
                    await ui.stream_llm(plan.get("reason") or "当前 Tool Agent 无法安全完成该请求。")
                    plan["status"] = "refused"
                    return plan

                if intent == "respond":
                    self._emit_workflow(ui, "正在整理工具结果...", state="running")
                    await ui.stream_llm(plan.get("answer") or "工具任务已完成。")
                    plan["status"] = "responded"
                    plan["observations"] = [observation.to_prompt_dict() for observation in observations]
                    return plan

                tool_name = str(plan.get("tool_name") or "").strip()
                arguments = normalize_tool_arguments(plan.get("arguments"))
                if not tool_name:
                    fallback = self._fallback_plan(user_input, "缺少 tool_name")
                    fallback["status"] = "invalid_plan"
                    return fallback
                resolved_tool_name = self._resolve_tool_name(tool_name, tools)
                if resolved_tool_name is None:
                    self._emit_workflow(ui, f"模型选择了不存在的工具：{tool_name}", state="warn")
                    fallback = self._fallback_plan(user_input, f"未知工具：{tool_name}")
                    fallback["status"] = "invalid_tool"
                    return fallback
                tool_name = resolved_tool_name
                arguments = self._coerce_arguments(arguments, self._tool_schema_for(tool_name, tools))
                planned_signature = json.dumps(
                    {"tool_name": tool_name, "arguments": arguments},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if planned_signature == last_tool_signature and observations:
                    summary_reason = "检测到重复工具调用规划，正在整理已有结果..."
                    summary_status = "repeated_tool_call_summarized"
                    break

                invocation = self._render_tool_invocation(tool_name, arguments)
                self._emit_workflow(ui, f"已选择工具：{self._shorten_text(invocation)}", state="done")
                decision = check_tool_permission(tool_name, **arguments)
                if decision.status == "deny":
                    self._emit_workflow(ui, "检测到高风险/越界工具调用，已拒绝执行", state="warn")
                    await ui.stream_llm(decision.reason)
                    return {
                        "intent": "refuse",
                        "status": "permission_denied",
                        "reason": decision.reason,
                        "tool_name": tool_name,
                        "arguments": arguments,
                    }

                if decision.status == "ask":
                    self._emit_workflow(ui, "工具调用需要用户确认", state="warn")
                    confirmed = await self._confirm_tool_call(tool_name, arguments, decision, ui)
                    if not confirmed:
                        self._emit_workflow(ui, "用户已取消工具调用", state="warn")
                        return {
                            "intent": "call_tool",
                            "status": "cancelled_by_user",
                            "tool_name": tool_name,
                            "arguments": arguments,
                        }
                    self._emit_workflow(ui, "已收到确认，准备调用工具", state="done")

                self._emit_workflow(ui, f"正在调用工具：{tool_name}", state="running")
                observation = await self._call_tool(tool_name, arguments)
                observations.append(observation)
                if observation.is_error:
                    self._emit_workflow(ui, f"工具返回错误：{self._shorten_text(observation.result)}", state="warn")
                # else:
                    # self._emit_workflow(ui, f"工具返回：{self._shorten_text(observation.result)}", state="info")

                if planned_signature == last_tool_signature:
                    repeated_tool_calls += 1
                else:
                    repeated_tool_calls = 0
                    last_tool_signature = planned_signature

                if repeated_tool_calls >= 1:
                    summary_reason = "检测到重复工具调用，正在整理已有结果..."
                    summary_status = "repeated_tool_call_summarized"
                    break

            self._emit_workflow(ui, summary_reason, state="warn")
            self._emit_workflow(ui, self._synthesis_workflow_text(observations), state="running")
            final_answer = await asyncio.to_thread(self._synthesize_answer, working_input, observations)
            await ui.stream_llm(final_answer)
            return {
                "intent": "respond",
                "status": summary_status,
                "answer": final_answer,
                "observations": [observation.to_prompt_dict() for observation in observations],
            }
        finally:
            try:
                await self.mcp_client.disconnect()
            except Exception:
                pass
            self.mcp_client = MCPClient(self._mcp_server_script_path)


__all__ = ["TOOL_AGENT_OUTPUT_SCHEMA", "TOOL_AGENT_SCHEMA_NAME", "ToolAgent"]
