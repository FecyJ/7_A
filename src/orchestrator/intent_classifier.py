"""
Task 2.2 - 结构化输出与意图分类模块
功能：接收用户自然语言输入，通过 LLM 进行意图分类，返回结构化 JSON 结果
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from typing import Any

from jsonschema import ValidationError, validate
from openai import OpenAIError

try:
    from .llm_client import (
        DEFAULT_MODEL,
        generate_text_response,
        get_llm_client,
        stream_text_response,
    )
except ImportError:
    from llm_client import (  # type: ignore
        DEFAULT_MODEL,
        generate_text_response,
        get_llm_client,
        stream_text_response,
    )


client = get_llm_client()
IntentResult = dict[str, Any]
INTENT_JSON_SCHEMA_NAME = "intent_routing_result"
SUBTASK_JSON_SCHEMA = {
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
        "agent": {
            "type": "string",
            "enum": ["shell_agent", "tool_agent", "memory_agent"],
        },
        "task_description": {"type": "string"},
        "context_passed": {"type": "array", "items": {"type": "string"}},
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
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

# === 1. JSON Schema 定义（用于 jsonschema 校验）===
INTENT_JSON_SCHEMA = {
    "type": "object",
    "required": [
        "reasoning",
        "confidence",
        "intent",
        "risk_level",
        "task_description",
        "context_passed",
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
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "task_description": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
        "context_passed": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
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
                {"type": "array", "items": SUBTASK_JSON_SCHEMA},
                {"type": "null"},
            ]
        },
    },
}

# Confidence 阈值常量
CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5


# === 2. 环境上下文收集 ===
def get_advanced_context() -> dict[str, str]:
    """获取当前操作系统、工作目录、文件列表和 Git 状态。"""
    context = {
        "os": platform.system(),
        "pwd": os.getcwd(),
        "shell": os.environ.get("SHELL", "unknown"),
        "files": "",
        "git_status": "",
    }

    try:
        ls_cmd = ["ls", "-la"] if platform.system() != "Windows" else ["dir"]
        ls_output = subprocess.check_output(ls_cmd, text=True, stderr=subprocess.DEVNULL)
        context["files"] = "\n".join(ls_output.splitlines()[:20])
    except Exception:
        context["files"] = "无法获取文件列表"

    try:
        git_output = subprocess.check_output(
            ["git", "status", "-s"], text=True, stderr=subprocess.DEVNULL
        )
        context["git_status"] = git_output.strip() or "干净的工作区 (没有未提交的更改)"
    except Exception:
        context["git_status"] = "当前不是 Git 仓库"

    return context


# === 3. System Prompt 生成 ===
def _format_extra_context(extra_context: dict[str, str] | None) -> str:
    """格式化 RAM / ROM 附加上下文。"""
    if not extra_context:
        return ""

    sections: list[str] = []
    short_term = str(extra_context.get("short_term_memory") or "").strip()
    long_term = str(extra_context.get("long_term_memory") or "").strip()

    if short_term:
        sections.append(f"""【短期会话上下文（RAM）】
以下是最近几轮对话摘要，请用它解决“那个文件 / 上一步结果 / 刚才提到的路径”等代词指代问题：
{short_term}""")

    if long_term:
        sections.append(f"""【长期记忆注入（ROM）】
以下是从持久化记忆库检索出的背景事实 / 用户偏好，请在规划任务时优先参考：
{long_term}""")

    last_file_path = str(extra_context.get("last_file_path") or "").strip()
    last_document_summary = str(extra_context.get("last_document_summary") or "").strip()
    last_document_content = str(extra_context.get("last_document_content") or "").strip()
    if last_file_path or last_document_summary or last_document_content:
        lines = ["【最近文档 Artifact】", "如果用户提到“前文所述 / 刚才那个 README / 该文档 / 这个文件”，默认优先指向以下最近一次成功读取/分析的文档："]
        if last_file_path:
            lines.append(f"- 最近文档路径: {last_file_path}")
        if last_document_summary:
            lines.append(f"- 最近文档摘要: {last_document_summary}")
        if last_document_content:
            lines.append(f"- 最近文档内容片段:\n{last_document_content}")
        sections.append("\n".join(lines))

    return ("\n\n" + "\n\n".join(sections)) if sections else ""


def get_system_prompt(context: dict[str, str], extra_context: dict[str, str] | None = None) -> str:
    """生成带环境上下文的系统提示词，要求 LLM 输出结构化 JSON。"""
    return f"""你是一个多 Agent 命令行系统的"总控大脑"。你的唯一任务是分析用户的自然语言输入，判断其真实意图，并输出严格的结构化 JSON 进行路由。

【当前环境上下文】
- 操作系统: {context['os']}
- Shell 类型: {context['shell']}
- 工作目录: {context['pwd']}
- 目录概览 (前20项):
{context['files']}
- Git 状态:
{context['git_status']}{_format_extra_context(extra_context)}

【意图分类标准与边界】
1. "shell_agent": 确定性的本地系统操作。
   - 边界：只需执行标准命令或管道（如 grep, awk, ls, zip）即可完成，不需要大模型的语义理解或复杂推理。
   - 示例："帮我看看当前目录有哪些文件"、"找出这段日志里的 error 行"。
2. "tool_agent": 需要认知能力、语义分析或调用外部资源的复杂任务。
   - 边界：涉及对文件内容的阅读理解、代码审查、翻译，或需要调用第三方 API。
   - 示例："读取 config.json 并总结里面的数据库配置"、"分析这段代码的复杂度"。
   - 强规则：凡是“先读取文件/代码/文档内容，再总结、分析、解释、翻译”的任务，一律优先判为 "tool_agent"。
   - 强规则：凡是“当前时间/日期/星期、天气、新闻、汇率、最新/实时数据”等依赖实时信息的问题，不允许判为 "direct_answer"，应交给可获取实时信息的 Agent。
3. "direct_answer": 纯知识问答或闲聊，无需系统执行任何操作。
   - 示例："什么是 Python 的 GIL？"、"多 Agent 系统怎么设计？"
   - 禁止场景：不要把“现在几点了”“今天天气”“最新新闻”“美元兑人民币汇率”等时效性问题判为 direct_answer。
4. "memory_agent": 显式的长期记忆读写请求。
   - 示例："记住我最喜欢的语言是 Python"、"你还记得我的默认开发语言吗"、"列出记忆"。
5. "multi_agent": 一个请求天然包含两个及以上子任务，需要拆成顺序执行的多个 A2A 子任务。
   - 示例："用我最喜欢的语言写一个 Hello World，并记住这个文件路径。"
6. "clarification": (最高优先级) 指令模糊或存在安全隐患，必须追问。
   - 边界：存在你认为无从确定的模糊指代；或者用户要求操作的实体在【当前环境上下文】中不存在并且你无法推断其含义；或者操作极度危险但意图不明。

【子任务拆分规则】
- 如果一个请求只需一个下级 Agent，就使用对应单一 intent。
- 如果一个请求同时包含“执行任务 + 记住结果 / 再查询别的资源”等多个动作，优先输出 intent="multi_agent"，并在 subtasks 中按执行顺序拆分。
- intent="multi_agent" 时，subtasks 至少要有 2 项；如果只有 1 项，说明这不是 multi_agent。
- subtasks 中每一项的 agent 只能是：shell_agent / tool_agent / memory_agent。
- memory_agent 子任务必须填写 memory_action（save/search/delete/list）；如果要引用上一步结果，可在 memory_content 中使用占位符：
  - {{last_result}}：上一子任务的结果摘要
  - {{last_path}}：上一子任务解析出的文件路径

【评估与校准规则】
你必须进行以下两项严格评估：
- confidence (置信度 0.0~1.0): 基础分根据你的判断打在 0.8~1.0之间。
    - 特殊处理规则：若用户指令有歧义扣 0.3；若要求的操作目标在环境中找不到，立刻扣 0.4。如果最终低于 0.5，必须将意图改为 "clarification"。
- risk_level (风险等级): 评估任务对系统的破坏性。枚举值："low"(安全查询), "medium"(修改普通文件), "high"(删除、格式化、修改系统配置)。

【输出要求】
你必须且只能输出一个合法的 JSON 对象，不要有任何 Markdown 标记（如 ```json）或其他文字。
为了彻底消除“事后合理化”的幻觉，你必须严格按照以下自回归逻辑链顺序输出字段：
先输出 reasoning (思考推演) -> 再输出 confidence (严格打分) -> 接着输出 intent (根据reasoning和confidence定论) -> 然后是 risk_level -> 最后动态输出专属字段。

请根据最终确定的 intent，选择以下几种 JSON 结构之一进行输出：

>>> 如果 intent 是 "shell_agent"、"tool_agent" 或 "memory_agent"：
{{
    "reasoning": "分析用户指令 -> 检查上下文文件是否存在 -> 评估风险 -> 决定委派给下级 Agent",
    "confidence": 0.95,
    "intent": "shell_agent" | "tool_agent" | "memory_agent",
    "risk_level": "low" | "medium" | "high",
    "task_description": "将用户的话转化为专业、清晰的任务指令，提供给下级 Agent",
    "context_passed": ["提取出的文件名、路径或关键参数"],
    "reply": null,
    "question": null,
    "options": null,
    "subtasks": null
}}

>>> 如果 intent 是 "multi_agent"：
{{
    "reasoning": "识别到这是复合请求，需要拆成多个 A2A 子任务顺序执行",
    "confidence": 0.92,
    "intent": "multi_agent",
    "risk_level": "low" | "medium" | "high",
    "task_description": "对复合请求的整体摘要",
    "context_passed": ["对整体请求有帮助的关键上下文"],
    "reply": null,
    "question": null,
    "options": null,
    "subtasks": [
        {{
            "agent": "shell_agent" | "tool_agent" | "memory_agent",
            "task_description": "该子任务的清晰说明",
            "context_passed": ["传给该子任务的参数"],
            "risk_level": "low" | "medium" | "high",
            "memory_action": null | "save" | "search" | "delete" | "list",
            "memory_content": null | "要写入/查询的记忆内容，支持 {{last_result}} / {{last_path}}"
        }},
        {{
            "agent": "shell_agent" | "tool_agent" | "memory_agent",
            "task_description": "第二个及之后的子任务说明",
            "context_passed": ["可引用上一步结果"],
            "risk_level": "low" | "medium" | "high",
            "memory_action": null | "save" | "search" | "delete" | "list",
            "memory_content": null | "要写入/查询的记忆内容，支持 {{last_result}} / {{last_path}}"
        }}
    ]
}}

>>> 如果 intent 是 "direct_answer"：
{{
    "reasoning": "识别为纯自然语言互动，无需操作系统",
    "confidence": 1.0,
    "intent": "direct_answer",
    "risk_level": "low",
    "task_description": null,
    "context_passed": null,
    "reply": "直接在这里写出完整的回答内容，一步到位",
    "question": null,
    "options": null,
    "subtasks": null
}}

>>> 如果 intent 是 "clarification"：
{{
    "reasoning": "发现指令缺失关键目标（如上下文无此文件），或指代不明，触发降级策略",
    "confidence": 0.4,
    "intent": "clarification",
    "risk_level": "low",
    "task_description": null,
    "context_passed": null,
    "reply": null,
    "question": "向用户提出礼貌的追问，例如：请问您要删除的是当前目录的 test.py 还是 src 目录下的？",
    "options": ["候选选项1", "候选选项2", "其他"],
    "subtasks": null
}}
"""


# === 4. 三层容错 JSON 解析 ===
def parse_llm_json(raw_text: str) -> IntentResult | None:
    """三层容错机制解析 LLM 返回的 JSON。"""
    raw_text = raw_text.strip()
    decoder = json.JSONDecoder()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    try:
        parsed, _ = decoder.raw_decode(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r"\{", raw_text)
        if match:
            parsed, _ = decoder.raw_decode(raw_text[match.start() :])
            if isinstance(parsed, dict):
                return parsed
    except json.JSONDecodeError:
        pass

    return None


# === 5. Schema 校验 ===
def validate_intent_result(data: IntentResult) -> tuple[bool, str]:
    """用 jsonschema 校验解析结果是否符合预期格式。"""
    try:
        validate(instance=data, schema=INTENT_JSON_SCHEMA)
        if data["confidence"] < CONFIDENCE_LOW and data["intent"] != "clarification":
            return False, "confidence < 0.5 时 intent 必须为 clarification"
        intent = data["intent"]
        subtasks = data.get("subtasks")

        if subtasks is not None:
            if not isinstance(subtasks, list) or not subtasks:
                return False, "subtasks 必须为非空数组或 null"
            for index, subtask in enumerate(subtasks, start=1):
                if not isinstance(subtask, dict):
                    return False, f"subtasks[{index}] 不是对象"
                if subtask["agent"] == "memory_agent" and subtask.get("memory_action") is None:
                    return False, f"subtasks[{index}] 的 memory_agent 必须提供 memory_action"

        if intent == "direct_answer":
            if data["risk_level"] != "low":
                return False, "direct_answer 的 risk_level 必须为 low"
            if not isinstance(data.get("reply"), str) or not data["reply"].strip():
                return False, "direct_answer 必须提供非空 reply"
            if subtasks is not None:
                return False, "direct_answer 不能携带 subtasks"
        elif intent == "clarification":
            if not isinstance(data.get("question"), str) or not data["question"].strip():
                return False, "clarification 必须提供非空 question"
            if not isinstance(data.get("options"), list):
                return False, "clarification 必须提供 options 列表"
            if subtasks is not None:
                return False, "clarification 不能携带 subtasks"
        elif intent in {"shell_agent", "tool_agent", "memory_agent"}:
            if subtasks is None:
                if not isinstance(data.get("task_description"), str) or not data["task_description"].strip():
                    return False, f"{intent} 必须提供非空 task_description"
                if not isinstance(data.get("context_passed"), list):
                    return False, f"{intent} 必须提供 context_passed 列表"
        elif intent == "multi_agent":
            if subtasks is None:
                return False, "multi_agent 必须提供 subtasks"
            if len(subtasks) < 2:
                return False, "multi_agent 至少需要 2 个子任务"

        return True, ""
    except ValidationError as exc:
        return False, f"Schema 校验失败: {exc.message}"


# === 6. 降级与后处理 ===
def _infer_fallback_risk(text: str) -> str:
    """基于候选命令/文本粗略推断风险级别。"""
    lowered = text.lower()
    high_risk_markers = [
        "rm ",
        "rm\t",
        "删除",
        "sudo",
        "mkfs",
        "dd ",
        "shutdown",
        "reboot",
        "poweroff",
        "chmod 777",
    ]
    medium_risk_markers = [
        "mv ",
        "cp ",
        "mkdir",
        "touch ",
        "写入",
        "修改",
        "重命名",
        "移动",
        "复制",
        "创建",
    ]

    if any(marker in lowered for marker in high_risk_markers):
        return "high"
    if any(marker in lowered for marker in medium_risk_markers):
        return "medium"
    return "low"


def _looks_like_shell_request(user_input: str) -> bool:
    """判断用户输入是否更像本地命令/文件系统任务。"""
    lowered = user_input.lower()
    shell_markers = [
        "删除",
        "列出",
        "查看",
        "执行",
        "运行",
        "查找",
        "搜索",
        "创建",
        "新建",
        "修改",
        "移动",
        "复制",
        "重命名",
        "打开文件",
        "读取文件",
        "当前目录",
        "目录",
        "文件",
        "终端",
        "命令",
        "shell",
        "ls",
        "pwd",
        "grep",
        "find",
        "cat ",
        "git ",
        "python ",
        "pip ",
    ]
    return any(marker in lowered for marker in shell_markers)


def _looks_like_semantic_file_task(*texts: str) -> bool:
    """判断是否属于“读取文件内容后再做语义理解”的任务，应优先交给 tool_agent。"""
    combined = "\n".join(text for text in texts if text).lower()
    if not combined:
        return False

    file_markers = [
        "读取",
        "读一下",
        "读出",
        "read",
        "open",
        "查看文件",
        "文件内容",
        "readme",
        "config",
        "json",
        "yaml",
        "toml",
        "md",
        "代码",
        "文档",
        "日志",
        ".py",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".log",
    ]
    semantic_markers = [
        "总结",
        "概括",
        "分析",
        "解释",
        "翻译",
        "review",
        "summarize",
        "summary",
        "analyze",
        "explain",
        "理解",
        "提炼",
        "归纳",
        "复杂度",
        "审查",
    ]
    return any(marker in combined for marker in file_markers) and any(
        marker in combined for marker in semantic_markers
    )


def _detect_memory_request(user_input: str) -> dict[str, str] | None:
    """识别显式的长期记忆读写请求。"""
    text = user_input.strip()
    if not text:
        return None

    save_triggers = ["记住", "记下", "保存记忆", "帮我记", "偏好", "偏向", "我的喜好是"]
    search_triggers = ["你记得", "记得吗", "之前说过", "我说过什么", "你还记得", "回忆一下"]
    delete_triggers = ["忘掉", "忘记", "删除记忆", "去掉记忆"]
    list_triggers = ["所有记忆", "列出记忆", "查看记忆", "有哪些记忆"]

    for trigger in save_triggers:
        if trigger in text:
            content = text.split(trigger, 1)[-1].strip() or text
            return {"action": "save", "content": content, "trigger": trigger}

    for trigger in search_triggers:
        if trigger in text:
            content = text.replace(trigger, "").strip() or text
            return {"action": "search", "content": content, "trigger": trigger}

    for trigger in delete_triggers:
        if trigger in text:
            content = text.split(trigger, 1)[-1].strip()
            if not content:
                return None
            return {"action": "delete", "content": content, "trigger": trigger}

    for trigger in list_triggers:
        if trigger in text:
            return {"action": "list", "content": "", "trigger": trigger}

    return None


def _looks_like_compound_request(user_input: str) -> bool:
    """粗略识别多动作复合请求。"""
    lowered = user_input.lower()
    connectors = ["并", "并且", "同时", "然后", "再", "之后", "接着", "and then", " then ", " also "]
    return any(connector in lowered for connector in connectors)


def _uses_generic_memory_reference(content: str) -> bool:
    lowered = content.strip().lower()
    generic_markers = [
        "这个",
        "那个",
        "它",
        "该",
        "刚才",
        "上一步",
        "上个结果",
        "文件路径",
        "这个文件路径",
        "结果",
        "路径",
    ]
    return any(marker in lowered for marker in generic_markers)


def _memory_template_from_content(content: str) -> str:
    cleaned = content.strip()
    if not cleaned:
        return "{{last_result}}"
    if _uses_generic_memory_reference(cleaned):
        if "路径" in cleaned or "文件" in cleaned:
            return "{{last_path}}"
        return "{{last_result}}"
    return cleaned


def _build_memory_subtask(memory_request: dict[str, str]) -> dict[str, Any]:
    """将记忆请求映射为标准子任务。"""
    action = memory_request["action"]
    raw_content = memory_request.get("content", "").strip()
    memory_content = _memory_template_from_content(raw_content)

    task_description_map = {
        "save": f"将以下信息写入长期记忆：{memory_content}",
        "search": f"检索与“{raw_content or memory_content}”相关的长期记忆",
        "delete": f"删除与“{raw_content or memory_content}”最相关的长期记忆",
        "list": "列出当前长期记忆中的重要条目",
    }
    context_passed = [raw_content] if raw_content else []

    return {
        "agent": "memory_agent",
        "task_description": task_description_map[action],
        "context_passed": context_passed,
        "risk_level": "low",
        "memory_action": action,
        "memory_content": memory_content if action != "list" else None,
    }


def _result_to_primary_subtask(result: IntentResult) -> dict[str, Any] | None:
    """将单一 agent 路由结果转换为第一个子任务。"""
    intent = result.get("intent")
    if intent not in {"shell_agent", "tool_agent", "memory_agent"}:
        return None
    task_description = str(result.get("task_description") or "").strip()
    if not task_description:
        return None

    memory_action = None
    memory_content = None
    if intent == "memory_agent":
        memory_request = _detect_memory_request(task_description) or {"action": "save", "content": task_description}
        memory_action = memory_request["action"]
        memory_content = _memory_template_from_content(memory_request.get("content", ""))

    return {
        "agent": intent,
        "task_description": task_description,
        "context_passed": list(result.get("context_passed") or []),
        "risk_level": str(result.get("risk_level") or "low"),
        "memory_action": memory_action,
        "memory_content": memory_content,
    }


def _collapse_single_subtask_multi_agent(
    user_input: str,
    result: IntentResult,
) -> IntentResult | None:
    """将“只有 1 个子任务的 multi_agent”收紧为单一 agent 路由。"""
    if result.get("intent") != "multi_agent":
        return None

    subtasks = result.get("subtasks")
    if not isinstance(subtasks, list) or len(subtasks) != 1:
        return None

    subtask = subtasks[0]
    if not isinstance(subtask, dict):
        return None

    agent = str(subtask.get("agent") or "").strip()
    if agent not in {"shell_agent", "tool_agent", "memory_agent"}:
        return None

    task_description = str(subtask.get("task_description") or result.get("task_description") or "").strip()
    context_passed = [
        str(item).strip()
        for item in list(subtask.get("context_passed") or result.get("context_passed") or [])
        if str(item).strip()
    ]
    risk_level = str(subtask.get("risk_level") or result.get("risk_level") or "low")

    reasoning = str(result.get("reasoning") or "").strip()
    if reasoning:
        reasoning += "；multi_agent 仅生成了 1 个子任务，已自动折叠为单一 Agent 路由。"
    else:
        reasoning = "multi_agent 仅生成了 1 个子任务，已自动折叠为单一 Agent 路由。"

    if agent == "shell_agent" and _looks_like_semantic_file_task(user_input, task_description):
        agent = "tool_agent"
        reasoning += "；该任务属于“读取后再分析/总结”的语义文件任务，进一步改路由为 tool_agent。"

    temporal_query_info = _detect_temporal_query(user_input, task_description)
    if agent in {"shell_agent", "tool_agent"} and temporal_query_info is not None:
        agent = "tool_agent"
        risk_level = "low"
        task_description = _build_temporal_tool_task(user_input, task_description)
        context_passed = [user_input]
        reasoning += f"；检测到这是“{temporal_query_info['label']}”类时效性问题，改路由为 tool_agent。"

    collapsed: IntentResult = {
        "reasoning": reasoning,
        "confidence": float(result.get("confidence") or 0.0),
        "intent": agent,
        "risk_level": risk_level,
        "task_description": task_description,
        "context_passed": context_passed,
        "reply": None,
        "question": None,
        "options": None,
        "subtasks": None,
    }

    if agent == "memory_agent":
        collapsed["memory_action"] = subtask.get("memory_action")
        collapsed["memory_content"] = subtask.get("memory_content")

    return collapsed


def _detect_temporal_query(*texts: str) -> dict[str, str] | None:
    """识别天气 / 新闻 / 汇率 / 时间等强时效性问题。"""
    combined = "\n".join(text for text in texts if text).lower()
    if not combined:
        return None

    if any(
        re.search(pattern, combined)
        for pattern in [
            r"现在几点",
            r"当前几点",
            r"现在时间",
            r"当前时间",
            r"几点了",
            r"今天几号",
            r"今天星期几",
            r"当前日期",
            r"现在日期",
            r"今天日期",
            r"\bcurrent time\b",
            r"\btime now\b",
            r"\bcurrent date\b",
            r"\btoday('?s)? date\b",
            r"\bwhat time is it\b",
        ]
    ):
        return {"category": "time", "label": "时间/日期"}

    if any(marker in combined for marker in ["天气", "气温", "温度", "weather", "forecast"]):
        return {"category": "weather", "label": "天气"}

    if any(marker in combined for marker in ["新闻", "头条", "news", "headline"]):
        return {"category": "news", "label": "新闻"}

    if any(marker in combined for marker in ["汇率", "币价", "exchange rate", "fx rate", "forex"]):
        return {"category": "exchange_rate", "label": "汇率"}
    if any(marker in combined for marker in ["美元", "人民币", "欧元", "日元", "英镑", "usd", "cny", "eur", "jpy", "gbp"]) and any(
        marker in combined for marker in ["汇率", "兑换", "兑", "对", "exchange", "rate", "convert"]
    ):
        return {"category": "exchange_rate", "label": "汇率"}

    return None


def _looks_like_realtime_time_query(*texts: str) -> bool:
    """判断是否是依赖当前时间/日期的实时查询。"""
    info = _detect_temporal_query(*texts)
    return bool(info and info.get("category") == "time")


def _wants_online_lookup(*texts: str) -> bool:
    combined = "\n".join(text for text in texts if text).lower()
    if not combined:
        return False
    markers = [
        "联网",
        "网络",
        "在线",
        "上网",
        "web",
        "internet",
        "online",
        "api",
    ]
    return any(marker in combined for marker in markers)


def _build_temporal_tool_task(user_input: str, task_description: str = "") -> str:
    info = _detect_temporal_query(user_input, task_description)
    category = info.get("category") if info else "temporal_live_data"
    online_hint = (
        "用户明确要求联网，优先使用 REST API 获取可验证的实时数据；"
        if _wants_online_lookup(user_input, task_description)
        else "优先使用 REST API 或其他确定性实时数据源；"
    )
    category_instruction = {
        "time": "获取当前时间/日期/星期，不要基于模型记忆直接回答。",
        "weather": "获取目标地点的最新天气信息，不要猜测天气。",
        "news": "获取最新新闻/头条，不要凭记忆编造“最新消息”。",
        "exchange_rate": "获取最新汇率，不要凭记忆输出币价。",
        "temporal_live_data": "获取最新的时效性数据，不要使用过期知识回答。",
    }.get(category, "获取最新的时效性数据，不要使用过期知识回答。")
    return (
        f"{category_instruction}"
        f"{online_hint}"
        "若参数不足则先澄清；若获取失败，要明确说明失败原因，不要虚构结果。"
    )


def _looks_like_human_answer(text: str) -> bool:
    """判断文本更像自然语言回答，而不是命令/错误/机器结果。"""
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    machine_markers = [
        '"command"',
        '"error"',
        '"stderr"',
        '"stdout"',
        '"returncode"',
        "traceback",
        "exception",
        "rm ",
        "ls ",
        "pwd",
    ]
    if stripped.startswith("{") or stripped.startswith("["):
        return False
    if any(marker in lowered for marker in machine_markers):
        return False
    return True


def _make_fallback(user_input: str, raw_text: str) -> IntentResult:
    """生成降级 fallback 结果。"""
    raw_text = raw_text.strip()
    extracted_text = raw_text
    extracted_key = ""
    parsed_raw: dict[str, Any] | None = None
    try:
        candidate = json.loads(raw_text)
        parsed_raw = candidate if isinstance(candidate, dict) else None
        if isinstance(parsed_raw, dict):
            for key in ("reply", "answer", "response", "message", "content", "text"):
                value = parsed_raw.get(key)
                if isinstance(value, str) and value.strip():
                    extracted_text = value.strip()
                    extracted_key = key
                    break
    except Exception:
        pass

    lowered_input = user_input.lower()
    looks_like_shell_request = _looks_like_shell_request(user_input)
    temporal_query_info = _detect_temporal_query(user_input)
    clarification_markers = [
        "请问",
        "能否",
        "哪个",
        "哪一个",
        "更多信息",
        "请提供",
        "具体",
        "是指",
        "还是",
    ]
    direct_answer_markers = [
        "你好",
        "你是谁",
        "什么",
        "是什么",
        "为啥",
        "为什么",
        "怎么",
        "如何",
        "介绍",
        "解释",
        "是谁",
    ]
    memory_request = _detect_memory_request(user_input)

    if parsed_raw:
        explicit_intent = parsed_raw.get("intent")
        if explicit_intent in {"shell_agent", "tool_agent", "memory_agent"}:
            task_description = parsed_raw.get("task_description") or parsed_raw.get("command")
            context_passed = parsed_raw.get("context_passed")
            if isinstance(task_description, str) and task_description.strip():
                return {
                    "intent": explicit_intent,
                    "reasoning": f"结构化解析失败，但模型已给出可路由任务，降级沿用 {explicit_intent}。原始内容: {raw_text[:200]}",
                    "confidence": 0.55,
                    "risk_level": parsed_raw.get("risk_level") or _infer_fallback_risk(task_description),
                    "task_description": task_description.strip(),
                    "context_passed": context_passed if isinstance(context_passed, list) else [],
                    "reply": None,
                    "question": None,
                    "options": None,
                    "subtasks": None,
                }

        if explicit_intent == "multi_agent":
            subtasks = parsed_raw.get("subtasks")
            if isinstance(subtasks, list) and len(subtasks) >= 2:
                return {
                    "intent": "multi_agent",
                    "reasoning": f"结构化解析失败，但模型已给出多子任务规划，降级沿用 multi_agent。原始内容: {raw_text[:200]}",
                    "confidence": 0.55,
                    "risk_level": parsed_raw.get("risk_level") or "medium",
                    "task_description": str(parsed_raw.get("task_description") or "复合请求"),
                    "context_passed": parsed_raw.get("context_passed") if isinstance(parsed_raw.get("context_passed"), list) else [],
                    "reply": None,
                    "question": None,
                    "options": None,
                    "subtasks": subtasks,
                }
            if isinstance(subtasks, list) and len(subtasks) == 1:
                collapsed = _collapse_single_subtask_multi_agent(
                    user_input,
                    {
                        "intent": "multi_agent",
                        "reasoning": f"结构化解析失败，且模型仅返回了 1 个子任务，尝试折叠为单一 Agent 路由。原始内容: {raw_text[:200]}",
                        "confidence": 0.55,
                        "risk_level": parsed_raw.get("risk_level") or "low",
                        "task_description": str(parsed_raw.get("task_description") or "复合请求"),
                        "context_passed": parsed_raw.get("context_passed")
                        if isinstance(parsed_raw.get("context_passed"), list)
                        else [],
                        "reply": None,
                        "question": None,
                        "options": None,
                        "subtasks": subtasks,
                    },
                )
                if collapsed is not None:
                    return collapsed

        command = parsed_raw.get("command")
        if isinstance(command, str) and command.strip():
            inferred_risk = _infer_fallback_risk(command)
            return {
                "intent": "shell_agent",
                "reasoning": f"结构化解析失败，但模型返回了命令候选，降级为 shell_agent。原始内容: {raw_text[:200]}",
                "confidence": 0.55,
                "risk_level": inferred_risk,
                "task_description": f"根据用户请求生成并安全执行与下述候选命令等价的 shell 操作：{command.strip()}",
                "context_passed": [command.strip()],
                "reply": None,
                "question": None,
                "options": None,
                "subtasks": None,
            }

        error_text = parsed_raw.get("error") or parsed_raw.get("reason")
        if looks_like_shell_request and isinstance(error_text, str) and error_text.strip():
            return {
                "intent": "clarification",
                "reasoning": f"结构化解析失败，且模型返回了拒绝/错误信息，保守降级为 clarification。原始内容: {raw_text[:200]}",
                "confidence": 0.0,
                "risk_level": "low",
                "task_description": None,
                "context_passed": None,
                "reply": None,
                "question": error_text.strip(),
                "options": ["补充目标路径", "补充更具体操作", "直接用 /命令 执行"],
                "subtasks": None,
            }

    if memory_request is not None and not _looks_like_compound_request(user_input):
        memory_subtask = _build_memory_subtask(memory_request)
        return {
            "intent": "memory_agent",
            "reasoning": "结构化解析失败，但检测到显式长期记忆请求，保守降级为 memory_agent。",
            "confidence": 0.55,
            "risk_level": "low",
            "task_description": memory_subtask["task_description"],
            "context_passed": memory_subtask["context_passed"],
            "reply": None,
            "question": None,
            "options": None,
            "subtasks": None,
        }

    looks_like_clarification = any(marker in raw_text for marker in clarification_markers)
    looks_like_clarification = looks_like_clarification or any(
        marker in extracted_text for marker in clarification_markers
    )
    looks_like_direct_answer = (
        bool(extracted_text)
        and (
            (
                extracted_key in {"reply", "answer", "response", "content", "text"}
                and _looks_like_human_answer(extracted_text)
            )
            or (
                _looks_like_human_answer(extracted_text)
                and (
                    any(marker in lowered_input for marker in direct_answer_markers)
                    or user_input.endswith(("?", "？"))
                )
            )
        )
        and not looks_like_shell_request
        and temporal_query_info is None
    )

    if temporal_query_info is not None:
        return {
            "intent": "tool_agent",
            "reasoning": (
                f"结构化解析失败，但检测到这是“{temporal_query_info['label']}”类时效性问题，"
                "保守降级为 tool_agent 处理。"
            ),
            "confidence": 0.55,
            "risk_level": "low",
            "reply": None,
            "task_description": _build_temporal_tool_task(user_input),
            "context_passed": [user_input],
            "question": None,
            "options": None,
            "subtasks": None,
        }

    if looks_like_direct_answer and not looks_like_clarification:
        return {
            "intent": "direct_answer",
            "reasoning": f"结构化解析失败，但模型已直接给出自然语言回答，降级为 direct_answer。原始内容: {raw_text[:200]}",
            "confidence": 0.6,
            "risk_level": "low",
            "reply": extracted_text,
            "task_description": None,
            "context_passed": None,
            "question": None,
            "options": None,
            "subtasks": None,
        }

    return {
        "intent": "clarification",
        "reasoning": f"JSON 解析/校验失败，自动降级为 clarification。原始内容: {raw_text[:200]}",
        "confidence": 0.0,
        "risk_level": "low",
        "task_description": None,
        "context_passed": None,
        "reply": None,
        "question": (
            extracted_text
            if looks_like_clarification and extracted_text
            else f"我暂时无法可靠判断你的意图。你是希望我处理这个任务吗：{user_input}"
        ),
        "options": (
            ["补充目标路径", "补充更具体操作", "直接用 /命令 执行"]
            if looks_like_shell_request
            else ["shell_agent", "tool_agent", "direct_answer", "其他"]
        ),
        "subtasks": None,
    }


def apply_confidence_policy(result: IntentResult) -> IntentResult:
    """根据 confidence 阈值做最终降级与补默认值。"""
    result = dict(result)
    confidence = float(result.get("confidence", 0.0))

    if confidence < CONFIDENCE_LOW:
        result["intent"] = "clarification"
        result["risk_level"] = "low"
        result.setdefault("question", "你的指令不太明确，能否提供更多细节？")
        if not isinstance(result.get("options"), list):
            result["options"] = []
        result["subtasks"] = None

    return result


def apply_intent_overrides(user_input: str, result: IntentResult) -> IntentResult:
    """基于确定性本地规则，对明显误路由结果做轻量纠偏。"""
    result = dict(result)
    intent = result.get("intent")
    task_description = str(result.get("task_description") or "")
    subtasks = result.get("subtasks")

    collapsed = _collapse_single_subtask_multi_agent(user_input, result)
    if collapsed is not None:
        return collapsed

    temporal_query_info = _detect_temporal_query(user_input, task_description)
    memory_request = _detect_memory_request(user_input)

    if memory_request is not None and not subtasks:
        if intent in {"shell_agent", "tool_agent"} and _looks_like_compound_request(user_input):
            primary_subtask = _result_to_primary_subtask(result)
            memory_subtask = _build_memory_subtask(memory_request)
            if primary_subtask is not None:
                reasoning = str(result.get("reasoning") or "").strip()
                result["intent"] = "multi_agent"
                result["subtasks"] = [primary_subtask, memory_subtask]
                result["reply"] = None
                result["question"] = None
                result["options"] = None
                if reasoning:
                    reasoning += "；检测到请求同时包含执行任务与长期记忆操作，已拆分为多子任务。"
                else:
                    reasoning = "检测到请求同时包含执行任务与长期记忆操作，已拆分为多子任务。"
                result["reasoning"] = reasoning
                return result

        if intent not in {"multi_agent", "shell_agent", "tool_agent"} or not _looks_like_compound_request(user_input):
            memory_subtask = _build_memory_subtask(memory_request)
            reasoning = str(result.get("reasoning") or "").strip()
            result["intent"] = "memory_agent"
            result["risk_level"] = "low"
            result["task_description"] = memory_subtask["task_description"]
            result["context_passed"] = memory_subtask["context_passed"]
            result["reply"] = None
            result["question"] = None
            result["options"] = None
            result["subtasks"] = None
            if reasoning:
                reasoning += "；检测到显式长期记忆指令，改路由为 memory_agent。"
            else:
                reasoning = "检测到显式长期记忆指令，改路由为 memory_agent。"
            result["reasoning"] = reasoning
            return result

    if temporal_query_info is not None:
        reasoning = str(result.get("reasoning") or "").strip()
        result["intent"] = "tool_agent"
        result["risk_level"] = "low"
        result["task_description"] = _build_temporal_tool_task(user_input, task_description)
        result["context_passed"] = [user_input]
        result["reply"] = None
        result["question"] = None
        result["options"] = None
        result["subtasks"] = None
        if reasoning:
            reasoning += (
                f"；检测到这是“{temporal_query_info['label']}”类时效性问题，"
                "禁止 direct_answer，改路由为 tool_agent。"
            )
        else:
            reasoning = f"检测到这是“{temporal_query_info['label']}”类时效性问题，禁止 direct_answer，改路由为 tool_agent。"
        result["reasoning"] = reasoning
        return result

    if intent == "shell_agent" and _looks_like_semantic_file_task(user_input, task_description):
        reasoning = str(result.get("reasoning") or "").strip()
        result["intent"] = "tool_agent"
        if reasoning:
            reasoning += "；检测到这是“读取文件后做语义理解/总结”的任务，按规则改路由为 tool_agent。"
        else:
            reasoning = "检测到这是“读取文件后做语义理解/总结”的任务，按规则路由为 tool_agent。"
        result["reasoning"] = reasoning

    if intent == "multi_agent" and isinstance(subtasks, list):
        normalized_subtasks: list[dict[str, Any]] = []
        for subtask in subtasks:
            if not isinstance(subtask, dict):
                continue
            agent = subtask.get("agent")
            subtask_description = str(subtask.get("task_description") or "")
            subtask_context = list(subtask.get("context_passed") or [])
            if agent == "shell_agent" and _looks_like_semantic_file_task(user_input, subtask_description):
                agent = "tool_agent"
            if agent in {"shell_agent", "tool_agent"} and _detect_temporal_query(user_input, subtask_description):
                agent = "tool_agent"
                subtask_description = _build_temporal_tool_task(user_input, subtask_description)
                subtask_context = [user_input]

            normalized_subtask = dict(subtask)
            normalized_subtask["agent"] = agent
            normalized_subtask["task_description"] = subtask_description
            normalized_subtask["context_passed"] = subtask_context
            normalized_subtasks.append(normalized_subtask)

        result["subtasks"] = normalized_subtasks or subtasks

    return result


# === 7. 核心意图分类 ===
def classify_intent(
    user_input: str,
    llm_client=None,
    model: str | None = None,
    max_retries: int = 1,
    verbose: bool = False,
    extra_context: dict[str, str] | None = None,
) -> IntentResult:
    """调用 LLM 进行意图分类，返回结构化结果 dict。"""
    llm_client = llm_client or client
    model = model or DEFAULT_MODEL
    context = get_advanced_context()
    system_prompt = get_system_prompt(context, extra_context=extra_context)

    for attempt in range(1 + max_retries):
        try:
            raw_text = generate_text_response(
                system_prompt,
                user_input,
                llm_client=llm_client,
                model=model,
                temperature=0.1,
                json_schema=INTENT_JSON_SCHEMA,
                json_schema_name=INTENT_JSON_SCHEMA_NAME,
            )
            if verbose:
                print(f"\n[LLM 原始返回]\n{raw_text}\n")

            parsed = parse_llm_json(raw_text)
            if parsed is None:
                if attempt < max_retries:
                    if verbose:
                        print(f"[重试 {attempt + 1}/{max_retries}] JSON 解析失败，正在重试...")
                    continue
                return _make_fallback(user_input, raw_text)

            valid, err_msg = validate_intent_result(parsed)
            if not valid:
                if attempt < max_retries:
                    if verbose:
                        print(f"[重试 {attempt + 1}/{max_retries}] {err_msg}，正在重试...")
                    continue
                return _make_fallback(user_input, raw_text)

            return parsed

        except OpenAIError as exc:
            if verbose:
                print(f"[API 错误] {exc}")
            return _make_fallback(user_input, str(exc))
        except Exception as exc:
            if verbose:
                print(f"[未知错误] {exc}")
            return _make_fallback(user_input, str(exc))

    return _make_fallback(user_input, "重试耗尽")


# === 8. 对外统一入口 ===
def handle_intent(
    user_input: str,
    llm_client=None,
    model: str | None = None,
    max_retries: int = 1,
    verbose: bool = False,
    extra_context: dict[str, str] | None = None,
) -> IntentResult:
    """分类并应用置信度策略，返回最终可路由结果。"""
    result = classify_intent(
        user_input,
        llm_client=llm_client,
        model=model,
        max_retries=max_retries,
        verbose=verbose,
        extra_context=extra_context,
    )
    result = apply_intent_overrides(user_input, result)
    return apply_confidence_policy(result)


# === 9. direct_answer 的流式回复 ===
def stream_direct_answer(
    user_input: str,
    llm_client=None,
    model: str | None = None,
    on_delta=None,
) -> str:
    """当 direct_answer 需要完整回答时，发起流式请求。"""
    llm_client = llm_client or client
    model = model or DEFAULT_MODEL

    return stream_text_response(
        "你是一个有用的 AI 助手，请直接回答用户的问题。",
        user_input,
        llm_client=llm_client,
        model=model,
        temperature=0.7,
        on_delta=on_delta,
    )
