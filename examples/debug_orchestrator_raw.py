"""总控 structured classifier 调试与测试矩阵脚本。

说明：本脚本直接评估 src/orchestrator/intent_classifier.py，
不经过 DualEngineRouter / 本地 Fast-Path。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestrator.intent_classifier import (  # noqa: E402
    INTENT_JSON_SCHEMA,
    INTENT_JSON_SCHEMA_NAME,
    _make_fallback,
    apply_confidence_policy,
    apply_intent_overrides,
    get_advanced_context,
    get_system_prompt,
    parse_llm_json,
    validate_intent_result,
)
from src.orchestrator.llm_client import (  # noqa: E402
    DEFAULT_MODEL,
    generate_text_response,
    get_api_mode,
    get_llm_client,
)


@dataclass(frozen=True)
class IntentCase:
    """一条分类测试用例。"""

    name: str
    prompt: str
    expected_intent: str
    min_confidence: float = 0.70


@dataclass
class IntentEvalRecord:
    """单条测试记录。"""

    name: str
    prompt: str
    expected_intent: str
    min_confidence: float
    raw_text: str
    parsed_ok: bool
    schema_ok: bool
    schema_error: str
    raw_intent: str | None
    final_intent: str
    confidence: float | None
    risk_level: str | None
    passed_intent: bool
    passed_confidence: bool
    passed: bool
    final_result: dict[str, Any]


DEFAULT_CASES: list[IntentCase] = [
    IntentCase(
        name="shell_agent / 基础目录查看",
        prompt="帮我看看当前目录有哪些文件",
        expected_intent="shell_agent",
    ),
    IntentCase(
        name="tool_agent / 读取后总结",
        prompt="读取 README.md 并总结这个项目的目标",
        expected_intent="tool_agent",
    ),
    IntentCase(
        name="tool_agent / 时效性天气",
        prompt="给出现在杭州的天气",
        expected_intent="tool_agent",
    ),
    IntentCase(
        name="memory_agent / 记忆写入",
        prompt="记住我最喜欢的语言是 Python",
        expected_intent="memory_agent",
    ),
    IntentCase(
        name="multi_agent / 执行并记忆",
        prompt="新建 hello.py，并记住这个文件路径",
        expected_intent="multi_agent",
        min_confidence=0.65,
    ),
    IntentCase(
        name="direct_answer / 常识问答",
        prompt="什么是 Git？",
        expected_intent="direct_answer",
    ),
    IntentCase(
        name="clarification / 模糊指代",
        prompt="把它删了",
        expected_intent="clarification",
        min_confidence=0.0,
    ),
]


def _shorten(text: str | None, limit: int = 60) -> str:
    if not text:
        return "-"
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _markdown_escape(text: str | None) -> str:
    return str(text or "-").replace("|", "\\|").replace("\n", "<br>")


def fetch_raw_orchestrator_output(user_input: str, *, model: str | None = None) -> str:
    """按当前 schema 直接请求分类模型，返回原始文本。"""
    llm_client = get_llm_client()
    context = get_advanced_context()
    system_prompt = get_system_prompt(context)
    return generate_text_response(
        system_prompt,
        user_input,
        llm_client=llm_client,
        model=model or DEFAULT_MODEL,
        temperature=0.1,
        json_schema=INTENT_JSON_SCHEMA,
        json_schema_name=INTENT_JSON_SCHEMA_NAME,
    )


def run_pipeline(user_input: str, *, model: str | None = None) -> dict[str, Any]:
    """单次调用，返回原始输出、解析结果和最终路由结果。"""
    raw_text = fetch_raw_orchestrator_output(user_input, model=model)
    parsed = parse_llm_json(raw_text)
    parsed_ok = parsed is not None
    schema_ok = False
    schema_error = ""

    if parsed is not None:
        schema_ok, schema_error = validate_intent_result(parsed)

    if parsed is None or not schema_ok:
        routed = _make_fallback(user_input, raw_text)
    else:
        routed = parsed

    final_result = apply_confidence_policy(apply_intent_overrides(user_input, routed))

    return {
        "raw_text": raw_text,
        "parsed": parsed,
        "parsed_ok": parsed_ok,
        "schema_ok": schema_ok,
        "schema_error": schema_error,
        "raw_intent": str(parsed.get("intent")) if isinstance(parsed, dict) and parsed.get("intent") else None,
        "final_result": final_result,
    }


def evaluate_case(case: IntentCase, *, model: str | None = None) -> IntentEvalRecord:
    payload = run_pipeline(case.prompt, model=model)
    final_result = dict(payload["final_result"])
    confidence_raw = final_result.get("confidence")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
    final_intent = str(final_result.get("intent") or "")
    risk_level = str(final_result.get("risk_level") or "") or None

    passed_intent = final_intent == case.expected_intent
    passed_confidence = confidence is not None and confidence >= case.min_confidence
    if case.min_confidence <= 0:
        passed_confidence = confidence is not None
    passed = passed_intent and passed_confidence

    return IntentEvalRecord(
        name=case.name,
        prompt=case.prompt,
        expected_intent=case.expected_intent,
        min_confidence=case.min_confidence,
        raw_text=str(payload["raw_text"]),
        parsed_ok=bool(payload["parsed_ok"]),
        schema_ok=bool(payload["schema_ok"]),
        schema_error=str(payload["schema_error"] or ""),
        raw_intent=payload["raw_intent"],
        final_intent=final_intent,
        confidence=confidence,
        risk_level=risk_level,
        passed_intent=passed_intent,
        passed_confidence=passed_confidence,
        passed=passed,
        final_result=final_result,
    )


def render_markdown(records: list[IntentEvalRecord], *, model: str, api_mode: str) -> str:
    lines = [
        "# Orchestrator 意图分类测试矩阵",
        "",
        f"- 模型：`{model}`",
        f"- API 模式：`{api_mode}`",
        f"- 测试用例数：`{len(records)}`",
        "- 评估对象：`src/orchestrator/intent_classifier.py`（不经过 DualEngineRouter）",
        "",
        "| 用例 | 输入 | 预期意图 | 原始意图 | 最终意图 | 置信度 | 风险 | Schema | 检验 |",
        "| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for record in records:
        schema_cell = "✅" if record.schema_ok else ("fallback" if record.parsed_ok else "parse-fallback")
        confidence_cell = f"{record.confidence:.2f}" if record.confidence is not None else "-"
        check_cell = "✅ PASS" if record.passed else "❌ FAIL"
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(record.name),
                    _markdown_escape(record.prompt),
                    _markdown_escape(record.expected_intent),
                    _markdown_escape(record.raw_intent),
                    _markdown_escape(record.final_intent),
                    confidence_cell,
                    _markdown_escape(record.risk_level),
                    schema_cell,
                    check_cell,
                ]
            )
            + " |"
        )

    total = len(records)
    passed = sum(record.passed for record in records)
    avg_conf = mean(record.confidence for record in records if record.confidence is not None)

    lines.extend(
        [
            "",
            "## 汇总",
            "",
            f"- 通过数：`{passed}/{total}`",
            f"- 通过率：`{(passed / total * 100):.1f}%`",
            f"- 平均置信度：`{avg_conf:.2f}`",
        ]
    )

    failed = [record for record in records if not record.passed]
    if failed:
        lines.extend(["", "## 失败样例", ""])
        for record in failed:
            lines.append(
                f"### {record.name}\n"
                f"- 输入：`{record.prompt}`\n"
                f"- 预期：`{record.expected_intent}`\n"
                f"- 实际：`{record.final_intent}`\n"
                f"- 置信度：`{record.confidence if record.confidence is not None else '-'} `\n"
                f"- Schema：`{record.schema_error or ('ok' if record.schema_ok else 'parse_failed')}`\n"
                f"- 原始输出：\n\n```json\n{record.raw_text.strip()}\n```\n"
            )

    return "\n".join(lines).rstrip() + "\n"


def print_raw_debug(prompt: str, *, model: str, api_mode: str) -> None:
    payload = run_pipeline(prompt, model=model)
    parsed = payload["parsed"]
    final_result = payload["final_result"]

    print(f"[debug] model={model} api_mode={api_mode}")
    print(f"[debug] prompt={prompt}")
    print("-" * 72)
    print("[raw]")
    print(payload["raw_text"], end="" if str(payload["raw_text"]).endswith("\n") else "\n")
    print("-" * 72)
    print("[parsed]")
    print(json.dumps(parsed, ensure_ascii=False, indent=2) if parsed is not None else "<parse failed>")
    print("-" * 72)
    print("[final]")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))
    print("-" * 72)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="调试总控分类原始输出，或批量生成分类测试矩阵")
    parser.add_argument(
        "prompt",
        nargs="*",
        help="提供用户输入时进入单条 raw 调试模式；不提供时默认运行测试矩阵",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="强制运行内置分类测试矩阵",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "docs" / "orchestrator_intent_matrix.md",
        help="矩阵模式下输出 Markdown 报告路径",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="矩阵模式下额外输出 JSON 原始结果",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="矩阵模式下只打印，不写文件",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    llm_client = get_llm_client()
    model = DEFAULT_MODEL
    api_mode = get_api_mode(llm_client)

    if args.prompt and not args.matrix:
        prompt = " ".join(args.prompt).strip()
        if not prompt:
            raise SystemExit("用户输入不能为空")
        print_raw_debug(prompt, model=model, api_mode=api_mode)
        return

    records = [evaluate_case(case, model=model) for case in DEFAULT_CASES]
    markdown = render_markdown(records, model=model, api_mode=api_mode)

    print(markdown)

    if not args.stdout_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Markdown 报告已写入：{args.output}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 结果已写入：{args.json_output}")


if __name__ == "__main__":
    main()
