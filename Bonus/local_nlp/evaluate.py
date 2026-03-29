"""Bonus 2 自动化测试评估脚本。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Literal

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from Bonus.local_nlp.client import CLOUD_MODEL, LOCAL_SLM_MODEL
    from Bonus.local_nlp.router import DualEngineRouter
else:
    from .client import CLOUD_MODEL, LOCAL_SLM_MODEL
    from .router import DualEngineRouter

from src.orchestrator.intent_classifier import handle_intent
from src.orchestrator.llm_client import get_default_model, get_llm_client


ExpectedEngine = Literal["local_fastpath", "cloud"]


@dataclass(frozen=True)
class EvalCase:
    category: str
    prompt: str
    expected_engine: ExpectedEngine
    expected_intent: str


@dataclass
class EvalRecord:
    category: str
    prompt: str
    expected_engine: ExpectedEngine
    expected_intent: str
    actual_engine: str
    actual_intent: str
    confidence: float | None
    latency_ms: float
    cloud_baseline_ms: float | None
    fallback_reason: str | None
    status_messages: list[str]
    passed: bool
    summary: str
    raw_result: dict[str, Any]


DEFAULT_TEST_CASES: list[EvalCase] = [
    EvalCase("基础文件操作", "帮我看看当前目录下有哪些文件", "local_fastpath", "shell_agent"),
    EvalCase("系统状态查询", "查看系统内存占用", "local_fastpath", "shell_agent"),
    EvalCase("基础常识问答", "什么是 Ubuntu？", "local_fastpath", "direct_answer"),
    EvalCase("项目理解", "读取 README.md 并总结这个项目的目标", "cloud", "tool_agent"),
    EvalCase("工具调用", "帮我查一下北京今天的天气", "cloud", "tool_agent"),
    EvalCase("模糊指令", "把它给我删了", "cloud", "clarification"),
]


def _shorten(text: str | None, max_length: int = 38) -> str:
    if not text:
        return "-"
    text = " ".join(text.split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _engine_label(engine: str) -> str:
    return {
        "local_fastpath": "🟢 本地 Ollama",
        "cloud": "🔵 云端大脑",
    }.get(engine, engine or "unknown")


def _summarize_result(result: dict[str, Any]) -> str:
    intent = str(result.get("intent", "unknown"))
    if intent in {"shell_agent", "tool_agent"}:
        return f"{intent} -> {_shorten(result.get('task_description'))}"
    if intent == "direct_answer":
        return f"{intent} -> {_shorten(result.get('reply'))}"
    if intent == "clarification":
        return f"{intent} -> {_shorten(result.get('question'))}"
    return intent


def _markdown_escape(value: str) -> str:
    return value.replace("\n", "<br>").replace("|", "\\|")


async def _measure_cloud_baseline(prompt: str, model: str) -> float:
    started = perf_counter()
    await asyncio.to_thread(
        handle_intent,
        prompt,
        get_llm_client(),
        model,
        1,
        False,
    )
    return round((perf_counter() - started) * 1000, 2)


async def evaluate_case(
    case: EvalCase,
    *,
    router: DualEngineRouter,
    with_cloud_baseline: bool,
) -> EvalRecord:
    status_messages: list[str] = []
    started = perf_counter()
    result = await router.route(case.prompt, status_callback=status_messages.append)
    latency_ms = round((perf_counter() - started) * 1000, 2)

    meta = result.get("_dual_engine", {}) if isinstance(result, dict) else {}
    actual_engine = str(meta.get("engine", "unknown"))
    actual_intent = str(result.get("intent", "unknown"))
    confidence_raw = result.get("confidence")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
    fallback_reason = meta.get("fallback_reason")
    passed = actual_engine == case.expected_engine and actual_intent == case.expected_intent

    cloud_baseline_ms = None
    if with_cloud_baseline:
        cloud_baseline_ms = await _measure_cloud_baseline(case.prompt, router.cloud_model)

    return EvalRecord(
        category=case.category,
        prompt=case.prompt,
        expected_engine=case.expected_engine,
        expected_intent=case.expected_intent,
        actual_engine=actual_engine,
        actual_intent=actual_intent,
        confidence=confidence,
        latency_ms=latency_ms,
        cloud_baseline_ms=cloud_baseline_ms,
        fallback_reason=str(fallback_reason) if fallback_reason else None,
        status_messages=status_messages,
        passed=passed,
        summary=_summarize_result(result),
        raw_result=result,
    )


def render_markdown(records: list[EvalRecord], *, with_cloud_baseline: bool) -> str:
    lines = [
        "# Bonus 2 自动化测试结果",
        "",
        f"- 本地模型：`{LOCAL_SLM_MODEL}`",
        f"- 云端模型：`{CLOUD_MODEL or get_default_model()}`",
        f"- 测试用例数：`{len(records)}`",
        "",
        "| 测试类别 | 自然语言输入 | 预期路由引擎 | 解析结果 | 结果与耗时 |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for record in records:
        expected = f"{_engine_label(record.expected_engine)} / `{record.expected_intent}`"
        result_cell = f"{'✅ 通过' if record.passed else '❌ 偏差'} ({record.latency_ms:.0f}ms)"
        if with_cloud_baseline and record.cloud_baseline_ms is not None:
            result_cell = (
                f"{'✅ 通过' if record.passed else '❌ 偏差'} "
                f"(双擎 {record.latency_ms:.0f}ms / 云端 {record.cloud_baseline_ms:.0f}ms)"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(record.category),
                    _markdown_escape(record.prompt),
                    _markdown_escape(expected),
                    _markdown_escape(
                        f"{record.summary} [{_engine_label(record.actual_engine)}]"
                    ),
                    _markdown_escape(result_cell),
                ]
            )
            + " |"
        )

    lines.extend(["", "## 测试结论与性能对比", ""])

    passed_count = sum(record.passed for record in records)
    local_records = [record for record in records if record.actual_engine == "local_fastpath"]
    cloud_records = [record for record in records if record.actual_engine == "cloud"]

    lines.append(f"- 通过率：`{passed_count}/{len(records)}`")
    lines.append(
        f"- 路由分布：本地 `{len(local_records)}` 条，云端 `{len(cloud_records)}` 条"
    )
    if local_records:
        lines.append(
            f"- 本地命中平均耗时：`{mean(record.latency_ms for record in local_records):.2f}ms`"
        )
    if cloud_records:
        lines.append(
            f"- 云端路径平均耗时：`{mean(record.latency_ms for record in cloud_records):.2f}ms`"
        )
    if with_cloud_baseline:
        cloud_baselines = [
            record.cloud_baseline_ms
            for record in records
            if record.cloud_baseline_ms is not None
        ]
        if cloud_baselines:
            lines.append(f"- 云端基线平均耗时：`{mean(cloud_baselines):.2f}ms`")
            if local_records:
                local_avg = mean(record.latency_ms for record in local_records)
                local_cloud_baselines = [
                    record.cloud_baseline_ms
                    for record in local_records
                    if record.cloud_baseline_ms is not None
                ]
                cloud_avg = mean(local_cloud_baselines) if local_cloud_baselines else mean(cloud_baselines)
                if cloud_avg > 0:
                    gain = (cloud_avg - local_avg) / cloud_avg * 100
                    lines.append(f"- 对本地命中请求，Fast-Path 平均延迟下降约：`{gain:.1f}%`")

    degraded = [record for record in records if record.fallback_reason]
    if degraded:
        lines.extend(["", "## 降级原因统计", ""])
        for record in degraded:
            reason = record.fallback_reason or "-"
            message = "；".join(record.status_messages) if record.status_messages else "-"
            lines.append(
                f"- `{record.prompt}` -> `{reason}`"
                + (f"（{message}）" if message != "-" else "")
            )

    return "\n".join(lines) + "\n"


async def run_evaluation(
    *,
    test_cases: list[EvalCase] | None = None,
    with_cloud_baseline: bool = False,
    verbose: bool = True,
) -> list[EvalRecord]:
    router = DualEngineRouter(
        cloud_llm_client=get_llm_client(),
        cloud_model=get_default_model(),
        max_retries=1,
        verbose=False,
    )

    records: list[EvalRecord] = []
    cases = test_cases or DEFAULT_TEST_CASES

    if verbose:
        print("🚀 开始执行批量意图转换测试...\n")

    for index, case in enumerate(cases, start=1):
        if verbose:
            print(f"[{index}/{len(cases)}] {case.category} :: {case.prompt}")
        record = await evaluate_case(
            case,
            router=router,
            with_cloud_baseline=with_cloud_baseline,
        )
        records.append(record)
        if verbose:
            engine = _engine_label(record.actual_engine)
            print(
                f"  -> {engine} | intent={record.actual_intent} | "
                f"latency={record.latency_ms:.0f}ms | "
                f"{'PASS' if record.passed else 'FAIL'}"
            )
            if record.status_messages:
                print(f"  -> status: {'；'.join(record.status_messages)}")
            print()

    return records


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 Bonus 2 自动化路由评估")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="将 Markdown 测试报告写入指定文件",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="将原始评估结果写入 JSON 文件",
    )
    parser.add_argument(
        "--cloud-baseline",
        action="store_true",
        help="额外测量纯云端 handle_intent 延迟，用于性能对比",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="关闭逐条进度打印，只输出最终 Markdown",
    )
    return parser


async def main() -> None:
    args = _build_argparser().parse_args()
    records = await run_evaluation(
        with_cloud_baseline=args.cloud_baseline,
        verbose=not args.quiet,
    )
    markdown = render_markdown(records, with_cloud_baseline=args.cloud_baseline)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Markdown 报告已写入：{args.output}")
    else:
        print(markdown)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 结果已写入：{args.json_output}")


if __name__ == "__main__":
    asyncio.run(main())
