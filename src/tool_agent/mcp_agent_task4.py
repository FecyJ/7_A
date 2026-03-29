"""Tool Agent demo / compatibility entry."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.tool_agent.tool_agent import ToolAgent
else:
    from .tool_agent import ToolAgent


class _ConsoleUI:
    def output_system(self, text: str, style: str = "") -> None:
        del style
        print(text)

    def output_workflow(self, text: str, state: str = "info") -> None:
        prefix = {"warn": "⚠ ", "error": "✖ "}.get(state, "")
        print(f"{prefix}{text}")

    def output_llm(self, content: str, markdown: bool | None = None, language: str | None = None) -> None:
        del markdown, language
        print(content)

    async def stream_llm(self, content: str, markdown: bool | None = None, language: str | None = None) -> None:
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
        print(f"\n[确认] {reason}\n- 风险：{risk_level}\n- 调用：{command}")
        if details:
            for detail in details:
                print(f"  {getattr(detail, 'plain', str(detail))}")
        answer = input(f"{confirm_label}? [y/N]: ").strip().lower()
        return answer in {"y", "yes"}

    async def prompt_clarification(
        self,
        *,
        question: str,
        options: list[str],
        allow_manual: bool = True,
        manual_prompt: str = "请输入补充信息...",
    ) -> str | None:
        print(f"\n{question}")
        for index, option in enumerate(options, start=1):
            print(f"[{index}] {option}")
        if allow_manual:
            print("[m] 手动输入")
        answer = input("请选择: ").strip()
        if answer.lower() == "m" and allow_manual:
            manual = input(f"{manual_prompt}: ").strip()
            return manual or None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        return answer or None


async def main() -> None:
    prompt = " ".join(sys.argv[1:]).strip() or input(">>> ").strip()
    if not prompt:
        print("请输入任务描述。")
        return

    agent = ToolAgent()
    ui = _ConsoleUI()
    routed_task = {
        "intent": "tool_agent",
        "risk_level": "low",
        "task_description": prompt,
        "context_passed": [],
    }
    result = await agent.run(routed_task, ui, user_input=prompt)
    print("\n[结果 JSON]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    await agent.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
