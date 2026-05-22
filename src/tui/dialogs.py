"""内嵌式交互面板：风险确认与交互式澄清。"""

from __future__ import annotations

import asyncio
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, Static

from .command_input import CommandInput


class InteractionPanel(Vertical, can_focus=True):
    """显示在日志区和输入框之间的正式 TUI 交互面板。"""

    MANUAL_INPUT_VALUE = "__manual_input__"

    DEFAULT_CSS = """
    InteractionPanel {
        display: none;
        height: auto;
        overflow-y: auto;
        margin: 0 1 0 1;
        padding: 0 1;
        border: round $primary;
        background: $surface;
    }

    InteractionPanel.-high {
        border: heavy $error;
        background: $error-muted 10%;
    }

    InteractionPanel.-medium {
        border: heavy $warning;
        background: $warning-muted 10%;
    }

    InteractionPanel.-clarification {
        border: round $primary;
        background: $panel;
    }

    InteractionPanel #panel_title {
        text-style: bold;
        margin-bottom: 0;
    }

    InteractionPanel.-high #panel_title,
    InteractionPanel.-high #panel_body {
        color: $error;
    }

    InteractionPanel #panel_body {
        display: none;
        max-height: 2;
        overflow-y: auto;
        margin-bottom: 0;
    }

    InteractionPanel #panel_command {
        display: none;
        border: tall $panel;
        background: $panel-darken-1;
        max-height: 4;
        overflow-y: auto;
        padding: 0 1;
        margin-bottom: 0;
    }

    InteractionPanel.-high #panel_command {
        border: tall $error;
        color: $error;
    }

    InteractionPanel #panel_options {
        display: none;
        margin-bottom: 0;
    }

    InteractionPanel #panel_details {
        display: none;
        color: $text-muted;
        max-height: 2;
        overflow-y: auto;
        margin-bottom: 0;
    }

    InteractionPanel #panel_input {
        display: none;
        margin-bottom: 0;
    }

    InteractionPanel #panel_hint {
        display: none;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._future: asyncio.Future[Any] | None = None
        self._option_values: list[Any] = []
        self._option_labels: list[str] = []
        self._selected_index = 0
        self._cancel_result: Any = None
        self._manual_prompt = "请输入补充信息..."
        self._variant = "clarification"

    def compose(self) -> ComposeResult:
        yield Static("", id="panel_title")
        yield Static("", id="panel_body")
        yield Static("", id="panel_command")
        yield Static("", id="panel_options")
        yield Static("", id="panel_details")
        yield Input(id="panel_input", placeholder="请输入补充信息...")
        yield Static("", id="panel_hint")

    def on_mount(self) -> None:
        self.display = False

    def _apply_dynamic_height(self) -> None:
        """将交互面板高度控制在屏幕的一半左右。"""
        screen_height = max(12, getattr(self.screen.size, "height", 24))
        max_height = min(max(10, screen_height - 6), max(14, screen_height // 2 + 4))
        self.styles.height = "auto"
        self.styles.max_height = max_height

    def _set_variant(self, variant: str) -> None:
        self._variant = variant
        self.remove_class("-high", "-medium", "-clarification")
        self.add_class(f"-{variant}")

    def _highlight_style(self) -> str:
        if self._variant == "high":
            return "bold white on red"
        if self._variant == "medium":
            return "bold black on yellow"
        return "bold white on blue"

    def _render_options(self) -> None:
        widget = self.query_one("#panel_options", Static)
        if not self._option_labels:
            widget.update("")
            widget.display = False
            self.refresh(layout=True)
            return

        text = Text()
        highlight_style = self._highlight_style()
        for index, label in enumerate(self._option_labels, start=1):
            line = Text(f"{index}. {label}")
            if index - 1 == self._selected_index:
                line.stylize(highlight_style)
            text.append_text(line)
            if index != len(self._option_labels):
                text.append("\n")

        widget.update(text)
        widget.display = True
        self.refresh(layout=True)

    def _show_body(self, body: str | None) -> None:
        widget = self.query_one("#panel_body", Static)
        body = (body or "").strip()
        if body:
            widget.update(body)
            widget.display = True
        else:
            widget.update("")
            widget.display = False
        self.refresh(layout=True)

    def _show_common(self, *, title: str, body: str, variant: str) -> None:
        self.display = True
        self._apply_dynamic_height()
        self._set_variant(variant)
        self.query_one("#panel_title", Static).update(title)
        self._show_body(body)
        self.refresh(layout=True)

    def _show_command_block(self, command: str | None) -> None:
        widget = self.query_one("#panel_command", Static)
        if command:
            widget.update(f"拟执行命令：\n{command}")
            widget.display = True
        else:
            widget.update("")
            widget.display = False
        self.refresh(layout=True)

    def _show_details(self, details: list[Any] | None) -> None:
        widget = self.query_one("#panel_details", Static)
        if details:
            text = Text()
            for index, detail in enumerate(details):
                if isinstance(detail, Text):
                    text.append_text(detail)
                else:
                    text.append(str(detail))
                if index != len(details) - 1:
                    text.append("\n")
            widget.update(text)
            widget.display = True
        else:
            widget.update("")
            widget.display = False
        self.refresh(layout=True)

    def _show_hint(self, hint: str | None) -> None:
        widget = self.query_one("#panel_hint", Static)
        if hint:
            widget.update(hint)
            widget.display = True
        else:
            widget.update("")
            widget.display = False
        self.refresh(layout=True)

    def _reveal_actions(self) -> None:
        if self.display:
            self.scroll_end(animate=False, immediate=True, x_axis=False)
            self.focus()

    def _focus_input(self) -> None:
        input_widget = self.query_one("#panel_input", Input)
        if input_widget.display:
            self.scroll_end(animate=False, immediate=True, x_axis=False)
            input_widget.focus()

    def _show_options(self, options: list[tuple[str, Any]]) -> None:
        self._option_labels = [label for label, _ in options]
        self._option_values = [value for _, value in options]
        self._selected_index = 0
        self.query_one("#panel_input", Input).display = False
        self._render_options()
        self.call_after_refresh(self._reveal_actions)

    def _show_manual_input(self, prompt: str) -> None:
        input_widget = self.query_one("#panel_input", Input)
        self.query_one("#panel_options", Static).display = False
        input_widget.display = True
        input_widget.placeholder = prompt
        input_widget.value = ""
        self._show_hint("请输入内容后按 Enter 确认，Esc 取消")
        self.call_after_refresh(self._focus_input)

    def _finish(self, result: Any) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(result)

        self.display = False
        self._option_labels = []
        self._option_values = []
        self._selected_index = 0
        self.query_one("#panel_command", Static).display = False
        self.query_one("#panel_details", Static).display = False
        self.query_one("#panel_options", Static).display = False
        self.query_one("#panel_input", Input).display = False
        self._show_hint(None)
        self.refresh(layout=True)

        try:
            self.app.query_one("#command_input", CommandInput).focus()
        except Exception:
            pass

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        if not self.display:
            return False
        if self.query_one("#panel_input", Input).display:
            return action == "cancel"
        return action in {"cursor_up", "cursor_down", "select", "cancel"}

    def action_cursor_up(self) -> None:
        if self._option_labels:
            self._selected_index = (self._selected_index - 1) % len(self._option_labels)
            self._render_options()

    def action_cursor_down(self) -> None:
        if self._option_labels:
            self._selected_index = (self._selected_index + 1) % len(self._option_labels)
            self._render_options()

    def action_select(self) -> None:
        if not self._option_values:
            return
        selected_value = self._option_values[self._selected_index]
        if selected_value == self.MANUAL_INPUT_VALUE:
            self._show_manual_input(self._manual_prompt)
            return
        self._finish(selected_value)

    def action_cancel(self) -> None:
        self._finish(self._cancel_result)

    async def request_confirmation(
        self,
        *,
        command: str,
        risk_level: str,
        reason: str,
        details: list[Any] | None = None,
        confirm_label: str = "Yes，继续执行",
    ) -> bool:
        self._future = asyncio.get_running_loop().create_future()
        self._cancel_result = False

        title = "⚠️ 高风险命令提示" if risk_level == "high" else "⚠️ 中风险命令提示"
        body = reason if risk_level == "high" else ""
        self._show_common(title=title, body=body, variant=risk_level)
        self._show_command_block(command)
        self._show_details(details)
        self._show_hint("↑↓ 选择 · Enter 确认 · Esc 取消")
        self._show_options(
            [
                (confirm_label, True),
                ("No，不执行该命令", False),
                ("取消", False),
            ]
        )
        await asyncio.sleep(0)
        return bool(await self._future)

    async def request_clarification(
        self,
        *,
        question: str,
        options: list[str],
        allow_manual: bool = True,
        manual_prompt: str = "请输入补充信息...",
    ) -> str | None:
        self._future = asyncio.get_running_loop().create_future()
        self._cancel_result = None
        self._manual_prompt = manual_prompt

        panel_options = [(option, option) for option in options]
        if allow_manual:
            panel_options.append(("手动输入...", self.MANUAL_INPUT_VALUE))
        panel_options.append(("取消", None))

        self._show_common(title="需要补充信息", body=question, variant="clarification")
        self._show_command_block(None)
        self._show_details(None)
        self._show_hint("↑↓ 选择 · Enter 确认 · Esc 取消")
        self._show_options(panel_options)
        await asyncio.sleep(0)
        return await self._future

    @on(Input.Submitted, "#panel_input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        self._finish(value or None)


__all__ = ["InteractionPanel"]
