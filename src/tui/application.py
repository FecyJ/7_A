"""TUI 应用壳"""

import asyncio
import os
import signal
from typing import Any, Awaitable, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Input

from .command_input import CommandInput
from .dialogs import InteractionPanel
from .footer import AgentFooter
from .log_view import AgentRichLog


class AgentCLI(App):
    """整体 TUI 界面"""

    CSS = """
    Screen { layout: vertical; }
    #log_area { height: 1fr; border: solid green; }
    #interaction_panel { height: auto; }
    #command_input { height: 3; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+d", "kill_process", "Kill", show=True, priority=True),
    ]

    def __init__(
        self,
        input_handler: Callable[[str, "AgentCLI"], Awaitable[None]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_handler = input_handler
        self.current_process: asyncio.subprocess.Process | None = None
        self.current_process_group_id: int | None = None
        self.current_process_input_fd: int | None = None
        self.current_process_terminated_by_user = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield AgentRichLog(id="log_area")
        yield InteractionPanel(id="interaction_panel")
        yield CommandInput(
            placeholder="输入 Prompt，或输入 / + CLI 命令以直接执行...",
            id="command_input",
        )
        yield AgentFooter()

    def on_ready(self) -> None:
        self.output_system("系统初始化完成，等待输入...", style="bold cyan")
        self.query_one("#command_input", CommandInput).focus()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """选择文本时，让 Ctrl+C 优先执行复制而不是退出"""
        # 原理：Textual 在判断一个 binding 快捷键是否可用时，会调用 check_action()
        # 返回 False 代表禁用 ctrl+c -> quit
        if action == "quit":
            selected_text = self.screen.get_selected_text()
            if selected_text:
                return False

            focused = self.focused
            if isinstance(focused, Input) and not focused.selection.is_empty:
                return False

        return True

    async def _force_stop_current_process(
        self,
        process: asyncio.subprocess.Process,
        process_group_id: int | None,
    ) -> None:
        """发送 SIGTERM 后等待退出，超时则升级为强杀"""
        try:
            await asyncio.wait_for(process.wait(), timeout=1.5)
            return
        except asyncio.TimeoutError:
            pass

        try:
            if os.name != "nt" and process_group_id is not None:
                os.killpg(process_group_id, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return

        await process.wait()

    def action_kill_process(self) -> None:
        """Ctrl + D 结束整个命令进程组"""
        process = self.current_process
        if process is None or process.returncode is not None:
            self.bell()
            return

        self.current_process_terminated_by_user = True
        process_group_id = self.current_process_group_id

        try:
            if os.name != "nt" and process_group_id is not None:
                os.killpg(process_group_id, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return

        self.run_worker(
            self._force_stop_current_process(process, process_group_id),
            name="kill-current-process",
            group="process-control",
            exclusive=True,
        )

    async def action_quit(self) -> None:
        """Ctrl + C 优雅退出：先结束当前子进程，再关闭 TUI。"""
        process = self.current_process
        if process is not None and process.returncode is None:
            self.current_process_terminated_by_user = True
            process_group_id = self.current_process_group_id
            try:
                if os.name != "nt" and process_group_id is not None:
                    os.killpg(process_group_id, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
            else:
                await self._force_stop_current_process(process, process_group_id)

        self.exit()

    def output_user(self, text: str) -> None:
        """输出用户输入"""
        self.query_one("#log_area", AgentRichLog).write_user_message(text)

    def output_system(self, text: str, style: str = "") -> None:
        """流式输出系统日志或 Shell 命令回显"""
        self.query_one("#log_area", AgentRichLog).write_system_message(text, style=style)

    def output_workflow(self, text: str, state: str = "info") -> None:
        """输出流程状态日志。"""
        self.query_one("#log_area", AgentRichLog).write_workflow_message(text, state=state)

    def output_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        """流式输出 LLM 的答复"""
        self.query_one("#log_area", AgentRichLog).write_llm_message(
            content,
            markdown=markdown,
            language=language,
        )

    async def stream_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        """以逐步刷新的形式输出 LLM 内容。"""
        await self.query_one("#log_area", AgentRichLog).stream_llm_message(
            content,
            markdown=markdown,
            language=language,
        )

    async def confirm_shell_command(
        self,
        *,
        command: str,
        risk_level: str,
        reason: str,
        details: list[Any] | None = None,
        confirm_label: str = "继续执行",
    ) -> bool:
        """在日志区和输入框之间显示风险确认面板。"""
        result = await self.query_one("#interaction_panel", InteractionPanel).request_confirmation(
            command=command,
            risk_level=risk_level,
            reason=reason,
            details=details,
            confirm_label=confirm_label,
        )
        return bool(result)

    async def prompt_clarification(
        self,
        *,
        question: str,
        options: list[str],
        allow_manual: bool = True,
        manual_prompt: str = "请输入补充信息...",
    ) -> str | None:
        """在日志区和输入框之间显示澄清面板。"""
        return await self.query_one("#interaction_panel", InteractionPanel).request_clarification(
            question=question,
            options=options,
            allow_manual=allow_manual,
            manual_prompt=manual_prompt,
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """接收输入框提交，并交给后端处理"""
        if event.input.id != "command_input":
            return

        user_input = event.value.strip()
        if not user_input:
            return

        input_widget = self.query_one("#command_input", CommandInput)
        input_widget.add_to_history(user_input)

        # 如果有子进程正在运行: 说明该输入是需要传给子进程的 stdin
        if self.current_process and self.current_process.returncode is None:
            self.output_user(user_input)
            input_widget.value = ""

            payload = (user_input + "\n").encode("utf-8")

            # PTY 模式：直接写入 master fd
            if self.current_process_input_fd is not None:
                input_fd = self.current_process_input_fd

                async def write_pty_stdin() -> None:
                    try:
                        await asyncio.to_thread(os.write, input_fd, payload)
                    except OSError:
                        pass

                self.run_worker(write_pty_stdin())
                return

            # PIPE 模式：写入 stdin 并 drain
            if self.current_process.stdin is not None:
                self.current_process.stdin.write(payload)

                async def drain_stdin() -> None:
                    try:
                        await self.current_process.stdin.drain()
                    except Exception:
                        pass

                self.run_worker(drain_stdin())
            return
        
        # 如果没有子进程在运行，视作系统命令或自然语言处理
        self.output_user(user_input)
        input_widget.value = ""

        self.run_worker(self.input_handler(user_input, self))
