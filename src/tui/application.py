"""TUI 应用壳。"""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Any, Awaitable, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Input, ListView, Static

from Bonus.memory_conv import SessionManager

from .command_input import CommandInput
from .dialogs import InteractionPanel
from .footer import AgentFooter
from .log_view import AgentRichLog
from .session_sidebar import SessionSidebar


class AgentCLI(App):
    """整体 TUI 界面。"""

    CSS = """
    Screen { layout: vertical; }

    #main_body {
        height: 1fr;
    }

    #sidebar_panel {
        width: 25%;
        min-width: 24;
        max-width: 40;
        border-right: solid $panel-lighten-1;
        padding: 0 1;
    }

    #sidebar_panel.-collapsed {
        display: none;
        width: 0;
        min-width: 0;
    }

    #sidebar_header {
        height: 3;
        content-align: left middle;
        color: $text-muted;
        text-style: bold;
    }

    #sidebar_tip {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    #sidebar {
        height: 1fr;
    }

    #conversation_panel {
        width: 1fr;
        height: 1fr;
    }

    #log_area { height: 1fr; border: solid green; }
    #interaction_panel { height: auto; }
    #command_input { height: 3; }
    """

    BINDINGS = [
        Binding("ctrl+n", "new_session", "New", show=True, priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", show=True, priority=True),
        Binding("ctrl+j", "focus_sidebar", "Sessions", show=True, priority=True),
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+d", "kill_process", "Kill", show=True, priority=True),
    ]

    def __init__(
        self,
        input_handler: Callable[[str, "AgentCLI"], Awaitable[Any]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_handler = input_handler
        self.orchestrator_agent = getattr(input_handler, "agent", None)
        self.session_manager = SessionManager()

        self.current_process: asyncio.subprocess.Process | None = None
        self.current_process_group_id: int | None = None
        self.current_process_input_fd: int | None = None
        self.current_process_terminated_by_user = False

        self.current_session_id: str | None = None
        self._session_messages: list[dict[str, Any]] = []
        self._suspend_session_capture = False
        self._session_busy = False
        self._sidebar_collapsed = False
        self._title_generation_tasks: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_body"):
            with Vertical(id="sidebar_panel"):
                yield Static("会话历史", id="sidebar_header")
                yield Static("Ctrl+N 新建 · Ctrl+B 折叠 · Ctrl+J 聚焦", id="sidebar_tip")
                yield SessionSidebar(id="sidebar")
            with Vertical(id="conversation_panel"):
                yield AgentRichLog(id="log_area")
                yield InteractionPanel(id="interaction_panel")
                yield CommandInput(
                    placeholder="输入 Prompt，或输入 / + CLI 命令以直接执行...",
                    id="command_input",
                )
        yield AgentFooter()

    def on_ready(self) -> None:
        self.run_worker(
            self._bootstrap_sessions(),
            name="bootstrap-sessions",
            group="sessions",
            exclusive=True,
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """选择文本时，让 Ctrl+C 优先执行复制而不是退出。"""
        del parameters
        if action == "quit":
            selected_text = self.screen.get_selected_text()
            if selected_text:
                return False

            focused = self.focused
            if isinstance(focused, Input) and not focused.selection.is_empty:
                return False

        return True

    def _is_process_running(self) -> bool:
        process = self.current_process
        return process is not None and process.returncode is None

    def _session_switch_allowed(self) -> bool:
        return not self._session_busy and not self._is_process_running()

    def _query_log(self) -> AgentRichLog:
        return self.query_one("#log_area", AgentRichLog)

    def _query_sidebar(self) -> SessionSidebar:
        return self.query_one("#sidebar", SessionSidebar)

    def _query_input(self) -> CommandInput:
        return self.query_one("#command_input", CommandInput)

    def _focus_input(self) -> None:
        self._query_input().focus()

    def _export_orchestrator_state(self) -> dict[str, Any]:
        if self.orchestrator_agent is None or not hasattr(self.orchestrator_agent, "export_session_state"):
            return {}
        try:
            return dict(self.orchestrator_agent.export_session_state())
        except Exception:
            return {}

    def _restore_orchestrator_state(self, state: dict[str, Any] | None) -> None:
        if self.orchestrator_agent is None:
            return
        if state and hasattr(self.orchestrator_agent, "restore_session_state"):
            self.orchestrator_agent.restore_session_state(state)
            return
        if hasattr(self.orchestrator_agent, "reset_session_state"):
            self.orchestrator_agent.reset_session_state()

    async def _persist_current_session(self) -> None:
        if self._suspend_session_capture or not self.current_session_id:
            return
        self.session_manager.save_session_snapshot(
            self.current_session_id,
            messages=self._session_messages,
            orchestrator_state=self._export_orchestrator_state(),
        )

    async def _refresh_sidebar(self) -> None:
        await self._query_sidebar().rebuild(
            self.session_manager.list_sessions(),
            active_session_id=self.current_session_id,
        )

    async def _render_session_messages(self, messages: list[dict[str, Any]]) -> None:
        log = self._query_log()
        self._suspend_session_capture = True
        try:
            log.clear()
            for entry in messages:
                if not isinstance(entry, dict):
                    continue
                kind = str(entry.get("kind") or "")
                content = str(entry.get("content") or "")
                if kind == "user":
                    log.write_user_message(content)
                elif kind == "system":
                    log.write_system_message(content, style=str(entry.get("style") or ""))
                elif kind == "workflow":
                    log.write_workflow_message(
                        content,
                        state=str(entry.get("state") or "info"),
                        animate=False,
                    )
                elif kind == "llm":
                    log.write_llm_message(
                        content,
                        markdown=bool(entry.get("markdown")) if "markdown" in entry else None,
                        language=str(entry.get("language") or "") or None,
                    )
        finally:
            self._suspend_session_capture = False

    def _append_session_entry(self, entry: dict[str, Any]) -> None:
        if self._suspend_session_capture:
            return
        self._session_messages.append(dict(entry))
        self.run_worker(
            self._persist_current_session(),
            name="persist-session",
            group="session-persist",
            exclusive=True,
        )

    def _history_has_substantive_exchange(self) -> bool:
        state = self._export_orchestrator_state()
        chat_history = list(state.get("chat_history") or [])
        has_user = any(str(item.get("role") or "") == "user" for item in chat_history if isinstance(item, dict))
        has_assistant = any(
            str(item.get("role") or "") == "assistant" for item in chat_history if isinstance(item, dict)
        )
        return has_user and has_assistant

    def _schedule_title_generation_if_needed(self) -> None:
        session_id = self.current_session_id
        if not session_id or session_id in self._title_generation_tasks:
            return

        session = self.session_manager.get_session(session_id)
        if not session:
            return

        current_title = str(session.get("title") or "").strip()
        if current_title and current_title != self.session_manager.DEFAULT_TITLE:
            return

        if not self._history_has_substantive_exchange():
            return

        self._title_generation_tasks.add(session_id)
        self.run_worker(
            self._generate_session_title(session_id),
            name=f"title-{session_id}",
            group="session-titles",
            exclusive=False,
        )

    async def _generate_session_title(self, session_id: str) -> None:
        try:
            state = self._export_orchestrator_state() if session_id == self.current_session_id else {}
            chat_history = list(state.get("chat_history") or [])
            await self.session_manager.generate_title_for_session(session_id, chat_history=chat_history)
            await self._refresh_sidebar()
        finally:
            self._title_generation_tasks.discard(session_id)

    async def _activate_session(self, session_id: str) -> None:
        session = self.session_manager.get_session(session_id)
        if session is None:
            return

        self.current_session_id = str(session.get("id") or session_id)
        self._session_messages = list(session.get("messages") or [])
        self._restore_orchestrator_state(session.get("orchestrator_state"))
        self.session_manager.set_active_session(self.current_session_id)
        await self._render_session_messages(self._session_messages)
        await self._refresh_sidebar()
        self._focus_input()

    async def _bootstrap_sessions(self) -> None:
        sessions = self.session_manager.list_sessions()
        if not sessions:
            created = self.session_manager.create_session()
            await self._activate_session(str(created["id"]))
            return

        active_session_id = self.session_manager.get_active_session_id()
        target = None
        if active_session_id:
            target = self.session_manager.get_session(active_session_id)
        if target is None:
            target = sessions[0]
        await self._activate_session(str(target.get("id") or ""))

    async def _create_new_session(self) -> None:
        created = self.session_manager.create_session()
        await self._activate_session(str(created["id"]))

    async def action_new_session(self) -> None:
        """Ctrl+N 新建空白会话。"""
        if not self._session_switch_allowed():
            self.bell()
            return
        await self._create_new_session()

    def action_toggle_sidebar(self) -> None:
        """Ctrl+B 折叠/展开侧边栏。"""
        panel = self.query_one("#sidebar_panel", Vertical)
        self._sidebar_collapsed = not self._sidebar_collapsed
        panel.set_class(self._sidebar_collapsed, "-collapsed")
        if self._sidebar_collapsed and self.focused is self._query_sidebar():
            self._focus_input()
        self.refresh(layout=True)

    def action_focus_sidebar(self) -> None:
        """Ctrl+J 聚焦会话列表。"""
        if self._sidebar_collapsed:
            self.action_toggle_sidebar()
        sidebar = self._query_sidebar()
        sidebar.focus()

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """点击或回车切换历史会话。"""
        if event.list_view.id != "sidebar":
            return

        session_id = self._query_sidebar().session_id_at(event.index)
        if not session_id:
            return
        if session_id == self.current_session_id:
            self._focus_input()
            return
        if not self._session_switch_allowed():
            self.bell()
            await self._refresh_sidebar()
            return
        await self._activate_session(session_id)

    async def _force_stop_current_process(
        self,
        process: asyncio.subprocess.Process,
        process_group_id: int | None,
    ) -> None:
        """发送 SIGTERM 后等待退出，超时则升级为强杀。"""
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
        """Ctrl + D 结束整个命令进程组。"""
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

        await self._persist_current_session()
        self.exit()

    def output_user(self, text: str) -> None:
        """输出用户输入。"""
        self._query_log().write_user_message(text)
        self._append_session_entry({"kind": "user", "content": text})

    def output_system(self, text: str, style: str = "") -> None:
        """流式输出系统日志或 Shell 命令回显。"""
        self._query_log().write_system_message(text, style=style)
        self._append_session_entry({"kind": "system", "content": text, "style": style})

    def output_workflow(self, text: str, state: str = "info") -> None:
        """输出流程状态日志。"""
        self._query_log().write_workflow_message(text, state=state)
        self._append_session_entry({"kind": "workflow", "content": text, "state": state})

    def output_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        """输出 LLM 的答复。"""
        self._query_log().write_llm_message(content, markdown=markdown, language=language)
        entry: dict[str, Any] = {"kind": "llm", "content": content}
        if markdown is not None:
            entry["markdown"] = markdown
        if language is not None:
            entry["language"] = language
        self._append_session_entry(entry)

    async def stream_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        """以逐步刷新的形式输出 LLM 内容。"""
        entry: dict[str, Any] = {"kind": "llm", "content": content}
        if markdown is not None:
            entry["markdown"] = markdown
        if language is not None:
            entry["language"] = language
        self._append_session_entry(entry)
        await self._query_log().stream_llm_message(
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

    async def _handle_user_input(self, user_input: str) -> None:
        self._session_busy = True
        try:
            await self.input_handler(user_input, self)
        finally:
            self._session_busy = False
            await self._persist_current_session()
            self._schedule_title_generation_if_needed()
            await self._refresh_sidebar()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """接收输入框提交，并交给后端处理。"""
        if event.input.id != "command_input":
            return

        user_input = event.value.strip()
        if not user_input:
            return

        input_widget = self._query_input()
        input_widget.add_to_history(user_input)

        # 如果有子进程正在运行: 说明该输入是需要传给子进程的 stdin
        if self._is_process_running():
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
            if self.current_process and self.current_process.stdin is not None:
                self.current_process.stdin.write(payload)

                async def drain_stdin() -> None:
                    try:
                        await self.current_process.stdin.drain()
                    except Exception:
                        pass

                self.run_worker(drain_stdin())
            return

        if self._session_busy:
            self.bell()
            return

        self.output_user(user_input)
        input_widget.value = ""
        self.run_worker(self._handle_user_input(user_input), group="user-input")
