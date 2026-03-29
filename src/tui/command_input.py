"""命令输入框组件"""

from __future__ import annotations

import os

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class CommandInput(TextArea):
    """支持多行输入、历史记录与 / 命令 Tab 补全的输入框。"""

    BINDINGS = [
        Binding("ctrl+c", "copy", "Copy", show=True, priority=True),
        Binding("up,ctrl+up", "history_previous", "Previous history", show=False),
        Binding("down,ctrl+down", "history_next", "Next history", show=False),
        Binding("tab", "auto_complete", "Auto complete", show=False, priority=True),
        Binding("enter", "submit", "Submit", show=False, priority=True),
        Binding("shift+enter", "insert_newline", "New line", show=False, priority=True),
        Binding("ctrl+j", "insert_newline", "New line", show=False, priority=True),
    ]

    class Submitted(Message):
        """提交输入内容。"""

        def __init__(self, input: "CommandInput", value: str) -> None:
            super().__init__()
            self.input = input
            self.value = value

        @property
        def control(self) -> "CommandInput":
            return self.input

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("show_line_numbers", False)
        kwargs.setdefault("highlight_cursor_line", False)
        kwargs.setdefault("soft_wrap", True)
        kwargs.setdefault("tab_behavior", "focus")
        super().__init__(*args, **kwargs)
        self.command_history: list[str] = []
        self.history_index: int | None = None
        self.history_draft = ""

    @property
    def value(self) -> str:
        """兼容旧 Input 风格接口。"""
        return self.text

    @value.setter
    def value(self, new_value: str) -> None:
        self.load_text(new_value)

    @property
    def cursor_position(self) -> int:
        """返回扁平化后的光标位置。"""
        return self._location_to_offset(self.cursor_location)

    @cursor_position.setter
    def cursor_position(self, offset: int) -> None:
        self.move_cursor(self._offset_to_location(offset))

    def add_to_history(self, command: str) -> None:
        """记录一条已提交命令，并重置历史浏览状态。"""
        if not command:
            return
        if not self.command_history or self.command_history[-1] != command:
            self.command_history.append(command)
        self.history_index = None
        self.history_draft = ""

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """仅当输入框存在选区时显示/启用 Copy。"""
        del parameters
        if action == "copy":
            return bool(self.selected_text)
        return True

    def _location_to_offset(self, location: tuple[int, int]) -> int:
        row, column = location
        lines = self.text.split("\n")
        row = max(0, min(row, len(lines) - 1))
        return sum(len(line) + 1 for line in lines[:row]) + min(column, len(lines[row]))

    def _offset_to_location(self, offset: int) -> tuple[int, int]:
        text = self.text
        offset = max(0, min(offset, len(text)))
        prefix = text[:offset]
        row = prefix.count("\n")
        column = len(prefix.rsplit("\n", 1)[-1])
        return (row, column)

    def _load_history_value(self, value: str) -> None:
        self.value = value
        self.cursor_position = len(value)

    def action_submit(self) -> None:
        """Enter 提交当前输入。"""
        self.post_message(self.Submitted(self, self.value))

    def action_insert_newline(self) -> None:
        """Shift+Enter / Ctrl+J 插入换行。"""
        self.insert("\n")

    def action_history_previous(self) -> None:
        """切到上一条历史命令；多行模式下保留方向键移动。"""
        if "\n" in self.value:
            self.action_cursor_up()
            return

        if not self.command_history:
            self.app.bell()
            return

        if self.history_index is None:
            self.history_draft = self.value
            self.history_index = len(self.command_history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        else:
            self.app.bell()

        self._load_history_value(self.command_history[self.history_index])

    def action_history_next(self) -> None:
        """切到下一条历史命令；多行模式下保留方向键移动。"""
        if "\n" in self.value:
            self.action_cursor_down()
            return

        if self.history_index is None:
            self.app.bell()
            return

        if self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self._load_history_value(self.command_history[self.history_index])
        else:
            self.history_index = None
            self._load_history_value(self.history_draft)

    def _command_token_context(self) -> tuple[int, int, str, bool] | None:
        """获取当前 / 命令下光标所在 token 的上下文。"""
        if not self.value.startswith("/"):
            return None

        command_text = self.value[1:]
        command_cursor = max(0, self.cursor_position - 1)
        command_cursor = min(command_cursor, len(command_text))

        token_start = command_cursor
        while token_start > 0 and not command_text[token_start - 1].isspace():
            token_start -= 1

        token_end = command_cursor
        while token_end < len(command_text) and not command_text[token_end].isspace():
            token_end += 1

        token_prefix = command_text[token_start:command_cursor]
        is_first_token = not command_text[:token_start].strip()

        return 1 + token_start, 1 + token_end, token_prefix, is_first_token

    def _complete_path_candidates(
        self,
        token_prefix: str,
        *,
        executable_only: bool = False,
    ) -> list[tuple[str, bool]]:
        """补全路径参数。"""
        expanded_prefix = os.path.expanduser(token_prefix)

        if token_prefix.endswith(os.sep):
            base_dir = expanded_prefix or "."
            partial_name = ""
            display_base = token_prefix
        else:
            base_dir = os.path.dirname(expanded_prefix) or "."
            partial_name = os.path.basename(expanded_prefix)
            display_base = token_prefix[: len(token_prefix) - len(partial_name)] if partial_name else token_prefix

        try:
            names = sorted(os.listdir(base_dir))
        except OSError:
            return []

        candidates: list[tuple[str, bool]] = []
        seen: set[str] = set()

        for name in names:
            if partial_name and not name.startswith(partial_name):
                continue

            full_path = os.path.join(base_dir, name)
            is_dir = os.path.isdir(full_path)

            if executable_only and not is_dir and not os.access(full_path, os.X_OK):
                continue

            completion = f"{display_base}{name}"
            if is_dir:
                completion += os.sep

            if completion not in seen:
                candidates.append((completion, is_dir))
                seen.add(completion)

        return candidates

    def _complete_command_candidates(self, token_prefix: str) -> list[tuple[str, bool]]:
        """补全 / 后的第一个 shell 命令。"""
        if os.sep in token_prefix or token_prefix.startswith("~"):
            return self._complete_path_candidates(token_prefix, executable_only=True)

        candidates: list[tuple[str, bool]] = []
        seen: set[str] = set()

        for directory in os.environ.get("PATH", "").split(os.pathsep):
            search_dir = directory or "."
            try:
                names = os.listdir(search_dir)
            except OSError:
                continue

            for name in names:
                if not name.startswith(token_prefix):
                    continue

                full_path = os.path.join(search_dir, name)
                if os.path.isdir(full_path) or not os.access(full_path, os.X_OK):
                    continue

                if name not in seen:
                    candidates.append((name, False))
                    seen.add(name)

        candidates.sort(key=lambda item: item[0])
        return candidates

    def _show_completion_candidates(self, candidates: list[str]) -> None:
        """将候选补全显示到日志区，避免 Tab 无反馈。"""
        if not hasattr(self.app, "output_system"):
            self.app.bell()
            return

        preview = candidates[:10]
        suffix = "" if len(candidates) <= 10 else " ..."
        self.app.output_system(f"{'    '.join(preview)}{suffix}", style="dim")

    def action_auto_complete(self) -> None:
        """仅对 / 开头的输入执行 Tab 补全。"""
        context = self._command_token_context()
        if context is None:
            self.app.bell()
            return

        token_start, token_end, token_prefix, is_first_token = context
        candidates = (
            self._complete_command_candidates(token_prefix)
            if is_first_token
            else self._complete_path_candidates(token_prefix)
        )

        if not candidates:
            self.app.bell()
            return

        candidate_values = [candidate for candidate, _ in candidates]
        common_prefix = os.path.commonprefix(candidate_values)

        if len(candidates) == 1:
            completion, is_dir = candidates[0]
            if not is_dir:
                completion += " "
            result = self.replace(
                completion,
                self._offset_to_location(token_start),
                self._offset_to_location(token_end),
                maintain_selection_offset=False,
            )
            self.move_cursor(result.end_location)
            return

        if common_prefix and common_prefix != token_prefix:
            result = self.replace(
                common_prefix,
                self._offset_to_location(token_start),
                self._offset_to_location(token_end),
                maintain_selection_offset=False,
            )
            self.move_cursor(result.end_location)
            return

        self._show_completion_candidates(candidate_values)
        self.app.bell()
