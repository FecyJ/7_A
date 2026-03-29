"""日志显示相关组件"""

import asyncio
import re
from dataclasses import dataclass

from rich.console import RenderableType
from rich.highlighter import ReprHighlighter
from rich.markdown import Markdown
from rich.measure import measure_renderables
from rich.pretty import Pretty
from rich.segment import Segment
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.binding import Binding
from textual.geometry import Size
from textual.selection import Selection
from textual.strip import Strip
from textual.timer import Timer
from textual.widgets import RichLog

ERROR_PATTERN = re.compile(r"(?i)\berror\b")
MARKDOWN_PATTERN = re.compile(
    r"(^#{1,6}\s)|(^>\s)|(^[-*+]\s)|(^\d+\.\s)|(```)|(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(\[[^\]]+\]\([^)]+\))",
    re.MULTILINE,
)


@dataclass
class _LLMStreamState:
    start_line: int
    buffer: str
    markdown: bool | None
    language: str | None


@dataclass
class _WorkflowAnimationState:
    start_line: int
    line_count: int
    content: str
    state: str
    frame: int = 0


def stylize_error_keywords(text: Text) -> Text:
    """将文本中的 error 关键字标红"""
    for match in ERROR_PATTERN.finditer(text.plain):
        text.stylize("bold red", *match.span())
    return text


class AgentRichLog(RichLog):
    """用户输入、系统输出、LLM 输出的日志组件"""

    ALLOW_SELECT = True
    BINDINGS = RichLog.BINDINGS + [
        Binding("ctrl+c", "copy_selection", "Copy", show=True, priority=True),
    ]
    markdown_theme = "github-dark"
    syntax_theme = "github-dark"

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("highlight", False)
        kwargs.setdefault("markup", False)
        kwargs.setdefault("wrap", True)
        super().__init__(*args, **kwargs)
        # 高亮器
        self.llm_highlighter = ReprHighlighter()
        # 保留每一行的纯文本，供复制使用
        self._plain_lines: list[str] = []
        self._llm_stream_state: _LLMStreamState | None = None
        self._workflow_animation: _WorkflowAnimationState | None = None
        self._workflow_timer: Timer | None = None
        self._last_entry_kind: str | None = None

    @staticmethod
    def is_markdown(content: str) -> bool:
        stripped = content.strip()
        if not stripped:
            return False
        return bool(MARKDOWN_PATTERN.search(stripped))

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """仅在日志存在选区时显示/启用 Copy"""
        if action == "copy_selection":
            selection = self.text_selection
            return selection is not None and bool(self.get_selection(selection))
        return True

    def write_user_message(self, content: str) -> None:
        """用户输入：只回显，不做 markdown / error / 语法高亮"""
        self._stop_workflow_animation()
        self._prepare_entry("user")
        prompt = Text("> ", style="bold green")
        prompt.append(content)
        self.write(prompt)
        self._last_entry_kind = "user"

    def write_system_message(self, content: str, style: str = "") -> None:
        """系统输出：保留样式并将 error 标红"""
        self._stop_workflow_animation()
        self._prepare_entry("output")
        self.write(stylize_error_keywords(Text(content, style=style)))
        self._last_entry_kind = "output"

    @staticmethod
    def _workflow_prefix(state: str) -> str:
        return {
            "running": "",
            "done": "",
            "info": "• ",
            "warn": "⚠ ",
            "error": "✖ ",
        }.get(state, "• ")

    @staticmethod
    def _workflow_base_style() -> str:
        return "color(250)"

    def _build_workflow_text(
        self,
        content: str,
        state: str,
        *,
        animated: bool = False,
        frame: int = 0,
    ) -> Text:
        """构造流程日志文本；running 支持扫描高亮动画。"""
        prefix = self._workflow_prefix(state)
        base_style = self._workflow_base_style()

        is_risk_warning = state == "warn" and ("高风险" in content or "风险" in content)
        warning_style = "bold red" if "高风险" in content else "bold yellow"

        text = Text()
        if is_risk_warning:
            text.append(prefix, style=warning_style)
            text.append(content, style=warning_style)
            return text

        prefix_style = "color(245)"
        text.append(prefix, style=prefix_style)
        text.append(content, style=base_style)

        if animated and state == "running" and content:
            body_start = len(prefix)
            band_width = max(4, min(10, len(content) // 3 or 4))
            travel = len(content) + band_width + 4
            band_start = (frame % max(1, travel)) - band_width
            band_end = band_start + band_width
            clip_start = max(0, band_start)
            clip_end = min(len(content), band_end)
            if clip_start < clip_end:
                text.stylize("bold color(255)", body_start + clip_start, body_start + clip_end)

        return text

    def _replace_lines(self, start_line: int, old_line_count: int, strips: list[Strip]) -> None:
        """替换指定日志区间。"""
        self.lines = self.lines[:start_line] + strips + self.lines[start_line + old_line_count :]
        self._widest_line_width = max((strip.cell_length for strip in self.lines), default=0)
        self.virtual_size = Size(self._widest_line_width, len(self.lines))
        self._line_cache.clear()
        self.refresh()
        self.scroll_end(animate=False, immediate=True, x_axis=False)
        if self._size_known:
            self._sync_plain_lines()

    def _stop_workflow_animation(self) -> None:
        """停止当前 running 工作流动画，并还原为静态灰色文本。"""
        if self._workflow_timer is not None:
            self._workflow_timer.stop()
            self._workflow_timer = None

        state = self._workflow_animation
        if state is not None and self._size_known:
            static_text = self._build_workflow_text(state.content, state.state, animated=False)
            strips = self._render_to_strips(static_text)
            self._replace_lines(state.start_line, state.line_count, strips)

        self._workflow_animation = None

    def _advance_workflow_animation(self) -> None:
        """推进 running 日志的高亮扫描动画。"""
        state = self._workflow_animation
        if state is None or not self._size_known:
            return

        animated_text = self._build_workflow_text(
            state.content,
            state.state,
            animated=True,
            frame=state.frame,
        )
        strips = self._render_to_strips(animated_text)
        self._replace_lines(state.start_line, state.line_count, strips)
        state.line_count = len(strips)
        state.frame += 1

    def write_workflow_message(
        self,
        content: str,
        state: str = "info",
        *,
        animate: bool | None = None,
    ) -> None:
        """输出流程状态日志。"""
        self._stop_workflow_animation()
        self._prepare_entry("workflow")

        if animate is None:
            animate = state == "running"

        text = self._build_workflow_text(content, state, animated=animate)
        start_line = len(self.lines) if self._size_known else 0
        self.write(text)
        self._last_entry_kind = "workflow"

        if animate and state == "running" and self._size_known:
            line_count = max(1, len(self.lines) - start_line)
            self._workflow_animation = _WorkflowAnimationState(
                start_line=start_line,
                line_count=line_count,
                content=content,
                state=state,
            )
            self._workflow_timer = self.set_interval(0.08, self._advance_workflow_animation)

    def build_llm_renderable(
        self,
        content: RenderableType | object,
        *,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> RenderableType:
        if isinstance(content, str):
            if language:
                return Syntax(
                    content.rstrip("\n"),
                    language,
                    theme=self.syntax_theme,
                    word_wrap=True,
                    line_numbers=False,
                )

            should_render_markdown = markdown if markdown is not None else self.is_markdown(content)
            if should_render_markdown:
                return Markdown(
                    content,
                    code_theme=self.markdown_theme,
                    inline_code_theme=self.syntax_theme,
                )

            text = Text(content)
            self.llm_highlighter.highlight(text)
            return text

        if isinstance(content, Text):
            text = content.copy()
            self.llm_highlighter.highlight(text)
            return text

        if hasattr(content, "__rich_console__") or hasattr(content, "__rich__"):
            return content

        return Pretty(content)

    def write_llm_message(
        self,
        content: RenderableType | object,
        *,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        """LLM 输出入口：可选 markdown / syntax 高亮"""
        self._stop_workflow_animation()
        self._prepare_entry("output")
        renderable = self._wrap_llm_renderable(
            self.build_llm_renderable(content, markdown=markdown, language=language)
        )
        self.write(renderable)
        self._last_entry_kind = "output"

    @staticmethod
    def _iter_stream_chunks(content: str, chunk_size: int = 6):
        """将文本切成较自然的小块，用于模拟流式输出。"""
        buffer = ""
        punctuation = "，。！？；：,.!?;:)]}】）"
        for char in content:
            buffer += char
            if char == "\n":
                yield buffer
                buffer = ""
                continue
            if len(buffer) >= chunk_size and (char.isspace() or char in punctuation):
                yield buffer
                buffer = ""
                continue
            if len(buffer) >= chunk_size * 2:
                yield buffer
                buffer = ""
        if buffer:
            yield buffer

    def _render_to_strips(self, renderable: RenderableType) -> list[Strip]:
        """将 renderable 渲染成 strips，供流式覆盖最后一段日志。"""
        renderable = self._make_renderable(renderable)
        console = self.app.console
        render_options = console.options

        if isinstance(renderable, Text) and not self.wrap:
            render_options = render_options.update(overflow="ignore", no_wrap=True)

        renderable_width = measure_renderables(console, render_options, [renderable]).maximum
        scrollable_content_width = self.scrollable_content_region.width
        render_width = renderable_width
        if scrollable_content_width and renderable_width > scrollable_content_width:
            render_width = min(renderable_width, scrollable_content_width)
        render_width = max(render_width, self.min_width)
        render_options = render_options.update_width(render_width)

        segments = console.render(renderable, render_options)
        lines = list(Segment.split_lines(segments))
        if not lines:
            return [Strip.blank(render_width)]

        strips = Strip.from_lines(lines)
        for strip in strips:
            strip.adjust_cell_length(render_width)
        return strips

    def _rerender_active_stream(self, *, final: bool = False) -> None:
        state = self._llm_stream_state
        if state is None:
            return

        renderable = self.build_llm_renderable(
            state.buffer,
            markdown=state.markdown if final else False,
            language=state.language if final else None,
        )
        strips = self._render_to_strips(self._wrap_llm_renderable(renderable))
        self.lines = self.lines[: state.start_line] + strips

        if self.max_lines is not None and len(self.lines) > self.max_lines:
            overflow = len(self.lines) - self.max_lines
            self._start_line += overflow
            self.lines = self.lines[overflow:]
            state.start_line = max(0, state.start_line - overflow)

        self._widest_line_width = max((strip.cell_length for strip in self.lines), default=0)
        self.virtual_size = Size(self._widest_line_width, len(self.lines))
        self._line_cache.clear()
        self.refresh()
        self.scroll_end(animate=False, immediate=True, x_axis=False)
        if self._size_known:
            self._sync_plain_lines()

    async def stream_llm_message(
        self,
        content: str,
        *,
        markdown: bool | None = None,
        language: str | None = None,
        chunk_delay: float = 0.01,
    ) -> None:
        """以单条日志逐步刷新的形式输出 LLM 内容。"""
        self._stop_workflow_animation()
        self._prepare_entry("output")
        if not isinstance(content, str) or not content:
            self.write_llm_message(content, markdown=markdown, language=language)
            return
        if not self._size_known:
            self.write_llm_message(content, markdown=markdown, language=language)
            return

        self._llm_stream_state = _LLMStreamState(
            start_line=len(self.lines),
            buffer="",
            markdown=markdown,
            language=language,
        )
        self._last_entry_kind = "output"
        for chunk in self._iter_stream_chunks(content):
            if self._llm_stream_state is None:
                break
            self._llm_stream_state.buffer += chunk
            self._rerender_active_stream(final=False)
            await asyncio.sleep(chunk_delay)

        if self._llm_stream_state is not None:
            self._rerender_active_stream(final=True)
        self._llm_stream_state = None

    def _trailing_blank_lines(self) -> int:
        """统计当前日志末尾连续空行数。"""
        count = 0
        for line in reversed(self._plain_lines):
            if line.strip():
                break
            count += 1
        return count

    def _set_trailing_blank_lines(self, desired: int) -> None:
        """将日志末尾空行数调整到目标值。"""
        if not self._size_known:
            for _ in range(desired):
                super().write("")
            return

        current = self._trailing_blank_lines()
        while current > desired and self.lines:
            self.lines.pop()
            current -= 1
        while current < desired:
            super().write("")
            current += 1

        self._widest_line_width = max((strip.cell_length for strip in self.lines), default=0)
        self.virtual_size = Size(self._widest_line_width, len(self.lines))
        self._line_cache.clear()
        self.refresh()
        self.scroll_end(animate=False, immediate=True, x_axis=False)
        self._sync_plain_lines()

    def _desired_gap_before(self, entry_kind: str) -> int:
        """根据上一条类型决定当前条目前的空行数。"""
        previous = self._last_entry_kind
        if previous is None:
            return 0
        if entry_kind == "user":
            return 2
        if entry_kind == "workflow":
            if previous == "workflow":
                return 0
            if previous == "output":
                return 1
            if previous == "user":
                return 2
        if entry_kind == "output":
            if previous == "workflow":
                return 1
            if previous == "output":
                return 0
            if previous == "user":
                return 2
        return 0

    def _prepare_entry(self, entry_kind: str) -> None:
        """在写入新顶层条目前，根据类型插入合适空行。"""
        desired_gap = self._desired_gap_before(entry_kind)
        self._set_trailing_blank_lines(desired_gap)

    @staticmethod
    def _wrap_llm_renderable(renderable: RenderableType) -> RenderableType:
        """为 agent 输出加上前导 bullet。"""
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=1, no_wrap=True)
        table.add_column(ratio=1)
        table.add_row(Text("•", style="color(245)"), renderable)
        return table

    def _sync_plain_lines(self) -> None:
        """同步当前日志的纯文本行，用于选择与复制is_markdown"""
        self._plain_lines = [
            "".join(segment.text for segment in strip if not segment.control)
            for strip in self.lines
        ]

    def write(self, *args, **kwargs):
        """写入日志后，同步可复制纯文本is_markdown"""
        result = super().write(*args, **kwargs)
        if self._size_known:
            self._sync_plain_lines()
        return result

    def clear(self):
        """清空日志与纯文本缓存is_markdown"""
        self._stop_workflow_animation()
        result = super().clear()
        self._plain_lines.clear()
        self._llm_stream_state = None
        self._last_entry_kind = None
        return result

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """返回当前选区中的纯文本is_markdown"""
        if not self._plain_lines:
            return None
        return selection.extract("\n".join(self._plain_lines)), "\n"

    def selection_updated(self, selection: Selection | None) -> None:
        """选区变化时刷新渲染与快捷键状态is_markdown"""
        self._line_cache.clear()
        self.refresh()
        self.refresh_bindings()

    def action_copy_selection(self) -> None:
        """复制当前日志选区"""
        self.screen.action_copy_text()

    def _render_line(self, y: int, scroll_x: int, width: int) -> Strip:
        """渲染单行，并在选择时应用选中高亮is_markdown"""
        if y >= len(self.lines):
            return Strip.blank(width, self.rich_style)

        selection = self.text_selection
        cache_key = (y + self._start_line, scroll_x, width, self._widest_line_width)
        if selection is None and cache_key in self._line_cache:
            return self._line_cache[cache_key]

        line = self.lines[y]
        if selection is not None:
            text = Text()
            for segment in line:
                if segment.control:
                    continue
                text.append(segment.text, segment.style)

            if (select_span := selection.get_span(y)) is not None:
                start, end = select_span
                if end == -1:
                    end = len(text)
                text.stylize(self.screen.selection_style, start, end)

            line = Strip(text.render(self.app.console), text.cell_len)

        line = line.crop_extend(scroll_x, scroll_x + width, self.rich_style).apply_offsets(scroll_x, y)

        if selection is None:
            self._line_cache[cache_key] = line
        return line
