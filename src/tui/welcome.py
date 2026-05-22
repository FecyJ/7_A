"""启动欢迎界面。"""

from __future__ import annotations

from rich.align import Align
from rich.console import Group, RenderableType
from rich.text import Text
from textual.widgets import Static


class WelcomeSplash(Static):
    """空白会话时显示的启动页。"""

    DEFAULT_CSS = """
    WelcomeSplash {
        height: 1fr;
        border: solid green;
        padding: 1 2;
        color: $text;
        background: $surface;
    }
    """

    def render(self) -> RenderableType:
        title = Text("欢迎使用 ACEE Multi-Agent CLI", style="bold color(255)")
        title.justify = "center"

        subtitle = Text("输入自然语言任务，或输入 / + 命令直接执行", style="color(248)")
        subtitle.justify = "center"

        hint = Text("Enter 提交 · Ctrl+J 换行 · Ctrl+N 新建会话", style="color(245)")
        hint.justify = "center"

        group = Group(
            title,
            Text(""),
            subtitle,
            Text(""),
            hint,
        )
        return Align.center(group, vertical="middle")
