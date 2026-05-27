"""Tool Agent 的 MCP Client。"""

from __future__ import annotations

import json
import os
import shutil
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool

try:
    from .security import check_tool_permission
except ImportError:
    from security import check_tool_permission  # type: ignore

load_dotenv(os.getenv("DOTENV_PATH") or None)


DEFAULT_SERVER_SCRIPT = Path(__file__).with_name("mcp_server.py")


def check_mcp_permission(tool_name: str, **kwargs: Any) -> bool:
    """兼容旧接口：返回该工具是否可继续进入调用流程。"""
    return check_tool_permission(tool_name, **kwargs).status != "deny"


class MCPClient:
    """基于 stdio 的 MCP 客户端。"""

    def __init__(self, server_script_path: str | Path | None = None) -> None:
        self.server_script_path = Path(server_script_path or DEFAULT_SERVER_SCRIPT).resolve()
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._tools_cache: list[Tool] | None = None

    @staticmethod
    def _build_server_params(server_script_path: Path) -> StdioServerParameters:
        suffix = server_script_path.suffix.lower()
        if suffix == ".py":
            command = sys.executable
        elif suffix == ".js":
            command = shutil.which("node")
            if command is None:
                raise FileNotFoundError("未找到 node，可执行 .js MCP server 失败。")
        else:
            raise ValueError("Server script must be a .py or .js file")

        if not server_script_path.exists():
            raise FileNotFoundError(f"MCP server 不存在：{server_script_path}")

        return StdioServerParameters(command=command, args=[str(server_script_path)], env=None)

    async def connect(self, server_script_path: str | Path | None = None) -> None:
        if self.session is not None:
            return

        if server_script_path is not None:
            self.server_script_path = Path(server_script_path).resolve()

        server_params = self._build_server_params(self.server_script_path)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()
        self._tools_cache = None

    async def connect_to_server(self, server_script_path: str) -> None:
        """兼容旧接口。"""
        await self.connect(server_script_path)

    async def ensure_connected(self) -> None:
        if self.session is None:
            await self.connect()

    async def list_tools(self, *, refresh: bool = False) -> list[Tool]:
        await self.ensure_connected()
        if self._tools_cache is not None and not refresh:
            return self._tools_cache
        assert self.session is not None
        response = await self.session.list_tools()
        self._tools_cache = list(response.tools)
        return self._tools_cache

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        await self.ensure_connected()
        assert self.session is not None
        return await self.session.call_tool(tool_name, arguments or {})

    async def disconnect(self) -> None:
        self.session = None
        self._tools_cache = None
        await self.exit_stack.aclose()
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.disconnect()

    @staticmethod
    def tool_to_prompt_dict(tool: Tool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        }

    @staticmethod
    def result_to_text(result: CallToolResult) -> str:
        parts: list[str] = []
        for block in result.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append(getattr(block, "text", ""))
            elif hasattr(block, "model_dump"):
                parts.append(json.dumps(block.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
            else:
                parts.append(str(block))
        if not parts and result.structuredContent is not None:
            parts.append(json.dumps(result.structuredContent, ensure_ascii=False, indent=2))
        text = "\n".join(part for part in parts if part).strip()
        return text or json.dumps(result.model_dump(mode="json", exclude_none=True), ensure_ascii=False)


__all__ = ["DEFAULT_SERVER_SCRIPT", "MCPClient", "check_mcp_permission"]
