"""Tool Agent 安全策略与权限检查。"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import socket
from typing import Any, Literal
from urllib.parse import urlparse

PermissionStatus = Literal["allow", "ask", "deny"]
RiskLevel = Literal["low", "medium", "high"]

SAFE_TOOLS = {
    "add",
    "multiply",
    "list_dir",
    "get_system_info",
    "get_current_time",
    "get_live_time",
    "get_weather",
    "get_news",
    "get_exchange_rate",
    "count_lines",
    "echo",
}


@dataclass(frozen=True)
class ToolPermissionDecision:
    """单次工具调用的权限判定结果。"""

    status: PermissionStatus
    risk_level: RiskLevel
    reason: str


def normalize_tool_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """确保 arguments 总是一个字典。"""
    return dict(arguments or {})


def is_safe_workspace_path(file_path: str) -> bool:
    """仅允许当前工作目录内的相对路径。"""
    file_path = str(file_path or "").strip()
    if not file_path:
        return False

    candidate = Path(file_path)
    if candidate.is_absolute() or file_path.startswith("~"):
        return False
    if any(part == ".." for part in candidate.parts):
        return False
    return True


def resolve_workspace_path(file_path: str, *, base_dir: str | Path | None = None) -> Path | None:
    """将相对路径解析到当前工作目录内；越界则返回 None。"""
    if not is_safe_workspace_path(file_path):
        return None

    workspace = Path(base_dir or Path.cwd()).resolve()
    target = (workspace / file_path).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        return None
    return target


def is_safe_public_http_url(url: str) -> bool:
    """仅允许访问公网 HTTP/HTTPS 地址，拒绝 localhost / 内网。"""
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False

    host = parsed.hostname.strip().strip("[]").split("%", 1)[0].lower()
    if not host or host in {"localhost", "0.0.0.0"} or host.endswith(".local"):
        return False

    try:
        ip = ipaddress.ip_address(host)
        return not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    except ValueError:
        pass

    try:
        addr_infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return True

    for _, _, _, _, sockaddr in addr_infos:
        candidate = str(sockaddr[0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def check_tool_permission(tool_name: str, **kwargs: Any) -> ToolPermissionDecision:
    """对工具调用执行本地权限判断。"""
    if tool_name in SAFE_TOOLS:
        return ToolPermissionDecision("allow", "low", f"{tool_name} 属于只读/低风险工具，可直接调用。")

    if tool_name == "read_file":
        file_path = str(kwargs.get("file_path", "")).strip()
        if not file_path:
            return ToolPermissionDecision("ask", "medium", "需要先明确要读取的文件路径。")
        if not is_safe_workspace_path(file_path):
            return ToolPermissionDecision(
                "deny",
                "high",
                "read_file 仅允许读取当前工作目录内的相对路径文件，禁止绝对路径、~ 或 ..。",
            )
        return ToolPermissionDecision("ask", "medium", f"准备读取文件：{file_path}")

    if tool_name == "http_request":
        url = str(kwargs.get("url", "")).strip()
        method = str(kwargs.get("method", "GET") or "GET").upper()
        if not url:
            return ToolPermissionDecision("ask", "medium", "需要先明确要访问的 URL。")
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            return ToolPermissionDecision("deny", "high", f"不支持的 HTTP 方法：{method}")
        if not is_safe_public_http_url(url):
            return ToolPermissionDecision(
                "deny",
                "high",
                "http_request 仅允许访问公网 HTTP/HTTPS 地址，禁止 localhost、内网、绝对本地目标或携带账号密码的 URL。",
            )

        risk_level: RiskLevel = "medium" if method in {"GET", "HEAD", "OPTIONS"} else "high"
        action_text = "读取外部网页/API" if risk_level == "medium" else "向外部服务发送修改型网络请求"
        return ToolPermissionDecision(
            "ask",
            risk_level,
            f"准备通过 {method} {url} {action_text}。",
        )

    return ToolPermissionDecision("ask", "medium", f"工具 {tool_name} 未在本地白名单中，调用前需要确认。")


__all__ = [
    "PermissionStatus",
    "RiskLevel",
    "SAFE_TOOLS",
    "ToolPermissionDecision",
    "check_tool_permission",
    "is_safe_workspace_path",
    "is_safe_public_http_url",
    "normalize_tool_arguments",
    "resolve_workspace_path",
]
