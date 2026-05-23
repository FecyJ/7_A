"""Tool Agent 使用的本地 MCP Server。"""

from __future__ import annotations

import ipaddress
import fnmatch
import json
import logging
import os
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import certifi
import httpx
from mcp.server.fastmcp import FastMCP

try:
    from .security import MAX_TEXT_WRITE_CHARS, is_protected_workspace_path, resolve_workspace_path
except ImportError:
    from security import MAX_TEXT_WRITE_CHARS, is_protected_workspace_path, resolve_workspace_path  # type: ignore

mcp = FastMCP("tool-agent-server")
logging.getLogger("mcp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

MAX_TEXT_READ_CHARS = 200_000
MAX_SEARCH_FILE_BYTES = 1_000_000


def _is_private_or_local_host(hostname: str | None) -> bool:
    """拒绝 localhost、内网 IP、链路本地地址等高风险目标。"""
    if not hostname:
        return True

    normalized = hostname.strip().strip("[]").split("%", 1)[0].lower()
    if not normalized:
        return True

    if normalized in {"localhost", "0.0.0.0"} or normalized.endswith(".local"):
        return True

    try:
        ip = ipaddress.ip_address(normalized)
        return (
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
        addr_infos = socket.getaddrinfo(normalized, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False

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
            return True
    return False


def _validate_public_http_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return False, "仅允许 http:// 或 https:// URL。"
    if not parsed.netloc or not parsed.hostname:
        return False, "URL 缺少有效主机名。"
    if parsed.username or parsed.password:
        return False, "URL 中不允许包含用户名或密码。"
    if _is_private_or_local_host(parsed.hostname):
        return False, "禁止访问 localhost、内网或其他本地网络地址。"
    return True, ""


def _stringify_mapping(data: dict[str, Any] | None) -> dict[str, str] | None:
    if not data:
        return None
    return {str(key): str(value) for key, value in data.items() if value is not None}


def _preview_response_body(response: httpx.Response, max_chars: int) -> str:
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            preview = json.dumps(response.json(), ensure_ascii=False, indent=2)
        except ValueError:
            preview = response.text
    elif response.text:
        preview = response.text
    else:
        preview = f"<{len(response.content)} bytes binary content>"

    if len(preview) > max_chars:
        return preview[:max_chars].rstrip() + "...[truncated]"
    return preview


def _workspace_relative(path: Path) -> str:
    workspace = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(workspace))
    except ValueError:
        return str(path)


def _is_hidden_or_git_path(path: Path) -> bool:
    return any(part == ".git" or part.startswith(".") for part in path.parts)


def _path_entry_payload(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    except OSError:
        size = None
        mtime = None

    if path.is_dir():
        path_type = "directory"
    elif path.is_file():
        path_type = "file"
    elif path.is_symlink():
        path_type = "symlink"
    else:
        path_type = "other"

    return {
        "name": path.name,
        "path": _workspace_relative(path),
        "type": path_type,
        "size_bytes": size,
        "modified_at": mtime,
    }


def _iter_workspace_entries(
    root: Path,
    *,
    recursive: bool,
    max_depth: int,
    include_hidden: bool,
):
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            continue

        for child in children:
            rel_parts = child.relative_to(root).parts
            if not include_hidden and _is_hidden_or_git_path(Path(*rel_parts)):
                continue
            yield child
            if recursive and child.is_dir() and depth < max_depth:
                stack.append((child, depth + 1))


def _request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 12.0,
    verify_ssl: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = httpx.get(
            url,
            params=params,
            timeout=timeout,
            follow_redirects=True,
            verify=certifi.where() if verify_ssl else False,
            headers={"User-Agent": "ACEE-ToolAgent/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload, None
        return {"data": payload}, None
    except Exception as exc:
        return None, str(exc)


@mcp.tool(description="原样返回输入文本。")
def echo(message: str) -> str:
    return message


@mcp.tool(description="计算两个整数的和。")
def add(a: int, b: int) -> int:
    return a + b


@mcp.tool(description="计算两个整数的乘积。")
def multiply(a: int, b: int) -> int:
    return a * b


@mcp.tool(
    description=(
        "列出当前工作目录内指定目录的文件和文件夹。"
        "path 必须是相对路径；recursive=true 时递归列出，max_depth 控制深度。"
    )
)
def list_dir(path: str = ".", recursive: bool = False, max_depth: int = 2, max_entries: int = 100, include_hidden: bool = False) -> str:
    target = resolve_workspace_path(path)
    if target is None:
        return json.dumps({"ok": False, "error": "非法路径，仅允许当前工作目录内的相对路径。", "path": path}, ensure_ascii=False)
    if not target.exists():
        return json.dumps({"ok": False, "error": f"目录不存在：{path}", "path": path}, ensure_ascii=False)
    if not target.is_dir():
        return json.dumps({"ok": False, "error": f"目标不是目录：{path}", "path": path}, ensure_ascii=False)

    max_depth = max(0, min(int(max_depth or 0), 8))
    max_entries = max(1, min(int(max_entries or 100), 1000))
    entries = []
    truncated = False
    for item in _iter_workspace_entries(target, recursive=bool(recursive), max_depth=max_depth, include_hidden=bool(include_hidden)):
        if len(entries) >= max_entries:
            truncated = True
            break
        entries.append(_path_entry_payload(item))

    return json.dumps(
        {
            "ok": True,
            "path": _workspace_relative(target),
            "recursive": bool(recursive),
            "max_depth": max_depth,
            "entries": entries,
            "truncated": truncated,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(description="查看当前工作目录内路径的元信息，包括是否存在、类型、大小和修改时间。path 必须是相对路径。")
def stat_path(path: str) -> str:
    target = resolve_workspace_path(path)
    if target is None:
        return json.dumps({"ok": False, "exists": False, "error": "非法路径，仅允许当前工作目录内的相对路径。", "path": path}, ensure_ascii=False)
    if not target.exists():
        return json.dumps({"ok": True, "exists": False, "path": path}, ensure_ascii=False, indent=2)

    payload = _path_entry_payload(target)
    payload.update(
        {
            "ok": True,
            "exists": True,
            "is_readable": os.access(target, os.R_OK),
            "is_writable": os.access(target, os.W_OK),
        }
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool(description="获取当前系统的基本信息，包括操作系统、工作目录和 CPU 数。")
def get_system_info() -> str:
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cwd": os.getcwd(),
        "cpu_count": os.cpu_count(),
    }
    return json.dumps(info, ensure_ascii=False)


@mcp.tool(description="获取当前时间，默认使用 Asia/Shanghai 时区。")
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return json.dumps(
            {
                "ok": False,
                "error": f"未知时区：{timezone}",
                "timezone": timezone,
            },
            ensure_ascii=False,
        )

    now = datetime.now(tz)
    return json.dumps(
        {
            "ok": True,
            "source": "local_system_clock",
            "timezone": timezone,
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
            "utc_offset": now.strftime("%z"),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(description="通过 REST API 获取指定时区的当前时间。")
def get_live_time(timezone: str = "Asia/Shanghai") -> str:
    payload, error = _request_json(
        "http://timeapi.io/api/Time/current/zone",
        params={"timeZone": timezone},
        verify_ssl=False,
    )
    if error is not None or payload is None:
        return json.dumps(
            {
                "ok": False,
                "source": "timeapi.io",
                "timezone": timezone,
                "error": f"获取实时时间失败：{error}",
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "ok": True,
            "source": "timeapi.io",
            "timezone": payload.get("timeZone") or timezone,
            "date": f"{int(payload.get('year', 0)):04d}-{int(payload.get('month', 0)):02d}-{int(payload.get('day', 0)):02d}"
            if all(payload.get(key) is not None for key in ("year", "month", "day"))
            else payload.get("date"),
            "time": payload.get("time"),
            "weekday": payload.get("dayOfWeek"),
            "iso": payload.get("dateTime"),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(description="通过 REST API 获取指定城市/地区的实时天气。")
def get_weather(location: str) -> str:
    location = str(location or "").strip()
    if not location:
        return json.dumps({"ok": False, "source": "wttr.in", "error": "缺少 location 参数。"}, ensure_ascii=False)

    payload, error = _request_json(
        f"http://wttr.in/{quote(location)}",
        params={"format": "j1"},
        verify_ssl=False,
    )
    if error is not None or payload is None:
        return json.dumps(
            {
                "ok": False,
                "source": "wttr.in",
                "location": location,
                "error": f"获取天气失败：{error}",
            },
            ensure_ascii=False,
        )

    current = (payload.get("current_condition") or [{}])[0]
    nearest = (payload.get("nearest_area") or [{}])[0]
    area_name = ((nearest.get("areaName") or [{}])[0].get("value") if isinstance(nearest, dict) else None) or location
    description = ((current.get("weatherDesc") or [{}])[0].get("value") if isinstance(current, dict) else None) or "未知"
    return json.dumps(
        {
            "ok": True,
            "source": "wttr.in",
            "query": location,
            "location": area_name,
            "description": description,
            "temperature_c": current.get("temp_C"),
            "feels_like_c": current.get("FeelsLikeC"),
            "humidity": current.get("humidity"),
            "wind_kmph": current.get("windspeedKmph"),
            "observation_time": current.get("localObsDateTime") or current.get("observation_time"),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(description="通过 REST API 获取最新新闻；topic 为空时返回最新头条。")
def get_news(topic: str | None = None, limit: int = 5) -> str:
    topic = str(topic or "").strip()
    limit = max(1, min(int(limit or 5), 10))
    if topic:
        payload, error = _request_json(
            "http://hn.algolia.com/api/v1/search_by_date",
            params={"query": topic, "tags": "story", "hitsPerPage": limit},
            verify_ssl=False,
        )
    else:
        payload, error = _request_json(
            "http://hn.algolia.com/api/v1/search",
            params={"tags": "front_page", "hitsPerPage": limit},
            verify_ssl=False,
        )

    if error is not None or payload is None:
        return json.dumps(
            {
                "ok": False,
                "source": "hn.algolia.com",
                "topic": topic or None,
                "error": f"获取新闻失败：{error}",
            },
            ensure_ascii=False,
        )

    hits = payload.get("hits") or []
    articles: list[dict[str, Any]] = []
    for hit in hits[:limit]:
        if not isinstance(hit, dict):
            continue
        title = hit.get("title") or hit.get("story_title")
        url = hit.get("url") or hit.get("story_url")
        if not title:
            continue
        articles.append(
            {
                "title": title,
                "url": url,
                "author": hit.get("author"),
                "published_at": hit.get("created_at"),
                "points": hit.get("points"),
            }
        )

    return json.dumps(
        {
            "ok": True,
            "source": "hn.algolia.com",
            "topic": topic or None,
            "articles": articles,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(description="通过 REST API 获取两个币种之间的最新汇率。")
def get_exchange_rate(base: str, target: str) -> str:
    base = str(base or "").upper().strip()
    target = str(target or "").upper().strip()
    if not base or not target:
        return json.dumps(
            {"ok": False, "source": "frankfurter.app", "error": "base 和 target 不能为空。"},
            ensure_ascii=False,
        )

    payload, error = _request_json(
        "http://api.frankfurter.app/latest",
        params={"from": base, "to": target},
        verify_ssl=False,
    )
    if error is not None or payload is None:
        return json.dumps(
            {
                "ok": False,
                "source": "frankfurter.app",
                "base": base,
                "target": target,
                "error": f"获取汇率失败：{error}",
            },
            ensure_ascii=False,
        )

    rate = (payload.get("rates") or {}).get(target)
    return json.dumps(
        {
            "ok": rate is not None,
            "source": "frankfurter.app",
            "base": payload.get("base", base),
            "target": target,
            "rate": rate,
            "date": payload.get("date"),
            "amount": payload.get("amount", 1),
            "error": None if rate is not None else "返回结果中缺少目标汇率。",
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(description="统计输入文本的行数。")
def count_lines(text: str) -> int:
    return len(str(text).splitlines())


@mcp.tool(
    name="read_file",
    description=(
        "读取当前工作目录内的 UTF-8 文本文件。仅允许相对路径。"
        "可用 start_line/end_line 读取 1-based 行区间，max_chars 控制最大返回字符数。"
    ),
)
def read_file_tool(file_path: str, start_line: int | None = None, end_line: int | None = None, max_chars: int = MAX_TEXT_READ_CHARS) -> str:
    target = resolve_workspace_path(file_path)
    if target is None:
        return "读取失败：非法路径，仅允许当前工作目录内的相对路径。"
    if not target.exists() or not target.is_file():
        return f"读取失败：文件不存在 -> {file_path}"

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"读取失败：{file_path} 不是 UTF-8 文本文件。"
    except OSError as exc:
        return f"读取失败：{exc}"

    if start_line is not None or end_line is not None:
        lines = content.splitlines(keepends=True)
        start = max(1, int(start_line or 1))
        end = int(end_line or len(lines))
        if end < start:
            return "读取失败：end_line 不能小于 start_line。"
        content = "".join(lines[start - 1 : end])

    max_chars = max(1_000, min(int(max_chars or MAX_TEXT_READ_CHARS), MAX_TEXT_READ_CHARS))
    if len(content) > max_chars:
        return content[:max_chars].rstrip() + f"\n...[truncated: read_file max_chars={max_chars}]"
    return content


@mcp.tool(
    name="search_files",
    description=(
        "在当前工作目录内搜索文件名和 UTF-8 文本内容。"
        "path 必须是相对路径；glob_pattern 可限制文件范围，如 *.py 或 src/**/*.py。"
    ),
)
def search_files(
    query: str,
    path: str = ".",
    glob_pattern: str | None = None,
    max_results: int = 50,
    include_hidden: bool = False,
    search_content: bool = True,
    search_names: bool = True,
) -> str:
    query = str(query or "")
    if not query:
        return json.dumps({"ok": False, "error": "query 不能为空。"}, ensure_ascii=False)

    root = resolve_workspace_path(path)
    if root is None:
        return json.dumps({"ok": False, "error": "非法路径，仅允许当前工作目录内的相对路径。", "path": path}, ensure_ascii=False)
    if not root.exists():
        return json.dumps({"ok": False, "error": f"路径不存在：{path}", "path": path}, ensure_ascii=False)

    max_results = max(1, min(int(max_results or 50), 200))
    query_lower = query.lower()
    matches: list[dict[str, Any]] = []
    truncated = False
    files_scanned = 0

    candidates = [root] if root.is_file() else _iter_workspace_entries(root, recursive=True, max_depth=20, include_hidden=include_hidden)
    for candidate in candidates:
        if len(matches) >= max_results:
            truncated = True
            break
        if not candidate.is_file():
            continue

        rel_path = _workspace_relative(candidate)
        if glob_pattern and not fnmatch.fnmatch(rel_path, glob_pattern):
            continue
        if not include_hidden and _is_hidden_or_git_path(Path(rel_path)):
            continue

        files_scanned += 1
        if search_names and query_lower in rel_path.lower():
            matches.append({"path": rel_path, "match_type": "name", "line_number": None, "line": None})
            if len(matches) >= max_results:
                truncated = True
                break

        if not search_content:
            continue

        try:
            if candidate.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue

        for line_number, line in enumerate(lines, start=1):
            if query_lower not in line.lower():
                continue
            matches.append(
                {
                    "path": rel_path,
                    "match_type": "content",
                    "line_number": line_number,
                    "line": line[:300],
                }
            )
            if len(matches) >= max_results:
                truncated = True
                break
        if truncated:
            break

    return json.dumps(
        {
            "ok": True,
            "query": query,
            "path": _workspace_relative(root),
            "glob_pattern": glob_pattern,
            "files_scanned": files_scanned,
            "matches": matches,
            "truncated": truncated,
        },
        ensure_ascii=False,
        indent=2,
    )


def _validate_text_write_path(path: str, *, expect_directory: bool = False) -> tuple[Path | None, str | None]:
    target = resolve_workspace_path(path)
    if target is None:
        return None, "非法路径，仅允许当前工作目录内的相对路径，禁止绝对路径、~ 或 ..。"
    if is_protected_workspace_path(path):
        return None, "禁止写入 .git 内部路径。"
    if expect_directory and target.exists() and not target.is_dir():
        return None, f"目标已存在且不是目录：{path}"
    if not expect_directory and target.exists() and target.is_dir():
        return None, f"目标已存在且是目录：{path}"
    return target, None


def _text_write_result(
    *,
    ok: bool,
    action: str,
    path: str,
    error: str | None = None,
    char_count: int | None = None,
    byte_count: int | None = None,
    created_dirs: bool = False,
) -> str:
    return json.dumps(
        {
            "ok": ok,
            "action": action,
            "path": path,
            "char_count": char_count,
            "byte_count": byte_count,
            "created_dirs": created_dirs,
            "error": error,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(
    name="create_directory",
    description="在当前工作目录内创建目录。仅允许相对路径，不允许 ..、~、绝对路径或 .git 内部路径。",
)
def create_directory_tool(dir_path: str, parents: bool = True, exist_ok: bool = True) -> str:
    target, error = _validate_text_write_path(dir_path, expect_directory=True)
    if target is None:
        return _text_write_result(ok=False, action="create_directory", path=dir_path, error=error)

    try:
        target.mkdir(parents=bool(parents), exist_ok=bool(exist_ok))
    except OSError as exc:
        return _text_write_result(ok=False, action="create_directory", path=dir_path, error=str(exc))

    return _text_write_result(ok=True, action="create_directory", path=dir_path)


@mcp.tool(
    name="write_file",
    description=(
        "向当前工作目录内的 UTF-8 文本文件写入内容。"
        "仅允许相对路径；overwrite=false 时不覆盖已有文件；create_dirs=true 时自动创建父目录。"
    ),
)
def write_file_tool(file_path: str, content: str, overwrite: bool = False, create_dirs: bool = False) -> str:
    content = str(content)
    if len(content) > MAX_TEXT_WRITE_CHARS:
        return _text_write_result(
            ok=False,
            action="write_file",
            path=file_path,
            error=f"单次写入内容超过 {MAX_TEXT_WRITE_CHARS} 字符上限。",
        )

    target, error = _validate_text_write_path(file_path)
    if target is None:
        return _text_write_result(ok=False, action="write_file", path=file_path, error=error)
    if target.exists() and not overwrite:
        return _text_write_result(
            ok=False,
            action="write_file",
            path=file_path,
            error="文件已存在；如需覆盖请设置 overwrite=true。",
        )
    created_dirs = False
    if not target.parent.exists():
        if not create_dirs:
            return _text_write_result(
                ok=False,
                action="write_file",
                path=file_path,
                error="父目录不存在；如需自动创建请设置 create_dirs=true。",
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            created_dirs = True
        except OSError as exc:
            return _text_write_result(ok=False, action="write_file", path=file_path, error=f"创建父目录失败：{exc}")

    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return _text_write_result(ok=False, action="write_file", path=file_path, error=str(exc))

    return _text_write_result(
        ok=True,
        action="write_file",
        path=file_path,
        char_count=len(content),
        byte_count=len(content.encode("utf-8")),
        created_dirs=created_dirs,
    )


@mcp.tool(
    name="append_file",
    description=(
        "向当前工作目录内的 UTF-8 文本文件追加内容。"
        "仅允许相对路径；create_dirs=true 时自动创建父目录。"
    ),
)
def append_file_tool(file_path: str, content: str, create_dirs: bool = False) -> str:
    content = str(content)
    if len(content) > MAX_TEXT_WRITE_CHARS:
        return _text_write_result(
            ok=False,
            action="append_file",
            path=file_path,
            error=f"单次写入内容超过 {MAX_TEXT_WRITE_CHARS} 字符上限。",
        )

    target, error = _validate_text_write_path(file_path)
    if target is None:
        return _text_write_result(ok=False, action="append_file", path=file_path, error=error)
    created_dirs = False
    if not target.parent.exists():
        if not create_dirs:
            return _text_write_result(
                ok=False,
                action="append_file",
                path=file_path,
                error="父目录不存在；如需自动创建请设置 create_dirs=true。",
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            created_dirs = True
        except OSError as exc:
            return _text_write_result(ok=False, action="append_file", path=file_path, error=f"创建父目录失败：{exc}")

    try:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        return _text_write_result(ok=False, action="append_file", path=file_path, error=str(exc))

    return _text_write_result(
        ok=True,
        action="append_file",
        path=file_path,
        char_count=len(content),
        byte_count=len(content.encode("utf-8")),
        created_dirs=created_dirs,
    )


@mcp.tool(
    name="replace_in_file",
    description=(
        "在当前工作目录内的 UTF-8 文本文件中做精确字符串替换。"
        "默认要求 old_text 恰好出现 1 次；expected_replacements 可指定期望替换次数。"
    ),
)
def replace_in_file_tool(
    file_path: str,
    old_text: str,
    new_text: str,
    expected_replacements: int = 1,
) -> str:
    old_text = str(old_text)
    new_text = str(new_text)
    expected_replacements = int(expected_replacements or 1)
    if not old_text:
        return _text_write_result(ok=False, action="replace_in_file", path=file_path, error="old_text 不能为空。")
    if expected_replacements < 1:
        return _text_write_result(ok=False, action="replace_in_file", path=file_path, error="expected_replacements 必须大于 0。")
    if len(new_text) > MAX_TEXT_WRITE_CHARS:
        return _text_write_result(
            ok=False,
            action="replace_in_file",
            path=file_path,
            error=f"new_text 超过 {MAX_TEXT_WRITE_CHARS} 字符上限。",
        )

    target, error = _validate_text_write_path(file_path)
    if target is None:
        return _text_write_result(ok=False, action="replace_in_file", path=file_path, error=error)
    if not target.exists() or not target.is_file():
        return _text_write_result(ok=False, action="replace_in_file", path=file_path, error=f"文件不存在：{file_path}")

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _text_write_result(ok=False, action="replace_in_file", path=file_path, error="目标不是 UTF-8 文本文件。")
    except OSError as exc:
        return _text_write_result(ok=False, action="replace_in_file", path=file_path, error=str(exc))

    actual_replacements = content.count(old_text)
    if actual_replacements != expected_replacements:
        return _text_write_result(
            ok=False,
            action="replace_in_file",
            path=file_path,
            error=f"old_text 出现 {actual_replacements} 次，不等于 expected_replacements={expected_replacements}，已取消写入。",
        )

    updated = content.replace(old_text, new_text, expected_replacements)
    if len(updated) > MAX_TEXT_WRITE_CHARS:
        return _text_write_result(
            ok=False,
            action="replace_in_file",
            path=file_path,
            error=f"替换后的文件内容超过 {MAX_TEXT_WRITE_CHARS} 字符上限。",
        )

    try:
        target.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return _text_write_result(ok=False, action="replace_in_file", path=file_path, error=str(exc))

    return _text_write_result(
        ok=True,
        action="replace_in_file",
        path=file_path,
        char_count=len(updated),
        byte_count=len(updated.encode("utf-8")),
    )


@mcp.tool(
    name="http_request",
    description=(
        "向公网 HTTP/HTTPS 地址发起网络请求，适合调用公开 API 或抓取网页文本。"
        "禁止访问 localhost、内网和其他本地网络地址。"
    ),
)
def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    body: str | None = None,
    timeout: float = 10.0,
    max_chars: int = 4000,
    verify_ssl: bool = True,
) -> str:
    allowed_methods = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
    method = str(method or "GET").upper()
    if method not in allowed_methods:
        return json.dumps(
            {
                "ok": False,
                "error": f"不支持的 HTTP 方法：{method}",
                "allowed_methods": sorted(allowed_methods),
            },
            ensure_ascii=False,
        )

    is_valid, reason = _validate_public_http_url(url)
    if not is_valid:
        return json.dumps({"ok": False, "error": reason}, ensure_ascii=False)

    timeout = max(1.0, min(float(timeout or 10.0), 30.0))
    max_chars = max(200, min(int(max_chars or 4000), 100000))

    try:
        response = httpx.request(
            method,
            url,
            headers=_stringify_mapping(headers),
            params=_stringify_mapping(params),
            json=json_body,
            content=body.encode("utf-8") if body is not None else None,
            timeout=timeout,
            follow_redirects=True,
            verify=certifi.where() if verify_ssl else False,
        )
    except Exception as exc:
        error_message = f"网络请求失败：{exc}"
        if verify_ssl and "CERTIFICATE_VERIFY_FAILED" in str(exc):
            error_message += "；如确认目标可信，可重试并设置 verify_ssl=false。"
        return json.dumps(
            {
                "ok": False,
                "error": error_message,
                "method": method,
                "url": url,
            },
            ensure_ascii=False,
        )

    payload = {
        "ok": response.is_success,
        "method": method,
        "url": url,
        "final_url": str(response.url),
        "status_code": response.status_code,
        "reason_phrase": response.reason_phrase,
        "content_type": response.headers.get("content-type", ""),
        "body_preview": _preview_response_body(response, max_chars),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
