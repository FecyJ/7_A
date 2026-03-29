"""Tool Agent 使用的本地 MCP Server。"""

from __future__ import annotations

import ipaddress
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
    from .security import resolve_workspace_path
except ImportError:
    from security import resolve_workspace_path  # type: ignore

mcp = FastMCP("tool-agent-server")
logging.getLogger("mcp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


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


@mcp.tool(description="列出当前工作目录下的文件和文件夹名称。")
def list_dir() -> str:
    entries = sorted(item.name + ("/" if item.is_dir() else "") for item in Path.cwd().iterdir())
    return json.dumps(entries, ensure_ascii=False)


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
    description="读取当前工作目录内的 UTF-8 文本文件。仅允许相对路径，不允许 .. 或绝对路径。",
)
def read_file_tool(file_path: str) -> str:
    target = resolve_workspace_path(file_path)
    if target is None:
        return "读取失败：非法路径，仅允许当前工作目录内的相对路径。"
    if not target.exists() or not target.is_file():
        return f"读取失败：文件不存在 -> {file_path}"

    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"读取失败：{file_path} 不是 UTF-8 文本文件。"
    except OSError as exc:
        return f"读取失败：{exc}"


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
