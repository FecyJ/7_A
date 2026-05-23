"""Structured per-session working memory for agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any


def _clip(text: Any, limit: int = 240) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _clip_block(text: Any, limit: int = 240) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _dedupe_append(items: list[Any], item: Any, *, max_items: int) -> None:
    if item in items:
        items.remove(item)
    items.append(item)
    del items[:-max_items]


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_string_list(values: Any, *, max_items: int) -> list[str]:
    result: list[str] = []
    for value in list(values or []):
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= max_items:
            break
    return result


def _coerce_dict_list(values: Any, *, max_items: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in list(values or []):
        if isinstance(value, dict):
            result.append(dict(value))
        if len(result) >= max_items:
            break
    return result


@dataclass
class WorkingMemory:
    """Compact execution state shared across routing and tool planning."""

    objective: str = ""
    active_paths: list[str] = field(default_factory=list)
    explored_paths: list[str] = field(default_factory=list)
    read_files: list[dict[str, Any]] = field(default_factory=list)
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    last_error: str = ""
    pending_steps: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    MAX_PATHS = 50
    MAX_READ_FILES = 30
    MAX_OBSERVATIONS = 24
    MAX_PENDING_STEPS = 24
    MAX_DECISIONS = 24

    @classmethod
    def from_dict(cls, payload: Any) -> "WorkingMemory":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            objective=str(payload.get("objective") or ""),
            active_paths=_coerce_string_list(payload.get("active_paths"), max_items=cls.MAX_PATHS),
            explored_paths=_coerce_string_list(payload.get("explored_paths"), max_items=cls.MAX_PATHS),
            read_files=_coerce_dict_list(payload.get("read_files"), max_items=cls.MAX_READ_FILES),
            tool_observations=_coerce_dict_list(payload.get("tool_observations"), max_items=cls.MAX_OBSERVATIONS),
            last_error=str(payload.get("last_error") or ""),
            pending_steps=_coerce_string_list(payload.get("pending_steps"), max_items=cls.MAX_PENDING_STEPS),
            decisions=_coerce_string_list(payload.get("decisions"), max_items=cls.MAX_DECISIONS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "active_paths": list(self.active_paths),
            "explored_paths": list(self.explored_paths),
            "read_files": [dict(item) for item in self.read_files],
            "tool_observations": [dict(item) for item in self.tool_observations],
            "last_error": self.last_error,
            "pending_steps": list(self.pending_steps),
            "decisions": list(self.decisions),
        }

    def has_content(self) -> bool:
        return any(
            [
                self.objective,
                self.active_paths,
                self.explored_paths,
                self.read_files,
                self.tool_observations,
                self.last_error,
                self.pending_steps,
                self.decisions,
            ]
        )

    def set_objective(self, objective: str) -> None:
        cleaned = _clip(objective, 500)
        if cleaned:
            self.objective = cleaned

    def record_decision(self, decision: str) -> None:
        cleaned = _clip(decision, 240)
        if cleaned:
            _dedupe_append(self.decisions, cleaned, max_items=self.MAX_DECISIONS)

    def record_error(self, message: str) -> None:
        self.last_error = _clip(message, 400)
        if self.last_error:
            self.record_decision(f"最近错误：{self.last_error}")

    def record_tool_observation(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        is_error: bool = False,
    ) -> None:
        summary = self._summarize_tool_result(tool_name, arguments, result, is_error=is_error)
        observation = {
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "summary": summary,
            "is_error": bool(is_error),
        }
        _dedupe_append(self.tool_observations, observation, max_items=self.MAX_OBSERVATIONS)
        if is_error:
            self.record_error(f"{tool_name} 调用失败：{summary}")

    def _record_path(self, path: str, *, explored: bool = False) -> None:
        cleaned = str(path or "").strip()
        if not cleaned:
            return
        _dedupe_append(self.active_paths, cleaned, max_items=self.MAX_PATHS)
        if explored:
            _dedupe_append(self.explored_paths, cleaned, max_items=self.MAX_PATHS)

    def _record_read_file(self, path: str, content: str) -> str:
        summary = self._summarize_file(path, content)
        entry = {
            "path": path,
            "summary": summary,
            "excerpt": _clip(content, 600),
        }
        self.read_files = [item for item in self.read_files if item.get("path") != path]
        self.read_files.append(entry)
        del self.read_files[:-self.MAX_READ_FILES]
        self._record_path(path)
        return summary

    def _summarize_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        *,
        is_error: bool,
    ) -> str:
        if is_error:
            return _clip(result, 360)

        payload = _safe_json_loads(result)
        if tool_name == "list_dir" and payload:
            return self._summarize_list_dir(payload)
        if tool_name == "stat_path" and payload:
            path = str(payload.get("path") or arguments.get("path") or "")
            self._record_path(path, explored=bool(payload.get("exists")))
            kind = str(payload.get("type") or "unknown")
            exists = bool(payload.get("exists"))
            return f"路径 {path} {'存在' if exists else '不存在'}，类型：{kind}"
        if tool_name == "read_file":
            path = str(arguments.get("file_path") or "")
            return self._record_read_file(path, result)
        if tool_name == "search_files" and payload:
            return self._summarize_search_files(payload)
        if tool_name in {"write_file", "append_file", "replace_in_file", "create_directory"} and payload:
            path = str(payload.get("path") or arguments.get("file_path") or arguments.get("dir_path") or "")
            if path:
                self._record_path(path)
            action = str(payload.get("action") or tool_name)
            ok = bool(payload.get("ok"))
            return f"{action} {'成功' if ok else '失败'}：{path or _clip(arguments, 120)}"

        return _clip(result, 360)

    def _summarize_list_dir(self, payload: dict[str, Any]) -> str:
        if not payload.get("ok"):
            error = str(payload.get("error") or "list_dir failed")
            path = str(payload.get("path") or "")
            self.record_error(f"list_dir {path}: {error}")
            return error

        path = str(payload.get("path") or ".")
        self._record_path(path, explored=True)
        entries = [item for item in list(payload.get("entries") or []) if isinstance(item, dict)]
        dirs = [str(item.get("path") or item.get("name") or "") for item in entries if item.get("type") == "directory"]
        files = [str(item.get("path") or item.get("name") or "") for item in entries if item.get("type") == "file"]
        for child_path in dirs[:20] + files[:20]:
            self._record_path(child_path)
        parts = [f"列出 {path}：{len(entries)} 项"]
        if dirs:
            parts.append(f"目录 {', '.join(dirs[:6])}")
        if files:
            parts.append(f"文件 {', '.join(files[:8])}")
        if payload.get("truncated"):
            parts.append("结果已截断")
        return "；".join(parts)

    def _summarize_search_files(self, payload: dict[str, Any]) -> str:
        if not payload.get("ok"):
            error = str(payload.get("error") or "search_files failed")
            self.record_error(error)
            return error

        query = str(payload.get("query") or "")
        matches = [item for item in list(payload.get("matches") or []) if isinstance(item, dict)]
        paths: list[str] = []
        for item in matches:
            path = str(item.get("path") or "").strip()
            if path and path not in paths:
                paths.append(path)
                self._record_path(path)
        return f"搜索 {query!r} 命中 {len(matches)} 处，相关路径：{', '.join(paths[:8]) or '无'}"

    @staticmethod
    def _summarize_file(path: str, content: str) -> str:
        suffix = Path(path).suffix.lower()
        if content.startswith("读取失败"):
            return _clip(content, 240)

        if suffix in {".html", ".htm"}:
            title_match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
            title = _clip(title_match.group(1), 80) if title_match else ""
            has_canvas = "<canvas" in content.lower()
            has_script = "<script" in content.lower()
            details = []
            if title:
                details.append(f"title={title}")
            if has_canvas:
                details.append("包含 canvas")
            if has_script:
                details.append("包含脚本")
            return f"{path} 是 HTML 文件" + (f"（{'，'.join(details)}）" if details else "")

        if suffix in {".md", ".markdown"}:
            heading = ""
            for line in content.splitlines():
                if line.strip().startswith("#"):
                    heading = line.strip("# ").strip()
                    break
            return f"{path} 是 Markdown 文档" + (f"（标题：{_clip(heading, 80)}）" if heading else "")

        if suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".css"}:
            lines = len(content.splitlines())
            return f"{path} 是代码/样式文件，共 {lines} 行"

        return f"{path} 已读取，约 {len(content)} 字符"

    def to_prompt(self, *, max_chars: int = 2500) -> str:
        if not self.has_content():
            return ""

        lines: list[str] = []
        if self.objective:
            lines.append(f"当前目标：{self.objective}")
        if self.explored_paths:
            lines.append("已探索路径：" + ", ".join(self.explored_paths[-12:]))
        if self.active_paths:
            lines.append("相关路径：" + ", ".join(self.active_paths[-16:]))
        if self.read_files:
            lines.append("已读文件摘要：")
            for item in self.read_files[-8:]:
                lines.append(f"- {item.get('path')}: {_clip(item.get('summary'), 180)}")
        if self.pending_steps:
            lines.append("待办：" + "；".join(self.pending_steps[-8:]))
        if self.last_error:
            lines.append(f"最近错误：{self.last_error}")
        if self.decisions:
            lines.append("关键决策：" + "；".join(self.decisions[-6:]))
        if self.tool_observations:
            lines.append("最近工具观察：")
            for item in self.tool_observations[-8:]:
                lines.append(f"- {item.get('tool_name')}: {_clip(item.get('summary'), 180)}")

        return _clip_block("\n".join(lines), max_chars)
