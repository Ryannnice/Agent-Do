import http.client
import json
import os
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import time
import uuid
from hashlib import sha256
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field


def load_local_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_local_env()


def env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


DATA_ROOT = Path(os.getenv("AGENT_DATA_ROOT", "./data")).resolve()
HOST_AGENT_DATA_ROOT = Path(env_or_default("HOST_AGENT_DATA_ROOT", str(DATA_ROOT))).resolve()
SESSIONS_ROOT = DATA_ROOT / "agent-sessions"
DATABASE_PATH = DATA_ROOT / "app.db"
STATIC_ROOT = Path(__file__).resolve().parent / "static"

CLAUDE_DOCKER_IMAGE = os.getenv("CLAUDE_DOCKER_IMAGE", "claude-runtime:latest")
CLAUDE_TIMEOUT_SECONDS = int(os.getenv("CLAUDE_TIMEOUT_SECONDS", "900"))
CLAUDE_MEMORY = os.getenv("CLAUDE_MEMORY", "2g")
CLAUDE_CPUS = os.getenv("CLAUDE_CPUS", "1")
APP_RUNTIME_IMAGE = env_or_default("APP_RUNTIME_IMAGE", "node:20-slim")
APP_RUNTIME_MEMORY = env_or_default("APP_RUNTIME_MEMORY", "2g")
APP_RUNTIME_CPUS = env_or_default("APP_RUNTIME_CPUS", "1")
APP_RUNTIME_INTERNAL_PORT = int(env_or_default("APP_RUNTIME_INTERNAL_PORT", "3000"))
PYTHON_RUNTIME_IMAGE = env_or_default("PYTHON_RUNTIME_IMAGE", "python:3.11-slim")
PYTHON_RUNTIME_INTERNAL_PORT = int(env_or_default("PYTHON_RUNTIME_INTERNAL_PORT", "8000"))
APP_RUNTIME_START_TIMEOUT_SECONDS = int(env_or_default("APP_RUNTIME_START_TIMEOUT_SECONDS", "30"))
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "demo-user")
CONTAINER_UID = env_or_default("CLAUDE_CONTAINER_UID", str(os.getuid()))
CONTAINER_GID = env_or_default("CLAUDE_CONTAINER_GID", str(os.getgid()))
CONTAINER_HOME = env_or_default("CLAUDE_CONTAINER_HOME", "/home/agent")
MAX_FILE_PREVIEW_BYTES = int(env_or_default("MAX_FILE_PREVIEW_BYTES", str(64 * 1024)))
DEFAULT_RUNTIME_PROFILE = env_or_default("DEFAULT_RUNTIME_PROFILE", "aliyun")
DEFAULT_CLAUDE_MODEL = env_or_default("DEFAULT_CLAUDE_MODEL", "sonnet")
ALIYUN_ANTHROPIC_BASE_URL = env_or_default(
    "ALIYUN_ANTHROPIC_BASE_URL",
    "https://dashscope.aliyuncs.com/apps/anthropic",
)
ALIYUN_ANTHROPIC_API_KEY = env_or_default("ALIYUN_ANTHROPIC_API_KEY", "")
ALIYUN_ANTHROPIC_AUTH_TOKEN = env_or_default("ALIYUN_ANTHROPIC_AUTH_TOKEN", "")
ALIYUN_ANTHROPIC_MODEL = env_or_default("ALIYUN_ANTHROPIC_MODEL", "qwen3-coder-next")
DEFAULT_APPEND_SYSTEM_PROMPT = (
    "When you create or modify files, verify the result before claiming success. "
    "Re-read the changed files and only say a file was updated if the workspace contents actually reflect the change. "
    "If no file was changed, say so explicitly. "
    "Agent-Do browser preview supports static HTML, Node web apps, and Python HTTP apps. "
    "If the user expects an in-browser preview, prefer HTML/Canvas/JavaScript or a web app. "
    "Avoid pygame, tkinter, curses, turtle, or other desktop Python UI/game frameworks unless the user explicitly asks for a desktop app. "
    "When generating a previewable Python project, prefer FastAPI, Flask, Streamlit, or another HTTP server, and add .agentdo/project.json with runtime/start/install/port when useful."
)
CLAUDE_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
]
HTTP_RUNTIME_MODES = {"node", "python_web"}
IGNORED_WORKSPACE_DIRS = {
    ".agentdo",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
}
PYTHON_DESKTOP_IMPORTS = {"arcade", "curses", "pygame", "pyglet", "tkinter", "turtle", "ursina"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_data_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_runtime_mount_source(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(DATA_ROOT)
    except ValueError:
        return resolved
    return (HOST_AGENT_DATA_ROOT / relative).resolve()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                workspace_path TEXT NOT NULL,
                home_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_active_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                exit_code INTEGER,
                duration_ms INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS runtimes (
                session_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                entry_file TEXT,
                container_name TEXT,
                host_port INTEGER,
                internal_port INTEGER,
                install_command TEXT,
                start_command TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            """
        )


def row_to_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def fetch_session(session_id: str) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return row_to_dict(row)


def has_assistant_reply(session_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM messages WHERE session_id = ? AND role = 'assistant' LIMIT 1",
            (session_id,),
        ).fetchone()
    return row is not None


def fetch_runtime_record(session_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM runtimes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return row_to_dict(row)


def upsert_runtime_record(
    session_id: str,
    mode: str,
    status: str,
    entry_file: str | None = None,
    container_name: str | None = None,
    host_port: int | None = None,
    internal_port: int | None = None,
    install_command: str | None = None,
    start_command: str | None = None,
    last_error: str | None = None,
) -> dict:
    existing = fetch_runtime_record(session_id)
    now = utc_now()
    payload = {
        "session_id": session_id,
        "mode": mode,
        "status": status,
        "entry_file": entry_file,
        "container_name": container_name,
        "host_port": host_port,
        "internal_port": internal_port,
        "install_command": install_command,
        "start_command": start_command,
        "last_error": last_error,
        "created_at": existing["created_at"] if existing else now,
        "updated_at": now,
    }
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO runtimes (
                session_id, mode, status, entry_file, container_name, host_port,
                internal_port, install_command, start_command, last_error, created_at, updated_at
            )
            VALUES (
                :session_id, :mode, :status, :entry_file, :container_name, :host_port,
                :internal_port, :install_command, :start_command, :last_error, :created_at, :updated_at
            )
            ON CONFLICT(session_id) DO UPDATE SET
                mode=excluded.mode,
                status=excluded.status,
                entry_file=excluded.entry_file,
                container_name=excluded.container_name,
                host_port=excluded.host_port,
                internal_port=excluded.internal_port,
                install_command=excluded.install_command,
                start_command=excluded.start_command,
                last_error=excluded.last_error,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        conn.commit()
    return fetch_runtime_record(session_id) or payload


def insert_message(
    session_id: str,
    role: str,
    content: str,
    exit_code: int | None = None,
    duration_ms: int | None = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (id, session_id, role, content, exit_code, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                session_id,
                role,
                content,
                exit_code,
                duration_ms,
                utc_now(),
            ),
        )
        conn.execute(
            "UPDATE sessions SET last_active_at = ? WHERE id = ?",
            (utc_now(), session_id),
        )
        conn.commit()


def delete_session_data(session_id: str) -> dict:
    session = fetch_session(session_id)
    runtime_record = fetch_runtime_record(session_id)
    container_name = runtime_record["container_name"] if runtime_record and runtime_record.get("container_name") else runtime_container_name(session_id)
    remove_runtime_container(container_name)

    session_root = Path(session["workspace_path"]).resolve().parent

    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM runtimes WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()

    shutil.rmtree(session_root, ignore_errors=True)
    return session


def snapshot_workspace(workspace_path: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not workspace_path.exists():
        return snapshot

    for path in sorted(workspace_path.rglob("*")):
        if not path.is_file():
            continue
        relpath = str(path.relative_to(workspace_path))
        digest = sha256(path.read_bytes()).hexdigest()
        snapshot[relpath] = digest
    return snapshot


def diff_workspace(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    before_keys = set(before)
    after_keys = set(after)

    added = sorted(after_keys - before_keys)
    deleted = sorted(before_keys - after_keys)
    modified = sorted(path for path in before_keys & after_keys if before[path] != after[path])
    changed_files = added + modified + deleted

    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
        "changed_files": changed_files,
    }


def should_flag_missing_changes(prompt: str, output: str) -> bool:
    text = f"{prompt}\n{output}".lower()
    indicators = [
        "创建",
        "修改",
        "改名",
        "重命名",
        "写入",
        "保存",
        "文件",
        "html",
        "readme",
        "created",
        "updated",
        "modified",
        "renamed",
        "saved",
        "wrote",
        "file",
    ]
    return any(token in text for token in indicators)


def maybe_annotate_output(prompt: str, output: str, workspace_diff: dict[str, list[str]]) -> str:
    if workspace_diff["changed_files"]:
        changed = ", ".join(workspace_diff["changed_files"][:10])
        suffix = f"\n\n[Backend note: Workspace changed files: {changed}]"
        return f"{output}{suffix}"

    if should_flag_missing_changes(prompt, output):
        suffix = "\n\n[Backend note: No workspace file changes were detected during this run.]"
        return f"{output}{suffix}"

    return output


def truncate_text(value: str, limit: int = 240) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(limit - 3, 0)]}..."


def strip_tool_error_markup(value: str) -> str:
    text = value or ""
    text = re.sub(r"</?tool_use_error>", "", text, flags=re.IGNORECASE)
    return text.strip()


def flatten_text_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [item for item in (flatten_text_value(item) for item in value) if item]
        return "\n".join(parts) if parts else None
    if isinstance(value, dict):
        for key in ("text", "content", "result", "output", "value"):
            if key in value:
                text = flatten_text_value(value[key])
                if text:
                    return text
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def extract_message_content_blocks(payload: dict) -> list[dict]:
    blocks: list[dict] = []

    def extend_from_container(container) -> None:
        if isinstance(container, dict):
            content = container.get("content")
            if isinstance(content, list):
                blocks.extend(item for item in content if isinstance(item, dict))
        elif isinstance(container, list):
            for item in container:
                extend_from_container(item)

    for key in ("message", "messages", "content"):
        extend_from_container(payload.get(key))

    return blocks


def extract_assistant_text(payload: dict) -> str | None:
    texts = [
        block["text"]
        for block in extract_message_content_blocks(payload)
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    if texts:
        return "".join(texts)

    result = payload.get("result")
    if isinstance(result, str):
        return result
    return None


def summarize_tool_input(value) -> str | None:
    if value is None:
        return None

    if isinstance(value, dict):
        preferred_keys = ("command", "cmd", "file_path", "path", "prompt", "query")
        parts: list[str] = []
        for key in preferred_keys:
            if key not in value:
                continue
            raw = value[key]
            if raw is None or raw == "":
                continue
            text = raw if isinstance(raw, str) else flatten_text_value(raw)
            if text:
                parts.append(f"{key}: {truncate_text(text, 120)}")
            if len(parts) >= 2:
                return "; ".join(parts)

        if parts:
            return "; ".join(parts)

        rendered = flatten_text_value(value)
        return truncate_text(rendered, 220) if rendered else None

    rendered = flatten_text_value(value)
    return truncate_text(rendered, 220) if rendered else None


def extract_stream_text_delta(payload: dict) -> str | None:
    if payload.get("type") != "stream_event":
        return None

    event = payload.get("event")
    if not isinstance(event, dict) or event.get("type") != "content_block_delta":
        return None

    delta = event.get("delta")
    if not isinstance(delta, dict):
        return None

    if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
        return delta["text"]
    return None


def extract_failure_message(payload: dict) -> str | None:
    tool_use_result = payload.get("tool_use_result")
    if isinstance(tool_use_result, str) and tool_use_result.strip():
        return strip_tool_error_markup(tool_use_result)

    if payload.get("type") == "user":
        for block in extract_message_content_blocks(payload):
            if block.get("type") != "tool_result":
                continue
            content = strip_tool_error_markup(flatten_text_value(block.get("content")) or "")
            if not content:
                continue
            if block.get("is_error") or "error" in content.lower():
                return content

    result_text = payload.get("result")
    if isinstance(result_text, str):
        cleaned = strip_tool_error_markup(result_text)
        if cleaned and payload.get("subtype") in {"error", "failed"}:
            return cleaned

    return None


def summarize_stream_event(payload: dict) -> list[dict]:
    event = payload.get("event")
    if not isinstance(event, dict):
        return []

    event_type = str(event.get("type") or "")
    if event_type == "content_block_start":
        block = event.get("content_block")
        if not isinstance(block, dict):
            return []
        block_type = str(block.get("type") or "")
        if block_type == "tool_use":
            tool_name = block.get("name") or block.get("tool_name") or "tool"
            return [
                {
                    "kind": "tool",
                    "summary": f"Calling {tool_name}",
                    "detail": summarize_tool_input(block.get("input")),
                }
            ]
        if block_type in {"thinking", "redacted_thinking"}:
            return [
                {
                    "kind": "thinking",
                    "summary": "workshop is analyzing the task",
                    "detail": None,
                }
            ]
        return []

    if event_type == "message_delta":
        return []

    # Ignore low-level content deltas such as input_json_delta.
    return []


def summarize_claude_event(payload: dict) -> list[dict]:
    event_type = str(payload.get("type") or "message")
    subtype = str(payload.get("subtype") or "").strip()
    entries: list[dict] = []

    if event_type == "stream_event":
        return summarize_stream_event(payload)

    if event_type == "system":
        detail_parts = []
        if payload.get("model"):
            detail_parts.append(f"model: {payload['model']}")
        if payload.get("cwd"):
            detail_parts.append(f"cwd: {payload['cwd']}")
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            detail_parts.append(f"tools: {len(tools)}")
        entries.append(
            {
                "kind": "system",
                "summary": "workshop runtime initialized",
                "detail": "; ".join(detail_parts) or (subtype or None),
            }
        )
        return entries

    if event_type == "assistant":
        for block in extract_message_content_blocks(payload):
            block_type = str(block.get("type") or "")
            if block_type == "tool_use":
                tool_name = block.get("name") or block.get("tool_name") or "tool"
                entries.append(
                    {
                        "kind": "tool",
                        "summary": f"Calling {tool_name}",
                        "detail": summarize_tool_input(block.get("input")),
                    }
                )
            elif block_type in {"thinking", "redacted_thinking"}:
                entries.append(
                    {
                        "kind": "thinking",
                        "summary": "workshop is analyzing the task",
                        "detail": None,
                    }
                )
        return entries

    if event_type == "user":
        for block in extract_message_content_blocks(payload):
            block_type = str(block.get("type") or "")
            if block_type == "tool_result":
                detail = strip_tool_error_markup(flatten_text_value(block.get("content")) or "")
                entries.append(
                    {
                        "kind": "error" if block.get("is_error") else "tool_result",
                        "summary": "Tool returned error" if block.get("is_error") else "Tool returned output",
                        "detail": truncate_text(detail, 220) or None,
                    }
                )
        return entries

    if event_type == "result":
        detail_parts = []
        if payload.get("duration_ms") is not None:
            detail_parts.append(f"duration: {payload['duration_ms']} ms")
        if payload.get("num_turns") is not None:
            detail_parts.append(f"turns: {payload['num_turns']}")
        entries.append(
            {
                "kind": "result",
                "summary": f"workshop finished ({subtype or 'done'})",
                "detail": "; ".join(detail_parts) or None,
            }
        )
        return entries

    entries.append(
        {
            "kind": "log",
            "summary": f"workshop event: {event_type}",
            "detail": subtype or None,
        }
    )
    return entries


def build_workspace_change_items(
    session_id: str,
    workspace_path: Path,
    workspace_diff: dict[str, list[str]],
) -> list[dict]:
    items: list[dict] = []
    for status, paths in (
        ("added", workspace_diff["added"]),
        ("modified", workspace_diff["modified"]),
        ("deleted", workspace_diff["deleted"]),
    ):
        for relpath in paths:
            item = {"path": relpath, "status": status}
            if status != "deleted":
                target = workspace_path / relpath
                if target.exists() and target.is_file():
                    item["size"] = target.stat().st_size
                    item["preview_url"] = f"/sessions/{session_id}/workspace/files/{quote(relpath, safe='/')}"
            items.append(item)
    return items


def read_workspace_file_preview(target: Path, workspace_path: Path) -> dict:
    raw = target.read_bytes()
    truncated = len(raw) > MAX_FILE_PREVIEW_BYTES
    preview_bytes = raw[:MAX_FILE_PREVIEW_BYTES]
    binary = b"\x00" in preview_bytes
    content = None

    if not binary:
        content = preview_bytes.decode("utf-8", errors="replace")

    return {
        "path": str(target.relative_to(workspace_path)),
        "size": len(raw),
        "binary": binary,
        "truncated": truncated,
        "content": content,
        "language": target.suffix.lstrip(".").lower() or "text",
    }


def run_command(
    command: list[str],
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Missing runtime dependency: {exc}") from exc

    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Command failed"
        raise HTTPException(status_code=500, detail=detail)
    return completed


def normalize_shell_command(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        return shlex.join([str(part) for part in value])
    return None


def normalize_relative_path(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("\\", "/")
    return text or None


def parse_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        return data
    return None


def detect_static_entry(workspace_path: Path) -> str | None:
    direct_index = workspace_path / "index.html"
    if direct_index.exists():
        return "index.html"

    html_files = sorted(
        str(path.relative_to(workspace_path))
        for path in workspace_path.rglob("*.html")
        if path.is_file() and ".agentdo" not in path.parts
    )
    if len(html_files) == 1:
        return html_files[0]
    return None


def resolve_workspace_relative_path(workspace_path: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    target = (workspace_path / relative_path).resolve()
    try:
        target.relative_to(workspace_path)
    except ValueError:
        return None
    return target


def normalize_runtime_mode(value) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "html": "static",
        "static": "static",
        "static_html": "static",
        "web_static": "static",
        "js": "node",
        "javascript": "node",
        "node": "node",
        "nodejs": "node",
        "python": "python_web",
        "python_http": "python_web",
        "python_web": "python_web",
        "fastapi": "python_web",
        "flask": "python_web",
        "streamlit": "python_web",
        "django": "python_web",
        "python_desktop": "python_desktop",
        "python_game": "python_desktop",
        "desktop_python": "python_desktop",
        "desktop_game": "python_desktop",
        "pygame": "python_desktop",
        "pyglet": "python_desktop",
        "arcade": "python_desktop",
        "tkinter": "python_desktop",
        "curses": "python_desktop",
        "turtle": "python_desktop",
    }
    return aliases.get(normalized)


def is_http_runtime_mode(mode: str | None) -> bool:
    return bool(mode in HTTP_RUNTIME_MODES)


def runtime_image_for_mode(mode: str) -> str:
    if mode == "node":
        return APP_RUNTIME_IMAGE
    if mode == "python_web":
        return PYTHON_RUNTIME_IMAGE
    raise HTTPException(status_code=500, detail=f"Unsupported runtime mode: {mode}")


def default_internal_port_for_mode(mode: str | None) -> int:
    return PYTHON_RUNTIME_INTERNAL_PORT if mode == "python_web" else APP_RUNTIME_INTERNAL_PORT


def should_ignore_workspace_path(path: Path) -> bool:
    return any(part in IGNORED_WORKSPACE_DIRS for part in path.parts)


def list_python_files(workspace_path: Path, preferred_entry: str | None = None) -> list[Path]:
    files = [
        path
        for path in workspace_path.rglob("*.py")
        if path.is_file() and not should_ignore_workspace_path(path.relative_to(workspace_path))
    ]
    preferred_target = resolve_workspace_relative_path(workspace_path, preferred_entry)

    def sort_key(path: Path) -> tuple[int, int, str]:
        relpath = path.relative_to(workspace_path)
        priority = 5
        if preferred_target and path == preferred_target:
            priority = 0
        elif relpath.name == "manage.py":
            priority = 1
        elif relpath.name in {"app.py", "main.py", "server.py", "run.py"}:
            priority = 2
        return (priority, len(relpath.parts), str(relpath))

    return sorted(files, key=sort_key)


def read_text_if_possible(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def extract_python_import_roots(source: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", source, re.MULTILINE)
    }


def python_module_from_path(workspace_path: Path, file_path: Path) -> str | None:
    relpath = file_path.relative_to(workspace_path).with_suffix("")
    parts = list(relpath.parts)
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def build_python_install_command(workspace_path: Path, packages: list[str] | None = None) -> str | None:
    requirements = workspace_path / "requirements.txt"
    if requirements.exists():
        return "python -m pip install --disable-pip-version-check -r requirements.txt"
    packages = [package for package in (packages or []) if package]
    if packages:
        deduped = list(dict.fromkeys(packages))
        return "python -m pip install --disable-pip-version-check " + shlex.join(deduped)
    return None


def build_python_desktop_spec(
    entry_file: str | None,
    message: str | None = None,
) -> dict:
    return {
        "mode": "python_desktop",
        "entry_file": entry_file,
        "install_command": None,
        "start_command": None,
        "internal_port": None,
        "message": message
        or "检测到 Python 桌面/pygame/tkinter 游戏项目，当前浏览器预览不支持这类本地窗口程序。请改为 HTML/Canvas、Node Web，或 Python HTTP 项目。",
    }


def detect_python_runtime_spec(
    workspace_path: Path,
    preferred_entry: str | None = None,
    runtime_hint: str | None = None,
    install_command_override: str | None = None,
    start_command_override: str | None = None,
    internal_port: int | None = None,
) -> dict | None:
    python_files = list_python_files(workspace_path, preferred_entry=preferred_entry)
    if not python_files and runtime_hint not in {"python_web", "python_desktop"}:
        return None

    port = internal_port or PYTHON_RUNTIME_INTERNAL_PORT
    first_python_file = None
    desktop_entry = None
    desktop_modules: set[str] = set()
    fastapi_candidate = None
    flask_candidate = None
    streamlit_candidate = None
    django_candidate = None

    for path in python_files:
        relpath = str(path.relative_to(workspace_path))
        source = read_text_if_possible(path)
        imports = extract_python_import_roots(source)
        if first_python_file is None:
            first_python_file = relpath

        blocked_modules = imports & PYTHON_DESKTOP_IMPORTS
        if blocked_modules and desktop_entry is None:
            desktop_entry = relpath
            desktop_modules = blocked_modules

        if django_candidate is None and path.name == "manage.py" and (
            "django" in imports or "execute_from_command_line" in source
        ):
            django_candidate = relpath

        if streamlit_candidate is None and "streamlit" in imports:
            streamlit_candidate = relpath

        if fastapi_candidate is None and "fastapi" in imports:
            match = re.search(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*FastAPI\s*\(", source, re.MULTILINE)
            if match:
                module_name = python_module_from_path(workspace_path, path)
                if module_name:
                    fastapi_candidate = {
                        "entry_file": relpath,
                        "app_var": match.group(1),
                        "module_name": module_name,
                    }

        if flask_candidate is None and "flask" in imports:
            match = re.search(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*Flask\s*\(", source, re.MULTILINE)
            if match:
                module_name = python_module_from_path(workspace_path, path)
                if module_name:
                    flask_candidate = {
                        "entry_file": relpath,
                        "app_var": match.group(1),
                        "module_name": module_name,
                    }

    has_web_candidate = any((fastapi_candidate, flask_candidate, streamlit_candidate, django_candidate))
    if runtime_hint == "python_desktop":
        return build_python_desktop_spec(entry_file=desktop_entry or first_python_file)
    if desktop_entry and not has_web_candidate:
        modules = ", ".join(sorted(desktop_modules))
        return build_python_desktop_spec(
            entry_file=desktop_entry,
            message=f"检测到 Python 桌面/小游戏框架（{modules}），当前浏览器预览不支持这类本地窗口程序。请改为 HTML/Canvas、Node Web，或 Python HTTP 项目。",
        )

    if fastapi_candidate:
        return {
            "mode": "python_web",
            "entry_file": fastapi_candidate["entry_file"],
            "install_command": install_command_override
            or build_python_install_command(workspace_path, ["fastapi", "uvicorn"]),
            "start_command": start_command_override
            or f"python -m uvicorn {fastapi_candidate['module_name']}:{fastapi_candidate['app_var']} --host 0.0.0.0 --port {port}",
            "internal_port": port,
            "message": None,
        }

    if flask_candidate:
        return {
            "mode": "python_web",
            "entry_file": flask_candidate["entry_file"],
            "install_command": install_command_override or build_python_install_command(workspace_path, ["flask"]),
            "start_command": start_command_override
            or f"python -m flask --app {flask_candidate['module_name']}:{flask_candidate['app_var']} run --host 0.0.0.0 --port {port}",
            "internal_port": port,
            "message": None,
        }

    if streamlit_candidate:
        return {
            "mode": "python_web",
            "entry_file": streamlit_candidate,
            "install_command": install_command_override
            or build_python_install_command(workspace_path, ["streamlit"]),
            "start_command": start_command_override
            or f"python -m streamlit run {shlex.quote(streamlit_candidate)} --server.address 0.0.0.0 --server.port {port}",
            "internal_port": port,
            "message": None,
        }

    if django_candidate:
        return {
            "mode": "python_web",
            "entry_file": django_candidate,
            "install_command": install_command_override or build_python_install_command(workspace_path, ["django"]),
            "start_command": start_command_override or f"python manage.py runserver 0.0.0.0:{port}",
            "internal_port": port,
            "message": None,
        }

    if runtime_hint == "python_web":
        return {
            "mode": "python_web",
            "entry_file": preferred_entry or first_python_file,
            "install_command": install_command_override or build_python_install_command(workspace_path),
            "start_command": start_command_override,
            "internal_port": port,
            "message": (
                None
                if start_command_override
                else "检测到 Python 项目，但未识别出可直接启动的 FastAPI、Flask、Streamlit 或 Django 入口。可在 .agentdo/project.json 里补充 runtime/start/install/port。"
            ),
        }

    return None


def infer_manifest_runtime_mode(
    runtime_hint: str | None,
    entry_file: str | None,
    install_command: str | None,
    start_command: str | None,
    package_json_present: bool,
) -> str | None:
    if runtime_hint:
        return runtime_hint

    text = " ".join(part for part in [entry_file, install_command, start_command] if part).lower()
    if any(token in text for token in ("npm", "node", "pnpm", "yarn", "vite", "next", "nuxt", "bun")):
        return "node"
    if entry_file and entry_file.endswith(".html"):
        return "static"
    if entry_file and entry_file.endswith(".py"):
        return "python_web"
    if any(token in text for token in ("uvicorn", "flask", "streamlit", "django", "gunicorn", "python", "pip")):
        return "python_web"
    if package_json_present:
        return "node"
    return None


def build_http_runtime_boot_command(spec: dict) -> str:
    install_command = spec.get("install_command")
    start_command = spec.get("start_command")
    if not start_command:
        raise HTTPException(status_code=500, detail="Missing runtime start command.")

    if spec["mode"] == "node":
        if install_command:
            return f"set -e; if [ ! -d node_modules ]; then {install_command}; fi; {start_command}"
        return start_command

    if spec["mode"] == "python_web":
        commands = [
            "set -e",
            "mkdir -p .agentdo",
            "if [ ! -x .agentdo/preview-venv/bin/python ]; then python -m venv .agentdo/preview-venv; fi",
            ". .agentdo/preview-venv/bin/activate",
        ]
        if install_command:
            commands.append(install_command)
        commands.append(start_command)
        return "; ".join(commands)

    raise HTTPException(status_code=500, detail=f"Unsupported runtime mode: {spec['mode']}")


def detect_runtime_spec(workspace_path: Path) -> dict:
    manifest = load_json_file(workspace_path / ".agentdo" / "project.json") or {}
    manifest_runtime_hint = normalize_runtime_mode(manifest.get("runtime"))
    manifest_entry = normalize_relative_path(manifest.get("entry"))
    manifest_install_command = normalize_shell_command(manifest.get("install"))
    manifest_start_command = normalize_shell_command(manifest.get("start"))
    manifest_port = parse_int(
        manifest.get("port"),
        PYTHON_RUNTIME_INTERNAL_PORT if manifest_runtime_hint == "python_web" else APP_RUNTIME_INTERNAL_PORT,
    )
    package_json = load_json_file(workspace_path / "package.json")

    if manifest_runtime_hint == "python_desktop":
        return build_python_desktop_spec(entry_file=manifest_entry)

    if manifest_start_command:
        inferred_mode = infer_manifest_runtime_mode(
            runtime_hint=manifest_runtime_hint,
            entry_file=manifest_entry,
            install_command=manifest_install_command,
            start_command=manifest_start_command,
            package_json_present=bool(package_json),
        )
        if inferred_mode == "static":
            entry_target = resolve_workspace_relative_path(workspace_path, manifest_entry)
            if entry_target and entry_target.is_file():
                return {
                    "mode": "static",
                    "entry_file": str(entry_target.relative_to(workspace_path)),
                    "install_command": None,
                    "start_command": None,
                    "internal_port": None,
                    "message": None,
                }
        if inferred_mode == "node":
            return {
                "mode": "node",
                "entry_file": detect_static_entry(workspace_path),
                "install_command": manifest_install_command or "npm install",
                "start_command": manifest_start_command,
                "internal_port": parse_int(manifest.get("port"), APP_RUNTIME_INTERNAL_PORT),
                "message": None,
            }

        python_manifest_spec = detect_python_runtime_spec(
            workspace_path,
            preferred_entry=manifest_entry,
            runtime_hint="python_web" if inferred_mode == "python_web" else manifest_runtime_hint,
            install_command_override=manifest_install_command,
            start_command_override=manifest_start_command,
            internal_port=parse_int(manifest.get("port"), PYTHON_RUNTIME_INTERNAL_PORT),
        )
        if python_manifest_spec:
            return python_manifest_spec

        return {
            "mode": inferred_mode or "unknown",
            "entry_file": manifest_entry,
            "install_command": manifest_install_command,
            "start_command": manifest_start_command,
            "internal_port": manifest_port if inferred_mode and inferred_mode != "static" else None,
            "message": "检测到了自定义运行命令，但无法判断这是哪种可预览项目。建议在 .agentdo/project.json 中显式设置 runtime。",
        }

    if manifest_entry:
        entry_target = resolve_workspace_relative_path(workspace_path, manifest_entry)
        if entry_target and entry_target.is_file() and entry_target.suffix.lower() == ".html":
            return {
                "mode": "static",
                "entry_file": str(entry_target.relative_to(workspace_path)),
                "install_command": None,
                "start_command": None,
                "internal_port": None,
                "message": None,
            }

    if package_json:
        scripts = package_json.get("scripts") or {}
        internal_port = APP_RUNTIME_INTERNAL_PORT
        install_command = "npm install"
        start_command = None

        if "dev" in scripts:
            start_command = f"npm run dev -- --host 0.0.0.0 --port {internal_port}"
        elif "start" in scripts:
            start_command = "npm run start"

        return {
            "mode": "node",
            "entry_file": detect_static_entry(workspace_path),
            "install_command": install_command,
            "start_command": start_command,
            "internal_port": internal_port,
            "message": None,
        }

    python_spec = detect_python_runtime_spec(
        workspace_path,
        preferred_entry=manifest_entry,
        runtime_hint=manifest_runtime_hint,
        install_command_override=manifest_install_command if manifest_runtime_hint == "python_web" else None,
        internal_port=parse_int(
            manifest.get("port"),
            PYTHON_RUNTIME_INTERNAL_PORT if manifest_runtime_hint == "python_web" else PYTHON_RUNTIME_INTERNAL_PORT,
        ),
    )
    if python_spec:
        return python_spec

    if manifest_runtime_hint == "node":
        return {
            "mode": "node",
            "entry_file": detect_static_entry(workspace_path),
            "install_command": manifest_install_command or "npm install",
            "start_command": manifest_start_command,
            "internal_port": parse_int(manifest.get("port"), APP_RUNTIME_INTERNAL_PORT),
            "message": "manifest 指定了 Node 项目，但当前 workspace 里没有可识别的 dev/start 脚本。请在 .agentdo/project.json 中补充 start 命令，或提供 package.json scripts。",
        }

    if manifest_runtime_hint == "python_web":
        return {
            "mode": "python_web",
            "entry_file": manifest_entry,
            "install_command": manifest_install_command or build_python_install_command(workspace_path),
            "start_command": manifest_start_command,
            "internal_port": parse_int(manifest.get("port"), PYTHON_RUNTIME_INTERNAL_PORT),
            "message": "manifest 指定了 Python Web 项目，但当前没有识别出可启动的 HTTP 入口。请在 .agentdo/project.json 中补充 start 命令。",
        }

    entry_file = detect_static_entry(workspace_path)
    if entry_file:
        return {
            "mode": "static",
            "entry_file": entry_file,
            "install_command": None,
            "start_command": None,
            "internal_port": None,
            "message": None,
        }

    return {
        "mode": "unknown",
        "entry_file": None,
        "install_command": None,
        "start_command": None,
        "internal_port": None,
        "message": None,
    }


def runtime_container_name(session_id: str) -> str:
    return f"agentdo-runtime-{session_id}"


def remove_runtime_container(container_name: str) -> None:
    run_command(["docker", "rm", "-f", container_name], check=False)


def read_runtime_logs(container_name: str, lines: int = 80) -> str:
    result = run_command(
        ["docker", "logs", "--tail", str(lines), container_name],
        check=False,
    )
    return (result.stdout or result.stderr or "").strip()


def inspect_container(container_name: str) -> dict | None:
    result = run_command(["docker", "inspect", container_name], check=False)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not data:
        return None
    return data[0]


def extract_host_port(inspect_data: dict, internal_port: int) -> int | None:
    ports = inspect_data.get("NetworkSettings", {}).get("Ports", {})
    bindings = ports.get(f"{internal_port}/tcp")
    if not bindings:
        return None
    host_port = bindings[0].get("HostPort")
    if not host_port:
        return None
    return int(host_port)


def wait_for_port(host: str, port: int, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def refresh_runtime_record(record: dict | None) -> dict | None:
    if not record or not is_http_runtime_mode(record["mode"]) or not record.get("container_name"):
        return record

    inspect_data = inspect_container(record["container_name"])
    if inspect_data is None:
        if record["status"] in {"running", "starting"}:
            return upsert_runtime_record(
                session_id=record["session_id"],
                mode=record["mode"],
                status="stopped",
                entry_file=record.get("entry_file"),
                container_name=record.get("container_name"),
                host_port=None,
                internal_port=record.get("internal_port"),
                install_command=record.get("install_command"),
                start_command=record.get("start_command"),
                last_error="Runtime container not found.",
            )
        return record

    state = inspect_data.get("State", {})
    running = bool(state.get("Running"))
    host_port = extract_host_port(
        inspect_data,
        int(record.get("internal_port") or default_internal_port_for_mode(record.get("mode"))),
    )

    if running:
        if record["status"] != "running" or record.get("host_port") != host_port:
            return upsert_runtime_record(
                session_id=record["session_id"],
                mode=record["mode"],
                status="running",
                entry_file=record.get("entry_file"),
                container_name=record.get("container_name"),
                host_port=host_port,
                internal_port=record.get("internal_port"),
                install_command=record.get("install_command"),
                start_command=record.get("start_command"),
                last_error=None,
            )
        return record

    final_status = "stopped" if record["status"] == "stopped" else "failed"
    last_error = read_runtime_logs(record["container_name"]) or record.get("last_error")
    return upsert_runtime_record(
        session_id=record["session_id"],
        mode=record["mode"],
        status=final_status,
        entry_file=record.get("entry_file"),
        container_name=record.get("container_name"),
        host_port=None,
        internal_port=record.get("internal_port"),
        install_command=record.get("install_command"),
        start_command=record.get("start_command"),
        last_error=last_error,
    )


def build_runtime_payload(session: dict) -> dict:
    workspace_path = Path(session["workspace_path"]).resolve()
    spec = detect_runtime_spec(workspace_path)
    record = refresh_runtime_record(fetch_runtime_record(session["id"]))
    if record and record["mode"] != spec["mode"]:
        record = None

    mode = spec["mode"]
    status = "not_available"
    can_start = False
    can_preview = False
    preview_url = None
    last_error = spec.get("message")
    host_port = None

    if mode == "static":
        status = "ready" if spec.get("entry_file") else "not_available"
        can_preview = bool(spec.get("entry_file"))
        preview_url = (
            f"/sessions/{session['id']}/preview/{quote(spec['entry_file'], safe='/')}"
            if can_preview
            else None
        )
    elif is_http_runtime_mode(mode):
        if spec.get("start_command") is None:
            status = "not_configured"
            if not last_error:
                if mode == "node":
                    last_error = "检测到了 package.json，但没有可识别的 dev/start 脚本。"
                else:
                    last_error = "检测到了 Python Web 项目，但没有可识别的启动命令。"
        else:
            can_start = True
            status = "stopped"
            if record:
                status = record["status"]
                last_error = record.get("last_error") or last_error
                host_port = record.get("host_port")
                if status == "running":
                    can_preview = True
                    preview_url = f"/sessions/{session['id']}/preview/"
    elif mode == "python_desktop":
        last_error = last_error or "检测到 Python 桌面项目，当前浏览器预览不支持。"
    else:
        last_error = "当前 workspace 中没有可预览的静态页面，也没有可运行的 Web 项目。"

    return {
        "session_id": session["id"],
        "mode": mode,
        "status": status,
        "entry_file": spec.get("entry_file"),
        "preview_url": preview_url,
        "can_preview": can_preview,
        "can_start": can_start,
        "host_port": host_port,
        "container_name": record.get("container_name") if record else None,
        "install_command": spec.get("install_command") or (record.get("install_command") if record else None),
        "start_command": spec.get("start_command") or (record.get("start_command") if record else None),
        "internal_port": spec.get("internal_port") or (record.get("internal_port") if record else None),
        "last_error": last_error,
    }


def start_runtime_for_session(session: dict) -> dict:
    workspace_path = Path(session["workspace_path"]).resolve()
    home_path = Path(session["home_path"]).resolve()
    runtime_workspace_path = resolve_runtime_mount_source(workspace_path)
    runtime_home_path = resolve_runtime_mount_source(home_path)
    spec = detect_runtime_spec(workspace_path)

    if spec["mode"] == "static":
        upsert_runtime_record(
            session_id=session["id"],
            mode="static",
            status="ready",
            entry_file=spec.get("entry_file"),
        )
        return build_runtime_payload(session)

    if not is_http_runtime_mode(spec["mode"]):
        raise HTTPException(
            status_code=400,
            detail=spec.get("message") or "当前 session 没有可在线运行的项目。",
        )

    if not spec.get("start_command"):
        raise HTTPException(
            status_code=400,
            detail=spec.get("message") or "检测到了可预览项目，但没有可运行的启动命令。",
        )

    container_name = runtime_container_name(session["id"])
    remove_runtime_container(container_name)
    upsert_runtime_record(
        session_id=session["id"],
        mode=spec["mode"],
        status="starting",
        entry_file=spec.get("entry_file"),
        container_name=container_name,
        host_port=None,
        internal_port=spec.get("internal_port"),
        install_command=spec.get("install_command"),
        start_command=spec.get("start_command"),
        last_error=None,
    )

    boot_command = build_http_runtime_boot_command(spec)

    run_command(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--memory",
            APP_RUNTIME_MEMORY,
            "--cpus",
            APP_RUNTIME_CPUS,
            "--user",
            f"{CONTAINER_UID}:{CONTAINER_GID}",
            "-e",
            f"HOME={CONTAINER_HOME}",
            "-e",
            "HOST=0.0.0.0",
            "-e",
            f"PORT={spec['internal_port']}",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-v",
            f"{runtime_workspace_path}:/workspace",
            "-v",
            f"{runtime_home_path}:{CONTAINER_HOME}",
            "-w",
            "/workspace",
            "-p",
            f"127.0.0.1::{spec['internal_port']}",
            runtime_image_for_mode(spec["mode"]),
            "sh",
            "-lc",
            boot_command,
        ],
        timeout=60,
    )

    inspect_data = inspect_container(container_name)
    host_port = extract_host_port(inspect_data or {}, int(spec["internal_port"])) if inspect_data else None
    if host_port is None or not wait_for_port("127.0.0.1", host_port, APP_RUNTIME_START_TIMEOUT_SECONDS):
        last_error = read_runtime_logs(container_name) or "Runtime did not become ready in time."
        remove_runtime_container(container_name)
        upsert_runtime_record(
            session_id=session["id"],
            mode=spec["mode"],
            status="failed",
            entry_file=spec.get("entry_file"),
            container_name=container_name,
            host_port=None,
            internal_port=spec.get("internal_port"),
            install_command=spec.get("install_command"),
            start_command=spec.get("start_command"),
            last_error=last_error,
        )
        raise HTTPException(status_code=500, detail=last_error)

    upsert_runtime_record(
        session_id=session["id"],
        mode=spec["mode"],
        status="running",
        entry_file=spec.get("entry_file"),
        container_name=container_name,
        host_port=host_port,
        internal_port=spec.get("internal_port"),
        install_command=spec.get("install_command"),
        start_command=spec.get("start_command"),
        last_error=None,
    )
    return build_runtime_payload(session)


def stop_runtime_for_session(session: dict) -> dict:
    record = fetch_runtime_record(session["id"])
    if record and record.get("container_name"):
        remove_runtime_container(record["container_name"])

    spec = detect_runtime_spec(Path(session["workspace_path"]).resolve())
    if spec["mode"] == "static":
        upsert_runtime_record(
            session_id=session["id"],
            mode="static",
            status="ready",
            entry_file=spec.get("entry_file"),
            last_error=None,
        )
    else:
        upsert_runtime_record(
            session_id=session["id"],
            mode=record["mode"] if record else spec["mode"],
            status="stopped",
            entry_file=record.get("entry_file") if record else spec.get("entry_file"),
            container_name=record.get("container_name") if record else None,
            host_port=None,
            internal_port=record.get("internal_port") if record else spec.get("internal_port"),
            install_command=record.get("install_command") if record else spec.get("install_command"),
            start_command=record.get("start_command") if record else spec.get("start_command"),
            last_error=None,
        )
    return build_runtime_payload(session)


def safe_workspace_file(workspace_path: Path, preview_path: str) -> Path:
    relative_path = preview_path.lstrip("/")
    target = (workspace_path / relative_path).resolve()
    try:
        target.relative_to(workspace_path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Preview file not found") from exc
    return target


def proxy_runtime_response(host_port: int, preview_path: str, request: Request) -> Response:
    upstream_path = "/" + preview_path.lstrip("/") if preview_path else "/"
    if request.url.query:
        upstream_path = f"{upstream_path}?{request.url.query}"

    conn = http.client.HTTPConnection("127.0.0.1", host_port, timeout=10)
    try:
        conn.request(
            request.method,
            upstream_path,
            headers={
                "Accept": request.headers.get("accept", "*/*"),
                "User-Agent": "Agent-Do-Preview-Proxy",
            },
        )
        response = conn.getresponse()
        body = response.read()
        headers = {}
        for key, value in response.getheaders():
            if key.lower() in {
                "content-type",
                "cache-control",
                "etag",
                "last-modified",
                "location",
            }:
                headers[key] = value
        headers.setdefault("Cache-Control", "no-store, max-age=0")
        return Response(content=body, status_code=response.status, headers=headers)
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"Preview proxy failed: {exc}") from exc
    finally:
        conn.close()


def resolve_runtime_profile(profile: str | None) -> str:
    if profile in {"default", "aliyun"}:
        return profile
    return DEFAULT_RUNTIME_PROFILE


def validate_runtime_env(profile: str) -> dict[str, str]:
    claude_env = resolve_claude_runtime_env(profile)
    if not claude_env.get("ANTHROPIC_API_KEY") and not claude_env.get("ANTHROPIC_AUTH_TOKEN"):
        raise HTTPException(
            status_code=500,
            detail=(
                f"workshop auth is not configured for runtime profile '{profile}'. "
                "Set ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN or the corresponding ALIYUN_ANTHROPIC_* variables."
            ),
        )
    return claude_env


def resolve_claude_runtime_env(profile: str) -> dict[str, str]:
    if profile == "aliyun":
        env_map = {
            "ANTHROPIC_BASE_URL": ALIYUN_ANTHROPIC_BASE_URL,
            "ANTHROPIC_MODEL": ALIYUN_ANTHROPIC_MODEL,
        }
        if ALIYUN_ANTHROPIC_API_KEY:
            env_map["ANTHROPIC_API_KEY"] = ALIYUN_ANTHROPIC_API_KEY
        if ALIYUN_ANTHROPIC_AUTH_TOKEN:
            env_map["ANTHROPIC_AUTH_TOKEN"] = ALIYUN_ANTHROPIC_AUTH_TOKEN
        return env_map

    return {name: value for name in CLAUDE_ENV_VARS if (value := os.getenv(name))}


def resolve_model(profile: str, requested_model: str | None) -> str:
    model = (requested_model or "").strip()
    if profile == "aliyun":
        if not model or model == "sonnet":
            return ALIYUN_ANTHROPIC_MODEL
        return model

    if not model:
        return DEFAULT_CLAUDE_MODEL
    return model


def build_claude_command(
    session: dict,
    prompt: str,
    model: str,
    max_turns: int,
    append_system_prompt: str | None,
    runtime_profile: str,
    output_format: str = "text",
    verbose: bool = False,
    include_partial_messages: bool = False,
) -> list[str]:
    claude_env = validate_runtime_env(runtime_profile)

    workspace_path = resolve_runtime_mount_source(session["workspace_path"])
    home_path = resolve_runtime_mount_source(session["home_path"])
    env_args = ["-e", f"HOME={CONTAINER_HOME}"]
    for name, value in claude_env.items():
        env_args.extend(["-e", f"{name}={value}"])

    command = [
        "docker",
        "run",
        "--rm",
        "--memory",
        CLAUDE_MEMORY,
        "--cpus",
        CLAUDE_CPUS,
        "--user",
        f"{CONTAINER_UID}:{CONTAINER_GID}",
        *env_args,
        "-v",
        f"{workspace_path}:/workspace",
        "-v",
        f"{home_path}:{CONTAINER_HOME}",
        "-w",
        "/workspace",
        CLAUDE_DOCKER_IMAGE,
        "claude",
        "--bare",
        "-p",
        prompt,
        "--output-format",
        output_format,
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--dangerously-skip-permissions",
    ]

    final_system_prompt = DEFAULT_APPEND_SYSTEM_PROMPT
    if append_system_prompt:
        final_system_prompt = f"{DEFAULT_APPEND_SYSTEM_PROMPT}\n\n{append_system_prompt}"
    command.extend(["--append-system-prompt", final_system_prompt])

    if verbose:
        command.append("--verbose")
    if include_partial_messages:
        command.append("--include-partial-messages")

    if has_assistant_reply(session["id"]):
        command.insert(command.index("-p"), "-c")

    return command


def run_claude(
    session: dict,
    prompt: str,
    model: str,
    max_turns: int,
    append_system_prompt: str | None,
    runtime_profile: str,
) -> tuple[str, int, int, dict[str, list[str]]]:
    workspace_path = Path(session["workspace_path"]).resolve()
    before_snapshot = snapshot_workspace(workspace_path)
    command = build_claude_command(
        session=session,
        prompt=prompt,
        model=model,
        max_turns=max_turns,
        append_system_prompt=append_system_prompt,
        runtime_profile=runtime_profile,
    )

    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Missing runtime dependency: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="workshop execution timed out") from exc

    duration_ms = int((perf_counter() - started) * 1000)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if completed.returncode != 0:
        detail = stderr or stdout or f"workshop failed with exit code {completed.returncode}"
        raise HTTPException(status_code=500, detail=detail)

    workspace_diff = diff_workspace(before_snapshot, snapshot_workspace(workspace_path))
    annotated_output = maybe_annotate_output(prompt, stdout, workspace_diff)

    return annotated_output, completed.returncode, duration_ms, workspace_diff


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def read_available(stream) -> bytes:
    try:
        if hasattr(stream, "read1"):
            chunk = stream.read1(4096)
        else:
            chunk = stream.read(4096)
    except BlockingIOError:
        return b""
    return chunk or b""


def stream_claude(
    session: dict,
    prompt: str,
    model: str,
    max_turns: int,
    append_system_prompt: str | None,
    runtime_profile: str,
):
    workspace_path = Path(session["workspace_path"]).resolve()
    before_snapshot = snapshot_workspace(workspace_path)
    last_workspace_snapshot = before_snapshot
    command = build_claude_command(
        session=session,
        prompt=prompt,
        model=model,
        max_turns=max_turns,
        append_system_prompt=append_system_prompt,
        runtime_profile=runtime_profile,
        output_format="stream-json",
        verbose=True,
        include_partial_messages=True,
    )

    started = perf_counter()
    yield sse_event(
        "started",
        {
            "session_id": session["id"],
            "timestamp": utc_now(),
            "runtime_profile": runtime_profile,
            "model": model,
        },
    )

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
        )
    except FileNotFoundError as exc:
        yield sse_event("error", {"message": f"Missing runtime dependency: {exc}"})
        return

    assert process.stdout is not None
    os.set_blocking(process.stdout.fileno(), False)

    output_buffer = ""
    raw_lines: list[str] = []
    assistant_text = ""
    last_failure_message: str | None = None
    seen_progress: set[str] = set()
    deadline = perf_counter() + CLAUDE_TIMEOUT_SECONDS
    next_workspace_scan = perf_counter() + 0.2

    def process_stream_line(line: str):
        nonlocal assistant_text, last_failure_message
        raw_lines.append(line)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return [
                sse_event(
                    "progress",
                    {
                        "kind": "log",
                        "summary": "Runtime output",
                        "detail": truncate_text(line, 260) or None,
                        "timestamp": utc_now(),
                    },
                )
            ]

        events: list[str] = []

        failure_message = extract_failure_message(payload)
        if failure_message:
            last_failure_message = failure_message

        delta_text = extract_stream_text_delta(payload)
        if delta_text:
            assistant_text += delta_text
            events.append(sse_event("response_delta", {"text": delta_text}))

        extracted_text = extract_assistant_text(payload)
        if extracted_text and payload.get("type") != "stream_event":
            if extracted_text.startswith(assistant_text):
                delta = extracted_text[len(assistant_text):]
            else:
                delta = extracted_text
            assistant_text = extracted_text
            if delta:
                events.append(sse_event("response_delta", {"text": delta}))

        for entry in summarize_claude_event(payload):
            fingerprint = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            if fingerprint in seen_progress:
                continue
            seen_progress.add(fingerprint)
            events.append(
                sse_event(
                    "progress",
                    {
                        **entry,
                        "timestamp": utc_now(),
                    },
                )
            )
        return events

    while True:
        now = perf_counter()
        if now > deadline:
            process.kill()
            process.wait()
            yield sse_event("error", {"message": "workshop execution timed out"})
            return

        if now >= next_workspace_scan:
            current_snapshot = snapshot_workspace(workspace_path)
            delta = diff_workspace(last_workspace_snapshot, current_snapshot)
            if delta["changed_files"]:
                yield sse_event(
                    "workspace",
                    {
                        "changes": build_workspace_change_items(session["id"], workspace_path, delta),
                        "changed_files": diff_workspace(before_snapshot, current_snapshot)["changed_files"],
                        "timestamp": utc_now(),
                    },
                )
                last_workspace_snapshot = current_snapshot
            next_workspace_scan = now + 0.3

        chunk = read_available(process.stdout)
        if chunk:
            output_buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in output_buffer:
                line, output_buffer = output_buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                for event_text in process_stream_line(line):
                    yield event_text
            continue

        if process.poll() is not None:
            tail = read_available(process.stdout)
            if tail:
                output_buffer += tail.decode("utf-8", errors="replace")
            if output_buffer.strip():
                for raw_line in output_buffer.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    for event_text in process_stream_line(line):
                        yield event_text
            current_snapshot = snapshot_workspace(workspace_path)
            delta = diff_workspace(last_workspace_snapshot, current_snapshot)
            if delta["changed_files"]:
                yield sse_event(
                    "workspace",
                    {
                        "changes": build_workspace_change_items(session["id"], workspace_path, delta),
                        "changed_files": diff_workspace(before_snapshot, current_snapshot)["changed_files"],
                        "timestamp": utc_now(),
                    },
                )
                last_workspace_snapshot = current_snapshot
            break

        time.sleep(0.05)

    duration_ms = int((perf_counter() - started) * 1000)
    exit_code = process.returncode or 0
    raw_output = "\n".join(raw_lines).strip()

    if exit_code != 0:
        detail = last_failure_message or f"workshop failed with exit code {exit_code}"
        yield sse_event(
            "error",
            {
                "message": detail,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
            },
        )
        return

    workspace_diff = diff_workspace(before_snapshot, last_workspace_snapshot)
    output = assistant_text.strip()
    annotated_output = maybe_annotate_output(prompt, output, workspace_diff)

    insert_message(
        session_id=session["id"],
        role="assistant",
        content=annotated_output,
        exit_code=exit_code,
        duration_ms=duration_ms,
    )

    yield sse_event(
        "done",
        {
            "output": annotated_output,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "changed_files": workspace_diff["changed_files"],
            "added_files": workspace_diff["added"],
            "modified_files": workspace_diff["modified"],
            "deleted_files": workspace_diff["deleted"],
            "workspace_changes": build_workspace_change_items(session["id"], workspace_path, workspace_diff),
        },
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_data_dirs()
    init_db()
    yield


app = FastAPI(title="Agent-Do MVP", lifespan=lifespan)


@app.middleware("http")
async def add_no_store_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/sessions/") or request.url.path in {"/sessions", "/runtime-profiles"}:
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


class CreateSessionRequest(BaseModel):
    user_id: str = Field(default=DEFAULT_USER_ID)
    title: str | None = None


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1)
    model: str | None = None
    max_turns: int = Field(default=8, ge=1, le=20)
    append_system_prompt: str | None = None
    runtime_profile: str | None = None


class RuntimeActionRequest(BaseModel):
    restart: bool = Field(default=False)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC_ROOT / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/healthz")
def healthcheck() -> dict:
    return {"status": "ok", "timestamp": utc_now()}


@app.post("/sessions")
def create_session(payload: CreateSessionRequest) -> dict:
    session_id = str(uuid.uuid4())
    session_root = SESSIONS_ROOT / session_id
    workspace_path = session_root / "workspace"
    home_path = session_root / "home"

    workspace_path.mkdir(parents=True, exist_ok=True)
    home_path.mkdir(parents=True, exist_ok=True)

    now = utc_now()
    session = {
        "id": session_id,
        "user_id": payload.user_id,
        "title": payload.title,
        "workspace_path": str(workspace_path),
        "home_path": str(home_path),
        "created_at": now,
        "last_active_at": now,
    }

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, user_id, title, workspace_path, home_path, created_at, last_active_at)
            VALUES (:id, :user_id, :title, :workspace_path, :home_path, :created_at, :last_active_at)
            """,
            session,
        )
        conn.commit()

    return session


@app.get("/sessions")
def list_sessions() -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY last_active_at DESC"
        ).fetchall()
    return {"items": [row_to_dict(row) for row in rows]}


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    return fetch_session(session_id)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    session = delete_session_data(session_id)
    return {"ok": True, "deleted_session_id": session_id, "title": session.get("title")}


@app.get("/sessions/{session_id}/messages")
def list_messages(session_id: str) -> dict:
    fetch_session(session_id)
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, role, content, exit_code, duration_ms, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        ).fetchall()
    return {"items": [row_to_dict(row) for row in rows]}


@app.get("/sessions/{session_id}/workspace/files/{file_path:path}")
def get_workspace_file(session_id: str, file_path: str) -> dict:
    session = fetch_session(session_id)
    workspace_path = Path(session["workspace_path"]).resolve()
    target = safe_workspace_file(workspace_path, file_path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Workspace file not found")
    return read_workspace_file_preview(target, workspace_path)


@app.get("/sessions/{session_id}/runtime")
def get_runtime(session_id: str) -> dict:
    session = fetch_session(session_id)
    return build_runtime_payload(session)


@app.post("/sessions/{session_id}/runtime/start")
def start_runtime(session_id: str, payload: RuntimeActionRequest | None = None) -> dict:
    session = fetch_session(session_id)
    existing = refresh_runtime_record(fetch_runtime_record(session["id"]))
    spec = detect_runtime_spec(Path(session["workspace_path"]).resolve())
    if (
        existing
        and existing["mode"] == spec["mode"]
        and is_http_runtime_mode(existing["mode"])
        and existing["status"] == "running"
        and not (payload and payload.restart)
    ):
        return build_runtime_payload(session)
    if payload and payload.restart:
        stop_runtime_for_session(session)
    return start_runtime_for_session(session)


@app.post("/sessions/{session_id}/runtime/stop")
def stop_runtime(session_id: str) -> dict:
    session = fetch_session(session_id)
    return stop_runtime_for_session(session)


@app.get("/sessions/{session_id}/runtime/logs")
def get_runtime_logs(session_id: str) -> dict:
    session = fetch_session(session_id)
    record = refresh_runtime_record(fetch_runtime_record(session["id"]))
    if not record or not record.get("container_name"):
        return {"session_id": session_id, "logs": "", "status": "not_running"}
    return {
        "session_id": session_id,
        "logs": read_runtime_logs(record["container_name"]),
        "status": record["status"],
    }


@app.get("/sessions/{session_id}/preview")
@app.get("/sessions/{session_id}/preview/{preview_path:path}")
def preview_session(session_id: str, request: Request, preview_path: str = ""):
    session = fetch_session(session_id)
    runtime = build_runtime_payload(session)

    if is_http_runtime_mode(runtime["mode"]):
        if runtime["status"] != "running" or not runtime.get("host_port"):
            raise HTTPException(status_code=409, detail="项目尚未运行，请先启动预览。")
        return proxy_runtime_response(int(runtime["host_port"]), preview_path, request)

    if runtime["mode"] == "static":
        workspace_path = Path(session["workspace_path"]).resolve()
        entry_file = runtime.get("entry_file")
        if not entry_file:
            raise HTTPException(status_code=404, detail="未找到可预览的 HTML 文件。")
        if not preview_path:
            quoted_entry = quote(entry_file)
            return RedirectResponse(url=f"/sessions/{session_id}/preview/{quoted_entry}")
        target = safe_workspace_file(workspace_path, preview_path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Preview file not found")
        media_type = "text/html; charset=utf-8" if target.suffix.lower() == ".html" else None
        return FileResponse(
            target,
            media_type=media_type,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    raise HTTPException(status_code=404, detail="当前 session 没有可预览内容。")


@app.post("/sessions/{session_id}/messages")
def send_message(session_id: str, payload: SendMessageRequest) -> dict:
    session = fetch_session(session_id)
    insert_message(session_id=session_id, role="user", content=payload.content)
    runtime_profile = resolve_runtime_profile(payload.runtime_profile)
    model = resolve_model(runtime_profile, payload.model)
    validate_runtime_env(runtime_profile)

    output, exit_code, duration_ms, workspace_diff = run_claude(
        session=session,
        prompt=payload.content,
        model=model,
        max_turns=payload.max_turns,
        append_system_prompt=payload.append_system_prompt,
        runtime_profile=runtime_profile,
    )

    insert_message(
        session_id=session_id,
        role="assistant",
        content=output,
        exit_code=exit_code,
        duration_ms=duration_ms,
    )

    return {
        "session_id": session_id,
        "output": output,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "runtime_profile": runtime_profile,
        "model": model,
        "changed_files": workspace_diff["changed_files"],
        "added_files": workspace_diff["added"],
        "modified_files": workspace_diff["modified"],
        "deleted_files": workspace_diff["deleted"],
    }


@app.post("/sessions/{session_id}/messages/stream")
def send_message_stream(session_id: str, payload: SendMessageRequest) -> StreamingResponse:
    session = fetch_session(session_id)
    insert_message(session_id=session_id, role="user", content=payload.content)
    runtime_profile = resolve_runtime_profile(payload.runtime_profile)
    model = resolve_model(runtime_profile, payload.model)
    validate_runtime_env(runtime_profile)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        stream_claude(
            session=session,
            prompt=payload.content,
            model=model,
            max_turns=payload.max_turns,
            append_system_prompt=payload.append_system_prompt,
            runtime_profile=runtime_profile,
        ),
        media_type="text/event-stream",
        headers=headers,
    )


@app.get("/runtime-profiles")
def list_runtime_profiles() -> dict:
    return {
        "default_runtime_profile": DEFAULT_RUNTIME_PROFILE,
        "items": [
            {
                "id": "default",
                "label": "workshop Default",
                "default_model": DEFAULT_CLAUDE_MODEL,
                "configured": bool(resolve_claude_runtime_env("default").get("ANTHROPIC_API_KEY") or resolve_claude_runtime_env("default").get("ANTHROPIC_AUTH_TOKEN")),
            },
            {
                "id": "aliyun",
                "label": "workshop via Aliyun",
                "default_model": ALIYUN_ANTHROPIC_MODEL,
                "configured": bool(ALIYUN_ANTHROPIC_API_KEY or ALIYUN_ANTHROPIC_AUTH_TOKEN),
                "base_url": ALIYUN_ANTHROPIC_BASE_URL,
            },
        ],
    }
