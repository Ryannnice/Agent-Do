import json
import os
import sqlite3
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field


def env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


DATA_ROOT = Path(os.getenv("AGENT_DATA_ROOT", "./data")).resolve()
SESSIONS_ROOT = DATA_ROOT / "agent-sessions"
DATABASE_PATH = DATA_ROOT / "app.db"
STATIC_ROOT = Path(__file__).resolve().parent / "static"

CLAUDE_DOCKER_IMAGE = os.getenv("CLAUDE_DOCKER_IMAGE", "claude-runtime:latest")
CLAUDE_TIMEOUT_SECONDS = int(os.getenv("CLAUDE_TIMEOUT_SECONDS", "900"))
CLAUDE_MEMORY = os.getenv("CLAUDE_MEMORY", "2g")
CLAUDE_CPUS = os.getenv("CLAUDE_CPUS", "1")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "demo-user")
CONTAINER_UID = env_or_default("CLAUDE_CONTAINER_UID", str(os.getuid()))
CONTAINER_GID = env_or_default("CLAUDE_CONTAINER_GID", str(os.getgid()))
CONTAINER_HOME = env_or_default("CLAUDE_CONTAINER_HOME", "/home/agent")
CLAUDE_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_data_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)


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


def build_claude_command(
    session: dict,
    prompt: str,
    model: str,
    max_turns: int,
    append_system_prompt: str | None,
) -> list[str]:
    claude_env = {name: os.getenv(name) for name in CLAUDE_ENV_VARS if os.getenv(name)}
    if not claude_env.get("ANTHROPIC_API_KEY") and not claude_env.get("ANTHROPIC_AUTH_TOKEN"):
        raise HTTPException(
            status_code=500,
            detail="Claude auth is not configured. Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN.",
        )

    workspace_path = Path(session["workspace_path"]).resolve()
    home_path = Path(session["home_path"]).resolve()
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
        "text",
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--dangerously-skip-permissions",
    ]

    if append_system_prompt:
        command.extend(["--append-system-prompt", append_system_prompt])

    if has_assistant_reply(session["id"]):
        command.insert(command.index("-p"), "-c")

    return command


def run_claude(
    session: dict,
    prompt: str,
    model: str,
    max_turns: int,
    append_system_prompt: str | None,
) -> tuple[str, int, int]:
    command = build_claude_command(
        session=session,
        prompt=prompt,
        model=model,
        max_turns=max_turns,
        append_system_prompt=append_system_prompt,
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
        raise HTTPException(status_code=504, detail="Claude execution timed out") from exc

    duration_ms = int((perf_counter() - started) * 1000)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if completed.returncode != 0:
        detail = stderr or stdout or f"Claude failed with exit code {completed.returncode}"
        raise HTTPException(status_code=500, detail=detail)

    return stdout, completed.returncode, duration_ms


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
):
    command = build_claude_command(
        session=session,
        prompt=prompt,
        model=model,
        max_turns=max_turns,
        append_system_prompt=append_system_prompt,
    )

    started = perf_counter()
    yield sse_event("started", {"session_id": session["id"], "timestamp": utc_now()})

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

    output_chunks: list[str] = []
    deadline = perf_counter() + CLAUDE_TIMEOUT_SECONDS

    while True:
        if perf_counter() > deadline:
            process.kill()
            process.wait()
            yield sse_event("error", {"message": "Claude execution timed out"})
            return

        chunk = read_available(process.stdout)
        if chunk:
            text = chunk.decode("utf-8", errors="replace")
            output_chunks.append(text)
            yield sse_event("chunk", {"text": text})
            continue

        if process.poll() is not None:
            tail = read_available(process.stdout)
            if tail:
                text = tail.decode("utf-8", errors="replace")
                output_chunks.append(text)
                yield sse_event("chunk", {"text": text})
            break

        time.sleep(0.05)

    duration_ms = int((perf_counter() - started) * 1000)
    output = "".join(output_chunks).strip()
    exit_code = process.returncode or 0

    if exit_code != 0:
        detail = output or f"Claude failed with exit code {exit_code}"
        yield sse_event(
            "error",
            {
                "message": detail,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
            },
        )
        return

    insert_message(
        session_id=session["id"],
        role="assistant",
        content=output,
        exit_code=exit_code,
        duration_ms=duration_ms,
    )

    yield sse_event(
        "done",
        {
            "output": output,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
        },
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_data_dirs()
    init_db()
    yield


app = FastAPI(title="Agent-Do MVP", lifespan=lifespan)


class CreateSessionRequest(BaseModel):
    user_id: str = Field(default=DEFAULT_USER_ID)
    title: str | None = None


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1)
    model: str = Field(default="sonnet")
    max_turns: int = Field(default=8, ge=1, le=20)
    append_system_prompt: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


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


@app.post("/sessions/{session_id}/messages")
def send_message(session_id: str, payload: SendMessageRequest) -> dict:
    session = fetch_session(session_id)
    insert_message(session_id=session_id, role="user", content=payload.content)

    output, exit_code, duration_ms = run_claude(
        session=session,
        prompt=payload.content,
        model=payload.model,
        max_turns=payload.max_turns,
        append_system_prompt=payload.append_system_prompt,
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
    }


@app.post("/sessions/{session_id}/messages/stream")
def send_message_stream(session_id: str, payload: SendMessageRequest) -> StreamingResponse:
    session = fetch_session(session_id)
    insert_message(session_id=session_id, role="user", content=payload.content)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        stream_claude(
            session=session,
            prompt=payload.content,
            model=payload.model,
            max_turns=payload.max_turns,
            append_system_prompt=payload.append_system_prompt,
        ),
        media_type="text/event-stream",
        headers=headers,
    )
