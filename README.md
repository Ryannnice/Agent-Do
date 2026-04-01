# Agent-Do FastAPI MVP

这是一个最小可跑的后端骨架，用来把 Claude Code 包装成一个云端 session 服务。

## MVP 能力

- `POST /sessions` 创建一个 session
- `GET /sessions` 查看 session 列表
- `GET /sessions/{session_id}` 查看单个 session
- `GET /sessions/{session_id}/messages` 查看消息历史
- `POST /sessions/{session_id}/messages` 在固定 workspace 内调用一次 Claude Code
- `POST /sessions/{session_id}/messages/stream` 以 SSE 流式返回 Claude 输出
- `GET /` 打开一个最小聊天页

每个 session 都会绑定两个本地目录：

- `workspace/`：用户项目和 Claude 改出来的文件
- `home/`：Claude 本地状态目录，用来延续同一个 session 的上下文

## 目录布局

```text
data/
  app.db
  agent-sessions/
    <session-id>/
      workspace/
      home/
```

## 先决条件

- Python 3.11+
- Docker
- 有效的 `ANTHROPIC_API_KEY`

## 1. 构建 Claude 运行时镜像

```bash
docker build -t claude-runtime:latest -f Dockerfile.claude .
```

## 2. 安装 Python 依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 配置环境变量

```bash
cp .env.example .env
export $(grep -v '^#' .env | xargs)
```

如果你不想用 `.env`，至少需要：

```bash
export ANTHROPIC_API_KEY=...
export CLAUDE_DOCKER_IMAGE=claude-runtime:latest
```

## 4. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

打开浏览器访问：

```text
http://127.0.0.1:8000/
```

## 5. 调用示例

创建 session：

```bash
curl -X POST http://127.0.0.1:8000/sessions \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","title":"demo"}'
```

给某个 session 发消息：

```bash
curl -X POST http://127.0.0.1:8000/sessions/<session-id>/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "content":"请分析当前目录并创建一个 README.md",
    "model":"sonnet",
    "max_turns":8
  }'
```

查看消息历史：

```bash
curl http://127.0.0.1:8000/sessions/<session-id>/messages
```

流式调用：

```bash
curl -N -X POST http://127.0.0.1:8000/sessions/<session-id>/messages/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "content":"请扫描当前目录并总结项目结构",
    "model":"sonnet",
    "max_turns":8
  }'
```

## 当前实现的取舍

- 这是阻塞式 API，请求会等 Claude 执行完再返回
- 同时也提供了一个 SSE 接口，前端可以逐步看到 Claude 输出
- 每次请求都启动一个临时 Docker 容器，不保活容器
- 为了让非交互模式先跑通，容器内命令启用了 `--dangerously-skip-permissions`
- 为避免 Claude Code 的 root 限制，runner 会默认以当前宿主机用户的 UID/GID 运行容器

最后这一点只适合内部 MVP。下一步如果要上线，优先补：

- 任务队列和超时重试
- 命令审批
- 更强的容器隔离
- 鉴权
- 用户级限流
