# Agent-Do FastAPI MVP

这是一个最小可跑的后端骨架，用来把 Claude Code 包装成一个云端 session 服务。

## MVP 能力

- `POST /sessions` 创建一个 session
- `GET /sessions` 查看 session 列表
- `GET /sessions/{session_id}` 查看单个 session
- `GET /sessions/{session_id}/messages` 查看消息历史
- `POST /sessions/{session_id}/messages` 在固定 workspace 内调用一次 Claude Code
- `POST /sessions/{session_id}/messages/stream` 以 SSE 流式返回 Claude 输出
- `GET /sessions/{session_id}/runtime` 查看当前 session 的预览状态
- `POST /sessions/{session_id}/runtime/start` 启动在线预览
- `POST /sessions/{session_id}/runtime/stop` 停止在线预览
- `GET /sessions/{session_id}/preview/...` 访问预览内容
- `GET /` 打开一个最小聊天页
- 后端会对比运行前后的 workspace，返回实际变更文件列表
- 支持在前端切换 Claude 运行配置：默认配置 / 阿里云 Anthropic 兼容接口

每个 session 都会绑定两个本地目录：

- `workspace/`：用户项目和 Claude 改出来的文件
- `home/`：Claude 本地状态目录，用来延续同一个 session 的上下文

预览分三类：

- 静态页面：如果 workspace 里有 `index.html`，或者只有一个 `.html` 文件，前端会直接预览
- Node 项目：如果 workspace 里有 `package.json` 且含 `dev` 或 `start` 脚本，后端会起一个长期 Docker 容器来跑项目
- Python Web 项目：如果 workspace 里能识别出 FastAPI、Flask、Streamlit 或 Django 入口，后端也会起一个长期 Docker 容器来跑项目

另外有一类会被明确标记为“不可浏览器预览”：

- Python 桌面/小游戏项目：例如 `pygame`、`tkinter`、`turtle`、`pyglet`、`arcade` 这类本地窗口程序。它们不是 HTTP 服务，当前前端不会尝试在线预览

如果你希望 Claude 生成的 Python 项目可以被右侧预览，建议让它生成 Web 版本，并在 workspace 里补一个 `.agentdo/project.json`，显式写出 `runtime`、`start`、`install` 和 `port`。

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

如果应用本身要跑在 Docker 里，还需要注意这两个路径变量：

```bash
export AGENTDO_HOST_DATA_ROOT=/absolute/path/on/host/agentdo-data
export AGENTDO_CONTAINER_DATA_ROOT=/var/lib/agentdo
```

原因是 Agent-Do 会继续调用宿主机 Docker 去启动 Claude 容器和预览容器。主应用容器里看到的数据目录路径，必须能映射回宿主机真实路径；因此 `AGENTDO_HOST_DATA_ROOT` 必须是宿主机绝对路径。

如果要通过阿里云百炼运行 Claude Code，请配置：

```bash
export ALIYUN_ANTHROPIC_BASE_URL=https://dashscope.aliyuncs.com/apps/anthropic
export ALIYUN_ANTHROPIC_API_KEY=...
export ALIYUN_ANTHROPIC_MODEL=qwen3-coder-next
export DEFAULT_RUNTIME_PROFILE=aliyun
```

## 4. Docker 打包与部署

项目里现在有两份 Docker 资产：

- [Dockerfile](/home/ryan/CUHKSZ/Agent-Do/Dockerfile)：主应用镜像，内含 Python 运行时和 Docker CLI
- [Dockerfile.claude](/home/ryan/CUHKSZ/Agent-Do/Dockerfile.claude)：Claude 执行容器镜像
- [docker-compose.yml](/home/ryan/CUHKSZ/Agent-Do/docker-compose.yml)：建议给团队对接时直接复用

推荐流程：

```bash
cp .env.example .env
mkdir -p /absolute/path/on/host/agentdo-data
```

然后编辑 `.env`，至少补齐：

```bash
ANTHROPIC_API_KEY=...
AGENTDO_HOST_DATA_ROOT=/absolute/path/on/host/agentdo-data
AGENTDO_CONTAINER_DATA_ROOT=/var/lib/agentdo
CLAUDE_CONTAINER_UID=1000
CLAUDE_CONTAINER_GID=1000
```

构建并启动：

```bash
docker compose build agentdo claude-runtime-image
docker compose up -d agentdo
```

查看状态：

```bash
docker compose ps
docker compose logs -f agentdo
```

这套 compose 默认做了几件事：

- 把宿主机 `docker.sock` 挂进主应用容器，允许它继续启动 Claude 和项目预览子容器
- 把宿主机数据目录绑定到容器内 `AGENTDO_CONTAINER_DATA_ROOT`
- 用 `HOST_AGENT_DATA_ROOT` 把子容器挂载源自动翻译回宿主机真实路径

如果你要接入团队已有 compose / k8s / CI 系统，最少保留下面这些约束：

- 主应用容器必须能执行 `docker` 命令
- 主应用容器必须能访问宿主机 Docker daemon，最简单是挂 `/var/run/docker.sock`
- 数据目录必须配置宿主机绝对路径，并传入 `HOST_AGENT_DATA_ROOT`
- `CLAUDE_CONTAINER_UID` / `CLAUDE_CONTAINER_GID` 最好显式设成团队机器上的普通用户 UID/GID，避免 Claude 子容器以 root 运行

## 5. 本地直接启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

应用启动时会自动读取项目根目录的 `.env`。

打开浏览器访问：

```text
http://127.0.0.1:8000/
```

## 6. 调用示例

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
- 阿里云模式走的是百炼官方 Anthropic 兼容接口，不是 OpenAI compatible-mode `/v1`
- 在线预览目前自动支持三类：静态 HTML、Node Web 项目、Python Web 项目
- `pygame` / `tkinter` / `turtle` 这类 Python 桌面项目会被明确标记为“不支持浏览器预览”
- Node 和 Python Web 项目的运行容器会长期保活，直到你手动停止预览

最后这一点只适合内部 MVP。下一步如果要上线，优先补：

- 任务队列和超时重试
- 命令审批
- 更强的容器隔离
- 鉴权
- 用户级限流
