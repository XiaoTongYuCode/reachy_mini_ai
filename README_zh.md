---
title: Reachy Mini 对话应用
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: 与 Reachy Mini 对话！
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini 对话应用

这是一个面向 Reachy Mini 机器人的对话应用，结合了实时语音后端、视觉处理流水线和编排好的动作库。

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## 目录
- [概览](#概览)
- [架构](#架构)
- [安装](#安装)
- [配置](#配置)
- [后端配置示例](#后端配置示例)
- [运行应用](#运行应用)
- [暴露给助手的 LLM 工具](#暴露给助手的-llm-工具)
- [高级功能](#高级功能)
- [排障](#排障)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 概览
- 基于 `fastrtc` 的实时音频对话循环，提供低延迟流式体验。支持的后端：
  - **Hugging Face**：默认方案，可使用内置 Hugging Face 服务器或你自己的本地端点。
  - **OpenAI Realtime**（`gpt-realtime`）：需要 `OPENAI_API_KEY`。
  - **Gemini Live**（`gemini-3.1-flash-live-preview`）：需要 `GEMINI_API_KEY`。
  - **Volcengine Realtime**：需要火山引擎语音服务的 `X-Api-*` 凭据。
  - **Aliyun DashScope Realtime**（`qwen3.5-omni-flash-realtime`）：需要 `DASHSCOPE_API_KEY`，支持模型侧 function calling。
- 视觉处理默认使用你选择的实时后端（调用相机工具时）；也可通过 `--local-vision` 启用基于 SmolVLM2 的本地视觉（CPU/GPU/MPS）。
- 分层动作系统会队列化主动作（舞蹈、情绪、goto 姿态、呼吸），并叠加说话响应式晃动与头部跟踪。
- 异步工具调度通过带实时转写的 Gradio Web UI 集成机器人动作、相机采集和可选的头部跟踪能力。
- 可选本地记忆会把对话和显式长期记忆存入 SQLite，并可在启用后向后续会话注入相关上下文。

## 架构

应用采用分层架构，将用户、AI 服务和机器人硬件连接起来：

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## 安装

> [!IMPORTANT]
> 使用本应用前，你需要先安装 [Reachy Mini 的 SDK](https://github.com/pollen-robotics/reachy_mini/)。<br>
> Windows 支持目前仍为实验性，尚未经过充分测试，请谨慎使用。

<details open>
<summary><b>使用 uv（推荐）</b></summary>

使用 [uv](https://docs.astral.sh/uv/) 快速完成项目初始化：

```bash
# macOS (Homebrew)
uv venv --python /opt/homebrew/bin/python3.12 .venv

# Linux / Windows (Python in PATH)
uv venv --python python3.12 .venv

source .venv/bin/activate
uv sync
```

> **说明：** 若要严格复现仓库 `uv.lock` 中的依赖集合，请运行 `uv sync --frozen`。这样 `uv` 会直接按 lockfile 安装，不会重新解析或升级版本。

**安装可选功能：**
```bash
uv sync --extra local_vision         # 本地 PyTorch/Transformers 视觉
uv sync --extra yolo_vision          # 用于头部跟踪的 YOLO 人脸检测后端
uv sync --extra mediapipe_vision     # 基于 MediaPipe 的头部跟踪
uv sync --extra all_vision           # 安装全部视觉功能
```

可组合 extras，或附带 dev 依赖：
```bash
uv sync --extra all_vision --group dev
```

</details>

<details>
<summary><b>使用 pip</b></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**安装可选功能：**
```bash
pip install -e .[local_vision]          # 本地视觉栈
pip install -e .[yolo_vision]           # 用于头部跟踪的 YOLO 人脸检测后端
pip install -e .[mediapipe_vision]      # 基于 MediaPipe 的视觉
pip install -e .[all_vision]            # 全部视觉功能
pip install -e .[dev]                   # 开发工具
```

某些 wheel（例如 PyTorch）体积较大，并且需要匹配 CUDA 或 CPU 构建；请确认你的平台与每个 extra 拉取的二进制兼容。

</details>

### 可选依赖分组

| Extra | 用途 | 说明 |
|-------|------|------|
| `local_vision` | 通过 PyTorch/Transformers 运行本地 VLM（SmolVLM2） | 推荐使用 GPU。请确保 PyTorch 构建与你的平台兼容。 |
| `yolo_vision` | 通过 `ultralytics` 和 `supervision` 使用 YOLOv11n 人脸检测 | 作为 `yolo` 头部跟踪后端。默认可在 CPU 运行，GPU 性能更好。 |
| `mediapipe_vision` | 使用 MediaPipe 的轻量级关键点跟踪 | 可在 CPU 运行。启用 `--head-tracker mediapipe`。 |
| `all_vision` | 一次性安装所有视觉 extra 的便捷别名 | 当你希望在所有视觉提供者之间灵活试验时使用。 |
| `dev` | 开发工具（`pytest`、`ruff`、`mypy`） | 仅开发环境需要。uv 使用 `--group dev`，pip 使用 `[dev]`。 |

**注意：** `dev` 是依赖分组（不是可选依赖）。uv 用 `--group dev`，pip 用 `[dev]`。

## 配置

默认配置使用 Hugging Face 后端，不需要 API Key。

当你希望切换后端、提供 API Key，或让 Hugging Face 指向你自己的本地端点时，请将 `.env.example` 复制为 `.env`。

| 变量 | 说明 |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI Realtime 模式必需。 |
| `GEMINI_API_KEY` | Gemini 模式必需。也支持 `GOOGLE_API_KEY`。可在 [aistudio.google.com](https://aistudio.google.com/apikey) 获取。 |
| `BACKEND_PROVIDER` | 要使用的实时后端：`huggingface`（默认）、`openai`、`gemini`、`ark` 或 `aliyun`。 |
| `MODEL_NAME` | OpenAI Realtime 或 Gemini Live 的可选模型覆盖。OpenAI 默认 `gpt-realtime`，Gemini 默认 `gemini-3.1-flash-live-preview`。Hugging Face 使用服务器侧模型选择。 |
| `HF_REALTIME_CONNECTION_MODE` | Hugging Face 连接模式：`deployed` 使用内置 Hugging Face 服务器；`local` 使用 `HF_REALTIME_WS_URL`。默认 `deployed`。 |
| `HF_REALTIME_LANGUAGE` | Hugging Face Realtime 的语音识别语言提示。默认 `zh`，适合中文；如需后端自动检测，可设为 `auto`。内置云端服务可能会忽略该值，因为实际 STT 由服务端启动参数决定；如果需要稳定中文识别，请使用本地网关，并在仓库根目录 `.env` 中配置 `GATEWAY_STT=faster-whisper` 和 `GATEWAY_LANGUAGE=zh`。 |
| `HF_REALTIME_WS_URL` | 你自己的 Hugging Face 后端的 websocket 端点。可填基础 URL（如 `ws://127.0.0.1:8765/v1`）或完整 websocket URL（`ws://127.0.0.1:8765/v1/realtime`）。在 `HF_REALTIME_CONNECTION_MODE=local` 时使用。 |
| `HF_REALTIME_AUTO_START` | 可选。与 `HF_REALTIME_CONNECTION_MODE=local` 一起设为 `true` 时，应用会在连接 `HF_REALTIME_WS_URL` 前启动 `reachy-mini-hf-realtime-gateway`。若存在 `services/hf_realtime_gateway/.venv/bin/reachy-mini-hf-realtime-gateway` 会优先使用，否则该命令必须已安装并在 `PATH` 中。默认 `false`。 |
| `HF_REALTIME_AUTO_START_TIMEOUT_SECONDS` | 可选。`HF_REALTIME_AUTO_START` 的就绪等待超时时间，单位秒。首次本地模型下载/预热可能需要几分钟。默认 `600`。 |
| `HF_HOME` | 本地 Hugging Face 下载缓存目录（仅 `--local-vision` 使用，默认 `./cache`）。 |
| `HF_TOKEN` | 可选的 Hugging Face 访问令牌（用于受限/私有资产）。 |
| `LOCAL_VISION_MODEL` | 本地视觉处理的 Hugging Face 模型路径（仅 `--local-vision` 使用，默认 `HuggingFaceTB/SmolVLM2-2.2B-Instruct`）。 |
| `REACHY_MINI_MEMORY_CONTEXT_ENABLED` | 可选。设为 `true` 时，每次用户转录后会刷新模型可见的长期记忆上下文。默认 `false`，以保证实时首答最快。 |
| `REACHY_MINI_MEMORY_AUTO_EXTRACT` | 可选。在运行流支持时启用自动记忆提取。默认 `false`；即使保持关闭，显式 `manage_memory` 工具调用仍可使用。 |
| `ARK_REALTIME_APP_ID` / `VOLCENGINE_REALTIME_APP_ID` / `VOLC_APP_ID` | `BACKEND_PROVIDER=ark` 时必需。火山引擎 Realtime 的 `X-Api-App-ID`。 |
| `ARK_REALTIME_ACCESS_KEY` / `VOLCENGINE_REALTIME_ACCESS_KEY` / `VOLCENGINE_REALTIME_ACCESS_TOKEN` / `VOLC_ACCESS_KEY` | `BACKEND_PROVIDER=ark` 时必需。火山引擎 Realtime 的 `X-Api-Access-Key`。 |
| `ARK_REALTIME_APP_KEY` / `VOLCENGINE_REALTIME_APP_KEY` / `VOLC_APP_KEY` | `BACKEND_PROVIDER=ark` 时必需。火山引擎 Realtime 的 `X-Api-App-Key`。 |
| `ARK_REALTIME_RESOURCE_ID` / `VOLCENGINE_REALTIME_RESOURCE_ID` / `VOLC_RESOURCE_ID` | `BACKEND_PROVIDER=ark` 时可选，默认 `volc.speech.dialog`。 |
| `ARK_REALTIME_WS_URL` | 可选火山引擎 Realtime websocket URL，默认 `wss://openspeech.bytedance.com/api/v3/realtime/dialogue`。 |
| `ARK_REALTIME_BOT_NAME` | 可选。发送给火山引擎 Realtime 的机器人展示名，默认 `Reachy Mini`。 |
| `ARK_REALTIME_INPUT_SAMPLE_RATE` | 可选。火山引擎 Realtime 输入音频采样率，默认 `16000`。 |
| `ARK_REALTIME_OUTPUT_SAMPLE_RATE` | 可选。火山引擎 Realtime 输出音频采样率，默认 `24000`。 |
| `DASHSCOPE_API_KEY` / `ALIYUN_API_KEY` | `BACKEND_PROVIDER=aliyun` 时必需。用于 Qwen realtime 的 DashScope API Key。 |
| `ALIYUN_REALTIME_MODEL` | 可选。阿里云百炼 / DashScope 模型覆盖，默认 `qwen3.5-omni-flash-realtime`。独立于 `MODEL_NAME`，避免跨供应商模型名污染。 |
| `ALIYUN_REALTIME_WS_URL` | 可选阿里云百炼 / DashScope realtime websocket URL，默认使用百炼控制台展示的 Qwen3.5 Omni realtime 端点。 |
| `ALIYUN_REALTIME_INPUT_SAMPLE_RATE` | 可选。阿里云 DashScope realtime 输入音频采样率，默认 `16000`。 |
| `ALIYUN_REALTIME_OUTPUT_SAMPLE_RATE` | 可选。阿里云 DashScope realtime 输出音频采样率，默认 `24000`。 |
| `ALIYUN_REALTIME_VIDEO_FPS` | 可选。检测到语音后，阿里云原生 `input_image_buffer.append` 视觉输入的摄像头抽帧帧率。默认 `1`；设为 `0` 可关闭自动语音窗口视觉帧。 |
| `ALIYUN_REALTIME_VIDEO_ACTIVE_SECONDS` | 可选。检测到语音后自动视觉帧保持活跃的秒数，默认 `10`。 |
| `OPENCLAW_GATEWAY_URL` | `ask_openclaw` 工具使用的 OpenClaw gateway URL，默认 `ws://localhost:18789`；为空时不加载该工具。 |
| `OPENCLAW_TOKEN` | OpenClaw gateway 认证 token；为空时不加载 `ask_openclaw`。 |
| `OPENCLAW_AGENT_ID` | 可选 OpenClaw agent ID，默认 `main`。 |
| `OPENCLAW_SESSION_KEY` | 可选 OpenClaw 会话 key，默认 `main`。 |
| `OPENCLAW_TIMEOUT_SECONDS` | `ask_openclaw` 单次请求超时时间，默认 `60` 秒。 |
| `VOLCENGINE_WEB_SEARCH_API_KEY` | 仅 backend 无关的 `web_search` 工具需要。使用火山引擎联网搜索产品的 APIKey 接入地址，不使用 Ark Responses 插件。 |
| `VOLCENGINE_WEB_SEARCH_API_URL` | 可选联网搜索 API URL，默认 `https://open.feedcoopapi.com/search_api/web_search`。 |
| `VOLCENGINE_WEB_SEARCH_TIMEOUT_SECONDS` | `web_search` 单次调用超时时间，默认 `30` 秒。 |
| `WEATHERAPI_API_KEY` | 可选 backend 无关 WeatherAPI.com Key。设置后，提示词基础信息和 `current_location_weather` 工具会包含实时天气。 |
| `SMTP_HOST` / `SMTP_PORT` | backend 无关的 `send_email` 工具 SMTP 服务配置。默认偏向 Gmail：`smtp.gmail.com`、`587`。 |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | `send_email` 的 SMTP 凭据。也支持 Gmail 别名：`GMAIL_EMAIL`、`GMAIL_ADDRESS`、`GMAIL_APP_PASSWORD`、`EMAIL_APP_PASSWORD`。 |
| `SMTP_FROM_EMAIL` / `SMTP_FROM_NAME` | `send_email` 的可选发件地址/发件名覆盖。 |
| `SMTP_USE_SSL` / `SMTP_USE_TLS` | 可选 SMTP 安全开关。端口 `465` 默认启用 SSL；未使用 SSL 时默认启用 TLS。 |
| `default_target_email` | 可选 backend 无关默认收件人；当 `send_email` 工具调用未提供 `target_email` 时使用。 |
| `REACHY_MINI_CUSTOM_PROFILE` | 可选启动 profile 名。若存在已保存启动设置或锁定 profile，可能会被覆盖。 |
| `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` | 可选外部 profile 根目录。 |
| `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` | 可选外部 tool 模块目录。 |
| `AUTOLOAD_EXTERNAL_TOOLS` | 可选。设为 `true` 时自动加载 `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` 下所有合法外部 tool 模块。默认 `false`。 |
| `REACHY_MINI_SKIP_DOTENV` | 可选。为真时跳过自动发现/加载 `.env`，只使用进程环境变量。 |

### Hugging Face 连接模式

通过应用托管的 Space 代理使用内置 Hugging Face 服务器。新安装默认如此；只有在你想从已保存的本地端点切回时才需要显式设置：

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=deployed
HF_REALTIME_LANGUAGE=zh
```

内置云端服务会自行选择 STT 后端。如果中文语音仍被识别成英文，请切换到本地网关，让仓库根目录 `.env` 控制 STT 模型和语言。

在与你的对话应用同一台机器上，运行 [speech-to-speech](https://github.com/huggingface/speech-to-speech) 作为自建实时语音后端：

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
HF_REALTIME_AUTO_START=true
```

如果 `reachy-mini-hf-realtime-gateway` 已安装在应用运行环境中，或已存在于 `services/hf_realtime_gateway/.venv`，也可以让应用自动启动它：

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
HF_REALTIME_AUTO_START=true
HF_REALTIME_AUTO_START_TIMEOUT_SECONDS=600
```

在你的笔记本上运行 Hugging Face 后端，并让 Reachy Mini Wireless 通过同一 Wi-Fi 网络连接：

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://<your-laptop-lan-ip>:8765/v1/realtime
```

在这种局域网部署下，请确保后端监听的是机器人可达地址，而不只是 `127.0.0.1`。

如果后端仍绑定在你笔记本的 loopback 上，也可以通过 SSH 转发到机器人：

```bash
ssh -N -R 8765:127.0.0.1:8765 <robot-user>@<robot-host>
```

然后在机器人上设置：

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

当使用无头设置 UI 时，选择 `Hugging Face` 可在内置服务器和本地 `host:port` 目标之间切换。UI 会自动写入 `HF_REALTIME_CONNECTION_MODE`，本地路径会写入 `HF_REALTIME_WS_URL`（默认 `localhost:8765`）。

## 后端配置示例

下面是常用 `.env` 最小配置片段。建议先复制 `.env.example` 为 `.env`，再只保留当前后端需要的字段。

### Hugging Face 内置后端

这是默认路径，不需要 API Key：

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=deployed
HF_REALTIME_LANGUAGE=zh
```

### 本地 Hugging Face 实时网关

当你希望完全控制 STT、TTS、语言和 LLM 端点时使用本地 gateway：

```bash
cd services/hf_realtime_gateway
uv sync
uv run reachy-mini-hf-realtime-gateway --dry-run
uv run reachy-mini-hf-realtime-gateway
```

然后让应用连接本地 realtime websocket：

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
GATEWAY_LLM_BASE_URL=http://127.0.0.1:8000/v1
GATEWAY_LLM_MODEL=your-model-name
GATEWAY_LLM_API_KEY=
```

如果希望应用自动拉起 gateway，设置 `HF_REALTIME_AUTO_START=true`。应用会优先查找 `services/hf_realtime_gateway/.venv/bin/reachy-mini-hf-realtime-gateway`，找不到时再使用 `PATH` 中的命令。

更多 gateway 侧 STT/TTS 配置见 [services/hf_realtime_gateway/README.md](services/hf_realtime_gateway/README.md)。

### OpenAI Realtime

```env
BACKEND_PROVIDER=openai
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-realtime
```

### Gemini Live

```env
BACKEND_PROVIDER=gemini
GEMINI_API_KEY=...
MODEL_NAME=gemini-3.1-flash-live-preview
```

`GOOGLE_API_KEY` 也可作为 `GEMINI_API_KEY` 的别名。

### 火山引擎 Realtime

```env
BACKEND_PROVIDER=ark
ARK_REALTIME_APP_ID=...
ARK_REALTIME_ACCESS_KEY=...
ARK_REALTIME_APP_KEY=...
ARK_REALTIME_RESOURCE_ID=volc.speech.dialog
```

Ark 路径现在直接使用火山引擎 realtime websocket 完成识别与回复，不再额外调用兼容 OpenAI 的 sidecar / OpenRouter 模型做本地 tool routing。

### 阿里云百炼 / DashScope Realtime

```env
BACKEND_PROVIDER=aliyun
DASHSCOPE_API_KEY=...
ALIYUN_REALTIME_MODEL=qwen3.5-omni-flash-realtime
```

阿里云路径使用 Qwen3.5 Omni 的 realtime websocket 与模型侧 function calling。当前 profile 启用的 tools 会随 session config 发送给模型，模型触发的 tool call 直接复用现有本地 `BackgroundToolManager` 执行路径。

当摄像头可用时，阿里云原生 websocket 路径只会在检测到语音后的短窗口内，按 `ALIYUN_REALTIME_VIDEO_FPS` 将 JPEG 帧通过 `input_image_buffer.append` 发送给模型。默认是 `1` FPS，持续 `10` 秒。模型也可以通过 `camera` 工具的 `duration_seconds` 启动异步 1 FPS 连续图像序列；工具会立即返回，并在后台继续发送最多 `120` 秒的帧。返回的 `tool_id` 可用 `task_status` 查看，并可通过 `task_cancel` 或 `cancel_aliyun_camera_sequence` 取消。

## 运行应用

激活你的虚拟环境后，启动：

```bash
reachy-mini-conversation-app
```

> [!TIP]
> 启动应用前请确保 Reachy Mini daemon 已运行。如果你看到 `TimeoutError`，说明 daemon 尚未启动。安装与启动方式请参考 [Reachy Mini 的 SDK](https://github.com/pollen-robotics/reachy_mini/)。

应用默认以控制台模式运行。添加 `--gradio` 可在 http://127.0.0.1:7860/ 启动 Web UI（仿真模式必需）。视觉和头部跟踪相关参数见下方 CLI 表格。

### 官方仿真 / 虚拟测试环境

Reachy Mini 已通过 `reachy-mini` SDK 提供官方 MuJoCo 仿真。当你需要无硬件测试环境时，建议使用它，而不是应用内本地 mock 机器人：

```bash
# 在当前环境安装官方仿真 extra
uv pip install "reachy-mini[mujoco]"

# 终端 1：启动官方仿真机器人 daemon
reachy-mini-daemon --sim

# 终端 2：让对话应用连接本地 daemon
reachy-mini-conversation-app --no-camera --gradio
```

在 macOS 上，MuJoCo 可能需要使用其 launcher，而非常规 daemon 入口：

```bash
mjpython -m reachy_mini.daemon.app.main --sim
```

如果你需要更轻量、无 MuJoCo 物理模拟的 daemon 层冒烟测试，SDK 还提供 `--mockup-sim`：

```bash
reachy-mini-daemon --mockup-sim
reachy-mini-conversation-app --no-camera --gradio
```

当应用连接到官方仿真 daemon 时，会从 SDK 状态中检测 `simulation_enabled` 或 `mockup_sim_enabled`，并在需要时自动启用 Gradio。

### CLI 选项

| 选项 | 默认值 | 说明 |
|--------|---------|-------------|
| `--head-tracker {none,yolo,mediapipe}` | `yolo` | 在相机可用时选择头部跟踪后端。`yolo` 使用本地 YOLO 人脸检测器，`mediapipe` 来自 `reachy_mini_toolbox` 包，`none` 会在保留相机采集的情况下禁用头部跟踪。需要安装对应 optional extra。 |
| `--no-camera` | `False` | 不使用相机采集和头部跟踪。 |
| `--local-vision` | `False` | 在相机工具请求时使用本地视觉模型（SmolVLM2），而不是所选实时后端。需要安装 `local_vision` extra。 |
| `--gradio` | `False` | 启动 Gradio Web UI。不加该参数时以控制台模式运行。仿真模式下必需。 |
| `--robot-name` | `None` | 可选。在同一子网运行多个 daemon 时按名称连接指定机器人。见 [同一子网的多机器人](#同一子网的多机器人)。 |
| `--debug` | `False` | 启用详细日志，便于排障。 |

### 示例

```bash
# 使用 MediaPipe 头部跟踪运行
reachy-mini-conversation-app --head-tracker mediapipe

# 使用默认 YOLO 人脸检测后端进行头部跟踪
reachy-mini-conversation-app

# 保留相机采集，但关闭头部跟踪
reachy-mini-conversation-app --head-tracker none

# 使用本地视觉处理运行（需要 local_vision extra）
reachy-mini-conversation-app --local-vision

# 仅音频对话（无相机）
reachy-mini-conversation-app --no-camera

# 启动 Gradio Web 界面
reachy-mini-conversation-app --gradio
```

YOLO tracker 首次启动可能较慢，因为它会在子进程里加载人脸检测模型。如果冷启动时超时，可以临时调大启动等待时间：

```bash
REACHY_MINI_YOLO_HEAD_TRACKER_START_TIMEOUT_SECONDS=180 reachy-mini-conversation-app --gradio
```

> [!WARNING]
> 在 Reachy Mini Wireless / Raspberry Pi 上直接运行对话应用时，不支持 `--local-vision`。如需本地视觉，请让 daemon 在机器人上运行，而将对话应用运行在你的笔记本或工作站上。

## 暴露给助手的 LLM 工具

| 工具 | 动作 | 依赖 |
|------|--------|--------------|
| `move_head` | 队列一个头部姿态变更（left/right/up/down/front）。 | 仅核心安装。 |
| `camera` | 抓取最新相机帧，并通过所选实时后端或本地视觉模型进行分析。Aliyun 下可用 `duration_seconds` 启动最多 120 秒的异步 1 FPS 序列。 | 需要 camera worker。启用 `--local-vision` 时使用本地视觉。 |
| `head_tracking` | 启用或禁用头部跟踪偏移（不是身份识别，仅检测并跟踪头部位置）。 | 需要 camera worker 且已配置头部跟踪器（`--head-tracker`）。 |
| `dance` | 从 `reachy_mini_dances_library` 队列一个舞蹈。 | 仅核心安装。 |
| `stop_dance` | 清空已队列的舞蹈。 | 仅核心安装。 |
| `play_emotion` | 通过 Hugging Face 数据集播放录制的情绪片段。 | 仅核心安装。默认使用开放情绪数据集：[`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)。 |
| `stop_emotion` | 清空已队列的情绪。 | 仅核心安装。 |
| `idle_do_nothing` | 在空闲轮次显式保持空闲。不用于正常对话轮次。 | 仅核心安装。 |
| `task_status` | 查看正在运行或最近完成的后台工具。 | 系统工具，所有 profile 都会加载。 |
| `task_cancel` | 通过工具 ID 取消正在运行的后台工具。 | 系统工具，所有 profile 都会加载。 |
| `cancel_aliyun_camera_sequence` | 取消最新或指定的 Aliyun 异步摄像头序列。 | 系统工具，所有 profile 都会加载；只取消 Aliyun camera sequence job。 |
| `manage_memory` | 记住、更新、忘记或搜索显式长期记忆。 | 系统工具，所有 profile 都会加载。需要本地 memory store 可用。 |
| `current_location_weather` | 获取最新近似当前位置和天气。 | 仅核心安装。实时天气需要 `WEATHERAPI_API_KEY`。 |
| `send_email` | 通过已配置的 SMTP 账户发送用户明确要求发送的邮件。 | 需要 SMTP 凭据，并且工具调用提供 `target_email` 或配置 `default_target_email`。 |
| `web_search` | 通过火山引擎联网搜索产品 API 搜索网页、网页总结或图片。 | 需要 `VOLCENGINE_WEB_SEARCH_API_KEY`；可选 `VOLCENGINE_WEB_SEARCH_API_URL` 和 `VOLCENGINE_WEB_SEARCH_TIMEOUT_SECONDS`。 |
| `ask_openclaw` | 把复杂请求转发给 OpenClaw agent，用于 OpenClaw 长期记忆、跨渠道上下文或本地没有的外部工具。 | 需要运行中的 OpenClaw gateway，且 `OPENCLAW_GATEWAY_URL` 和 `OPENCLAW_TOKEN` 都非空；从 `tools.txt` 移除即可禁用。 |

工具是否可用受 profile 控制。写在 `profiles/<profile>/tools.txt` 中的工具，只有在对应 profile 本地文件、内置模块或外部工具模块存在时才会加载。系统工具（`task_status`、`task_cancel`、`cancel_aliyun_camera_sequence`、`manage_memory`）会自动加入每个 profile。`web_search`、`current_location_weather`、`send_email` 是普通的 backend 无关工具；它们的 API key 和默认收件人配置不依赖 `BACKEND_PROVIDER`。`ask_openclaw` 还会额外检查 `OPENCLAW_GATEWAY_URL` 和 `OPENCLAW_TOKEN`。

### 持久化记忆

应用会创建名为 `memory.sqlite3` 的本地 SQLite memory store。在源码检出目录中，默认路径位于 `src/reachy_mini_conversation_app/storage/`；打包或实例化运行时可能使用对应 instance storage 路径。

记忆分为两层：
- 对话历史：本地保留 session 和消息片段，用于搜索和后续上下文构建。
- 显式长期记忆：助手可通过 `manage_memory` 记住、更新、忘记或搜索长期事实、偏好、任务和备注。

默认情况下，memory store 可供工具使用，但为了保证实时首答速度，模型可见的记忆上下文默认关闭。启用上下文注入：

```env
REACHY_MINI_MEMORY_CONTEXT_ENABLED=true
```

除非你明确需要运行流自动抽取记忆，否则建议保持 `REACHY_MINI_MEMORY_AUTO_EXTRACT=false`。更可控的方式是让用户明确说“记住……”，由助手调用 `manage_memory`。

### OpenClaw 桥接

`ask_openclaw` 用于把复杂请求委托给外部 OpenClaw agent，例如跨渠道上下文、更大的工具生态或外部长期记忆。只有以下两个值都非空时才会暴露：

```env
OPENCLAW_GATEWAY_URL=ws://localhost:18789
OPENCLAW_TOKEN=your-openclaw-token
OPENCLAW_AGENT_ID=main
OPENCLAW_SESSION_KEY=main
OPENCLAW_TIMEOUT_SECONDS=60
```

即使已经配置，如果某个 profile 不希望启用它，也可以在该 profile 的 `tools.txt` 中删除或注释 `ask_openclaw`。

### 邮件发送

`send_email` 只应在用户明确要求发送邮件时使用。启用前先配置 SMTP：

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your.sender@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_EMAIL=your.sender@gmail.com
SMTP_FROM_NAME=Reachy Mini
SMTP_USE_SSL=false
SMTP_USE_TLS=true
default_target_email=recipient@example.com
```

如果使用 Gmail，请使用应用专用密码，不要使用账号登录密码。

## 高级功能

内置动作内容以开放 Hugging Face 数据集方式发布：
- Emotions：[`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- Dances：[`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library)

<details>
<summary><b>自定义配置档（Profiles）</b></summary>

你可以创建带有专属指令和启用工具集的自定义 profile。

在常规使用中，可在 UI 中选择 profile 并保存为启动配置。该选择会持久化到 `startup_settings.json`。

如果尚未保存启动配置，也可通过环境变量 `REACHY_MINI_CUSTOM_PROFILE=<name>` 预设启动 profile（加载 `profiles/<name>/`）。若两者都未设置，则使用 `default` profile。

每个 profile 应包含 `instructions.txt`（提示词文本）。建议同时包含 `tools.txt`（允许工具列表）。若非默认 profile 缺失该文件，应用会回退到 `profiles/default/tools.txt`。Profile 也可包含自定义工具实现。

**自定义指令：**

在 `instructions.txt` 中编写纯文本提示。若要复用共享提示片段，可加入如下行：
```
[passion_for_lobster_jokes]
[identities/witty_identity]
```
每个占位符会拉取 `src/reachy_mini_conversation_app/prompts/` 下对应文件（支持嵌套路径）。可参考 `profiles/example/` 的目录结构。

**启用工具：**

在 `tools.txt` 中逐行列出启用的工具。以 `#` 前缀表示注释掉：
```
play_emotion
# move_head

# 我在本地定义的自定义工具
sweep_look
```
工具会先从 profile 目录中的 Python 文件解析（自定义工具），再从核心库 `src/reachy_mini_conversation_app/tools/` 解析（如 `dance`、`head_tracking`）。

**自定义工具：**

除核心库内置工具外，你还可以在 profile 文件夹中添加 Python 文件，实现该 profile 专属的自定义工具。
自定义工具必须继承 `reachy_mini_conversation_app.tools.core_tools.Tool`（见 `profiles/example/sweep_look.py`）。

**在 UI 中编辑人格：**

使用 `--gradio` 启动后，打开 "Personality" 折叠面板：
- 在可用 profile（`profiles/` 下文件夹）与内置默认配置之间切换。
- 点击 "Apply" 可实时更新当前会话指令。
- 输入名称和指令文本可创建新人格。应用会在 `profiles/<name>/` 下写入文件，并从 `default` profile 复制 `tools.txt`。

注意："Personality" 面板只更新对话指令。工具集在启动时从 `tools.txt` 加载，不支持热重载。

</details>

<details>
<summary><b>锁定 profile 模式</b></summary>

如果你想创建一个不能切换 profile 的锁定版本，请编辑 `src/reachy_mini_conversation_app/config.py`，将 `LOCKED_PROFILE` 常量设为目标 profile 名：
```python
LOCKED_PROFILE: str | None = "mars_rover"  # 锁定到该 profile
```
设置 `LOCKED_PROFILE` 后，应用将始终使用该 profile，并忽略已保存的启动设置、`REACHY_MINI_CUSTOM_PROFILE` 和 Gradio UI。UI 会显示 "(locked)"，并禁用所有 profile 编辑控件。
这适合创建固定人格的专用应用克隆。克隆脚本只需修改这个常量即可锁定变体。

</details>

<details>
<summary><b>外部 profiles 与 tools</b></summary>

你可以通过仓库默认目录之外的 profiles/tools 扩展应用。

- 核心 profiles 位于 `profiles/`。
- 核心 tools 位于 `src/reachy_mini_conversation_app/tools/`。

**推荐目录结构：**

```text
external_content/
├── external_profiles/
│   └── my_profile/
│       ├── instructions.txt
│       ├── tools.txt        # 可选（见下方回退行为）
│       └── voice.txt        # 可选
└── external_tools/
    └── my_custom_tool.py
```

**环境变量：**

当你希望通过环境变量驱动外部 profile/tool 选择时，在 `.env` 中设置：

```env
# 可选：回退/手动 profile 选择器
REACHY_MINI_CUSTOM_PROFILE=my_profile
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=./external_content/external_profiles
REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=./external_content/external_tools
# 可选：便捷模式
# AUTOLOAD_EXTERNAL_TOOLS=1
```

**加载行为：**

- **默认/严格模式**：`tools.txt` 显式定义启用工具。`tools.txt` 里的每个名称都必须能解析到内置工具（`src/reachy_mini_conversation_app/tools/`）或 `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` 下的外部工具模块。
- **便捷模式**（`AUTOLOAD_EXTERNAL_TOOLS=1`）：会自动加入 `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` 下所有有效的 `*.py` 工具文件。
- **外部 profile 回退**：如果选中的外部 profile 没有 `tools.txt`，应用会回退到内置 `profiles/default/tools.txt`。

这同时支持：
1. 下载的外部工具搭配内置/默认 profile 使用。
2. 下载的外部 profile 搭配内置默认工具使用。

</details>

<details>
<summary><b>提示词片段与基础信息</b></summary>

Profile 指令可以用方括号语法复用共享提示片段：

```text
[passion_for_lobster_jokes]
[identities/witty_identity]
```

片段会从 `src/reachy_mini_conversation_app/prompts/` 解析。应用还会构建当前时间、近似位置、可选 WeatherAPI 天气等基础信息。`current_location_weather` 使用同一套基础信息路径，因此提示词中的上下文和按需工具结果保持一致。

</details>

<details>
<summary><b>同一子网的多机器人</b></summary>

如果你在同一网络里运行多个 Reachy Mini daemon，可使用：

```bash
reachy-mini-conversation-app --robot-name <name>
```

`<name>` 必须与 daemon 的 `--robot-name` 值一致，应用才会连接到正确机器人。

</details>

## 排障

### Reachy Mini daemon 连接失败

如果启动时报 `TimeoutError` 或 `ConnectionError`，先确认 daemon 已启动，并确认应用连接的是目标机器人：

```bash
reachy-mini-daemon
reachy-mini-conversation-app --debug
```

同一子网存在多个 daemon 时，传入 daemon 名称：

```bash
reachy-mini-conversation-app --robot-name <name>
```

macOS 仿真优先使用：

```bash
mjpython -m reachy_mini.daemon.app.main --sim
```

如果端口 `8000` 已被占用，请停止已有 daemon，或确认你当前要测试的是哪一个 daemon 配置。

### Hugging Face 本地 gateway 无法启动

先直接检查 service 环境：

```bash
cd services/hf_realtime_gateway
uv run reachy-mini-hf-realtime-gateway --dry-run
uv run reachy-mini-hf-realtime-gateway --healthcheck
```

常见原因：
- 缺少 `GATEWAY_LLM_BASE_URL` 或 `GATEWAY_LLM_MODEL`。
- 没有在 service 目录执行 `uv sync` 创建本地 `.venv`。
- 首次模型下载或预热时间超过 `HF_REALTIME_AUTO_START_TIMEOUT_SECONDS`。
- 根目录 `HF_HOME` 指向了非预期缓存。调试 gateway 时，优先把模型/cache 配置放在 gateway 运行环境中确认。

### 中文语音识别不稳定

内置 Hugging Face 云端服务可能会忽略本地 `HF_REALTIME_LANGUAGE`，因为实际 STT 由服务端启动参数决定。如果需要稳定中文识别，请改用本地 gateway，并设置 gateway 侧 STT/语言配置，例如 `GATEWAY_STT=faster-whisper` 和 `GATEWAY_LANGUAGE=zh`。

### 本地视觉失败或速度过慢

`--local-vision` 会加载 PyTorch/Transformers，不适合直接在 Reachy Mini Wireless / Raspberry Pi 上运行。建议让 daemon 在机器人上运行，而把本应用放在笔记本或工作站上运行。如果导入崩溃或 GPU 显存不足，请去掉 `--local-vision`，让相机分析走当前选择的 realtime 后端。

### 头部跟踪启动超时

YOLO 头部跟踪会在子进程中加载检测器。冷启动较慢时可以调大等待时间：

```bash
REACHY_MINI_YOLO_HEAD_TRACKER_START_TIMEOUT_SECONDS=180 reachy-mini-conversation-app --gradio
```

使用 `--head-tracker none` 可保留相机采集但关闭头部跟踪；使用 `--no-camera` 会同时关闭相机采集和头部跟踪。

### Profile 或工具未加载

先确认当前 profile 和工具 allowlist：

```bash
ls profiles/<profile>
sed -n '1,120p' profiles/<profile>/tools.txt
```

对于共享工具，仅新增 `src/reachy_mini_conversation_app/tools/<tool>.py` 不够：还需要把工具名写进当前 profile 的 `tools.txt`，除非它是系统工具，或通过 `AUTOLOAD_EXTERNAL_TOOLS=1` 加载。

### 开发检查

安装 dev 依赖分组后运行主要检查：

```bash
uv sync --group dev
uv run ruff check .
uv run mypy
uv run pytest
```

迭代时可先跑更窄的测试：

```bash
uv run pytest tests/test_config_name_collisions.py tests/test_external_loading.py
uv run pytest tests/test_memory.py
uv run pytest tests/tools
```

## 参与贡献

欢迎提交 bug 修复、新功能、profiles 以及文档改进。请先阅读我们的
[贡献指南](CONTRIBUTING.md)，其中包含分支规范、质量检查和 PR 流程。

快速开始：
- Fork 并克隆仓库
- 按 [安装步骤](#安装) 配置环境（包含 `dev` 依赖分组）
- 运行 [CONTRIBUTING.md](CONTRIBUTING.md) 中列出的贡献者检查项

## 许可证

Apache 2.0
