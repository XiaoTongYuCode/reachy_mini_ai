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
- [运行应用](#运行应用)
- [暴露给助手的 LLM 工具](#暴露给助手的-llm-工具)
- [高级功能](#高级功能)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 概览
- 基于 `fastrtc` 的实时音频对话循环，提供低延迟流式体验。支持的后端：
  - **Hugging Face**：默认方案，可使用内置 Hugging Face 服务器或你自己的本地端点。
  - **OpenAI Realtime**（`gpt-realtime`）：需要 `OPENAI_API_KEY`。
  - **Gemini Live**（`gemini-3.1-flash-live-preview`）：需要 `GEMINI_API_KEY`。
- 视觉处理默认使用你选择的实时后端（调用相机工具时）；也可通过 `--local-vision` 启用基于 SmolVLM2 的本地视觉（CPU/GPU/MPS）。
- 分层动作系统会队列化主动作（舞蹈、情绪、goto 姿态、呼吸），并叠加说话响应式晃动与头部跟踪。
- 异步工具调度通过带实时转写的 Gradio Web UI 集成机器人动作、相机采集和可选的头部跟踪能力。

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
| `BACKEND_PROVIDER` | 要使用的实时后端：`huggingface`（默认）、`openai` 或 `gemini`。 |
| `MODEL_NAME` | OpenAI Realtime 或 Gemini Live 的可选模型覆盖。OpenAI 默认 `gpt-realtime`，Gemini 默认 `gemini-3.1-flash-live-preview`。Hugging Face 使用服务器侧模型选择。 |
| `HF_REALTIME_CONNECTION_MODE` | Hugging Face 连接模式：`deployed` 使用内置 Hugging Face 服务器；`local` 使用 `HF_REALTIME_WS_URL`。默认 `deployed`。 |
| `HF_REALTIME_WS_URL` | 你自己的 Hugging Face 后端的 websocket 端点。可填基础 URL（如 `ws://127.0.0.1:8765/v1`）或完整 websocket URL（`ws://127.0.0.1:8765/v1/realtime`）。在 `HF_REALTIME_CONNECTION_MODE=local` 时使用。 |
| `HF_HOME` | 本地 Hugging Face 下载缓存目录（仅 `--local-vision` 使用，默认 `./cache`）。 |
| `HF_TOKEN` | 可选的 Hugging Face 访问令牌（用于受限/私有资产）。 |
| `LOCAL_VISION_MODEL` | 本地视觉处理的 Hugging Face 模型路径（仅 `--local-vision` 使用，默认 `HuggingFaceTB/SmolVLM2-2.2B-Instruct`）。 |

### Hugging Face 连接模式

通过应用托管的 Space 代理使用内置 Hugging Face 服务器。新安装默认如此；只有在你想从已保存的本地端点切回时才需要显式设置：

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=deployed
```

在与你的对话应用同一台机器上，运行 [speech-to-speech](https://github.com/huggingface/speech-to-speech) 作为自建实时语音后端：

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
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
| `--head-tracker {yolo,mediapipe}` | `None` | 在相机可用时选择头部跟踪后端。`yolo` 使用本地 YOLO 人脸检测器，`mediapipe` 来自 `reachy_mini_toolbox` 包。需要安装对应 optional extra。 |
| `--no-camera` | `False` | 不使用相机采集和头部跟踪。 |
| `--local-vision` | `False` | 在相机工具请求时使用本地视觉模型（SmolVLM2），而不是所选实时后端。需要安装 `local_vision` extra。 |
| `--gradio` | `False` | 启动 Gradio Web UI。不加该参数时以控制台模式运行。仿真模式下必需。 |
| `--robot-name` | `None` | 可选。在同一子网运行多个 daemon 时按名称连接指定机器人。见 [同一子网的多机器人](#同一子网的多机器人)。 |
| `--debug` | `False` | 启用详细日志，便于排障。 |

### 示例

```bash
# 使用 MediaPipe 头部跟踪运行
reachy-mini-conversation-app --head-tracker mediapipe

# 使用 YOLO 人脸检测后端进行头部跟踪
reachy-mini-conversation-app --head-tracker yolo

# 使用本地视觉处理运行（需要 local_vision extra）
reachy-mini-conversation-app --local-vision

# 仅音频对话（无相机）
reachy-mini-conversation-app --no-camera

# 启动 Gradio Web 界面
reachy-mini-conversation-app --gradio
```

> [!WARNING]
> 在 Reachy Mini Wireless / Raspberry Pi 上直接运行对话应用时，不支持 `--local-vision`。如需本地视觉，请让 daemon 在机器人上运行，而将对话应用运行在你的笔记本或工作站上。

## 暴露给助手的 LLM 工具

| 工具 | 动作 | 依赖 |
|------|--------|--------------|
| `move_head` | 队列一个头部姿态变更（left/right/up/down/front）。 | 仅核心安装。 |
| `camera` | 抓取最新相机帧，并通过所选实时后端或本地视觉模型进行分析。 | 需要 camera worker。启用 `--local-vision` 时使用本地视觉。 |
| `head_tracking` | 启用或禁用头部跟踪偏移（不是身份识别，仅检测并跟踪头部位置）。 | 需要 camera worker 且已配置头部跟踪器（`--head-tracker`）。 |
| `dance` | 从 `reachy_mini_dances_library` 队列一个舞蹈。 | 仅核心安装。 |
| `stop_dance` | 清空已队列的舞蹈。 | 仅核心安装。 |
| `play_emotion` | 通过 Hugging Face 数据集播放录制的情绪片段。 | 仅核心安装。默认使用开放情绪数据集：[`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)。 |
| `stop_emotion` | 清空已队列的情绪。 | 仅核心安装。 |
| `idle_do_nothing` | 在空闲轮次显式保持空闲。不用于正常对话轮次。 | 仅核心安装。 |

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
<summary><b>同一子网的多机器人</b></summary>

如果你在同一网络里运行多个 Reachy Mini daemon，可使用：

```bash
reachy-mini-conversation-app --robot-name <name>
```

`<name>` 必须与 daemon 的 `--robot-name` 值一致，应用才会连接到正确机器人。

</details>

## 参与贡献

欢迎提交 bug 修复、新功能、profiles 以及文档改进。请先阅读我们的
[贡献指南](CONTRIBUTING.md)，其中包含分支规范、质量检查和 PR 流程。

快速开始：
- Fork 并克隆仓库
- 按 [安装步骤](#安装) 配置环境（包含 `dev` 依赖分组）
- 运行 [CONTRIBUTING.md](CONTRIBUTING.md) 中列出的贡献者检查项

## 许可证

Apache 2.0