---
title: Reachy Mini Conversation App
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Talk with Reachy Mini!
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini conversation app

Conversational app for the Reachy Mini robot combining realtime voice backends, vision pipelines, and choreographed motion libraries.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Table of contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Backend recipes](#backend-recipes)
- [Running the app](#running-the-app)
- [LLM tools](#llm-tools-exposed-to-the-assistant)
- [Advanced features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview
- Real-time audio conversation loop with `fastrtc` for low-latency streaming. Supported backends:
  - **Hugging Face** - default, using the built-in Hugging Face server or your own local endpoint.
  - **OpenAI Realtime** (`gpt-realtime`) - requires `OPENAI_API_KEY`.
  - **Gemini Live** (`gemini-3.1-flash-live-preview`) - requires `GEMINI_API_KEY`.
  - **Volcengine Realtime** - requires Volcengine Speech `X-Api-*` credentials.
  - **Aliyun DashScope Realtime** (`qwen3.5-omni-flash-realtime`) - requires `DASHSCOPE_API_KEY` and supports provider-side function calling.
- Vision processing uses the selected realtime backend by default (when the camera tool is used), with optional on-device local vision using SmolVLM2 (CPU/GPU/MPS) via `--local-vision`.
- Layered motion system queues primary moves (dances, emotions, goto poses, breathing) while blending speech-reactive wobble and head-tracking.
- Async tool dispatch integrates robot motion, camera capture, and optional head-tracking capabilities through a Gradio web UI with live transcripts.
- Optional local memory stores conversations and explicit long-term memories in SQLite, and can inject relevant context into future sessions when enabled.

## Architecture

The app follows a layered architecture connecting the user, AI services, and robot hardware:

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Installation

> [!IMPORTANT]
> Before using this app, you need to install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/).<br>
> Windows support is currently experimental and has not been extensively tested. Use with caution.

<details open>
<summary><b>Using uv (recommended)</b></summary>

Set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
# macOS (Homebrew)
uv venv --python /opt/homebrew/bin/python3.12 .venv

# Linux / Windows (Python in PATH)
uv venv --python python3.12 .venv

source .venv/bin/activate
uv sync
```

> **Note:** To reproduce the exact dependency set from this repo's `uv.lock`, run `uv sync --frozen`. This ensures `uv` installs directly from the lockfile without re-resolving or updating any versions.

**Install optional features:**
```bash
uv sync --extra local_vision         # Local PyTorch/Transformers vision
uv sync --extra yolo_vision          # YOLO face-detection backend for head tracking
uv sync --extra mediapipe_vision     # MediaPipe-based head-tracking
uv sync --extra all_vision           # All vision features
```

Combine extras or include dev dependencies:
```bash
uv sync --extra all_vision --group dev
```

</details>

<details>
<summary><b>Using pip</b></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Install optional features:**
```bash
pip install -e .[local_vision]          # Local vision stack
pip install -e .[yolo_vision]           # YOLO face-detection backend for head tracking
pip install -e .[mediapipe_vision]      # MediaPipe-based vision
pip install -e .[all_vision]            # All vision features
pip install -e .[dev]                   # Development tools
```

Some wheels (like PyTorch) are large and require compatible CUDA or CPU builds—make sure your platform matches the binaries pulled in by each extra.

</details>

### Optional dependency groups

| Extra | Purpose | Notes |
|-------|---------|-------|
| `local_vision` | Run the local VLM (SmolVLM2) through PyTorch/Transformers | GPU recommended. Ensure compatible PyTorch builds for your platform. |
| `yolo_vision` | YOLOv11n face detection via `ultralytics` and `supervision` | Used as the `yolo` head-tracking backend. Runs on CPU (default). GPU improves performance. |
| `mediapipe_vision` | Lightweight landmark tracking with MediaPipe | Works on CPU. Enables `--head-tracker mediapipe`. |
| `all_vision` | Convenience alias installing every vision extra | Install when you want the flexibility to experiment with every provider. |
| `dev` | Developer tooling (`pytest`, `ruff`, `mypy`) | Development-only dependencies. Use `--group dev` with uv or `[dev]` with pip. |

**Note:** `dev` is a dependency group (not an optional dependency). With uv, use `--group dev`. With pip, use `[dev]`.

## Configuration

The default setup uses the Hugging Face backend and does not require an API key.

Copy `.env.example` to `.env` when you want to switch backends, provide API keys, or point Hugging Face at your own local endpoint.

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for OpenAI Realtime mode. |
| `GEMINI_API_KEY` | Required for Gemini mode. Also accepts `GOOGLE_API_KEY`. Get one at [aistudio.google.com](https://aistudio.google.com/apikey). |
| `BACKEND_PROVIDER` | Realtime backend to use: `huggingface` (default), `openai`, `gemini`, `ark`, or `aliyun`. |
| `MODEL_NAME` | Optional model override for OpenAI Realtime or Gemini Live. Defaults to `gpt-realtime` for OpenAI and `gemini-3.1-flash-live-preview` for Gemini. Hugging Face uses the server's model selection. |
| `HF_REALTIME_CONNECTION_MODE` | Hugging Face connection selector: `deployed` uses the built-in Hugging Face server; `local` uses `HF_REALTIME_WS_URL`. Defaults to `deployed`. |
| `HF_REALTIME_LANGUAGE` | Speech recognition language hint for Hugging Face Realtime. Defaults to `zh` for Chinese; set to `auto` to let the backend detect the language. The built-in deployed server may ignore this if its server-side STT is not configured for the requested language; for reliable Chinese recognition, use the local gateway with `GATEWAY_STT=faster-whisper` and `GATEWAY_LANGUAGE=zh`. |
| `HF_REALTIME_WS_URL` | Direct websocket endpoint for your own Hugging Face backend. Accepts either a base URL like `ws://127.0.0.1:8765/v1` or the full websocket URL `ws://127.0.0.1:8765/v1/realtime`. Used when `HF_REALTIME_CONNECTION_MODE=local`. |
| `HF_REALTIME_AUTO_START` | Optional. When `true` with `HF_REALTIME_CONNECTION_MODE=local`, the app starts `reachy-mini-hf-realtime-gateway` before connecting to `HF_REALTIME_WS_URL`. The app uses `services/hf_realtime_gateway/.venv/bin/reachy-mini-hf-realtime-gateway` when present, otherwise the command must be available on `PATH`. Defaults to `false`. |
| `HF_REALTIME_AUTO_START_TIMEOUT_SECONDS` | Optional readiness timeout for `HF_REALTIME_AUTO_START`, in seconds. First local model download/warmup can take several minutes. Defaults to `600`. |
| `HF_HOME` | Cache directory for local Hugging Face downloads (only used with `--local-vision` flag, defaults to `./cache`). |
| `HF_TOKEN` | Optional token for Hugging Face access (for gated/private assets). |
| `LOCAL_VISION_MODEL` | Hugging Face model path for local vision processing (only used with `--local-vision` flag, defaults to `HuggingFaceTB/SmolVLM2-2.2B-Instruct`). |
| `REACHY_MINI_MEMORY_CONTEXT_ENABLED` | Optional. When `true`, each user transcript can refresh model-visible long-term memory context. Defaults to `false` for the fastest realtime response path. |
| `REACHY_MINI_MEMORY_AUTO_EXTRACT` | Optional. Enables automatic memory extraction when supported by the runtime flow. Defaults to `false`; explicit `manage_memory` tool calls still work when memory is available. |
| `ARK_REALTIME_APP_ID` / `VOLCENGINE_REALTIME_APP_ID` / `VOLC_APP_ID` | Required for `BACKEND_PROVIDER=ark`. Volcengine Realtime `X-Api-App-ID`. |
| `ARK_REALTIME_ACCESS_KEY` / `VOLCENGINE_REALTIME_ACCESS_KEY` / `VOLCENGINE_REALTIME_ACCESS_TOKEN` / `VOLC_ACCESS_KEY` | Required for `BACKEND_PROVIDER=ark`. Volcengine Realtime `X-Api-Access-Key`. |
| `ARK_REALTIME_APP_KEY` / `VOLCENGINE_REALTIME_APP_KEY` / `VOLC_APP_KEY` | Required for `BACKEND_PROVIDER=ark`. Volcengine Realtime `X-Api-App-Key`. |
| `ARK_REALTIME_RESOURCE_ID` / `VOLCENGINE_REALTIME_RESOURCE_ID` / `VOLC_RESOURCE_ID` | Optional for `BACKEND_PROVIDER=ark`; defaults to `volc.speech.dialog`. |
| `ARK_REALTIME_WS_URL` | Optional Volcengine Realtime websocket URL; defaults to `wss://openspeech.bytedance.com/api/v3/realtime/dialogue`. |
| `ARK_REALTIME_BOT_NAME` | Optional bot display name sent to Volcengine Realtime. Defaults to `Reachy Mini`. |
| `ARK_REALTIME_INPUT_SAMPLE_RATE` | Optional input audio sample rate for Volcengine Realtime. Defaults to `16000`. |
| `ARK_REALTIME_OUTPUT_SAMPLE_RATE` | Optional output audio sample rate for Volcengine Realtime. Defaults to `24000`. |
| `DASHSCOPE_API_KEY` / `ALIYUN_API_KEY` | Required for `BACKEND_PROVIDER=aliyun`. DashScope API key used for Qwen realtime. |
| `ALIYUN_REALTIME_MODEL` | Optional Aliyun DashScope model override. Defaults to `qwen3.5-omni-flash-realtime`. Kept separate from `MODEL_NAME` to avoid cross-provider model collisions. |
| `ALIYUN_REALTIME_WS_URL` | Optional Aliyun DashScope realtime websocket URL; defaults to the Qwen3.5 Omni realtime endpoint shown in the Bailian console. |
| `ALIYUN_REALTIME_INPUT_SAMPLE_RATE` | Optional input audio sample rate for Aliyun DashScope realtime. Defaults to `16000`. |
| `ALIYUN_REALTIME_OUTPUT_SAMPLE_RATE` | Optional output audio sample rate for Aliyun DashScope realtime. Defaults to `24000`. |
| `ALIYUN_REALTIME_VIDEO_FPS` | Optional camera frame rate for Aliyun native `input_image_buffer.append` vision input after speech is detected. Defaults to `1`; set to `0` to disable automatic speech-window visual frames. |
| `ALIYUN_REALTIME_VIDEO_ACTIVE_SECONDS` | Optional duration, in seconds, to keep automatic Aliyun visual frames active after speech is detected. Defaults to `10`. |
| `OPENCLAW_GATEWAY_URL` | OpenClaw gateway URL used by the `ask_openclaw` tool. Defaults to `ws://localhost:18789`; the tool is not loaded when this is blank. |
| `OPENCLAW_TOKEN` | OpenClaw gateway auth token. The `ask_openclaw` tool is not loaded when this is blank. |
| `OPENCLAW_AGENT_ID` | Optional OpenClaw agent ID. Defaults to `main`. |
| `OPENCLAW_SESSION_KEY` | Optional OpenClaw session key. Defaults to `main`. |
| `OPENCLAW_TIMEOUT_SECONDS` | Optional timeout for each `ask_openclaw` request. Defaults to `60`. |
| `VOLCENGINE_WEB_SEARCH_API_KEY` | Required only for the backend-independent `web_search` tool. Uses the Volcengine Web Search product APIKey endpoint, not the Ark Responses plugin. |
| `VOLCENGINE_WEB_SEARCH_API_URL` | Optional Web Search API URL; defaults to `https://open.feedcoopapi.com/search_api/web_search`. |
| `VOLCENGINE_WEB_SEARCH_TIMEOUT_SECONDS` | Optional timeout for each `web_search` call; defaults to `30`. |
| `WEATHERAPI_API_KEY` | Optional backend-independent WeatherAPI.com key. When set, prompt-time base info and the `current_location_weather` tool include live weather. |
| `SMTP_HOST` / `SMTP_PORT` | Optional backend-independent SMTP server settings for the `send_email` tool. Defaults are Gmail-oriented (`smtp.gmail.com`, `587`). |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | SMTP credentials for `send_email`. Gmail aliases `GMAIL_EMAIL`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, and `EMAIL_APP_PASSWORD` are also accepted. |
| `SMTP_FROM_EMAIL` / `SMTP_FROM_NAME` | Optional sender address/name overrides for `send_email`. |
| `SMTP_USE_SSL` / `SMTP_USE_TLS` | Optional SMTP security flags. SSL defaults to `true` on port `465`; TLS defaults to enabled when SSL is not used. |
| `default_target_email` | Optional backend-independent default recipient used by `send_email` when the tool call does not provide `target_email`. |
| `REACHY_MINI_CUSTOM_PROFILE` | Optional startup profile name. Ignored when a saved startup setting or locked profile overrides it. |
| `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` | Optional filesystem directory containing external profile folders. |
| `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` | Optional filesystem directory containing external tool modules. |
| `AUTOLOAD_EXTERNAL_TOOLS` | Optional. When `true`, auto-load every valid external tool module in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`. Defaults to `false`. |
| `REACHY_MINI_SKIP_DOTENV` | Optional. When truthy, skips automatic `.env` discovery/loading and uses the process environment only. |

### Hugging Face Connection Modes

Use the built-in Hugging Face server through the app-managed Space proxy. This is the default for a new install; set it explicitly only when you want to switch back from a saved local endpoint:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=deployed
HF_REALTIME_LANGUAGE=zh
```

The deployed server chooses its own STT backend. If Chinese speech is recognized as English, switch to the local gateway so the STT model and language are controlled by the repository root `.env`.

Run your own realtime voice backend using [speech-to-speech](https://github.com/huggingface/speech-to-speech) on the same machine as the conversation app:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
HF_REALTIME_AUTO_START=true
```

If `reachy-mini-hf-realtime-gateway` is installed in the app environment or in `services/hf_realtime_gateway/.venv`, the app can start it automatically:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
HF_REALTIME_AUTO_START=true
HF_REALTIME_AUTO_START_TIMEOUT_SECONDS=600
```

Run your own Hugging Face backend on your laptop and connect to it from Reachy Mini Wireless over the same Wi-Fi network:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://<your-laptop-lan-ip>:8765/v1/realtime
```

For that LAN setup, make sure the backend listens on an address reachable from the robot, not only on `127.0.0.1`.

If the backend stays bound to loopback on your laptop, you can forward it into the robot over SSH instead:

```bash
ssh -N -R 8765:127.0.0.1:8765 <robot-user>@<robot-host>
```

Then set this on the robot:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

When using the headless settings UI, selecting `Hugging Face` lets you choose either the built-in server or a local `host:port` target. The UI writes `HF_REALTIME_CONNECTION_MODE` for you, and the local path writes `HF_REALTIME_WS_URL` with a default of `localhost:8765`.

## Backend recipes

Use these minimal `.env` snippets as starting points. Copy `.env.example` to `.env`, then keep only the lines needed for your chosen backend.

### Hugging Face deployed backend

This is the default path and requires no API key:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=deployed
HF_REALTIME_LANGUAGE=zh
```

### Local Hugging Face realtime gateway

Use this when you want full local control over STT, TTS, language, and the LLM endpoint:

```bash
cd services/hf_realtime_gateway
uv sync
uv run reachy-mini-hf-realtime-gateway --dry-run
uv run reachy-mini-hf-realtime-gateway
```

Then point the app at the local realtime websocket:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
GATEWAY_LLM_BASE_URL=http://127.0.0.1:8000/v1
GATEWAY_LLM_MODEL=your-model-name
GATEWAY_LLM_API_KEY=
```

For app-managed startup, set `HF_REALTIME_AUTO_START=true`. The app first looks for `services/hf_realtime_gateway/.venv/bin/reachy-mini-hf-realtime-gateway`, then falls back to `PATH`.

See [services/hf_realtime_gateway/README.md](services/hf_realtime_gateway/README.md) for gateway-specific STT/TTS settings.

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

`GOOGLE_API_KEY` is accepted as an alias for `GEMINI_API_KEY`.

### Volcengine Realtime

```env
BACKEND_PROVIDER=ark
ARK_REALTIME_APP_ID=...
ARK_REALTIME_ACCESS_KEY=...
ARK_REALTIME_APP_KEY=...
ARK_REALTIME_RESOURCE_ID=volc.speech.dialog
```

The Ark path now talks directly to the Volcengine realtime websocket for recognition and replies. It no longer calls an OpenAI-compatible sidecar / OpenRouter model to decide local tool routing first.

### Aliyun DashScope Realtime

```env
BACKEND_PROVIDER=aliyun
DASHSCOPE_API_KEY=...
ALIYUN_REALTIME_MODEL=qwen3.5-omni-flash-realtime
```

The Aliyun path uses Qwen3.5 Omni's realtime websocket with provider-side function calling. Tools enabled by the active profile are sent in the session config, so tool calls are executed by the existing local `BackgroundToolManager` path.

When a camera is available, the native Aliyun websocket path sends JPEG frames through `input_image_buffer.append` at `ALIYUN_REALTIME_VIDEO_FPS` frames per second only for a short window after speech is detected. The default is `1` FPS for `10` seconds. The `camera` tool can also start an asynchronous 1 FPS image sequence with `duration_seconds` when the model needs more visual context; sequence requests return immediately and continue sending frames in the background for up to `120` seconds. The returned `tool_id` can be inspected with `task_status` and cancelled with `task_cancel` or `cancel_aliyun_camera_sequence`.

## Running the app

Activate your virtual environment, then launch:

```bash
reachy-mini-conversation-app
```

> [!TIP]
> Make sure the Reachy Mini daemon is running before launching the app. If you see a `TimeoutError`, it means the daemon isn't started. See [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) for setup instructions.

The app runs in console mode by default. Add `--gradio` to launch a web UI at http://127.0.0.1:7860/ (required for simulation mode). Vision and head-tracking options are described in the CLI table below.

### Official simulation / virtual test environment

Reachy Mini already provides an official MuJoCo simulation through the `reachy-mini` SDK. Use it instead of an app-local mock robot when you need a hardware-free test environment:

```bash
# Install the official simulation extra in your active environment
uv pip install "reachy-mini[mujoco]"

# Terminal 1: start the official simulated robot daemon
reachy-mini-daemon --sim

# Terminal 2: run the conversation app against that local daemon
reachy-mini-conversation-app --no-camera --gradio
```

On macOS, MuJoCo may need its launcher instead of the regular daemon entrypoint:

```bash
mjpython -m reachy_mini.daemon.app.main --sim
```

For a lighter daemon-level smoke test without MuJoCo physics, the SDK also exposes `--mockup-sim`:

```bash
reachy-mini-daemon --mockup-sim
reachy-mini-conversation-app --no-camera --gradio
```

When the app connects to an official simulated daemon, it detects `simulation_enabled` or `mockup_sim_enabled` from the SDK status and automatically enables Gradio if needed.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--head-tracker {none,yolo,mediapipe}` | `yolo` | Select a head-tracking backend when a camera is available. `yolo` uses a local YOLO face detector, `mediapipe` comes from the `reachy_mini_toolbox` package, and `none` disables head tracking while keeping camera capture. Requires the matching optional extra. |
| `--no-camera` | `False` | Run without camera capture or head tracking. |
| `--local-vision` | `False` | Use the local vision model (SmolVLM2) for camera-tool requests instead of the selected realtime backend. Requires `local_vision` extra to be installed. |
| `--gradio` | `False` | Launch the Gradio web UI. Without this flag, runs in console mode. Required when running in simulation mode. |
| `--robot-name` | `None` | Optional. Connect to a specific robot by name when running multiple daemons on the same subnet. See [Multiple robots on the same subnet](#advanced-features). |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |

### Examples

```bash
# Run with MediaPipe head tracking
reachy-mini-conversation-app --head-tracker mediapipe

# Run with the default YOLO face-detection backend for head tracking
reachy-mini-conversation-app

# Disable head tracking while keeping camera capture
reachy-mini-conversation-app --head-tracker none

# Run with local vision processing (requires local_vision extra)
reachy-mini-conversation-app --local-vision

# Audio-only conversation (no camera)
reachy-mini-conversation-app --no-camera

# Launch with Gradio web interface
reachy-mini-conversation-app --gradio
```

The YOLO tracker can take longer to start the first time because it loads the face-detection model in a subprocess. If startup times out on a slow or cold machine, increase the startup wait:

```bash
REACHY_MINI_YOLO_HEAD_TRACKER_START_TIMEOUT_SECONDS=180 reachy-mini-conversation-app --gradio
```

> [!WARNING]
> `--local-vision` is not supported when running the conversation app directly on Reachy Mini Wireless / the Raspberry Pi. For local vision, keep the daemon running on the robot and start the conversation app from your laptop or workstation instead.

## LLM tools exposed to the assistant

| Tool | Action | Dependencies |
|------|--------|--------------|
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `camera` | Capture the latest camera frame and analyze it with the selected realtime backend or the local vision model. With Aliyun, `duration_seconds` can start an async 1 FPS sequence for up to 120 seconds. | Requires camera worker. Uses local vision when `--local-vision` is enabled. |
| `head_tracking` | Enable or disable head-tracking offsets (not identity recognition - only detects and tracks head position). | Camera worker with configured head tracker (`--head-tracker`). |
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face datasets. | Core install only. Uses the default open emotions dataset: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library). |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `idle_do_nothing` | Explicitly remain idle during an idle turn. Not intended for normal conversation turns. | Core install only. |
| `task_status` | Inspect currently running or recently completed background tools. | System tool, loaded for every profile. |
| `task_cancel` | Cancel a running background tool by ID. | System tool, loaded for every profile. |
| `cancel_aliyun_camera_sequence` | Cancel the latest or specified Aliyun async camera sequence. | System tool, loaded for every profile; only cancels Aliyun camera sequence jobs. |
| `manage_memory` | Remember, update, forget, or search explicit long-term memories. | System tool, loaded for every profile. Requires local memory store availability. |
| `current_location_weather` | Fetch the latest approximate current address and weather. | Core install only. Live weather requires `WEATHERAPI_API_KEY`. |
| `send_email` | Send a user-requested email through the configured SMTP account. | Requires SMTP credentials and either `target_email` in the tool call or `default_target_email`. |
| `web_search` | Search current web pages, web summaries, or images through the Volcengine Web Search product API. | Requires `VOLCENGINE_WEB_SEARCH_API_KEY`; optional `VOLCENGINE_WEB_SEARCH_API_URL` and `VOLCENGINE_WEB_SEARCH_TIMEOUT_SECONDS`. |
| `ask_openclaw` | Forward complex requests to an OpenClaw agent for external memory, cross-channel context, or tools not available locally. | Requires a running OpenClaw gateway plus non-empty `OPENCLAW_GATEWAY_URL` and `OPENCLAW_TOKEN`; disable by removing it from `tools.txt`. |

Tool availability is profile-gated. A tool listed in `profiles/<profile>/tools.txt` is loaded if the corresponding profile-local file, built-in module, or external tool module exists. System tools (`task_status`, `task_cancel`, `cancel_aliyun_camera_sequence`, `manage_memory`) are added automatically for every profile. `web_search`, `current_location_weather`, and `send_email` are normal backend-independent tools; their API keys and default recipient settings do not depend on `BACKEND_PROVIDER`. `ask_openclaw` is additionally gated by `OPENCLAW_GATEWAY_URL` and `OPENCLAW_TOKEN`.

### Persistent memory

The app creates a local SQLite memory store named `memory.sqlite3`. In a source checkout, the default path is under `src/reachy_mini_conversation_app/storage/`; packaged or instance-specific runs may use the instance storage path.

Memory has two distinct layers:
- Conversation history: sessions and message snippets can be retained locally for search and future context building.
- Explicit long-term memory: the assistant can use `manage_memory` to remember, update, forget, or search durable facts, preferences, tasks, and notes.

By default, memory storage is available to tools, but model-visible memory context is kept off for the fastest realtime response path. Enable context injection with:

```env
REACHY_MINI_MEMORY_CONTEXT_ENABLED=true
```

Keep `REACHY_MINI_MEMORY_AUTO_EXTRACT=false` unless you intentionally want automatic extraction behavior in runtime paths that support it. For predictable behavior, prefer explicit user requests such as "remember that..." so the assistant uses `manage_memory`.

### OpenClaw bridge

`ask_openclaw` is intended for complex requests that should be delegated to an external OpenClaw agent, for example cross-channel context, larger tool ecosystems, or external long-term memory. It is not exposed unless both values are non-empty:

```env
OPENCLAW_GATEWAY_URL=ws://localhost:18789
OPENCLAW_TOKEN=your-openclaw-token
OPENCLAW_AGENT_ID=main
OPENCLAW_SESSION_KEY=main
OPENCLAW_TIMEOUT_SECONDS=60
```

To disable it for a profile even when configured, remove or comment `ask_openclaw` in that profile's `tools.txt`.

### Email sending

`send_email` only sends when the user explicitly asks for an email. Configure SMTP before enabling it in a profile:

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

For Gmail, use an app password instead of the account login password.

## Advanced features

Built-in motion content is published as open Hugging Face datasets:
- Emotions: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- Dances: [`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library)

<details>
<summary><b>Custom profiles</b></summary>

Create custom profiles with dedicated instructions and enabled tools.

For normal usage, select a profile from the UI and save it for startup. That selection is persisted in `startup_settings.json`.

If no startup settings have been saved yet, you can still seed startup from the environment with `REACHY_MINI_CUSTOM_PROFILE=<name>` to load `profiles/<name>/`. If neither is set, the `default` profile is used.

Each profile should include `instructions.txt` (prompt text). `tools.txt` (list of allowed tools) is recommended. If missing for a non-default profile, the app falls back to `profiles/default/tools.txt`. Profiles can optionally contain custom tool implementations.

**Custom instructions:**

Write plain-text prompts in `instructions.txt`. To reuse shared prompt pieces, add lines like:
```
[passion_for_lobster_jokes]
[identities/witty_identity]
```
Each placeholder pulls the matching file under `src/reachy_mini_conversation_app/prompts/` (nested paths allowed). See `profiles/example/` for a reference layout.

**Enabling tools:**

List enabled tools in `tools.txt`, one per line. Prefix with `#` to comment out:
```
play_emotion
# move_head

# My custom tool defined locally
sweep_look
```
Tools are resolved first from Python files in the profile folder (custom tools), then from the core library `src/reachy_mini_conversation_app/tools/` (like `dance`, `head_tracking`).

**Custom tools:**

On top of built-in tools found in the core library, you can implement custom tools specific to your profile by adding Python files in the profile folder.
Custom tools must subclass `reachy_mini_conversation_app.tools.core_tools.Tool` (see `profiles/example/sweep_look.py`).

**Edit personalities from the UI:**

When running with `--gradio`, open the "Personality" accordion:
- Select among available profiles (folders under `profiles/`) or the built‑in default.
- Click "Apply" to update the current session instructions live.
- Create a new personality by entering a name and instructions text. It stores files under `profiles/<name>/` and copies `tools.txt` from the `default` profile.

Note: The "Personality" panel updates the conversation instructions. Tool sets are loaded at startup from `tools.txt` and are not hot‑reloaded.

</details>

<details>
<summary><b>Locked profile mode</b></summary>

To create a locked variant of the app that cannot switch profiles, edit `src/reachy_mini_conversation_app/config.py` and set the `LOCKED_PROFILE` constant to the desired profile name:
```python
LOCKED_PROFILE: str | None = "mars_rover"  # Lock to this profile
```
When `LOCKED_PROFILE` is set, the app always uses that profile, ignoring saved startup settings, `REACHY_MINI_CUSTOM_PROFILE`, and the Gradio UI. The UI shows "(locked)" and disables all profile editing controls.
This is useful for creating dedicated clones of the app with a fixed personality. Clone scripts can simply edit this constant to lock the variant.

</details>

<details>
<summary><b>External profiles and tools</b></summary>

You can extend the app with profiles/tools stored outside the repository defaults.

- Core profiles are under `profiles/`.
- Core tools are under `src/reachy_mini_conversation_app/tools/`.

**Recommended layout:**

```text
external_content/
├── external_profiles/
│   └── my_profile/
│       ├── instructions.txt
│       ├── tools.txt        # optional (see fallback behavior below)
│       └── voice.txt        # optional
└── external_tools/
    └── my_custom_tool.py
```

**Environment variables:**

Set these values in your `.env` when you want env-driven external profile/tool selection:

```env
# Optional fallback/manual profile selector:
REACHY_MINI_CUSTOM_PROFILE=my_profile
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=./external_content/external_profiles
REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=./external_content/external_tools
# Optional convenience mode:
# AUTOLOAD_EXTERNAL_TOOLS=1
```

**Loading behavior:**

- **Default/strict mode**: `tools.txt` defines enabled tools explicitly. Every name in `tools.txt` must resolve to either a built-in tool (`src/reachy_mini_conversation_app/tools/`) or an external tool module in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`.
- **Convenience mode** (`AUTOLOAD_EXTERNAL_TOOLS=1`): all valid `*.py` tool files in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` are auto-added.
- **External profile fallback**: if the selected external profile has no `tools.txt`, the app falls back to built-in `profiles/default/tools.txt`.

This supports both:
1. Downloaded external tools used with built-in/default profile.
2. Downloaded external profiles used with built-in default tools.

</details>

<details>
<summary><b>Prompt snippets and base information</b></summary>

Profile instructions can include reusable snippets with bracket syntax:

```text
[passion_for_lobster_jokes]
[identities/witty_identity]
```

Snippets resolve from `src/reachy_mini_conversation_app/prompts/`. The app also builds base information such as current time, approximate location, and optional WeatherAPI data. `current_location_weather` uses the same base-info path, so prompt-time context and on-demand tool results stay aligned.

</details>

<details>
<summary><b>Multiple robots on the same subnet</b></summary>

If you run multiple Reachy Mini daemons on the same network, use:

```bash
reachy-mini-conversation-app --robot-name <name>
```

`<name>` must match the daemon's `--robot-name` value so the app connects to the correct robot.

</details>

## Troubleshooting

### Reachy Mini daemon connection failures

If startup fails with `TimeoutError` or `ConnectionError`, start the daemon first and make sure the app is targeting the right robot:

```bash
reachy-mini-daemon
reachy-mini-conversation-app --debug
```

For multiple daemons on one subnet, pass the daemon name:

```bash
reachy-mini-conversation-app --robot-name <name>
```

On macOS simulation, prefer:

```bash
mjpython -m reachy_mini.daemon.app.main --sim
```

If port `8000` is already in use, stop the existing daemon or pick the daemon setup you intend to test.

### Hugging Face local gateway does not start

Check the service environment directly:

```bash
cd services/hf_realtime_gateway
uv run reachy-mini-hf-realtime-gateway --dry-run
uv run reachy-mini-hf-realtime-gateway --healthcheck
```

Common causes:
- `GATEWAY_LLM_BASE_URL` or `GATEWAY_LLM_MODEL` is missing.
- The service-local `.venv` was not created with `uv sync`.
- First model download or warmup takes longer than `HF_REALTIME_AUTO_START_TIMEOUT_SECONDS`.
- A root-level `HF_HOME` points at an unexpected cache. Keep gateway model/cache settings in the gateway environment when debugging.

### Chinese speech recognition is inaccurate

The deployed Hugging Face server may ignore local `HF_REALTIME_LANGUAGE` if its server-side STT is configured differently. For reliable Chinese recognition, use the local gateway and set gateway STT/language settings, for example `GATEWAY_STT=faster-whisper` and `GATEWAY_LANGUAGE=zh`.

### Local vision fails or is slow

`--local-vision` loads PyTorch/Transformers and is not suitable for running directly on Reachy Mini Wireless / Raspberry Pi. Run the daemon on the robot and run this app from a laptop or workstation. If imports crash or GPU memory is insufficient, run without `--local-vision` so camera analysis uses the selected realtime backend.

### Head tracking startup timeout

YOLO head tracking loads the detector in a subprocess. On cold machines, increase startup wait:

```bash
REACHY_MINI_YOLO_HEAD_TRACKER_START_TIMEOUT_SECONDS=180 reachy-mini-conversation-app --gradio
```

Use `--head-tracker none` to keep camera capture while disabling head tracking, or `--no-camera` to disable both camera capture and head tracking.

### Profile or tool does not load

Confirm the selected profile and tool allowlist:

```bash
ls profiles/<profile>
sed -n '1,120p' profiles/<profile>/tools.txt
```

For shared tools, adding `src/reachy_mini_conversation_app/tools/<tool>.py` is not enough: the tool also needs to be listed in the active profile's `tools.txt`, unless it is a system tool or loaded through `AUTOLOAD_EXTERNAL_TOOLS=1`.

### Development checks

Install the dev dependency group, then run the main checks:

```bash
uv sync --group dev
uv run ruff check .
uv run mypy
uv run pytest
```

Use narrower tests while iterating:

```bash
uv run pytest tests/test_config_name_collisions.py tests/test_external_loading.py
uv run pytest tests/test_memory.py
uv run pytest tests/tools
```

## Contributing

We welcome bug fixes, features, profiles, and documentation improvements. Please review our
[contribution guide](CONTRIBUTING.md) for branch conventions, quality checks, and PR workflow.

Quick start:
- Fork and clone the repo
- Follow the [installation steps](#installation) (include the `dev` dependency group)
- Run contributor checks listed in [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache 2.0
