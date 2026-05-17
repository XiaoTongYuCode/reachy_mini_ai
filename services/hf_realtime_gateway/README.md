# Reachy Mini HF 实时网关

该服务将 Hugging Face `speech-to-speech` 封装为一个本地 OpenAI Realtime
兼容网关，供 Reachy Mini 对话应用使用。它保留机器人应用作为实时客户端，
并在独立进程中运行本地 STT/TTS，以及外部 OpenAI 兼容 LLM 端点。

## 默认配置

- STT：`faster-whisper`、`large-v3`、`zh`
- LLM：`responses-api`，通过 `GATEWAY_LLM_*` 配置
- TTS：`qwen3`、`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`、`Vivian`
- Realtime 端点：`ws://127.0.0.1:8765/v1/realtime`

## 安装

```bash
cd services/hf_realtime_gateway
uv sync
```

从仓库根目录复制示例配置，并在根目录 `.env` 中设置 LLM 字段：

```bash
cd ../..
cp .env.example .env
```

必填字段：

```env
GATEWAY_LLM_BASE_URL=http://127.0.0.1:8000/v1
GATEWAY_LLM_MODEL=your-model-name
GATEWAY_LLM_API_KEY=
```

如果使用 API 提供商，请设置 `GATEWAY_LLM_API_KEY`。如果使用本地 `vLLM`
或 `llama.cpp`，且服务端不要求鉴权，可留空。

如果兼容 Responses API 的服务端不支持 `chat_template_kwargs`，请关闭
speech-to-speech 的 provider-side thinking 参数：

```env
GATEWAY_RESPONSES_API_DISABLE_THINKING=false
```

默认 STT 使用 `faster-whisper`，避免 STT 和 Qwen3 TTS 在 Apple Silicon 上同时
占用 MLX/Metal 导致本地进程崩溃。若确认本机 `whisper-mlx` 稳定，也可以显式设置：

```env
GATEWAY_STT=whisper-mlx
GATEWAY_STT_MODEL=large-v3
```

## 运行

在不启动模型的情况下检查生成命令：

```bash
uv run reachy-mini-hf-realtime-gateway --dry-run
```

启动网关：

```bash
uv run reachy-mini-hf-realtime-gateway
```

探测已运行的网关：

```bash
uv run reachy-mini-hf-realtime-gateway --healthcheck
```

将 Reachy 应用配置为连接此服务：

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

当 Reachy Mini 通过局域网连接时，请绑定到所有网卡：

```env
GATEWAY_HOST=0.0.0.0
```

然后将应用指向该机器的局域网地址：

```env
HF_REALTIME_WS_URL=ws://<gateway-lan-ip>:8765/v1/realtime
```

## 说明

该封装有意不实现 realtime WebSocket 协议本身。协议处理、VAD/STT/LLM/TTS
编排、实时转录、音频增量，以及函数调用事件，仍由 `speech-to-speech` 负责。
