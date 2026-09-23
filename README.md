# TurboFieldfare

Native Qwen inference for Apple Silicon. TurboFieldfare runs local MLX + Metal models on Mac with a simple chat app and a dedicated OpenAI-compatible server for Hermes/OpenClaw agent workflows.

## Requirements

- Apple Silicon Mac
- macOS 26+
- Xcode + Swift 6.2+
- Metal Toolchain

Install the toolchain once if needed:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Quick start

### 1) Start the local agent server

Use the dedicated Hermes/OpenClaw server mode:

```bash
bash Scripts/setup-qwen-agents.sh
python3 Scripts/serve-qwen-agents.py --check
python3 Scripts/serve-qwen-agents.py
```

Optional custom model path:

```bash
python3 Scripts/serve-qwen-agents.py --model /absolute/path/to/qwen3.8-27b-8bit.mlx
```

Optional prefix cache for repeated prompts:

```bash
python3 Scripts/serve-qwen-agents.py --prefix-cache-gb 4
```

Optional concurrency override:

```bash
python3 Scripts/serve-qwen-agents.py --concurrency 3 --context 65536
```

### 2) Verify the server is live

```bash
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
```

### 3) Connect Hermes

In Hermes, open your model/provider settings and choose a custom endpoint. Then set:

- Base URL: `http://127.0.0.1:8080/v1`
- API key: `local`
- Model: `qwen3.8-27b-8bit.mlx`
- Context window: `65536`

Or use the CLI flow in Hermes:

```bash
hermes model
```

Then select `Custom endpoint`, enter the values above, and save.

### 4) Connect OpenClaw

Add a provider similar to this:

```json
{
  "provider": "openai",
  "name": "local-qwen",
  "base_url": "http://127.0.0.1:8080/v1",
  "api_key": "local",
  "model": "qwen3.8-27b-8bit.mlx"
}
```

Then point your parent/child agent config at that provider.

## Basic validation

```bash
python3 Scripts/serve-qwen-agents.py --list-models
python3 Scripts/serve-qwen-agents.py --check
bash Scripts/test.sh --qwen-agents
bash -n Scripts/setup-qwen-agents.sh Scripts/test.sh
```

## Native app quick start

If you want the Mac app instead of the agent server:

```bash
bash Scripts/build-qwen.sh
.build/release/TurboFieldfareMac
```

Then choose a model in the app, download it, and load it.

## Project notes

- This project is built for Apple Silicon and uses MLX + Metal.
- The agent/server path is meant for Hermes/OpenClaw local subagents.
- The native app remains a separate path for direct interactive chat.
- Full details are in [docs/QWEN_AGENT_SERVER.md](docs/QWEN_AGENT_SERVER.md) and [docs/LOCAL_AGENT_FLEET.md](docs/LOCAL_AGENT_FLEET.md).

## License

[Apache License 2.0](LICENSE).
