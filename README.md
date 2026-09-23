<p align="center">
  <img src="docs/assets/turbofieldfare-logo-rounded.png" alt="TurboFieldfare" width="220">
</p>

# TurboFieldfare — Qwen for Apple Silicon

The native Mac app now uses **Qwen3.8-27B** through **MLX Swift and Metal**, with
the existing chat interface, sidebar, conversation history, image attachments,
streaming responses, cancellation, sampling controls, and memory HUD.
The sibling `TurboFieldfareDecodeService` owns the model; the foreground app
does not load a second copy of its weights. MLX runs on the Apple GPU using
the Mac's unified memory.

## Build and open

Requires Apple Silicon, macOS 26+, Xcode with Swift 6.2+, and the Metal Toolchain.
If Xcode reports that the Metal Toolchain is missing, install it once:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Then build the app, its sibling service, and MLX's GPU shaders:

```bash
bash Scripts/build-qwen.sh
.build/release/TurboFieldfareMac
```

Command-line `swift build` alone does not compile MLX's Metal shaders. The build
script places `mlx.metallib` beside both executables and reuses it while the
shader sources and compiler remain unchanged. Keep these executables, their
resource bundles, and `mlx.metallib` together.

Choose **Download**, then **Load Model**. In a source checkout the installation
goes to `scratch/qwen3.8-27b.mlx`; otherwise it lives under the app's Application
Support directory. Start with **4K context** on the 18 GB M3 Pro. Larger contexts
and image turns require additional memory.

Use the model dropdown in the status bar or installation screen to choose an
installed model or a compatible package under **Available to Download**:

| Model | Download | Format | Input |
| --- | --- | --- | --- |
| [Qwen3.5 4B](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit) | 3.06 GB | MLX 4-bit | Text and images |
| [Qwen3.5 9B](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | 5.98 GB | MLX 4-bit | Text and images |
| [Qwen3 14B](https://huggingface.co/mlx-community/Qwen3-14B-4bit) | 8.32 GB | MLX 4-bit | Text only |
| Qwen3.8 27B | 12.73 GB | MLX 3-bit | Text and images |
| [Qwen3.8 27B, 64 GB Mac option](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit) | 29.53 GB | MLX 8-bit | Text and images |
| [Qwen3 32B](https://huggingface.co/mlx-community/Qwen3-32B-4bit) | 18.45 GB | MLX 4-bit | Text only |

Choose a model, click **Download** (or **Resume**), then **Load Model**. Selecting
a model does not start a download. The installer shows download size, required
disk space, and progress. Download sizes are not runtime memory requirements.
Each variant has a separate installation directory, pinned revision, verified
file hashes, resumable download, and conversation history. The 27B launch
default is unchanged. The 14B and 32B models use the text runtime and reject
image attachments, including during history replay. Qwen chat formatting preserves earlier reasoning to
match the app's retained token history.

On a 64 GB Mac, choose **Qwen3.8 27B MLX 8-bit (64 GB Mac)** for the
higher-precision package. It installs separately in `scratch/qwen3.8-27b-8bit.mlx`,
pinned to revision `815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9`, with a
29,531,519,120-byte download. Selecting it does not download or load it automatically.
It needs roughly 30 GB for weights plus conversation state, image processing,
temporary allocations, and macOS; the smaller 3-bit option remains available.

Both packages use the same native MLX/Metal path: quantized weights in Apple
Silicon unified memory, one model owner in the sibling service, bounded text
prefill, and reuse of the live conversation cache. Switching unloads the previous
model before loading the next. The reusable allocation cache defaults to 64 MiB;
New Chat and unload release unused cached allocations, and failed/cancelled loads
also clean up. These controls do not shrink the model's weight storage or promise
a measured peak RAM figure. Loading refuses a package whose weights alone exceed
Metal's recommended working-set budget; passing that check does not guarantee
that a long conversation will fit. Start with a modest context and increase it after
checking memory pressure on the target Mac.

This model option provides chat and image inference. Autonomous browser, terminal,
SSH, email, and scheduled job execution require a separate agent/tool integration;
adding this package does not enable those actions.

For Hermes/OpenClaw subagents, the [dedicated Qwen API server](docs/QWEN_AGENT_SERVER.md)
uses continuous batching with one model and two active requests by default.
Run it instead of the native app; it reuses the verified 8-bit installation.
The server also accepts the other five verified Qwen packages via `--model`.
For repeated agent prompts, `--prefix-cache-gb 4` enables a bounded SSD prefix
cache while leaving the additional hot RAM cache disabled.

For a local specialist fleet, use [prompt routing and concurrent models](docs/LOCAL_AGENT_FLEET.md).
It adds Qwen3-Coder, GLM research, a small extraction worker, embeddings, and
reranking behind one Hermes/OpenClaw endpoint. Different models may generate
concurrently when their combined memory reservation fits; idle models are
reclaimed as needed. The optional 3-bit general worker leaves more room for
overlapping large specialists. Install/launch commands and local client settings
are included in the guide.

Installed-model discovery scans the current model's parent, the checkout's
`scratch` directory (when available), and
`~/Library/Application Support/TurboFieldfare`. Incomplete packages and vision
companions are excluded. The list refreshes when the app becomes active; use
**Refresh Installed Models** in the inspector to rescan manually. Switching
releases the previous model and is disabled during active operations.

## Model

The default 27B installation is pinned to
[`leonsarmiento/Qwen3.8-27B-3bit-mlx`](https://huggingface.co/leonsarmiento/Qwen3.8-27B-3bit-mlx),
revision `5fc234d9e6080b8388a11286380e801b7c9f535c`.
The complete download is **12,729,681,276 bytes**, including vision and tokenizer
files. Every file has a pinned SHA-256 checksum in
[the installer catalog](Sources/TurboFieldfareApp/Core/Resources/qwen-model.json).

This is a mixed 3-bit MLX quantization of the requested 27B model, selected for
the 18 GB Mac. It is a different weight encoding from Ollama's
`qwen3.8:27b-mlx` and `qwen3.8:27b-mlx-bf16` tags. BF16 is not installed by this
app. Download size is not peak runtime memory, and Gemma's earlier 2–4 GB
memory measurements do not apply to this dense model.

The installer writes one MLX installation, verifies each completed file, and
publishes its completion receipt only after all files pass. Cancellation keeps
verified complete files for the next attempt; an interrupted file restarts.
**Discard saved download** removes only a recognized partial Qwen installation.
The original Gemma model is neither converted nor deleted.

## Conversation history

The app keeps its local conversation sidebar and records Qwen's exact prompt
and generated token IDs. Browsing history leaves the active lineage alone;
continuing a saved chat replays its recorded inputs. Model identity and template
identity keep Qwen and Gemma token histories separate.

Text continuations reuse Qwen's attention and recurrent state. A turn that adds
another image replays the conversation so image positions remain correct; the
HUD reports zero reused tokens for that replay. **New Chat** clears the active
conversation. Reloading or unloading releases the live model context.

Image support is included in the model download. Images are processed within a
512-pixel bounding box, with a 256-token maximum per image. Saved chats retain
the prepared Qwen image tensors with checksums, plus display thumbnails. Missing
or modified image inputs refuse replay rather than silently becoming text-only
conversations.

## Controls

- Context, temperature, Top-K, and Top-P remain in the inspector.
- **Allocation cache** sets MLX's reusable allocation budget; model weights and
  conversation state are additional memory. Reload to change the budget.
- **Prefill** uses bounded chunks for text. Turning it off processes text one
  token at a time, including history replay. Intermediate text chunks evaluate
  only KV/recurrent state; only the final chunk evaluates vocabulary scores.
  Image prefill uses MLX's multimodal preparation path.
- **Stop** keeps a reply completed through the last token boundary. Cancellation
  or failure before decoding rewinds the turn for retry.
- The HUD reports service memory, generation rate, context use, and token reuse.
  Last-run diagnostics include prefill and time to first token.

Qwen uses the pinned model's thinking template with low reasoning effort and
preserved reasoning history. Gemma-specific expert-cache and RDADVISE controls
are replaced by the applicable MLX controls; their former memory estimates are
not presented as Qwen measurements.

## Validation

Run focused tests through the repository's serial runner:

```bash
bash Scripts/test.sh -c release --filter Qwen
```

Tokenizer-asset tests additionally accept `QWEN_TOKENIZER_TEST_DIRECTORY`, a
directory containing the pinned tokenizer and preprocessing files. They do not
load model weights. [Validation notes](docs/QWEN_VALIDATION.md) distinguish
compiled and tested behavior from real-model measurements.
The [8-bit validation report](docs/QWEN_8BIT_VALIDATION.md) records the release
build and model-free checks against the 64 GB option's actual assets.

## Original Gemma tools

This conversion targets the Mac app and its decode service. The original
`TurboFieldfareCLI`, `TurboFieldfareServer`, `TurboFieldfareRepack`, and `.gturbo`
runtime still target Gemma. Their previous documentation is preserved in
[the Gemma runtime reference](docs/GEMMA_RUNTIME.md),
[the server guide](docs/OPENAI_SERVER.md), and
[the community benchmark protocol](docs/COMMUNITY_BENCHMARKS.md).

## License

[Apache License 2.0](LICENSE). MLX Swift and MLX Swift LM are upstream dependencies
with their own licenses. The model repository publishes its model license.
