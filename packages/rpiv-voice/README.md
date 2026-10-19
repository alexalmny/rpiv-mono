# rpiv-voice — faster-whisper Base fork

This fork keeps [rpiv-voice](https://github.com/juicesharp/rpiv-mono)'s `/voice` interface and Decibri microphone capture, replacing sherpa-onnx with **faster-whisper and multilingual Whisper Base**.

- Local speech recognition: microphone audio is never uploaded.
- One Python worker stays model-loaded for the entire dictation session, including live partial previews.
- Automatically uses NVIDIA CUDA with `int8_float16`, or CPU `int8` if CUDA initialization or inference fails.
- On Windows, reuses CUDA/cuDNN DLLs from an installed Torch package when available, without importing Torch.
- Greedy decoding (`beam_size=1`), no word timestamps, no previous-segment conditioning, and no redundant second VAD pass.
- Existing live preview, hallucination filter, pause, settings and review-before-send behavior remain intact.

## Install from a checkout

Requires Pi, Node.js 22+, Python 3.10+ with compatible faster-whisper wheels, and microphone access. Windows x64, Apple Silicon macOS and glibc Linux x64/arm64 are supported by the existing capture dependency; Python-wheel availability must also match your platform. CUDA acceleration is optional.

From `packages/rpiv-voice`:

```sh
python -m pip install -r requirements.txt
npm ci --workspaces=false --omit=dev --legacy-peer-deps
pi install .
```

Pi supplies its own host packages. Restart Pi after installing, then run `/voice`.
This uses your selected/shared Python installation: **no virtual environment is created** and the extension never runs pip automatically.

If Pi's `python` is not the interpreter containing faster-whisper, set `RPIV_VOICE_PYTHON` to the full executable path before starting Pi. It is an executable path, not a shell command.

The first run uses an existing Hugging Face cached `Systran/faster-whisper-base` model if present; otherwise it downloads approximately **148 MB** of model/tokenizer files. Subsequent inference is offline. The model cache respects the Hugging Face environment variables; it is not the upstream sherpa model directory.

## Use

| Key | Action |
| --- | --- |
| `/voice` | Open dictation |
| Enter | Finish dictation and paste the transcript into Pi's editor; does not send it |
| Esc | Discard dictation |
| Space | Pause/resume transcription |
| Tab | Open settings |

Pi's remapped confirm/cancel keys are respected. Live partials update about once a second; phrase boundaries commit final text. The model is loaded when `/voice` opens and released when it closes. Shutdown/reload also terminates active workers.

## Settings and diagnostics

Settings are unchanged: `~/.config/rpiv-voice/voice.json` (or the XDG config path) supports `hallucinationFilterEnabled`, `equalizerEnabled` and JSON-only `numThreads` (default 4, range 1–16). Threads primarily affect CPU inference.

Language follows the optional rpiv-i18n locale, or is auto-detected per utterance if that package is absent. The multilingual Base model is fixed.

`~/.config/rpiv-voice/errors.log` records the selected backend (`faster-whisper base cuda` or `cpu`), fallback reasons, startup errors and decode latency. Raw microphone audio and transcripts are not logged by this extension. CPU fallback may be slower; this is not a guarantee that every machine beats upstream sherpa-onnx.

See [configuration](docs/configuration.md), [keys](docs/keybindings.md), and [model/runtime setup](docs/model.md).

## Development

With the monorepo's development dependencies installed:

```sh
npx vitest run --config vitest.voice.config.ts
python test/faster-whisper-worker.test.py
```

The voice-only test setup does not initialize unrelated monorepo extensions. Python tests use fake inference backends and never download models.

## License

MIT, retaining upstream attribution. faster-whisper and its model/runtime dependencies have their own licenses.
