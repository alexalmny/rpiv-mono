# faster-whisper Base runtime and model

This fork replaces upstream sherpa-onnx and its ONNX Base archive with the CTranslate2 implementation provided by faster-whisper. The model is **multilingual Base**, not `base.en`.

## Dependencies

- Python 3.10+ with compatible binary wheels, and `faster-whisper==1.2.1`; install the package's `requirements.txt` into your chosen interpreter.
- Existing Node/Decibri microphone capture dependencies, unchanged.
- Optional NVIDIA GPU with compatible CUDA/cuDNN libraries. CPU inference requires no CUDA.

No Python environment, installer, PyTorch download, system FFmpeg installation or tar extraction is created/run by the extension. PyAV is an upstream faster-whisper dependency, even though this worker accepts PCM directly.

## Model download and cache

The worker first resolves `Systran/faster-whisper-base` in the standard Hugging Face cache with `local_files_only=True`. If the cache is unavailable it shows a download phase and calls faster-whisper's model downloader. Expect approximately **148 MB** for model weights, tokenizer, vocabulary and configuration. INT8 compute does not shrink the standard model download.

On a default setup the cache is under `~/.cache/huggingface/hub/models--Systran--faster-whisper-base/`. Hugging Face's `HF_HOME`/`HF_HUB_CACHE` overrides are respected. The extension never deletes or rewrites another application's cached model to repair a failure. It does not touch `~/.pi/models/whisper-base/` from upstream.

After a complete cache download, inference requires no network. For strict offline operation, launch Pi with `HF_HUB_OFFLINE=1`; a missing cache then fails instead of downloading. Startup errors go to `~/.config/rpiv-voice/errors.log` and the overlay closes with an error notification. Repair missing/corrupt model files through the Hugging Face tooling rather than deleting shared caches automatically.

## Worker and acceleration

One Python subprocess is started when `/voice` opens. It loads Base once, warms up inference, then accepts FIFO-serialized mono 16 kHz float32 audio requests over private stdin/stdout pipes. There is no HTTP server or listening port. Each audio request is bounded to 30 seconds; rpiv's existing segment cap is 12 seconds.

The worker chooses CUDA `int8_float16` when a usable NVIDIA GPU exists, otherwise CPU `int8`. GPU warmup failures trigger CPU fallback. A GPU failure during transcription retries that same request on CPU and keeps subsequent requests on CPU. Backend selection/fallback is written to the diagnostic log.

On Windows, if Torch is installed, its `lib` directory is added to DLL lookup and the worker's PATH without importing Torch. This can reuse already-installed CUDA libraries; no CUDA download is performed. Compatibility is still checked by actual inference, not merely by detecting a GPU.

The default executable is `python` from Pi's PATH. Set `RPIV_VOICE_PYTHON` to a full executable path if necessary. `numThreads` controls CPU threads. Startup is bounded to five minutes (including any first download), and an individual decode to one minute. Commit drain keeps the existing ten-second UI bound. Closing/cancelling the session or shutting down Pi terminates the worker and rejects outstanding requests.

## Recognition settings

- `beam_size=1`, `best_of=1`, `temperature=0` for low-latency greedy decoding.
- `condition_on_previous_text=False` to keep independent phrase/partial decodes independent.
- `word_timestamps=False` because the UI consumes text only.
- `vad_filter=False` because Decibri and the existing pipeline already perform voice/silence segmentation.
- Language follows rpiv-i18n when installed; otherwise Whisper auto-detects per utterance.

These settings trade some beam-search accuracy for responsiveness. There is no claim that CTranslate2 is faster than sherpa-onnx on every device.

## Platforms

The unchanged Decibri 3.x capture binaries support Windows x64, Apple Silicon macOS and glibc Linux x64/arm64. Intel macOS and musl Linux remain unsupported by capture. faster-whisper/CTranslate2/PyAV binary-wheel support must also be available for the chosen Python/platform combination. GPU acceleration is optional; Apple Silicon uses CPU, not CUDA.
