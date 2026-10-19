"""Private JSON-lines worker: cached multilingual Base, local inference only."""
import argparse
import base64
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys

# Keep third-party diagnostics off the protocol stream (and Pi's terminal).
PROTOCOL = sys.stdout
DLL_HANDLES = []


def emit(message):
    PROTOCOL.write(json.dumps(message, ensure_ascii=True) + "\n")
    PROTOCOL.flush()


def configure_cuda_dlls():
    if sys.platform != "win32":
        return
    # Reuse installed Torch DLLs without importing Torch or downloading CUDA.
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        directory = Path(spec.origin).parent / "lib"
        if directory.is_dir():
            DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")


def decode(model, samples, language):
    segments, _ = model.transcribe(
        samples,
        language=language,
        beam_size=1,
        best_of=1,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=False,  # Decibri already segments speech before it reaches us.
        word_timestamps=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def read_audio(request, np):
    if not isinstance(request, dict):
        raise ValueError("Expected an object")
    identifier = request.get("id")
    if type(identifier) is not int or identifier < 1:
        raise ValueError("Invalid request id")
    if request.get("sampleRate") != 16000:
        raise ValueError("Expected 16000 Hz mono float32 PCM")
    encoded = request.get("audio")
    if not isinstance(encoded, str) or len(encoded) > 16000 * 30 * 4 * 2:
        raise ValueError("Invalid or oversized audio")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) % 4 or len(raw) > 16000 * 30 * 4:
        raise ValueError("Invalid PCM length (maximum 30 seconds)")
    samples = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(samples).all():
        raise ValueError("Non-finite audio samples")
    return samples


def serve(threads, language):
    configure_cuda_dlls()
    import ctranslate2
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model

    try:
        model_path = download_model("base", local_files_only=True)
    except Exception:
        emit({"kind": "progress", "message": "Downloading faster-whisper Base (~148 MB)…"})
        model_path = download_model("base")

    def load(device):
        compute_type = "int8_float16" if device == "cuda" else "int8"
        model = WhisperModel(
            model_path, device=device, compute_type=compute_type,
            cpu_threads=threads, num_workers=1, local_files_only=True,
        )
        # CUDA DLLs are loaded lazily: test a decode before claiming readiness.
        decode(model, np.zeros(16000, dtype=np.float32), language or "en")
        return model

    device = "cpu"
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            model = load("cuda")
            device = "cuda"
        else:
            model = load("cpu")
    except Exception as error:
        emit({"kind": "diagnostic", "message": f"GPU unavailable; using CPU INT8: {error}"})
        model = load("cpu")

    emit({"kind": "ready", "device": device, "model": "base"})
    # A closed parent pipe ends the worker; no server or background daemon.
    while True:
        line = sys.stdin.buffer.readline(4 * 1024 * 1024 + 1)
        if not line:
            return
        if len(line) > 4 * 1024 * 1024:
            raise ValueError("Oversized protocol message")
        request = None
        try:
            request = json.loads(line)
            samples = read_audio(request, np)
            try:
                text = decode(model, samples, language) if samples.size else ""
            except Exception as error:
                if device != "cuda":
                    raise
                emit({"kind": "diagnostic", "message": f"GPU decode failed; retrying on CPU: {error}"})
                del model
                model = load("cpu")
                device = "cpu"
                text = decode(model, samples, language) if samples.size else ""
            emit({"id": request["id"], "text": text})
        except Exception as error:
            identifier = request.get("id") if isinstance(request, dict) else None
            emit({"id": identifier, "error": f"{type(error).__name__}: {error}"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 17))
    parser.add_argument("--language", default=None)
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    with contextlib.redirect_stdout(sys.stderr):
        try:
            serve(args.threads, args.language)
        except Exception as error:
            emit({"kind": "fatal", "error": f"{type(error).__name__}: {error}"})
            raise SystemExit(1)


if __name__ == "__main__":
    main()
