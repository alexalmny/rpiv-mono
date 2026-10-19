"""Offline tests: no model download, microphone access, or CUDA initialization."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    "voice_worker", Path(__file__).parents[1] / "packages/rpiv-voice/audio/faster-whisper-worker.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def request(identifier=1, samples=None):
    if samples is None:
        samples = np.array([0.25, -0.5], dtype="<f4")
    return {"id": identifier, "sampleRate": 16000, "audio": base64.b64encode(samples.tobytes()).decode()}


class WorkerTests(unittest.TestCase):
    def test_audio_validation(self):
        np.testing.assert_array_equal(worker.read_audio(request(), np), [0.25, -0.5])
        for change in [
            {"sampleRate": 44100}, {"id": True}, {"audio": "!!!"},
            {"audio": base64.b64encode(b"a").decode()},
            {"audio": request(samples=np.array([np.nan], dtype="<f4"))["audio"]},
            {"audio": request(samples=np.zeros(480001, dtype="<f4"))["audio"]},
        ]:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    worker.read_audio({**request(), **change}, np)

    def serve(self, cuda_count=1, cuda_failure_at=None, missing=False):
        calls = []
        download = Mock(side_effect=[FileNotFoundError(), "/model"] if missing else None, return_value="/model")

        class Model:
            def __init__(self, path, **kwargs):
                self.device = kwargs["device"]
                self.count = 0
                calls.append((path, kwargs))

            def transcribe(self, samples, **kwargs):
                self.count += 1
                if self.device == "cuda" and self.count == cuda_failure_at:
                    raise RuntimeError("CUDA test failure")
                assert kwargs["beam_size"] == 1
                assert kwargs["condition_on_previous_text"] is False
                assert kwargs["vad_filter"] is False
                return iter([types.SimpleNamespace(text=" Bonjour ")]), None

        modules = {
            "ctranslate2": types.SimpleNamespace(get_cuda_device_count=lambda: cuda_count),
            "faster_whisper": types.SimpleNamespace(WhisperModel=Model),
            "faster_whisper.utils": types.SimpleNamespace(download_model=download),
        }
        # Invalid input must return an error without losing the following request.
        payload = "\n".join(json.dumps(r) for r in [request(), {**request(2), "sampleRate": 8}, request(3)]) + "\n"
        output = io.StringIO()
        with patch.dict(sys.modules, modules), patch.object(worker, "configure_cuda_dlls"), \
                patch.object(worker, "PROTOCOL", output), \
                patch.object(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(payload.encode()))):
            worker.serve(4, "fr")
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        return calls, messages, download

    def test_gpu_int8_fp16_and_cache_reuse(self):
        calls, messages, download = self.serve()
        self.assertEqual(calls[0][1]["compute_type"], "int8_float16")
        self.assertEqual(calls[0][1]["local_files_only"], True)
        self.assertEqual(messages[0]["device"], "cuda")
        self.assertEqual(messages[1], {"id": 1, "text": "Bonjour"})
        self.assertIn("error", messages[2])
        self.assertEqual(messages[3], {"id": 3, "text": "Bonjour"})
        download.assert_called_once_with("base", local_files_only=True)

    def test_cpu_without_cuda(self):
        calls, messages, _ = self.serve(cuda_count=0)
        self.assertEqual(calls[0][1]["compute_type"], "int8")
        self.assertEqual(messages[0]["device"], "cpu")

    def test_gpu_warmup_failure_falls_back(self):
        calls, messages, _ = self.serve(cuda_failure_at=1)
        self.assertEqual([c[1]["device"] for c in calls], ["cuda", "cpu"])
        self.assertTrue(any(m.get("device") == "cpu" for m in messages))
        self.assertTrue(any(m.get("text") == "Bonjour" for m in messages))

    def test_gpu_decode_failure_retries_same_audio_on_cpu(self):
        calls, messages, _ = self.serve(cuda_failure_at=2)
        self.assertEqual([c[1]["device"] for c in calls], ["cuda", "cpu"])
        self.assertIn({"id": 1, "text": "Bonjour"}, messages)

    def test_missing_cache_emits_progress_then_downloads(self):
        _, messages, download = self.serve(missing=True)
        self.assertEqual(messages[0]["kind"], "progress")
        self.assertEqual(download.call_args_list[1].args, ("base",))


if __name__ == "__main__":
    unittest.main()
