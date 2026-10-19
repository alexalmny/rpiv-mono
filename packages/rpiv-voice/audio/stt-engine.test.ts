import { EventEmitter } from "node:events";
import { PassThrough, Writable } from "node:stream";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ spawn: vi.fn() }));
vi.mock("node:child_process", () => ({ spawn: mocks.spawn }));
import { createSttEngine, releaseAllSttEngines } from "./stt-engine.ts";

class Worker extends EventEmitter {
	stdout = new PassThrough();
	stderr = new PassThrough();
	requests: Array<{ id: number; audio: string; sampleRate: number }> = [];
	stdin = new Writable({
		write: (chunk, _encoding, callback) => {
			this.requests.push(JSON.parse(chunk.toString()));
			callback();
		},
	});
	kill = vi.fn(() => true);
	send(message: unknown): void {
		this.stdout.write(`${JSON.stringify(message)}\n`);
	}
}

let worker: Worker;
const tick = async (): Promise<void> => {
	await Promise.resolve();
	await Promise.resolve();
	await Promise.resolve();
};
async function start() {
	const pending = createSttEngine({});
	worker.send({ kind: "ready", model: "base", device: "cuda" });
	return pending;
}

beforeEach(() => {
	worker = new Worker();
	mocks.spawn.mockReset().mockReturnValue(worker);
});
afterEach(() => {
	releaseAllSttEngines();
	vi.useRealTimers();
	vi.unstubAllEnvs();
});

describe("faster-whisper worker bridge", () => {
	it("waits for model readiness, forwards progress and uses an explicit Python path safely", async () => {
		vi.stubEnv("RPIV_VOICE_PYTHON", "C:/Python With Spaces/python.exe");
		const progress = vi.fn();
		const pending = createSttEngine({ language: "fr", numThreads: 8, onProgress: progress });
		let ready = false;
		void pending.then(() => { ready = true; });
		await tick();
		expect(ready).toBe(false);
		worker.send({ kind: "progress", message: "Downloading model" });
		expect(progress).toHaveBeenCalledWith("Downloading model");
		worker.send({ kind: "ready", model: "base", device: "cpu" });
		await pending;
		expect(mocks.spawn.mock.calls[0][0]).toBe("C:/Python With Spaces/python.exe");
		expect(mocks.spawn.mock.calls[0][1]).toEqual([
			"-u", expect.stringContaining("faster-whisper-worker.py"), "--threads", "8", "--language", "fr",
		]);
		expect(mocks.spawn.mock.calls[0][2]).toMatchObject({ windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
	});

	it("transfers the exact PCM view and returns trimmed Unicode text", async () => {
		const engine = await start();
		const source = new Float32Array([9, 0.25, -0.5, 9]);
		const pending = engine.recognize(source.subarray(1, 3), 16000);
		await tick();
		const request = worker.requests[0];
		expect(request.sampleRate).toBe(16000);
		expect(Buffer.from(request.audio, "base64")).toEqual(Buffer.from(new Float32Array([0.25, -0.5]).buffer));
		worker.send({ id: request.id, text: "  Bonjour à tous  " });
		await expect(pending).resolves.toBe("Bonjour à tous");
	});

	it("does not send empty audio and rejects unsupported rate or excessive length", async () => {
		const engine = await start();
		await expect(engine.recognize(new Float32Array(), 16000)).resolves.toBe("");
		await expect(engine.recognize(new Float32Array(1), 44100)).rejects.toThrow("16000 Hz");
		await expect(engine.recognize(new Float32Array(480001), 16000)).rejects.toThrow("30 seconds");
		expect(worker.requests).toHaveLength(0);
	});

	it("serializes decodes and keeps the queue usable after a per-request failure", async () => {
		const engine = await start();
		const first = engine.recognize(new Float32Array([0.5]), 16000);
		const firstCheck = expect(first).rejects.toThrow("decode failed");
		const second = engine.recognize(new Float32Array([0.25]), 16000);
		await tick();
		expect(worker.requests).toHaveLength(1);
		worker.send({ id: 1, error: "decode failed" });
		await firstCheck;
		await tick();
		expect(worker.requests).toHaveLength(2);
		worker.send({ id: 2, text: "recovered" });
		await expect(second).resolves.toBe("recovered");
	});

	it("release is idempotent and rejects active and queued decodes", async () => {
		const engine = await start();
		const first = expect(engine.recognize(new Float32Array([1]), 16000)).rejects.toThrow("released");
		const second = expect(engine.recognize(new Float32Array([1]), 16000)).rejects.toThrow("released");
		await tick();
		engine.release();
		engine.release();
		await Promise.all([first, second]);
		expect(worker.kill).toHaveBeenCalledOnce();
		await expect(engine.recognize(new Float32Array(), 16000)).rejects.toThrow("released");
	});

	it("surfaces Python startup errors and terminates the worker", async () => {
		const pending = createSttEngine({});
		const check = expect(pending).rejects.toThrow("missing dependency");
		worker.send({ kind: "fatal", error: "missing dependency" });
		await check;
		expect(worker.kill).toHaveBeenCalledOnce();
	});

	it("surfaces a missing Python executable", async () => {
		const pending = createSttEngine({});
		const check = expect(pending).rejects.toThrow("ENOENT");
		worker.emit("error", new Error("ENOENT"));
		await check;
	});

	it("reports unexpected exit with bounded stderr", async () => {
		const engine = await start();
		const check = expect(engine.recognize(new Float32Array([1]), 16000)).rejects.toThrow("native failure");
		await tick();
		worker.stderr.write("native failure");
		worker.emit("exit", 1, null);
		await check;
	});

	it("fails closed on invalid protocol messages", async () => {
		const engine = await start();
		const check = expect(engine.recognize(new Float32Array([1]), 16000)).rejects.toThrow();
		await tick();
		worker.stdout.write("not JSON\n");
		await check;
	});

	it("bounds startup and decode waits", async () => {
		vi.useFakeTimers();
		const startup = expect(createSttEngine({})).rejects.toThrow("startup timed out");
		await vi.advanceTimersByTimeAsync(300_000);
		await startup;
		worker = new Worker();
		mocks.spawn.mockReturnValue(worker);
		const engine = await start();
		const decode = expect(engine.recognize(new Float32Array([1]), 16000)).rejects.toThrow("decode timed out");
		await vi.advanceTimersByTimeAsync(60_000);
		await decode;
	});
});


it("notifies the cache exactly once when a worker is stopped", async () => {
	const onStop = vi.fn();
	const pending = createSttEngine({ onStop });
	worker.send({ kind: "ready", model: "base", device: "cuda" });
	const engine = await pending;
	engine.release();
	engine.release();
	worker.emit("exit", 1);
	expect(onStop).toHaveBeenCalledOnce();
});
