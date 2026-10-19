import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { DEFAULT_NUM_THREADS } from "../config/voice-config.ts";
import { appendDiagnosticLog } from "./error-log.ts";

export interface SttEngineConfig {
	language?: string;
	numThreads?: number;
	onProgress?: (message: string) => void;
}

export interface SttEngine {
	recognize(samples: Float32Array, sampleRate: number): Promise<string>;
	release(): void;
}

const activeEngines = new Set<SttEngine>();

export function releaseAllSttEngines(): void {
	for (const engine of activeEngines) engine.release();
}

/** One model-loaded Python worker per dictation session, never per phrase. */
export async function createSttEngine(config: SttEngineConfig): Promise<SttEngine> {
	const worker = fileURLToPath(new URL("./faster-whisper-worker.py", import.meta.url));
	const args = ["-u", worker, "--threads", String(config.numThreads ?? DEFAULT_NUM_THREADS)];
	if (config.language) args.push("--language", config.language);
	const child = spawn(process.env.RPIV_VOICE_PYTHON || "python", args, {
		stdio: ["pipe", "pipe", "pipe"],
		windowsHide: true,
		env: { ...process.env, PYTHONIOENCODING: "utf-8" },
	});
	const lines = createInterface({ input: child.stdout });
	const pending = new Map<number, { resolve: (text: string) => void; reject: (error: Error) => void }>();
	let nextId = 0;
	let stopped: Error | undefined;
	let stderr = "";
	let readyResolve!: () => void;
	let readyReject!: (error: Error) => void;
	const ready = new Promise<void>((resolve, reject) => {
		readyResolve = resolve;
		readyReject = reject;
	});
	let queue: Promise<unknown> = Promise.resolve();

	const stop = (error: Error): void => {
		if (stopped) return;
		stopped = error;
		clearTimeout(startupTimer);
		readyReject(error);
		for (const request of pending.values()) request.reject(error);
		pending.clear();
		lines.close();
		child.kill();
		activeEngines.delete(engine);
	};
	const startupTimer = setTimeout(() => stop(new Error("faster-whisper startup timed out")), 300_000);
	child.stderr.on("data", (chunk: Buffer) => {
		stderr = (stderr + chunk.toString("utf8")).slice(-8192);
	});
	child.on("error", (error) => stop(error));
	child.stdin.on("error", (error) => stop(error));
	child.on("exit", (code, signal) => {
		stop(new Error(`faster-whisper worker exited (${code ?? signal}): ${stderr}`));
	});
	lines.on("line", (line) => {
		try {
			const message = JSON.parse(line);
			if (message.kind === "ready") {
				clearTimeout(startupTimer);
				appendDiagnosticLog("stt.backend", `faster-whisper ${message.model} ${message.device}`);
				readyResolve();
			} else if (message.kind === "progress") {
				config.onProgress?.(String(message.message));
			} else if (message.kind === "diagnostic") {
				appendDiagnosticLog("stt.backend", String(message.message));
			} else if (message.kind === "fatal") {
				stop(new Error(String(message.error)));
			} else {
				const request = pending.get(message.id);
				if (!request) throw new Error("Unknown faster-whisper response id");
				pending.delete(message.id);
				if (typeof message.error === "string") request.reject(new Error(message.error));
				else if (typeof message.text === "string") request.resolve(message.text.trim());
				else {
					const error = new Error("Invalid faster-whisper response");
					request.reject(error);
					throw error;
				}
			}
		} catch (error) {
			stop(error instanceof Error ? error : new Error(String(error)));
		}
	});

	const engine: SttEngine = {
		async recognize(samples, sampleRate) {
			if (stopped) throw stopped;
			if (sampleRate !== 16000 || samples.length > 16000 * 30) {
				throw new Error("Expected at most 30 seconds of 16000 Hz mono PCM");
			}
			if (samples.length === 0) return "";
			// Snapshot before queueing: the caller may reuse its Float32Array.
			const audio = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength).toString("base64");
			const run = async (): Promise<string> => {
				if (stopped) throw stopped;
				const id = ++nextId;
				let timer: ReturnType<typeof setTimeout> | undefined;
				try {
					return await new Promise<string>((resolve, reject) => {
						pending.set(id, { resolve, reject });
						timer = setTimeout(() => stop(new Error("faster-whisper decode timed out")), 60_000);
						child.stdin.write(`${JSON.stringify({ id, audio, sampleRate })}\n`, (error) => {
							if (error) stop(error);
						});
					});
				} finally {
					clearTimeout(timer);
				}
			};
			const result = queue.then(run);
			queue = result.catch(() => {});
			return result;
		},
		release() {
			stop(new Error("faster-whisper engine released"));
		},
	};
	activeEngines.add(engine);
	await ready;
	return engine;
}
