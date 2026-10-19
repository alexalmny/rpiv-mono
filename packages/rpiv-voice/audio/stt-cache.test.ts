import { beforeEach, describe, expect, it, vi } from "vitest";
import { SttCache } from "./stt-cache.ts";
import { createSttEngine, type SttEngine, type SttEngineConfig } from "./stt-engine.ts";

vi.mock("./stt-engine.ts", () => ({ createSttEngine: vi.fn() }));
const create = vi.mocked(createSttEngine);
let engine: SttEngine;
beforeEach(() => {
	engine = { recognize: vi.fn(async () => "hello"), release: vi.fn() };
	create.mockReset().mockResolvedValue(engine);
});

describe("model retention", () => {
	it("loads lazily, shares in-flight startup and reuses the same model", async () => {
		const cache = new SttCache();
		expect(create).not.toHaveBeenCalled();
		const first = cache.get({ numThreads: 4 });
		expect(cache.get({ numThreads: 4 })).toBe(first);
		expect(await first).toBe(engine);
		expect(await cache.get({ numThreads: 4 })).toBe(engine);
		expect(create).toHaveBeenCalledOnce();
		expect(engine.release).not.toHaveBeenCalled();
	});

	it("releases the old model when language or CPU threads change", async () => {
		const cache = new SttCache();
		await cache.get({ language: "fr", numThreads: 4 });
		await cache.get({ language: "en", numThreads: 4 });
		await cache.get({ language: "en", numThreads: 8 });
		expect(create).toHaveBeenCalledTimes(3);
		expect(engine.release).toHaveBeenCalledTimes(2);
	});

	it("evicts failed startup so the next dictation can retry", async () => {
		create.mockRejectedValueOnce(new Error("startup failed"));
		const cache = new SttCache();
		await expect(cache.get({})).rejects.toThrow("startup failed");
		expect(await cache.get({})).toBe(engine);
		expect(create).toHaveBeenCalledTimes(2);
	});

	it("evicts a worker that exits after startup", async () => {
		const cache = new SttCache();
		await cache.get({});
		create.mock.calls[0][0].onStop?.();
		await cache.get({});
		expect(create).toHaveBeenCalledTimes(2);
	});

	it("does not let an old worker exit invalidate its replacement", async () => {
		const cache = new SttCache();
		await cache.get({ language: "fr" });
		const oldConfig: SttEngineConfig = create.mock.calls[0][0];
		await cache.get({ language: "en" });
		oldConfig.onStop?.();
		await cache.get({ language: "en" });
		expect(create).toHaveBeenCalledTimes(2);
	});

	it("release is idempotent and the next dictation loads a new worker", async () => {
		const cache = new SttCache();
		await cache.get({});
		cache.release();
		cache.release();
		await Promise.resolve();
		expect(engine.release).toHaveBeenCalledOnce();
		await cache.get({});
		expect(create).toHaveBeenCalledTimes(2);
	});

	it("releases a still-loading model as soon as startup settles", async () => {
		let ready!: (value: SttEngine) => void;
		create.mockReturnValueOnce(new Promise((resolve) => { ready = resolve; }));
		const cache = new SttCache();
		const pending = cache.get({});
		cache.release();
		ready(engine);
		await pending;
		expect(engine.release).toHaveBeenCalledOnce();
	});
});
