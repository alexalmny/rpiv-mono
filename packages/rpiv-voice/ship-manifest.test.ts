import { existsSync, readFileSync } from "node:fs";
import { verifyShipManifest } from "@juicesharp/rpiv-test-utils";
import { describe, expect, it } from "vitest";

describe("publish manifest", () => {
	it("`package.json` `files` array covers every production .ts module across the tree", () => {
		expect(verifyShipManifest(import.meta.url).missing).toEqual([]);
	});
	it("ships the Python worker and dependency declaration", () => {
		const pkg = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8"));
		for (const path of ["audio/faster-whisper-worker.py", "requirements.txt"]) {
			expect(pkg.files).toContain(path);
			expect(existsSync(new URL(path, import.meta.url))).toBe(true);
		}
	});
});
