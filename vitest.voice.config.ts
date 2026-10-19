import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
	resolve: {
		alias: {
			"@juicesharp/rpiv-test-utils": fileURLToPath(new URL("./packages/test-utils/index.ts", import.meta.url)),
		},
	},
	test: {
		include: ["packages/rpiv-voice/**/*.test.ts", "packages/test-utils/manifest.test.ts"],
		setupFiles: ["./test/voice-setup.ts"],
		maxWorkers: 4,
		clearMocks: true,
		restoreMocks: true,
		unstubGlobals: true,
		testTimeout: 15_000,
	},
});
