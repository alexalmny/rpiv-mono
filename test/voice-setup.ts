import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeEach } from "vitest";

// Voice-only suite: never import or initialize the other monorepo extensions.
const home = mkdtempSync(join(tmpdir(), "rpiv-voice-test-"));
process.env.HOME = home;
process.env.USERPROFILE = home;
delete process.env.PI_CODING_AGENT_DIR;
delete process.env.XDG_CONFIG_HOME;

beforeEach(() => {
	delete (globalThis as Record<symbol, unknown>)[Symbol.for("rpiv-voice")];
	delete (globalThis as Record<symbol, unknown>)[Symbol.for("rpiv-i18n")];
	rmSync(join(home, ".config"), { recursive: true, force: true });
});
afterAll(() => rmSync(home, { recursive: true, force: true }));
