import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { appendErrorLog } from "../audio/error-log.ts";
import { createMic, type DecibriLike } from "../audio/mic-source.ts";
import { createSttEngine, type SttEngine } from "../audio/stt-engine.ts";
import {
	isHallucinationFilterEnabled,
	loadVoiceConfig,
	resolveNumThreads,
	type VoiceConfig,
} from "../config/voice-config.ts";
import { getActiveLocale, t } from "../state/i18n-bridge.ts";
import type { VoiceResult } from "../state/state-reducer.ts";
import { VoiceSession } from "../state/voice-session.ts";
import { STATUS_BAR_PULSE_FRAME_INTERVAL_MS } from "../view/components/status-bar-view.ts";
import { type PipelineHandle, startDictationPipeline } from "./pipeline-runner.ts";
import { runWithSplash } from "./splash-runner.ts";

export const VOICE_COMMAND_NAME = "voice";

// Locales the bundled Whisper base multilingual model recognizes well. Mapped
// from i18n locale codes; entries not in this set fall back to Whisper's
// auto-detect (the multilingual model handles ~99 languages, but this keeps
// us aligned to locales we actively translate to and have tested).
const WHISPER_SUPPORTED_LANGUAGES: ReadonlySet<string> = new Set([
	"de",
	"en",
	"es",
	"fr",
	"it",
	"ja",
	"pt",
	"ru",
	"uk",
	"zh",
]);

// IETF locale tags have an optional region subtag after the first hyphen
// (e.g. "pt-BR" → "pt"). Whisper's language hint expects just the base.
const LOCALE_REGION_SEPARATOR = "-";
const LOCALE_BASE_INDEX = 0;

function baseLanguage(locale: string): string {
	return locale.split(LOCALE_REGION_SEPARATOR)[LOCALE_BASE_INDEX] ?? locale;
}

function isWhisperSupported(language: string): boolean {
	return WHISPER_SUPPORTED_LANGUAGES.has(language);
}

function whisperLanguageForLocale(locale: string | undefined): string | undefined {
	if (!locale) return undefined;
	const base = baseLanguage(locale);
	return isWhisperSupported(base) ? base : undefined;
}

// Upper bound on the commit drain: mic shutdown plus one final Whisper decode
// of a sub-12 s segment (typically well under a second). A hung native decode
// or a mic that never emits a terminal event must not park the handler forever
// with the overlay already closed — on timeout the reducer's commit merge,
// already the floor, is pasted as-is and a breadcrumb lands in errors.log.
const COMMIT_DRAIN_TIMEOUT_MS = 10_000;

interface Preflight {
	sttEngine: SttEngine;
	mic: DecibriLike;
}

export function registerVoiceCommand(pi: ExtensionAPI): void {
	pi.registerCommand(VOICE_COMMAND_NAME, {
		description: t("command.description", "Dictate text with your voice — local STT, no cloud"),
		handler: (_args: string, ctx: ExtensionCommandContext) => handleVoiceCommand(ctx),
	});
}

async function handleVoiceCommand(ctx: ExtensionCommandContext): Promise<void> {
	if (!ctx.hasUI) {
		ctx.ui.notify(t("error.requires_interactive", "/voice requires interactive mode"), "error");
		return;
	}

	// One config snapshot per invocation: engine construction (numThreads) and
	// the dictation session (VoiceSession's persistedConfig, the hallucination
	// filter default) must read the same load, so it happens once, here.
	// Settings saves are the exception — they re-read the file at save time so
	// mid-session hand edits of JSON-only keys round-trip.
	const persistedConfig = loadVoiceConfig();

	const preflight = await runPreflight(ctx, persistedConfig);
	if (!preflight) return;

	try {
		const result = await runDictationSession(ctx, preflight.sttEngine, preflight.mic, persistedConfig);
		if (result.intent === "commit" && result.transcript) {
			ctx.ui.pasteToEditor(result.transcript);
		}
	} finally {
		try {
			preflight.mic.stop();
		} finally {
			preflight.sttEngine.release();
		}
	}
}

async function runPreflight(ctx: ExtensionCommandContext, persistedConfig: VoiceConfig): Promise<Preflight | null> {
	let sttEngine: SttEngine | undefined;
	const status: { stage: "engine" | "mic" } = { stage: "engine" };
	try {
		return await runWithSplash<Preflight>(ctx, { initialPhase: { kind: "loading_engine" } }, async (controller) => {
			sttEngine = await createSttEngine({
				language: whisperLanguageForLocale(getActiveLocale()),
				numThreads: resolveNumThreads(persistedConfig),
				onProgress: (message) => controller.setPhase({ kind: "downloading", message }),
			});
			status.stage = "mic";
			controller.setPhase({ kind: "initializing_mic" });
			const mic = await createMic();
			return { sttEngine, mic };
		});
	} catch (error) {
		sttEngine?.release();
		appendErrorLog(`stt.preflight.${status.stage}`, error);
		ctx.ui.notify(
			status.stage === "mic"
				? t("error.mic_unavailable", "Microphone unavailable. Check the input device and microphone permissions.")
				: t("error.engine_load_failed", "Failed to load STT model."),
			"error",
		);
		return null;
	}
}

async function runDictationSession(
	ctx: ExtensionCommandContext,
	sttEngine: SttEngine,
	mic: DecibriLike,
	persistedConfig: VoiceConfig,
): Promise<VoiceResult> {
	const controller = new AbortController();

	let pipelineHandle: PipelineHandle | undefined;
	let pulseTick: ReturnType<typeof setInterval> | undefined;

	try {
		const result = await ctx.ui.custom<VoiceResult>((tui, theme, _kb, done) => {
			const session = new VoiceSession({
				tui,
				theme,
				persistedConfig,
				deps: {
					pasteToEditor: (text) => ctx.ui.pasteToEditor(text),
					notify: (message, level) => ctx.ui.notify(message, level),
					abort: () => controller.abort(),
					stopMic: () => pipelineHandle?.stop(),
					setPipelinePaused: (paused) => pipelineHandle?.setPaused(paused),
					setHallucinationFilterEnabled: (enabled) => pipelineHandle?.setHallucinationFilterEnabled(enabled),
				},
				done,
			});
			pipelineHandle = startDictationPipeline(mic, sttEngine, session, controller.signal, {
				hallucinationFilterEnabled: isHallucinationFilterEnabled(persistedConfig),
			});
			pulseTick = setInterval(() => session.tickPulse(), STATUS_BAR_PULSE_FRAME_INTERVAL_MS);
			return session.component;
		});

		if (pulseTick) clearInterval(pulseTick);
		let finalResult: VoiceResult = result;
		if (result.intent === "commit") {
			// Commit-drain merge floor. The reducer's commit merge — committed
			// finals plus the visible partial (`state.partialTranscript`, an
			// accumulator in neither pipeline store) — is the floor.
			// finalTranscriptPromise resolves only after the whole finals chain
			// settles, so the drained text equals the committed finals plus the
			// drain-time final only when that final was accepted; on
			// hallucination/RMS disagreement or a drain-time decode error the drain
			// collapses to the prior finals — text the merge already extends.
			// Replace the merge only when the drain extends beyond it as text; an
			// empty drain fails `if (drained)` outright.
			//
			// Ordering: stop the mic and drain BEFORE aborting. The drain-time
			// final is the one decode whose outcome the user consumes; the pipeline
			// treats an aborted signal as cancel teardown (it suppresses error
			// breadcrumbs and skips stranded decode work), so the commit drain must
			// run while the signal is live. Cancel inverts this: the reducer aborts
			// first and the pipeline short-circuits the drain fire-and-forget.
			pipelineHandle?.stop();
			const drained = await boundedDrain(pipelineHandle?.finalTranscriptPromise);
			if (drained && !result.transcript.startsWith(drained)) {
				finalResult = { ...result, transcript: drained };
			}
		}
		if (!controller.signal.aborted) controller.abort();
		return finalResult;
	} finally {
		if (pulseTick) clearInterval(pulseTick);
		if (!controller.signal.aborted) controller.abort();
	}
}

// Bounds the commit-drain await so a mic that never emits a terminal event or
// a hung native decode cannot park the handler after the overlay closed.
// Resolves `undefined` on timeout — the caller then keeps the reducer merge.
async function boundedDrain(drain: Promise<string> | undefined): Promise<string | undefined> {
	if (!drain) return undefined;
	let timer: ReturnType<typeof setTimeout> | undefined;
	const timeout = new Promise<undefined>((resolve) => {
		timer = setTimeout(() => resolve(undefined), COMMIT_DRAIN_TIMEOUT_MS);
	});
	try {
		const drained = await Promise.race([drain, timeout]);
		if (drained === undefined) {
			appendErrorLog("stt.drain", new Error(`commit drain timed out after ${COMMIT_DRAIN_TIMEOUT_MS}ms`));
		}
		return drained;
	} finally {
		clearTimeout(timer);
	}
}
