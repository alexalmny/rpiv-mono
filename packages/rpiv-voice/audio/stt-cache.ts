import { createSttEngine, type SttEngine, type SttEngineConfig } from "./stt-engine.ts";

/** Session-scoped, single-flight model cache; never owns a microphone. */
export class SttCache {
	private pending?: Promise<SttEngine>;
	private key?: string;

	get(config: SttEngineConfig): Promise<SttEngine> {
		const key = JSON.stringify([config.language, config.numThreads]);
		if (this.key !== key) this.release();
		if (!this.pending) {
			this.key = key;
			const pending = createSttEngine({
				...config,
				onStop: () => {
					if (this.pending === pending) this.pending = undefined;
				},
			});
			this.pending = pending;
			void pending.catch(() => {
				if (this.pending === pending) this.pending = undefined;
			});
		}
		return this.pending;
	}

	release(): void {
		const pending = this.pending;
		this.pending = undefined;
		this.key = undefined;
		void pending?.then((engine) => engine.release(), () => {});
	}
}
