// Optional retro 8-bit chiptune office feedback. Disabled (muted) by default.

class RetroSoundEngine {
  private ctx: AudioContext | null = null;
  private muted = true;

  public isMuted(): boolean {
    return this.muted;
  }

  public setMuted(muted: boolean): void {
    this.muted = muted;
  }

  public playTone(freq: number, durationMs = 70, type: OscillatorType = 'square'): void {
    if (this.muted || typeof window === 'undefined') return;
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return;
    try {
      if (!this.ctx) {
        this.ctx = new AudioCtx();
      }
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = type;
      osc.frequency.value = freq;
      gain.gain.value = 0.03;
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + durationMs / 1000);
    } catch {
      // Ignore audio permission errors cleanly
    }
  }

  public playEventCue(eventType: string): void {
    if (this.muted) return;
    if (eventType === 'agent_completed' || eventType === 'experiment_completed') {
      this.playTone(660, 75, 'triangle');
    } else if (eventType === 'agent_failed' || eventType === 'experiment_failed') {
      this.playTone(190, 120, 'sawtooth');
    } else if (eventType === 'groq_request_started' || eventType === 'agent_parallel_started') {
      this.playTone(440, 50, 'square');
    }
  }
}

export const soundEngine = new RetroSoundEngine();
