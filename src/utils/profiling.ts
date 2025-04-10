import { TimerData, TimerResult } from '../types';

export function createTimer() {
  const timers: Record<string, TimerData> = {};

  return {
    start(name: string): void {
      timers[name] = {
        start: process.hrtime(),
        splits: [],
        lastDuration: undefined, // Reset last duration on start
      };
    },

    split(name: string, label: string | null = null): number {
      // ... (blijft hetzelfde)
      const timer = timers[name];
      if (!timer) {
        throw new Error(`Timer ${name} not started`);
      }
      const elapsed = this.getElapsed(name);
      timer.splits.push({ label, elapsed });
      return elapsed;
    },

    stop(name: string): TimerResult {
      const timer = timers[name];
      if (!timer) {
        throw new Error(`Timer ${name} not started`);
      }

      const elapsed = this.getElapsed(name);
      timer.lastDuration = elapsed; // Store the duration

      if (timer.splits.length === 0) {
        timer.splits.push({ label: null, elapsed });
      }

      const result: TimerResult = {
        total: elapsed,
        splits: timer.splits,
      };

      // Don't delete the timer data immediately if getDuration is used
      // delete timers[name];
      // Instead, maybe reset splits? Or keep it for inspection?
      // For simplicity here, we keep it until next start.

      return result;
    },

    getElapsed(name: string): number {
      // ... (blijft hetzelfde)
      const timer = timers[name];
      if (!timer) {
        // Return 0 or throw error? Returning 0 might be safer in some contexts.
        // console.warn(`Timer ${name} not started or already stopped when calling getElapsed.`);
        return 0;
        // throw new Error(`Timer ${name} not started`);
      }
      const [seconds, nanoseconds] = process.hrtime(timer.start);
      return seconds * 1000 + nanoseconds / 1000000; // Convert to milliseconds
    },

    /**
     * Get the duration of the last completed run for this timer name.
     * Returns undefined if the timer hasn't been stopped yet.
     */
    getDuration(name: string): number | undefined {
      return timers[name]?.lastDuration;
    },

    isRunning(name: string): boolean {
      return timers.hasOwnProperty(name);
    },

    getActiveTimers(): string[] {
      return Object.keys(timers);
    },
  };
}

export type Timer = ReturnType<typeof createTimer>;
