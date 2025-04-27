import { TimerResult } from '../types';
export declare function createTimer(): {
    start(name: string): void;
    split(name: string, label?: string | null): number;
    stop(name: string): TimerResult;
    getElapsed(name: string): number;
    /**
     * Get the duration of the last completed run for this timer name.
     * Returns undefined if the timer hasn't been stopped yet.
     */
    getDuration(name: string): number | undefined;
    isRunning(name: string): boolean;
    getActiveTimers(): string[];
};
export type Timer = ReturnType<typeof createTimer>;
