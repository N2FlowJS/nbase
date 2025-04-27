"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createTimer = void 0;
function createTimer() {
    const timers = {};
    return {
        start(name) {
            timers[name] = {
                start: process.hrtime(),
                splits: [],
                lastDuration: undefined, // Reset last duration on start
            };
        },
        split(name, label = null) {
            // ... (blijft hetzelfde)
            const timer = timers[name];
            if (!timer) {
                throw new Error(`Timer ${name} not started`);
            }
            const elapsed = this.getElapsed(name);
            timer.splits.push({ label, elapsed });
            return elapsed;
        },
        stop(name) {
            const timer = timers[name];
            if (!timer) {
                throw new Error(`Timer ${name} not started`);
            }
            const elapsed = this.getElapsed(name);
            timer.lastDuration = elapsed; // Store the duration
            if (timer.splits.length === 0) {
                timer.splits.push({ label: null, elapsed });
            }
            const result = {
                total: elapsed,
                splits: timer.splits,
            };
            // Don't delete the timer data immediately if getDuration is used
            // delete timers[name];
            // Instead, maybe reset splits? Or keep it for inspection?
            // For simplicity here, we keep it until next start.
            return result;
        },
        getElapsed(name) {
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
        getDuration(name) {
            return timers[name]?.lastDuration;
        },
        isRunning(name) {
            return timers.hasOwnProperty(name);
        },
        getActiveTimers() {
            return Object.keys(timers);
        },
    };
}
exports.createTimer = createTimer;
//# sourceMappingURL=profiling.js.map