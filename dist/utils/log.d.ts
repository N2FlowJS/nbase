/**
 * General purpose logger
 */
type LogLevel = 'debug' | 'info' | 'warn' | 'error';
export declare const log: (level: LogLevel, message: string, ...args: any[]) => void;
export {};
