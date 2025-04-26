import * as fs from 'fs';
import * as path from 'path';

/**
 * General purpose logger
 */
type LogLevel = 'debug' | 'info' | 'warn' | 'error';
const isDev = process.env.NODE_ENV !== 'production';
// Color codes for console output
const colors = {
    debug: '#6c757d',  // gray
    info: '#0d6efd',   // blue
    warn: '#ffc107',   // yellow
    error: '#dc3545',  // red
    success: '#198754', // green
    api: '#6610f2',    // purple
    response: '#20c997', // teal
    request: '#fd7e14'  // orange
};

const LOG_DIR = path.resolve(__dirname, '../../logs');
const MAX_LINES = 1000;

let currentFileIndex = 1;
let currentLineCount = 0;
let logStream: fs.WriteStream | null = null;

function ensureLogDir() {
    if (!fs.existsSync(LOG_DIR)) {
        fs.mkdirSync(LOG_DIR, { recursive: true });
    }
}

function getLogFilePath(index: number) {
    return path.join(LOG_DIR, `log_${index}.txt`);
}

function openLogStream() {
    ensureLogDir();
    if (logStream) logStream.end();
    logStream = fs.createWriteStream(getLogFilePath(currentFileIndex), { flags: 'a' });
}

function writeLogLine(line: string) {
    if (!logStream) openLogStream();
    if (currentLineCount >= MAX_LINES) {
        currentFileIndex++;
        currentLineCount = 0;
        openLogStream();
    }
    logStream!.write(line + '\n');
    currentLineCount++;
}

export const log = (level: LogLevel, message: string, ...args: any[]) => {
    if (isDev) {
        const color = colors[level];
        switch (level) {
            case 'debug':
                console.debug(`%c${message}`, `color: ${color}`, ...args);
                break;
            case 'info':
                console.info(`%c${message}`, `color: ${color}`, ...args);
                break;
            case 'warn':
                console.warn(`%c${message}`, `color: ${color}`, ...args);
                break;
            case 'error':
                console.error(`%c${message}`, `color: ${color}`, ...args);
                break;
        }
    } else {
        const timestamp = new Date().toISOString();
        const logLine = `[${timestamp}] [${level.toUpperCase()}] ${message}` + (args.length ? ' ' + args.map(a => JSON.stringify(a)).join(' ') : '');
        writeLogLine(logLine);
    }
};

// Optional: close log stream on process exit
process.on('exit', () => {
    if (logStream) logStream.end();
});
process.on('SIGINT', () => {
    if (logStream) logStream.end();
    process.exit();
});
