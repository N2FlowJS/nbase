import * as fs from 'node:fs';
import * as path from 'node:path';

/**
 * General purpose logger
 */
type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_DIR = process.env['NBASE_LOG_DIR'] ? path.resolve(process.env['NBASE_LOG_DIR']) : path.resolve(__dirname, '../../logs');
const MAX_LINES = 1000;

/**
 * Whether to write to a rotating log file as well as the console.
 *
 * Controlled by `NBASE_LOG_TO_FILE`; the previous `1 == 1 ||` short-circuit
 * made the file-writing branch unreachable, so the rotation machinery below
 * (and the stream lifecycle) was dead code.
 */
const logToFile = process.env['NBASE_LOG_TO_FILE'] === 'true';

// Color codes for console output
const colors = {
  debug: '#6c757d', // gray
  info: '#0d6efd', // blue
  warn: '#ffc107', // yellow
  error: '#dc3545', // red
};

let currentFileIndex = 1;
let currentLineCount = 0;
let logStream: fs.WriteStream | null = null;

function ensureLogDir(): void {
  if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
  }
}

function getLogFilePath(index: number): string {
  return path.join(LOG_DIR, `log_${index}.txt`);
}

function openLogStream(): void {
  ensureLogDir();
  if (logStream) logStream.end();
  logStream = fs.createWriteStream(getLogFilePath(currentFileIndex), { flags: 'a' });
}

function writeLogLine(line: string): void {
  if (!logStream) openLogStream();
  if (currentLineCount >= MAX_LINES) {
    currentFileIndex++;
    currentLineCount = 0;
    openLogStream();
  }
  logStream!.write(line + '\n');
  currentLineCount++;
}

export const log = (level: LogLevel, message: string, ...args: unknown[]): void => {
  // Always emit to the console; a library should not go silent in a host app.
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

  if (logToFile) {
    const timestamp = new Date().toISOString();
    const logLine = `[${timestamp}] [${level.toUpperCase()}] ${message}` + (args.length ? ' ' + args.map((a) => JSON.stringify(a)).join(' ') : '');
    writeLogLine(logLine);
  }
};

/**
 * Closes the rotating log stream.
 *
 * Exposed rather than wired to `process.on('SIGINT', ...)`: this module is
 * imported by anything that touches the library, and registering a SIGINT
 * handler at import time made Ctrl-C exit the *host* application whenever
 * nbase happened to be a transitive dependency.
 */
export function closeLogStream(): void {
  if (logStream) {
    logStream.end();
    logStream = null;
  }
}
