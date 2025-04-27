"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.log = void 0;
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const NBASE_DB_PATH = `${process.env.NBASE_DB_PATH}`;
const isDev = 1 == 1 || NBASE_DB_PATH.length > 0;
// Color codes for console output
const colors = {
    debug: '#6c757d', // gray
    info: '#0d6efd', // blue
    warn: '#ffc107', // yellow
    error: '#dc3545', // red
    success: '#198754', // green
    api: '#6610f2', // purple
    response: '#20c997', // teal
    request: '#fd7e14' // orange
};
const LOG_DIR = path.resolve(__dirname, '../../logs');
const MAX_LINES = 1000;
let currentFileIndex = 1;
let currentLineCount = 0;
let logStream = null;
function ensureLogDir() {
    if (!fs.existsSync(LOG_DIR)) {
        fs.mkdirSync(LOG_DIR, { recursive: true });
    }
}
function getLogFilePath(index) {
    return path.join(LOG_DIR, `log_${index}.txt`);
}
function openLogStream() {
    ensureLogDir();
    if (logStream)
        logStream.end();
    logStream = fs.createWriteStream(getLogFilePath(currentFileIndex), { flags: 'a' });
}
function writeLogLine(line) {
    if (!logStream)
        openLogStream();
    if (currentLineCount >= MAX_LINES) {
        currentFileIndex++;
        currentLineCount = 0;
        openLogStream();
    }
    logStream.write(line + '\n');
    currentLineCount++;
}
const log = (level, message, ...args) => {
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
    }
    else {
        const timestamp = new Date().toISOString();
        const logLine = `[${timestamp}] [${level.toUpperCase()}] ${message}` + (args.length ? ' ' + args.map(a => JSON.stringify(a)).join(' ') : '');
        writeLogLine(logLine);
    }
};
exports.log = log;
// Optional: close log stream on process exit
process.on('exit', () => {
    if (logStream)
        logStream.end();
});
process.on('SIGINT', () => {
    if (logStream)
        logStream.end();
    process.exit();
});
//# sourceMappingURL=log.js.map