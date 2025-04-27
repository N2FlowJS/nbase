"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ensureDatabaseReady = exports.loggingMiddleware = exports.addRequestId = void 0;
/**
 * Add a request ID to each incoming request
 */
const addRequestId = (req, res, next) => {
    const requestId = req.headers['x-request-id'] || `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    res.setHeader('X-Request-ID', requestId);
    req.headers['x-request-id'] = requestId;
    next();
};
exports.addRequestId = addRequestId;
/**
 * Log request information
 */
const loggingMiddleware = (req, res, next) => {
    const start = Date.now();
    const requestId = req.headers['x-request-id'] || 'no-ip';
    console.log(`[${requestId}] ${req.method} ${req.url} started`);
    res.on('finish', () => {
        const duration = Date.now() - start;
        console.log(`[${requestId}] ${req.method} ${req.url} ${res.statusCode} - ${duration}ms`);
    });
    next();
};
exports.loggingMiddleware = loggingMiddleware;
/**
 * Middleware to check if database is ready
 * Returns 503 Service Unavailable if database is not ready
 */
function ensureDatabaseReady(database) {
    return (req, res, next) => {
        if (!database.IsReady()) {
            res.status(503).json({ success: false, error: 'Database is not ready.' });
            return;
        }
        next();
    };
}
exports.ensureDatabaseReady = ensureDatabaseReady;
//# sourceMappingURL=common.js.map