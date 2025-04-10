import { Database } from '../../database/database';
import { Request, Response, NextFunction } from 'express';

/**
 * Add a request ID to each incoming request
 */
export const addRequestId = (req: Request, res: Response, next: NextFunction) => {
  const requestId = req.headers['x-request-id'] || `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  res.setHeader('X-Request-ID', requestId);
  req.headers['x-request-id'] = requestId as string;
  next();
};

/**
 * Log request information
 */
export const loggingMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const start = Date.now();
  const requestId = req.headers['x-request-id'] || 'no-ip';
  console.log(`[${requestId}] ${req.method} ${req.url} started`);

  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`[${requestId}] ${req.method} ${req.url} ${res.statusCode} - ${duration}ms`);
  });

  next();
};

/**
 * Middleware to check if database is ready
 * Returns 503 Service Unavailable if database is not ready
 */
export function ensureDatabaseReady(database: Database) {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!database.IsReady()) {
      res.status(503).json({ success: false, error: 'Database is not ready.' });
      return;
    }
    next();
  };
}
