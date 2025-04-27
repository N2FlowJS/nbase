import { Database } from '../../database/database';
import { Request, Response, NextFunction } from 'express';
/**
 * Add a request ID to each incoming request
 */
export declare const addRequestId: (req: Request, res: Response, next: NextFunction) => void;
/**
 * Log request information
 */
export declare const loggingMiddleware: (req: Request, res: Response, next: NextFunction) => void;
/**
 * Middleware to check if database is ready
 * Returns 503 Service Unavailable if database is not ready
 */
export declare function ensureDatabaseReady(database: Database): (req: Request, res: Response, next: NextFunction) => void;
