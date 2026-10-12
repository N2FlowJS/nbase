import { Router, Request, Response } from 'express';
import { ApiContext } from '../../types';
import config from '../../config';
import { publicErrorMessage } from '../utils/http_error';

export function indexRoutes(context: ApiContext) {
  const router = Router();
  const { database } = context;
  // API Health Check
  router.get('/health', (_req: Request, res: Response) => {
    res.json({
      status: 'ok',
      version: config.version || '1.0.0',
      timestamp: new Date().toISOString(),
    });
  });
  // Statistics endpoint
  router.get('/stats', async (_req: Request, res: Response) => {
    try {
      const stats = await database.getStats();

      res.json(stats);
    } catch (error) {
      console.error('Error getting stats:', error);
      res.status(500).json({
        error: publicErrorMessage(error),
      });
    }
  });
  return router;
}
