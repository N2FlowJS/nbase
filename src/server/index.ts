import cors from 'cors';
import express, { NextFunction, Request, Response } from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import net from 'net';
import { promisify } from 'util';
import { default as config, default as configDefaults } from '../config';
import { createTimer } from '../utils/profiling';

// Import Database class
import { Database } from '../database/database';

// Import middleware
import { loggingMiddleware } from './middleware/common';

// Import routes
import { indexRoutes } from './routes/index';
import { searchRoutes } from './routes/search';
import { vectorRoutes } from './routes/vectors';

// Import helpers
import { ApiContext, IServerInstance, IServerOptions } from '../types';
import { createFilterFunction } from './utils/filters';

// Export request interfaces for type checking
export type { AddVectorRequest, BatchSearchRequest, BulkAddRequest, DatabaseOptions, FilterConfig, IServerOptions, SaveLoadDatabaseRequest, SearchRequest, TrainIndexRequest, UpdateMetadataRequest } from '../types';

/**
 * Checks if a port is in use
 * @param port Port to check
 * @returns Promise resolving to true if port is in use, false otherwise
 */
async function isPortInUse(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net
      .createServer()
      .once('error', (err: any) => {
        if (err.code === 'EADDRINUSE') {
          resolve(true);
        } else {
          resolve(false);
        }
      })
      .once('listening', () => {
        server.close();
        resolve(false);
      })
      .listen(port);
  });
}


/**
 * Creates and configures an Express server instance with the provided options.
 *
 * @param options - Configuration options for the server. If not provided, defaults will be used.
 * @returns An object representing the server instance, including the Express app,
 *          a graceful shutdown handler, the database instance, and the API context.
 *
 * ### Options
 * - `port` (number): The port on which the server will listen. Defaults to `configDefaults.server.port`.
 * - `host` (string): The host address for the server. Defaults to `configDefaults.server.host`.
 * - `database` (object): Database configuration options, including clustering, indexing, partitioning,
 *   persistence, and monitoring. Defaults to `configDefaults`.
 * - `rateLimit` (object): Rate limiting configuration. Defaults to `configDefaults.server.rateLimit`.
 * - `middleware` (array): Custom middleware functions to apply to the server. Defaults to an empty array.
 * - `debug` (boolean): Enables debug logging if set to `true`. Defaults to `false`.
 * - `errorHandler` (function): Custom error handling middleware. Defaults to a generic error handler.
 *
 * ### Features
 * - Configures security headers using `helmet`.
 * - Enables CORS with `cors`.
 * - Supports large JSON payloads with a size limit of 50MB.
 * - Adds logging middleware for request logging.
 * - Supports custom middleware injection.
 * - Implements rate limiting for API routes.
 * - Provides debug logging for incoming requests when `debug` is enabled.
 * - Initializes a database instance with the provided configuration.
 * - Registers API routes for vectors, search, and index.
 * - Includes error handling and a "Not Found" handler for unregistered routes.
 * - Provides a graceful shutdown mechanism to close database connections properly.
 *
 * ### Returns
 * The returned `IServerInstance` object includes:
 * - `app`: The configured Express application.
 * - `gracefulShutdown`: A function to handle graceful shutdown of the server.
 * - `database`: The initialized database instance.
 * - `context`: The API context containing shared resources.
 */
function createServer(options: IServerOptions = {}): IServerInstance {
  const app = express();

  // Merge default config with provided options
  const serverOptions: Required<IServerOptions> = {
    port: options.port || configDefaults.server.port,
    host: options.host || configDefaults.server.host,
    database: {
      ...{
        clustering: configDefaults.clustering,
        indexing: configDefaults.indexing,
        partitioning: configDefaults.partitioning,
        persistence: configDefaults.persistence,
        monitoring: configDefaults.monitoring,
      },
      ...options.database,
    },
    rateLimit: { ...configDefaults.server.rateLimit, ...options.rateLimit },
    middleware: options.middleware || [],
    debug: options.debug || false,
    errorHandler:
      options.errorHandler ||
      ((err: Error, req: Request, res: Response, next: NextFunction) => {
        console.error('API Error:', err);
        res.status(500).json({
          error: err.message,
          stack: process.env.NODE_ENV === 'development' ? err.stack : undefined,
        });
      }),
  };

  // Request timer for performance monitoring
  const timer = createTimer();

  // Configure middleware
  app.use(helmet()); // Security headers
  app.use(cors()); // Cross-origin resource sharing
  app.use(express.json({ limit: '50mb' })); // JSON body parser with large payload support

  // Add logging middleware
  app.use(loggingMiddleware);

  // Apply custom middleware
  serverOptions.middleware.forEach((middleware) => app.use(middleware));

  // Rate limiting
  if (serverOptions.rateLimit.enable) {
    const apiLimiter = rateLimit({
      windowMs: serverOptions.rateLimit.windowMs,
      max: serverOptions.rateLimit.maxRequestsPerMinute,
      standardHeaders: true,
      legacyHeaders: false,
      message: 'Too many requests, please try again later',
    });
    app.use('/api/', apiLimiter);
  }

  // Debug logging
  if (serverOptions.debug) {
    // Add debug logging for request URLs
    app.use((req: Request, res: Response, next: NextFunction) => {
      console.log(`DEBUG: Incoming request: ${req.method} ${req.originalUrl}`);
      next();
    });
  }

  // Initialize database instance with options
  const database = new Database(serverOptions.database);

  // Create context object with shared resources
  const apiContext: ApiContext = {
    timer,
    createFilterFunction,
    database,
  };

  /**
   * API Routes
   */
  // Register route handlers
  app.use('/api/vectors', vectorRoutes(apiContext));
  app.use('/api/search', searchRoutes(apiContext));
  app.use('/', indexRoutes(apiContext));

  // Error handling middleware
  app.use(serverOptions.errorHandler);

  // Not Found handler
  app.use((req: Request, res: Response) => {
    res.status(404).json({
      error: `Route not found: ${req.method} ${req.url}`,
    });
  });

  // Graceful shutdown handler for proper database closing
  const gracefulShutdown = async () => {
    console.log('Closing database connections...');
    await database.close().catch(console.error);
    console.log('Database connections closed');
  };

  // For backward compatibility with tests - attach gracefulShutdown to app
  (app as any).gracefulShutdown = gracefulShutdown;

  // Return both for new code that expects the new format
  // but also make app the default export for backward compatibility
  const result: IServerInstance = {
    app,
    gracefulShutdown,
    database,
    context: apiContext,
  };
  Object.defineProperty(result, 'default', {
    value: app,
  });

  return result;
}

/**
 * Start server
 */
if (require.main === module) {
  const startServer = async () => {
    const { app, gracefulShutdown } = createServer({
      database: {
        persistence: config.persistence,
        partitioning: config.partitioning,
        indexing: config.indexing,
        clustering: {
          ...config.clustering,
          useCompression: false,
        },
      },
    }) as IServerInstance;
    const PORT = process.env.PORT || config.server.port || 1307;
    const HOST = process.env.HOST || config.server.host || 'localhost';

    // Check if port is in use and kill process if necessary
    const portInUse = await isPortInUse(Number(PORT));
    if (portInUse) {
      console.log('Port is used');

      return

    }

    const server = app.listen(PORT, () => {
      console.log(`API Server running on http://${HOST}:${PORT}`);
    });

    // Graceful shutdown
    process.on('SIGTERM', async () => {
      console.log('SIGTERM received, shutting down gracefully');
      await gracefulShutdown();
      server.close(() => {
        console.log('HTTP server closed');
        process.exit(0);
      });
    });
  };

  startServer().catch((err) => {
    console.error('Failed to start server:', err);
    process.exit(1);
  });
}

// Export server factory function
export default createServer;
