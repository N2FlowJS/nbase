import cors from 'cors';
import express, { NextFunction, Request, Response } from 'express';
import type { Express } from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import net from 'net';
import { createHash, timingSafeEqual } from 'node:crypto';
import { default as config, default as configDefaults } from '../config';
import { createTimer } from '../utils/profiling';
import { log } from '../utils/log';

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
import { publicErrorMessage } from './utils/http_error';

// Export request interfaces for type checking
export type { AddVectorRequest, BatchSearchRequest, BulkAddRequest, DatabaseOptions, FilterConfig, IServerOptions, SaveLoadDatabaseRequest, SearchRequest, TrainIndexRequest, UpdateMetadataRequest } from '../types';

/**
 * Checks if a port is in use
 * @param port Port to check
 * @returns Promise resolving to true if port is in use, false otherwise
 */
async function isPortInUse(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    // The probe socket must be closed on every path, otherwise it stays bound
    // and leaks a file descriptor for the life of the process.
    const settle = (inUse: boolean): void => {
      server.close(() => resolve(inUse));
    };
    server
      .once('error', (err: NodeJS.ErrnoException) => {
        if (err.code === 'EADDRINUSE') {
          settle(true);
        } else {
          settle(false);
        }
      })
      .once('listening', () => {
        settle(false);
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
    cors: options.cors ?? false,
    auth: { ...(options.auth ?? {}) },
    trustProxy: options.trustProxy ?? false,
    // 10k vectors is ~50M pair comparisons; beyond that the O(n^2) graph
    // endpoints are refused rather than allowed to stall the event loop.
    maxGraphExtractionVectors: options.maxGraphExtractionVectors ?? 10000,
    middleware: options.middleware || [],
    debug: options.debug || false,
    errorHandler:
      options.errorHandler ||
      ((err: Error & { status?: number; statusCode?: number }, _req: Request, res: Response, _next: NextFunction) => {
        log('error', 'API Error:', err);
        const isDevelopment = process.env['NODE_ENV'] === 'development';
        // body-parser sets `status`/`statusCode` for a malformed body (400) or
        // an oversized one (413). Those were reported as 500, which told the
        // caller the server broke rather than that its request did.
        const status = err.status ?? err.statusCode ?? 500;
        res.status(status).json({
          error: publicErrorMessage(err, status),
          stack: isDevelopment ? err.stack : undefined,
        });
      }),
  };

  // Configure middleware
  app.use(helmet()); // Security headers

  // CORS is opt-in. It used to be mounted as a bare `cors()`, which answers
  // every origin with `Access-Control-Allow-Origin: *` — that is a browser
  // policy, not an access control, and it defeated the api-key middleware that
  // hosts add through `middleware` (a key in a browser app is readable by any
  // site that can talk to the API). With no `cors` option nothing is added and
  // the browser's same-origin default applies.
  if (serverOptions.cors === true) {
    app.use(cors());
  } else if (Array.isArray(serverOptions.cors) && serverOptions.cors.length > 0) {
    app.use(cors({ origin: serverOptions.cors }));
  }

  // `trust proxy` feeds `X-Forwarded-For` into `req.ip`, which is the key the
  // rate limiter buckets on. Unset by default so a client cannot forge it.
  app.set('trust proxy', serverOptions.trustProxy);

  app.use(express.json({ limit: '50mb' })); // JSON body parser with large payload support

  // Add logging middleware
  app.use(loggingMiddleware);

  // Apply custom middleware
  serverOptions.middleware.forEach((middleware) => app.use(middleware));

  // Rate limiting
  if (serverOptions.rateLimit.enable) {
    // Omit absent keys: express-rate-limit's Options type does not accept an
    // explicit undefined under exactOptionalPropertyTypes.
    const apiLimiter = rateLimit({
      ...(serverOptions.rateLimit.windowMs !== undefined ? { windowMs: serverOptions.rateLimit.windowMs } : {}),
      ...(serverOptions.rateLimit.maxRequestsPerMinute !== undefined ? { max: serverOptions.rateLimit.maxRequestsPerMinute } : {}),
      standardHeaders: true,
      legacyHeaders: false,
      message: 'Too many requests, please try again later',
    });
    app.use('/api/', apiLimiter);
    // `/stats` is mounted at `/` by `indexRoutes`, so it was outside the
    // limiter entirely — and it is an O(all vectors) call that stringifies
    // every metadata object, which makes it the cheapest thing to loop on.
    app.use('/', apiLimiter);
  }

  // API key authentication (opt-in).
  const apiKeys = serverOptions.auth.apiKeys ?? [];
  if (serverOptions.auth.requireAuth === true && apiKeys.length === 0) {
    throw new Error('auth.requireAuth is set but auth.apiKeys is empty: the API would reject every request.');
  }
  if (apiKeys.length > 0) {
    app.use('/api', createApiKeyAuth(apiKeys));
  } else {
    log('warn', '[Server] No API keys configured: /api/* accepts unauthenticated requests. ' + 'Set `auth: { apiKeys: [...] }` (or `auth: { requireAuth: true }` to refuse to start) before exposing this process.');
  }

  // Debug logging
  if (serverOptions.debug) {
    // Add debug logging for request URLs
    app.use((req: Request, _res: Response, next: NextFunction) => {
      log('info', `DEBUG: Incoming request: ${req.method} ${req.originalUrl}`);
      next();
    });
  }

  // Initialize database instance with options
  const database = new Database(serverOptions.database);

  // Create context object with shared resources.
  // `createTimer` hands each request its own Timer; a single shared instance
  // meant concurrent requests overwrote each other's start timestamps.
  const apiContext: ApiContext = {
    createTimer,
    createFilterFunction,
    database,
    maxGraphExtractionVectors: serverOptions.maxGraphExtractionVectors,
  };

  /**
   * API Routes
   */
  // Register route handlers
  app.use('/api/vectors', vectorRoutes(apiContext));
  app.use('/api/search', searchRoutes(apiContext));
  app.use('/', indexRoutes(apiContext));

  // Not Found handler first, then the error handler. The order used to be the
  // other way round, so anything that threw inside the 404 handler (or inside
  // middleware the host appends after `createServer` returned) fell through to
  // Express's default handler, which answers with the stack trace.
  app.use((_req: Request, res: Response) => {
    res.status(404).json({
      error: 'Route not found',
    });
  });

  // Error handling middleware
  app.use(serverOptions.errorHandler);

  // Graceful shutdown handler for proper database closing
  const gracefulShutdown = async () => {
    console.log('Closing database connections...');
    await database.close().catch(console.error);
    console.log('Database connections closed');
  };

  // For backward compatibility with tests - attach gracefulShutdown to app
  (app as Express & { gracefulShutdown?: () => Promise<void> }).gracefulShutdown = gracefulShutdown;

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
 * Rejects a request whose `Authorization: Bearer <key>` (or `x-api-key`) does
 * not match one of `apiKeys`.
 *
 * Comparison uses `timingSafeEqual` over a fixed-length hash of both sides:
 * a plain `===` leaks the key length and prefix through response timing.
 */
function createApiKeyAuth(apiKeys: string[]): express.RequestHandler {
  const keyHashes = apiKeys.map((key) => createHash('sha256').update(key).digest());

  return (req: Request, res: Response, next: NextFunction) => {
    const header = req.headers.authorization;
    const bearer = typeof header === 'string' && header.startsWith('Bearer ') ? header.slice('Bearer '.length) : undefined;
    const provided = bearer ?? (typeof req.headers['x-api-key'] === 'string' ? req.headers['x-api-key'] : undefined);

    if (provided === undefined) {
      res.status(401).json({ error: 'Missing API key' });
      return;
    }

    const providedHash = createHash('sha256').update(provided).digest();
    const matches = keyHashes.some((hash) => timingSafeEqual(hash, providedHash));
    if (!matches) {
      res.status(403).json({ error: 'Invalid API key' });
      return;
    }

    next();
  };
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
    const PORT = process.env['PORT'] || config.server.port || 1307;
    const HOST = process.env['HOST'] || config.server.host || 'localhost';

    // Check if port is in use and kill process if necessary
    const portInUse = await isPortInUse(Number(PORT));
    if (portInUse) {
      log('error', `Port ${PORT} is already in use.`);
      // `createServer()` already built the Database (auto-save interval,
      // monitor interval, open partition data). Returning here without closing
      // it left all of that running and unflushed.
      await gracefulShutdown().catch((err) => log('error', 'Error during cleanup:', err));
      process.exit(1);
    }

    // The host has to be passed through: `app.listen(PORT)` binds every
    // interface, so a process configured for `localhost` (the default) was in
    // fact reachable from the network while logging `http://localhost:PORT`.
    const server = app.listen(Number(PORT), HOST, () => {
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
