"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const cors_1 = __importDefault(require("cors"));
const express_1 = __importDefault(require("express"));
const express_rate_limit_1 = __importDefault(require("express-rate-limit"));
const helmet_1 = __importDefault(require("helmet"));
const net_1 = __importDefault(require("net"));
const config_1 = __importDefault(require("../config"));
const profiling_1 = require("../utils/profiling");
// Import Database class
const database_1 = require("../database/database");
// Import middleware
const common_1 = require("./middleware/common");
// Import routes
const index_1 = require("./routes/index");
const search_1 = require("./routes/search");
const vectors_1 = require("./routes/vectors");
const filters_1 = require("./utils/filters");
/**
 * Checks if a port is in use
 * @param port Port to check
 * @returns Promise resolving to true if port is in use, false otherwise
 */
async function isPortInUse(port) {
    return new Promise((resolve) => {
        const server = net_1.default
            .createServer()
            .once('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                resolve(true);
            }
            else {
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
function createServer(options = {}) {
    const app = (0, express_1.default)();
    // Merge default config with provided options
    const serverOptions = {
        port: options.port || config_1.default.server.port,
        host: options.host || config_1.default.server.host,
        database: {
            ...{
                clustering: config_1.default.clustering,
                indexing: config_1.default.indexing,
                partitioning: config_1.default.partitioning,
                persistence: config_1.default.persistence,
                monitoring: config_1.default.monitoring,
            },
            ...options.database,
        },
        rateLimit: { ...config_1.default.server.rateLimit, ...options.rateLimit },
        middleware: options.middleware || [],
        debug: options.debug || false,
        errorHandler: options.errorHandler ||
            ((err, req, res, next) => {
                console.error('API Error:', err);
                res.status(500).json({
                    error: err.message,
                    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined,
                });
            }),
    };
    // Request timer for performance monitoring
    const timer = (0, profiling_1.createTimer)();
    // Configure middleware
    app.use((0, helmet_1.default)()); // Security headers
    app.use((0, cors_1.default)()); // Cross-origin resource sharing
    app.use(express_1.default.json({ limit: '50mb' })); // JSON body parser with large payload support
    // Add logging middleware
    app.use(common_1.loggingMiddleware);
    // Apply custom middleware
    serverOptions.middleware.forEach((middleware) => app.use(middleware));
    // Rate limiting
    if (serverOptions.rateLimit.enable) {
        const apiLimiter = (0, express_rate_limit_1.default)({
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
        app.use((req, res, next) => {
            console.log(`DEBUG: Incoming request: ${req.method} ${req.originalUrl}`);
            next();
        });
    }
    // Initialize database instance with options
    const database = new database_1.Database(serverOptions.database);
    // Create context object with shared resources
    const apiContext = {
        timer,
        createFilterFunction: filters_1.createFilterFunction,
        database,
    };
    /**
     * API Routes
     */
    // Register route handlers
    app.use('/api/vectors', (0, vectors_1.vectorRoutes)(apiContext));
    app.use('/api/search', (0, search_1.searchRoutes)(apiContext));
    app.use('/', (0, index_1.indexRoutes)(apiContext));
    // Error handling middleware
    app.use(serverOptions.errorHandler);
    // Not Found handler
    app.use((req, res) => {
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
    app.gracefulShutdown = gracefulShutdown;
    // Return both for new code that expects the new format
    // but also make app the default export for backward compatibility
    const result = {
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
                persistence: config_1.default.persistence,
                partitioning: config_1.default.partitioning,
                indexing: config_1.default.indexing,
                clustering: {
                    ...config_1.default.clustering,
                    useCompression: false,
                },
            },
        });
        const PORT = process.env.PORT || config_1.default.server.port || 1307;
        const HOST = process.env.HOST || config_1.default.server.host || 'localhost';
        // Check if port is in use and kill process if necessary
        const portInUse = await isPortInUse(Number(PORT));
        if (portInUse) {
            console.log('Port is used');
            return;
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
exports.default = createServer;
//# sourceMappingURL=index.js.map