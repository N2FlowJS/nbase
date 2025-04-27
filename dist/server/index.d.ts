import { IServerInstance, IServerOptions } from '../types';
export type { AddVectorRequest, BatchSearchRequest, BulkAddRequest, DatabaseOptions, FilterConfig, IServerOptions, SaveLoadDatabaseRequest, SearchRequest, TrainIndexRequest, UpdateMetadataRequest } from '../types';
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
declare function createServer(options?: IServerOptions): IServerInstance;
export default createServer;
