import { ApiContext } from '../../types';
/**
 * Creates and configures Express router with vector-related API endpoints.
 *
 * Sets up the following endpoints:
 * - `POST /api/vectors` - Add a single vector or bulk vectors
 * - `GET /api/vectors/:id` - Get a vector by ID
 * - `GET /api/vectors/:id/exists` - Check if a vector exists
 * - `PATCH /api/vectors/:id/metadata` - Update vector metadata
 * - `DELETE /api/vectors/:id` - Delete a vector
 * - `GET /api/vectors/:id/similar` - Find similar vectors to a given vector
 *
 * Each endpoint includes proper error handling, database readiness checks,
 * and timing metrics. The endpoints support both string and numeric IDs,
 * with automatic type conversion attempts when a lookup fails.
 *
 * @param context - The API context containing database and timer instances
 * @returns An Express router configured with vector-related endpoints
 */
export declare function vectorRoutes(context: ApiContext): import("express-serve-static-core").Router;
