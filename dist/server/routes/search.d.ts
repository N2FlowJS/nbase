import { ApiContext } from '../../types';
/**
 * Configures and returns the search-related routes for the API.
 *
 * @param context - The API context containing dependencies such as the database, timer, and filter creation function.
 *
 * @returns An Express router with the following endpoints:
 *
 * ### POST `/`
 * Performs a nearest neighbor search using the provided query vector(s) and options.
 *
 * Request Body:
 * - `query` (array): The query vector(s) to search for. **Required**.
 * - `k` (number): The number of nearest neighbors to retrieve.
 * - `method` (string): The search method to use (`hnsw` or `clustered`).
 * - `partitionIds` (array): Optional array of partition IDs to restrict the search.
 * - `efSearch` (number): The efSearch parameter for HNSW search.
 * - `distanceMetric` (string): The distance metric to use for the search.
 * - `rerank` (boolean): Whether to apply reranking to the results.
 * - `rerankingMethod` (string): The method to use for reranking.
 * - `rerankLambda` (number): The lambda parameter for reranking.
 * - `filters` (object | array): Filters to apply to the search results.
 * - `includeMetadata` (boolean): Whether to include metadata in the results.
 * - `includeVectors` (boolean): Whether to include vectors in the results.
 * - `skipCache` (boolean): Whether to skip the cache for this search.
 * - `searchTimeoutMs` (number): Timeout for the search operation in milliseconds.
 *
 * Response:
 * - `results` (array): The search results.
 * - `count` (number): The number of results returned.
 * - `duration` (number): The time taken to perform the search in milliseconds.
 * - `searchOptions` (object): Details about the search options used.
 *
 * ### POST `/metadata`
 * Searches for metadata entries that match specific criteria across all loaded partitions.
 *
 * Request Body:
 * - `criteria` (string | array | object): The criteria to match. **Required**.
 * - `values` (any): Optional values to match (used with string/array criteria).
 * - `includeVectors` (boolean): Whether to include vectors in the response.
 * - `limit` (number): Optional limit on the number of results to return.
 *
 * Response:
 * - `results` (array): The metadata entries matching the criteria.
 * - `count` (number): The number of results returned.
 * - `duration` (number): The time taken to perform the search in milliseconds.
 *
 * Errors:
 * - Returns a 400 status code for invalid requests (e.g., missing or invalid `query` or `criteria`).
 * - Returns a 500 status code for internal server errors, with optional stack trace in development mode.
 *
 * ### POST `/relationships`
 * Finds relationships between vectors based on a distance threshold.
 *
 * Request Body:
 * - `threshold` (number): The maximum distance between vectors to consider them related. **Required**.
 * - `metric` (string): Distance metric to use (e.g., 'cosine', 'euclidean'). Default depends on database implementation.
 * - `partitionIds` (array): Optional array of partition IDs to restrict the relationship search.
 *
 * Response:
 * - `relationships` (array): Array of relationships between vectors.
 * - `count` (number): The number of relationships found.
 * - `duration` (number): The time taken to extract relationships in milliseconds.
 *
 * Errors:
 * - Returns a 400 status code for invalid requests (e.g., missing or invalid `threshold`).
 * - Returns a 500 status code for internal server errors, with optional stack trace in development mode.
 */
export declare function searchRoutes(context: ApiContext): import("express-serve-static-core").Router;
