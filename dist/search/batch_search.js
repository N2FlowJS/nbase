"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.BatchEngineSearch = void 0;
const config_1 = __importDefault(require("../config")); // Keep this for fallback default timeout
const profiling_1 = require("../utils/profiling");
/**
 * BatchEngineSearch processes multiple vector queries in parallel using a PartitionedVectorDBInterface.
 * This version uses the optimized type definitions.
 */
/**
 * Manages efficient processing of multiple vector search queries in batches.
 *
 * The BatchEngineSearch class provides functionality to handle concurrent vector
 * search operations against a partitioned vector database. It automatically manages
 * large batches by splitting them into smaller chunks, processes queries in parallel,
 * and handles timeout and error scenarios gracefully.
 *
 * Key features:
 * - Efficient batching of multiple vector search queries
 * - Automatic chunking of large query batches
 * - Concurrent query execution with configurable timeouts
 * - Support for preserving original query order in results
 * - Integration with different search methods (HNSW or clustered)
 * - Error handling and graceful degradation
 *
 * @example
 * ```typescript
 * const searchEngine = new PartitionedVectorDB(...);
 * const batchSearch = new BatchEngineSearch(searchEngine, {
 *   maxBatchSize: 32,
 *   defaultSearchTimeoutMs: 10000
 * });
 *
 * const queries = [
 *   { query: vectorA, k: 5, options: { useHNSW: true } },
 *   { query: vectorB, k: 10, options: { filter: { category: "tech" } } }
 * ];
 *
 * const results = await batchSearch.searchBatch(queries);
 * ```
 */
class BatchEngineSearch {
    constructor(searchEngine, options = {} // Accept the new configuration type
    ) {
        this.searchEngine = searchEngine;
        // Define default values using keys from BatchSearchConfiguration
        const defaults = {
            maxBatchSize: 64,
            prioritizeOrder: true,
            groupSimilarQueries: false,
            defaultSearchTimeoutMs: config_1.default.batchSearch?.defaultSearchTimeoutMs || 15000,
        };
        // Merge provided options with defaults
        this.options = { ...defaults, ...options };
    }
    /**
     * Process multiple search queries in a batch.
     * @param queries - Array of search queries using the new BatchSearchQuery interface.
     * @returns Results for each query.
     */
    async searchBatch(queries) {
        if (!queries || !queries.length)
            return [];
        const timer = (0, profiling_1.createTimer)();
        timer.start('batch_search_partitioned');
        // Split into smaller batches if too large, using the new option key
        if (queries.length > this.options.maxBatchSize) {
            const results = [];
            console.log(`Batch too large (${queries.length}), splitting into chunks of ${this.options.maxBatchSize}`);
            for (let i = 0; i < queries.length; i += this.options.maxBatchSize) {
                const batchQueries = queries.slice(i, i + this.options.maxBatchSize);
                const batchResults = await this.searchBatch(batchQueries); // Recursive call
                results.push(...batchResults);
            }
            timer.stop('batch_search_partitioned');
            console.log(`Finished processing large batch (${queries.length}) in ${timer.getElapsed('batch_search_partitioned')}ms`);
            return results;
        }
        // Process the current batch (or sub-batch)
        const results = await this._processBatch(queries);
        timer.stop('batch_search_partitioned');
        console.log(`Processed batch of ${queries.length} queries in ${timer.getElapsed('batch_search_partitioned')}ms`);
        return results;
    }
    /**
     * Process batch queries using the PartitionedVectorDB directly.
     * Leverages Promise.all for concurrency across queries and relies on the
     * PartitionedVectorDB's internal parallelism.
     */
    async _processBatch(queries) {
        const timer = (0, profiling_1.createTimer)();
        timer.start('process_batch');
        let processedQueries = queries;
        if (this.options.groupSimilarQueries) {
            processedQueries = this._groupSimilarQueries(queries);
        }
        // Use Promise.all to run searches concurrently
        const resultsPromises = processedQueries.map(async (queryData, originalIndex) => {
            // Retain originalIndex for reordering if needed
            // Destructure from BatchSearchQuery
            const { query, k, options = {} } = queryData;
            const queryTimer = (0, profiling_1.createTimer)();
            queryTimer.start(`query_${originalIndex}`); // Use original index for tracking
            let methodUsed = 'unknown';
            try {
                // Build options object to pass into the search engine method
                // Combine fields from BaseSearchOptions and SearchExecutionOptions
                const engineSearchOptions = {
                    // BaseSearchOptions
                    k: k, // k is usually passed separately but can be included if the engine API requires it
                    filter: options.filter,
                    includeMetadata: false, // Metadata is usually not needed in raw search
                    distanceMetric: options.distanceMetric, // Allow overriding the default metric
                    // SearchExecutionOptions
                    partitionIds: options.partitionIds,
                    efSearch: options.efSearch, // For HNSW search
                };
                let queryResult;
                // Decide the method based on the options *of each query*
                if (options.useHNSW && typeof this.searchEngine.findNearestHNSW === 'function') {
                    methodUsed = 'hnsw';
                    queryResult = await this.searchEngine.findNearestHNSW(query, k, engineSearchOptions // Pass the merged options
                    );
                }
                else if (typeof this.searchEngine.findNearest === 'function') {
                    methodUsed = 'clustered';
                    // Ensure HNSW-specific parameters are not passed to findNearest
                    const { efSearch, ...clusteredOptions } = engineSearchOptions;
                    queryResult = await this.searchEngine.findNearest(query, k, clusteredOptions);
                }
                else {
                    throw new Error('Search engine provides neither findNearestHNSW nor findNearest.');
                }
                queryTimer.stop(`query_${originalIndex}`);
                console.log(`Query ${originalIndex} (k=${k}, method=${methodUsed}) took ${queryTimer.getElapsed(`query_${originalIndex}`)}ms`);
                return {
                    originalIndex: originalIndex, // Retain original index for reordering
                    result: queryResult,
                    error: null,
                };
            }
            catch (error) {
                queryTimer.stop(`query_${originalIndex}`);
                console.error(`Error processing query ${originalIndex} after ${queryTimer.getElapsed(`query_${originalIndex}`)}ms:`, error);
                return {
                    originalIndex: originalIndex,
                    result: [], // Return an empty array in case of error
                    error: error instanceof Error ? error.message : String(error),
                };
            }
        });
        // Add timeout to each promise using the new option key
        const timedPromises = resultsPromises.map((p) => Promise.race([
            p,
            new Promise((_, reject) => setTimeout(() => reject(new Error(`Query timed out after ${this.options.defaultSearchTimeoutMs}ms`)), this.options.defaultSearchTimeoutMs // Use the new key
            )),
        ]).catch((error) => {
            console.error('Batch query failed or timed out:', error);
            return {
                originalIndex: -1,
                result: [],
                error: error.message,
            };
        }));
        // Wait for all queries to complete or timeout
        const settledResults = await Promise.all(timedPromises);
        timer.stop('process_batch');
        // Reconstruct results array, potentially reordering
        const finalResults = new Array(queries.length);
        if (this.options.prioritizeOrder) {
            // Use originalIndex to place results correctly
            for (const res of settledResults) {
                if (res.originalIndex !== -1 && res.originalIndex < finalResults.length) {
                    finalResults[res.originalIndex] = res.result;
                    if (res.error) {
                        console.warn(`Query at original index ${res.originalIndex} failed: ${res.error}`);
                    }
                }
                else if (res.originalIndex === -1 && res.error) {
                    console.error(`A query timed out or failed without recoverable index: ${res.error}`);
                }
            }
            for (let i = 0; i < finalResults.length; i++) {
                if (finalResults[i] === undefined) {
                    console.warn(`Result for original index ${i} is missing (likely due to unrecoverable error/timeout).`);
                    finalResults[i] = [];
                }
            }
        }
        else {
            settledResults.forEach((res, i) => {
                finalResults[i] = res.result;
                if (res.error) {
                    console.warn(`Query at result index ${i} (order not prioritized) failed: ${res.error}`);
                }
            });
            while (finalResults.length < queries.length) {
                finalResults.push([]);
            }
            finalResults.length = queries.length;
        }
        return finalResults;
    }
    /**
     * Placeholder for grouping similar queries.
     * @private
     */
    _groupSimilarQueries(queries) {
        if (this.options.groupSimilarQueries) {
            console.log('Query grouping requested but basic implementation used.');
        }
        return queries;
    }
    /**
     * Clean up resources (no-op in this version).
     */
    shutdown() {
        console.log('PartitionedBatchSearch shutdown.');
    }
}
exports.BatchEngineSearch = BatchEngineSearch;
//# sourceMappingURL=batch_search.js.map