"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.SearchReranker = void 0;
const distanceMetrics = __importStar(require("../utils/distance_metrics")); // Import distance functions
/**
 * SearchReranker provides various methods to reorder search results
 * to improve diversity, relevance, or other custom criteria.
 */
/**
 * A utility class for reranking search results using various strategies.
 *
 * The SearchReranker provides algorithms to refine initial search results
 * beyond simple distance/similarity sorting. This can improve result relevance
 * and user experience by considering factors like diversity or weighted attributes.
 *
 * @class SearchReranker
 *
 * Supports three reranking strategies:
 * - `standard`: Preserves the original ranking, optionally limiting to top k results
 * - `diversity`: Implements Maximal Marginal Relevance (MMR) to balance relevance and diversity
 * - `weighted`: Adjusts ranking based on weighted metadata attributes
 *
 * @example
 * ```typescript
 * const reranker = new SearchReranker();
 *
 * // Standard reranking (limit to top 5)
 * const topResults = reranker.rerank(initialResults, { method: 'standard', k: 5 });
 *
 * // Diversity reranking
 * const diverseResults = reranker.rerank(initialResults, {
 *   method: 'diversity',
 *   queryVector: query,
 *   vectorsMap: vectors,
 *   lambda: 0.7
 * });
 *
 * // Weighted reranking based on metadata
 * const weightedResults = reranker.rerank(initialResults, {
 *   method: 'weighted',
 *   metadataMap: metadata,
 *   weights: { recency: 0.3, popularity: 0.5 }
 * });
 * ```
 */
class SearchReranker {
    /**
     * Rerank search results using the specified method.
     * This is the main public entry point for reranking.
     *
     * @param results The initial list of search results, typically sorted by distance/similarity.
     * @param options Configuration for the reranking process, including the method to use.
     * @returns A new list of reranked search results.
     */
    rerank(results, options = {}) {
        const { method = 'standard' } = options; // Default to standard if no method specified
        // Ensure results is an array before proceeding
        if (!Array.isArray(results)) {
            console.error('Reranker received invalid input: results is not an array.');
            return [];
        }
        // Dispatch to the appropriate private reranking method
        switch (method) {
            case 'diversity':
                console.log('Dispatching to diversity reranking...'); // Debug log
                return this._diversityReranking(results, options);
            case 'weighted':
                console.log('Dispatching to weighted reranking...'); // Debug log
                return this._weightedReranking(results, options);
            case 'standard':
            default: // Fallback to standard reranking
                console.log('Dispatching to standard reranking (default)...'); // Debug log
                return this._standardReranking(results, options);
        }
    }
    /**
     * Basic reranking: Returns the results as is or potentially capped at k.
     * Does not change the order based on content or metadata.
     */
    _standardReranking(results, options) {
        const { k = results.length } = options;
        // Simple copy and slice to avoid modifying original results and apply k limit
        return results.slice(0, k);
    }
    /**
     * Diversity-based reranking using Maximal Marginal Relevance (MMR) concept.
     * Requires actual vectors for calculation.
     */
    _diversityReranking(initialResults, options) {
        const { k = initialResults.length, queryVector, lambda = 0.7, // Default balance: more towards relevance
        vectorsMap, distanceMetric = 'euclidean', // Default distance metric
         } = options;
        // --- Input Validation ---
        if (!queryVector || !vectorsMap || vectorsMap.size === 0 || initialResults.length <= 1) {
            console.warn('Diversity reranking skipped: Missing queryVector, vectorsMap, or insufficient results.');
            return initialResults.slice(0, k); // Return original top K
        }
        // Add more validation as needed (e.g., lambda range)
        // --- Setup ---
        const distanceFunc = distanceMetrics.getDistanceFunction(distanceMetric);
        const typedQueryVector = queryVector instanceof Float32Array ? queryVector : new Float32Array(queryVector);
        const remainingResults = new Map();
        const resultVectors = new Map();
        initialResults.forEach((res) => {
            const vector = vectorsMap.get(res.id);
            if (vector) {
                remainingResults.set(res.id, res);
                resultVectors.set(res.id, vector);
            }
            else {
                console.warn(`Vector for result ID ${res.id} not found in vectorsMap. Skipping for diversity rerank.`);
            }
        });
        if (remainingResults.size === 0) {
            console.warn('No results with available vectors for diversity reranking.');
            return initialResults.slice(0, k);
        }
        const finalResults = [];
        const selectedIds = new Set();
        // --- MMR Algorithm ---
        // 1. Select the first result
        let firstResult = null;
        let minInitialDist = Infinity;
        for (const res of remainingResults.values()) {
            if (res.dist < minInitialDist) {
                minInitialDist = res.dist;
                firstResult = res;
            }
        }
        if (!firstResult) {
            console.error('Could not determine the first result for MMR.');
            return initialResults.slice(0, k);
        }
        finalResults.push(firstResult);
        selectedIds.add(firstResult.id);
        remainingResults.delete(firstResult.id);
        // 2. Iteratively select remaining results
        while (finalResults.length < k && remainingResults.size > 0) {
            let bestCandidateId = null;
            let maxMmrScore = -Infinity;
            for (const [candidateId, candidateResult] of remainingResults.entries()) {
                const candidateVector = resultVectors.get(candidateId);
                if (!candidateVector)
                    continue;
                // Calculate Relevance Score (using similarity proxy from distance)
                const relevanceScore = 1.0 / (1.0 + candidateResult.dist);
                // Calculate Diversity Score (Min Distance to Selected)
                let minDistanceToSelected = Infinity;
                for (const selectedId of selectedIds) {
                    const selectedVector = resultVectors.get(selectedId);
                    if (selectedVector) {
                        const distToSelected = distanceFunc(candidateVector, selectedVector);
                        minDistanceToSelected = Math.min(minDistanceToSelected, distToSelected);
                    }
                }
                const diversityScore = minDistanceToSelected; // Higher is more diverse
                // Combine scores using lambda
                const mmrScore = lambda * relevanceScore + (1 - lambda) * diversityScore;
                if (mmrScore > maxMmrScore) {
                    maxMmrScore = mmrScore;
                    bestCandidateId = candidateId;
                }
            }
            // Add the best candidate found
            if (bestCandidateId !== null) {
                const bestResult = remainingResults.get(bestCandidateId);
                finalResults.push(bestResult);
                selectedIds.add(bestCandidateId);
                remainingResults.delete(bestCandidateId);
            }
            else {
                console.warn('MMR iteration finished without selecting a candidate.');
                break; // No more suitable candidates
            }
        }
        return finalResults;
    }
    /**
     * Weighted reranking based on metadata attributes.
     * Requires metadataMap in options.
     */
    _weightedReranking(results, options) {
        const { k = results.length, weights = {}, metadataMap } = options; // Use metadataMap from options
        if (!metadataMap || metadataMap.size === 0 || Object.keys(weights).length === 0) {
            console.warn('Weighted reranking skipped: Missing metadataMap or weights.');
            return results.slice(0, k); // Apply k limit even if not reranking
        }
        // Create weighted scores
        const weightedResults = results.map((result) => {
            const itemMetadata = metadataMap.get(result.id) || {};
            let weightedScore = result.dist; // Start with original distance
            // Apply weights
            for (const [key, weight] of Object.entries(weights)) {
                if (key in itemMetadata && typeof itemMetadata[key] === 'number') {
                    weightedScore -= itemMetadata[key] * weight;
                }
            }
            return { ...result, weightedScore };
        });
        // Sort by weighted score and take top k
        return weightedResults.sort((a, b) => a.weightedScore - b.weightedScore).slice(0, k);
    }
}
exports.SearchReranker = SearchReranker;
exports.default = SearchReranker;
//# sourceMappingURL=reranking.js.map