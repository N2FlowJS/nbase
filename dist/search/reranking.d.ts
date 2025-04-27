import { RerankingOptions, SearchResult } from '../types';
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
export declare class SearchReranker {
    /**
     * Rerank search results using the specified method.
     * This is the main public entry point for reranking.
     *
     * @param results The initial list of search results, typically sorted by distance/similarity.
     * @param options Configuration for the reranking process, including the method to use.
     * @returns A new list of reranked search results.
     */
    rerank(results: SearchResult[], options?: RerankingOptions): SearchResult[];
    /**
     * Basic reranking: Returns the results as is or potentially capped at k.
     * Does not change the order based on content or metadata.
     */
    private _standardReranking;
    /**
     * Diversity-based reranking using Maximal Marginal Relevance (MMR) concept.
     * Requires actual vectors for calculation.
     */
    private _diversityReranking;
    /**
     * Weighted reranking based on metadata attributes.
     * Requires metadataMap in options.
     */
    private _weightedReranking;
}
export default SearchReranker;
