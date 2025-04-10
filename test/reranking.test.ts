import { expect } from 'chai';
import { SearchReranker } from '../src/search/reranking';
import { SearchResult } from '../src/types';

describe('SearchReranker', () => {
    let reranker: SearchReranker;
    let testResults: SearchResult[];
    
    beforeEach(() => {
        reranker = new SearchReranker();
        
        // Setup test results
        testResults = [
            { id: 1, dist: 0.1, metadata: {} },
            { id: 2, dist: 0.2, metadata: {} },
            { id: 3, dist: 0.3, metadata: {} },
            { id: 4, dist: 0.4, metadata: {} },
            { id: 5, dist: 0.5, metadata: {} }
        ];
    });

    describe('rerank method', () => {
        it('should return empty array when input is not an array', () => {
            // @ts-ignore - Intentionally passing invalid input
            const result = reranker.rerank({});
            expect(result).to.be.an('array').that.is.empty;
        });

        it('should default to standard reranking when no method is specified', () => {
            const result = reranker.rerank(testResults);
            expect(result).to.deep.equal(testResults);
        });
    });

    describe('standard reranking', () => {
        it('should return all results when k is not specified', () => {
            const result = reranker.rerank(testResults, { method: 'standard' });
            expect(result).to.have.lengthOf(testResults.length);
            expect(result).to.deep.equal(testResults);
        });

        it('should limit results to k when specified', () => {
            const k = 3;
            const result = reranker.rerank(testResults, { method: 'standard', k });
            expect(result).to.have.lengthOf(k);
            expect(result).to.deep.equal(testResults.slice(0, k));
        });
    });

    describe('diversity reranking', () => {
        it('should fall back to standard when required params are missing', () => {
            const result = reranker.rerank(testResults, { method: 'diversity' });
            expect(result).to.deep.equal(testResults);
        });
        
        it('should perform MMR reranking when all required parameters are provided', () => {
            // Create test vectors
            const vectorsMap = new Map();
            const vectors = [
                new Float32Array([0.1, 0.2, 0.3]),
                new Float32Array([0.4, 0.5, 0.6]), 
                new Float32Array([0.7, 0.8, 0.9]),
                new Float32Array([0.2, 0.3, 0.4]),
                new Float32Array([0.5, 0.6, 0.7])
            ];
            
            testResults.forEach((result, idx) => {
                vectorsMap.set(result.id, vectors[idx]);
            });
            
            const queryVector = new Float32Array([0.3, 0.3, 0.3]);
            
            const result = reranker.rerank(testResults, { 
                method: 'diversity',
                queryVector,
                vectorsMap,
                lambda: 0.5
            });
            
            // Verify we got the right number of results
            expect(result).to.have.lengthOf(testResults.length);
            
            // Verify first result is still the most relevant one
            expect(result[0].id).to.equal(1);
            
            // Verify the exact order is different from the original
            const originalIds = testResults.map(r => r.id);
            const rerankedIds = result.map(r => r.id);
            expect(rerankedIds).to.not.deep.equal(originalIds);
        });
    });

    describe('weighted reranking', () => {
        it('should fall back to standard when metadataMap is missing', () => {
            const result = reranker.rerank(testResults, { 
                method: 'weighted',
                weights: { popularity: 0.5 }
            });
            expect(result).to.deep.equal(testResults);
        });
        
        it('should rerank based on weighted metadata attributes', () => {
            // Setup metadata map
            const metadataMap = new Map();
            metadataMap.set(1, { popularity: 5, recency: 2 });
            metadataMap.set(2, { popularity: 10, recency: 1 });
            metadataMap.set(3, { popularity: 2, recency: 8 });
            metadataMap.set(4, { popularity: 6, recency: 4 });
            metadataMap.set(5, { popularity: 4, recency: 5 });
            
            const result = reranker.rerank(testResults, {
                method: 'weighted',
                metadataMap,
                weights: { popularity: 0.1, recency: 0.05 }
            });
            
            // Item 2 should come first due to high popularity
            expect(result[0].id).to.equal(2);
            
            // Verify all results are returned but in different order
            expect(result).to.have.lengthOf(testResults.length);
            const rerankedIds = result.map(r => r.id);
            const originalIds = testResults.map(r => r.id);
            expect(rerankedIds).to.not.deep.equal(originalIds);
        });
        
        it('should limit results to k when specified', () => {
            const metadataMap = new Map();
            testResults.forEach((r, i) => {
                metadataMap.set(r.id, { score: 10 - i });
            });
            
            const k = 2;
            const result = reranker.rerank(testResults, {
                method: 'weighted',
                metadataMap,
                weights: { score: 0.1 },
                k
            });
            
            expect(result).to.have.lengthOf(k);
        });
    });
});