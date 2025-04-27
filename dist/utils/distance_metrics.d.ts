import { Vector } from '../types';
/**
 * Euclidean distance (L2 norm)
 * @param a First vector
 * @param b Second vector
 * @returns Euclidean distance between vectors
 */
export declare function euclidean(a: Vector, b: Vector): number;
/**
 * Manhattan distance (L1 norm)
 * @param a First vector
 * @param b Second vector
 * @returns Manhattan distance between vectors
 */
export declare function manhattan(a: Vector, b: Vector): number;
/**
 * Cosine distance (1 - cosine similarity)
 * @param a First vector
 * @param b Second vector
 * @returns Cosine distance between vectors
 */
export declare function cosine(a: Vector, b: Vector): number;
/**
 * Dot product (inner product) similarity
 * This returns a similarity rather than a distance (higher is more similar)
 * @param a First vector
 * @param b Second vector
 * @returns Dot product similarity between vectors
 */
export declare function dotProduct(a: Vector, b: Vector): number;
/**
 * Inner product distance (negative dot product)
 * Since dot product is a similarity, we negate it to get a distance
 * @param a First vector
 * @param b Second vector
 * @returns Inner product distance between vectors
 */
export declare function innerProduct(a: Vector, b: Vector): number;
/**
 * Chebyshev distance (L-infinity norm, maximum coordinate difference)
 * @param a First vector
 * @param b Second vector
 * @returns Chebyshev distance between vectors
 */
export declare function chebyshev(a: Vector, b: Vector): number;
/**
 * Squared Euclidean distance
 * Same as Euclidean but without the square root, faster for comparisons
 * @param a First vector
 * @param b Second vector
 * @returns Squared Euclidean distance between vectors
 */
export declare function squaredEuclidean(a: Vector, b: Vector): number;
/**
 * Hamming distance (number of positions where values differ)
 * @param a First vector
 * @param b Second vector
 * @returns Hamming distance between vectors
 */
export declare function hamming(a: Vector, b: Vector): number;
/**
 * Get a distance function by name
 * @param name Name of the distance function
 * @returns Distance function
 */
export declare function getDistanceFunction(name: string): (a: Vector, b: Vector) => number;
declare const _default: {
    euclidean: typeof euclidean;
    manhattan: typeof manhattan;
    cosine: typeof cosine;
    dotProduct: typeof dotProduct;
    innerProduct: typeof innerProduct;
    chebyshev: typeof chebyshev;
    squaredEuclidean: typeof squaredEuclidean;
    hamming: typeof hamming;
    getDistanceFunction: typeof getDistanceFunction;
};
export default _default;
