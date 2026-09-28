import { Vector } from '../types';

/**
 * Euclidean distance (L2 norm)
 * @param a First vector
 * @param b Second vector
 * @returns Euclidean distance between vectors
 */
export function euclidean(a: Vector, b: Vector): number {
  return Math.sqrt(squaredEuclidean(a, b));
}

/**
 * Manhattan distance (L1 norm)
 * @param a First vector
 * @param b Second vector
 * @returns Manhattan distance between vectors
 */
export function manhattan(a: Vector, b: Vector): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  let i = 0;

  for (; i < len; i++) {
    const d = a[i] - b[i];
    sum += (d < 0 ? -d : d);
  }

  return sum;
}

/**
 * Cosine distance (1 - cosine similarity)
 * @param a First vector
 * @param b Second vector
 * @returns Cosine distance between vectors
 */
export function cosine(a: Vector, b: Vector): number {
  let dotProductVal = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(a.length, b.length);
  let i = 0;

  for (; i < len; i++) {
    const ai = a[i], bi = b[i];
    dotProductVal += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
  }

  if (normA === 0 || normB === 0) {
    return 1;
  }

  const similarity = dotProductVal / Math.sqrt(normA * normB);
  return 1 - Math.max(-1, Math.min(1, similarity));
}

/**
 * Dot product (inner product) similarity
 * This returns a similarity rather than a distance (higher is more similar)
 * @param a First vector
 * @param b Second vector
 * @returns Dot product similarity between vectors
 */
export function dotProduct(a: Vector, b: Vector): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  let i = 0;

  for (; i < len; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

/**
 * Inner product distance (negative dot product)
 * Since dot product is a similarity, we negate it to get a distance
 * @param a First vector
 * @param b Second vector
 * @returns Inner product distance between vectors
 */
export function innerProduct(a: Vector, b: Vector): number {
  return -dotProduct(a, b);
}

/**
 * Chebyshev distance (L-infinity norm, maximum coordinate difference)
 * @param a First vector
 * @param b Second vector
 * @returns Chebyshev distance between vectors
 */
export function chebyshev(a: Vector, b: Vector): number {
  let max = 0;
  const len = Math.min(a.length, b.length);

  for (let i = 0; i < len; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > max) max = diff;
  }

  return max;
}

/**
 * Squared Euclidean distance
 * Same as Euclidean but without the square root, faster for comparisons
 * @param a First vector
 * @param b Second vector
 * @returns Squared Euclidean distance between vectors
 */
export function squaredEuclidean(a: Vector, b: Vector): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  let i = 0;

  for (; i < len; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

/**
 * Hamming distance (number of positions where values differ)
 * @param a First vector
 * @param b Second vector
 * @returns Hamming distance between vectors
 */
export function hamming(a: Vector, b: Vector): number {
  let count = 0;
  const len = Math.min(a.length, b.length);

  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) {
      count++;
    }
  }

  return count;
}

/**
 * Get a distance function by name
 * @param name Name of the distance function
 * @returns Distance function
 */
export function getDistanceFunction(name: string): (a: Vector, b: Vector) => number {
  switch (name.toLowerCase()) {
    case 'euclidean':
      return euclidean;
    case 'manhattan':
      return manhattan;
    case 'cosine':
      return cosine;
    case 'dotproduct':
    case 'dot':
      return dotProduct;
    case 'innerproduct':
    case 'inner':
      return innerProduct;
    case 'chebyshev':
    case 'infinity':
      return chebyshev;
    case 'squaredeuclidean':
    case 'squared':
      return squaredEuclidean;
    case 'hamming':
      return hamming;
    default:
      return euclidean; // Default to Euclidean
  }
}

export default {
  euclidean,
  manhattan,
  cosine,
  dotProduct,
  innerProduct,
  chebyshev,
  squaredEuclidean,
  hamming,
  getDistanceFunction,
};
