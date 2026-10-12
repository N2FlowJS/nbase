import { expect } from 'chai';
import { DistanceMetric, Vector } from '../../src/types';

/**
 * A deterministic PRNG (mulberry32). Every dataset in the suite is seeded, so a
 * failure is reproducible and a search that starts returning the wrong ids is a
 * real regression rather than a different random draw.
 */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A reproducible vector set: `count` vectors of `dimension` floats in [0, 1). */
export function seededVectors(count: number, dimension: number, seed = 1): Float32Array[] {
  const random = createRandom(seed);
  const vectors: Float32Array[] = [];
  for (let i = 0; i < count; i++) {
    const vector = new Float32Array(dimension);
    for (let d = 0; d < dimension; d++) {
      vector[d] = random();
    }
    vectors.push(vector);
  }
  return vectors;
}

/** A query vector taken from inside the same distribution. */
export function seededQuery(dimension: number, seed = 99): Float32Array {
  const random = createRandom(seed);
  const vector = new Float32Array(dimension);
  for (let d = 0; d < dimension; d++) {
    vector[d] = random();
  }
  return vector;
}

/**
 * Reference implementations, deliberately naive and written in float64.
 *
 * They exist to be obviously correct rather than fast: the library has four
 * separate Euclidean kernels and they had already drifted apart (three of them
 * add a dimension-mismatch penalty the reference does not have), so the oracle
 * cannot be another copy of the same arithmetic.
 */
export function referenceDistance(metric: DistanceMetric, a: Vector, b: Vector): number {
  if (metric === 'cosine') {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
      const left = a[i] ?? 0;
      const right = b[i] ?? 0;
      dot += left * right;
      normA += left * left;
      normB += right * right;
    }
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 1 : 1 - dot / denominator;
  }

  let sum = 0;
  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i++) {
    const delta = (a[i] ?? 0) - (b[i] ?? 0);
    sum += delta * delta;
  }
  // Cross-dimension vectors are comparable only by padding the shorter one.
  for (let i = length; i < Math.max(a.length, b.length); i++) {
    const delta = i < a.length ? (a[i] ?? 0) : 0;
    sum += delta * delta;
  }
  return Math.sqrt(sum);
}

export interface BruteForceHit {
  id: number | string;
  dist: number;
}

/** Exact nearest neighbours, sorted by distance then id for a stable order. */
export function bruteForceNearest(ids: ReadonlyArray<number | string>, vectors: ReadonlyMap<number | string, Float32Array>, query: Vector, k: number, metric: DistanceMetric = 'euclidean', excluded: ReadonlySet<number | string> = new Set()): BruteForceHit[] {
  const hits: BruteForceHit[] = [];
  for (const id of ids) {
    if (excluded.has(id)) continue;
    const vector = vectors.get(id);
    if (!vector) continue;
    hits.push({ id, dist: referenceDistance(metric, query, vector) });
  }
  hits.sort((a, b) => a.dist - b.dist || String(a.id).localeCompare(String(b.id)));
  return hits.slice(0, k);
}

/** Fraction of the true top-k that the implementation returned, order ignored. */
export function recallAt(actual: ReadonlyArray<{ id: number | string }>, expected: ReadonlyArray<BruteForceHit>, k: number): number {
  const expectedIds = new Set(expected.slice(0, k).map((hit) => hit.id));
  if (expectedIds.size === 0) return 1;
  const returned = new Set(actual.slice(0, k).map((result) => result.id));
  let hits = 0;
  for (const id of expectedIds) {
    if (returned.has(id)) hits++;
  }
  return hits / expectedIds.size;
}

/** 1 when the ids match exactly and in order, 0 otherwise. */
export function exactIdSequence(actual: ReadonlyArray<{ id: number | string }>): string {
  return actual.map((result) => String(result.id)).join(',');
}

/**
 * Asserts that an approximate search found at least `minRecall` of the true
 * top-k. The ids are reported on failure: a bare percentage does not say which
 * neighbour was missed, which is the first thing anyone looks at.
 */
export function expectRecall(actual: ReadonlyArray<{ id: number | string }>, expected: ReadonlyArray<BruteForceHit>, k: number, minRecall: number, label = 'search'): void {
  const recall = recallAt(actual, expected, k);
  const missing = expected.slice(0, k).filter((hit) => !actual.some((result) => String(result.id) === String(hit.id)));
  expect(recall, `${label}: recall@${k} was ${recall.toFixed(3)} (need ${minRecall}), missed [${missing.map((hit) => `${hit.id}@${hit.dist.toFixed(4)}`).join(' ')}]`).to.be.at.least(minRecall);
}

/** Convenience for building the `vectors` map a brute-force call needs. */
export function toVectorMap(vectors: ReadonlyArray<Float32Array>, prefix = 'v'): Map<number | string, Float32Array> {
  const map = new Map<number | string, Float32Array>();
  vectors.forEach((vector, index) => {
    map.set(`${prefix}${index}`, vector);
  });
  return map;
}
