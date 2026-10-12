/**
 * Request-payload validation for the HTTP API.
 *
 * Every numeric field used to be forwarded to the search engine unchecked, and
 * two of the failures are remote denial of service rather than bad input:
 *
 * - `k: "abc"` produced `NaN`, and HNSW computes `ef = Math.max(k, efSearch)`,
 *   so the beam's termination test `results.size >= ef` was never true and the
 *   query scanned the whole index. `k: 1e9` did the same directly.
 * - `threshold` on `/relationships` and `/communities` only had a `> 0` check.
 *   Both endpoints are O(n^2) in the vector count, so one request could build a
 *   multi-gigabyte result set and be OOM-killed — losing whatever the database
 *   had not saved yet.
 *
 * These helpers answer with a plain `number` or `null`, so a route reads as
 * "reject with 400, or continue".
 */

/** Thrown for a payload that cannot be served. Carries the client-facing text. */
export class InvalidRequestError extends Error {
  public readonly status = 400;

  constructor(message: string) {
    super(message);
    this.name = 'InvalidRequestError';
  }
}

export const MAX_K = 1000;
export const MAX_EF_SEARCH = 10_000;
export const MAX_LIMIT = 10_000;
/** Ceiling for a distance threshold on the O(n^2) graph endpoints. */
export const MAX_GRAPH_THRESHOLD = 1_000_000;
export const MAX_VECTOR_DIMENSIONS = 8192;

/**
 * An integer within `[min, max]`.
 *
 * `Number.isInteger` is the check that matters: `parseInt('abc')`, `NaN` and
 * `Infinity` all fail it, and so do fractional values, which would otherwise be
 * truncated at some arbitrary layer.
 */
export function requireInteger(value: unknown, name: string, min: number, max: number): number | undefined {
  if (value === undefined || value === null) return undefined;
  const numeric = typeof value === 'string' && value.trim() !== '' ? Number(value) : value;
  if (typeof numeric !== 'number' || !Number.isInteger(numeric)) {
    throw new InvalidRequestError(`Invalid ${name}: expected an integer, received ${JSON.stringify(value)}`);
  }
  if (numeric < min || numeric > max) {
    throw new InvalidRequestError(`Invalid ${name}: must be between ${min} and ${max}, received ${numeric}`);
  }
  return numeric;
}

/** A finite number within `[min, max]` (thresholds, lambdas). */
export function requireNumber(value: unknown, name: string, min: number, max: number): number | undefined {
  if (value === undefined || value === null) return undefined;
  const numeric = typeof value === 'string' && value.trim() !== '' ? Number(value) : value;
  if (typeof numeric !== 'number' || !Number.isFinite(numeric)) {
    throw new InvalidRequestError(`Invalid ${name}: expected a finite number, received ${JSON.stringify(value)}`);
  }
  if (numeric < min || numeric > max) {
    throw new InvalidRequestError(`Invalid ${name}: must be between ${min} and ${max}, received ${numeric}`);
  }
  return numeric;
}

/**
 * A vector id: a non-empty string, or a safe integer.
 *
 * An object or `null` used to be accepted and stored as a `Map` key across
 * `memoryStorage`, `metadata` and `vectorDimensions`; because
 * `typeof vectorId === 'number'` was false, the auto-id counter did not move
 * and the next generated id silently overwrote it.
 */
export function requireVectorId(value: unknown, name = 'id'): string | number | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'string') {
    if (value.length === 0 || value.length > 256) {
      throw new InvalidRequestError(`Invalid ${name}: string ids must be 1-256 characters`);
    }
    return value;
  }
  if (typeof value === 'number' && Number.isSafeInteger(value)) {
    return value;
  }
  throw new InvalidRequestError(`Invalid ${name}: expected a string or a safe integer, received ${JSON.stringify(value)}`);
}

/** A query/add vector: a non-empty array of finite numbers within the dimension cap. */
export function requireVector(value: unknown, name = 'vector'): number[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new InvalidRequestError(`Invalid ${name}: expected a non-empty array of numbers`);
  }
  if (value.length > MAX_VECTOR_DIMENSIONS) {
    throw new InvalidRequestError(`Invalid ${name}: at most ${MAX_VECTOR_DIMENSIONS} dimensions are supported, received ${value.length}`);
  }
  for (const component of value) {
    if (typeof component !== 'number' || !Number.isFinite(component)) {
      throw new InvalidRequestError(`Invalid ${name}: every component must be a finite number`);
    }
  }
  return value as number[];
}

/**
 * Strips the keys that would change an object's prototype when spread.
 *
 * `{...metadata}` copies `__proto__` as an own property rather than assigning
 * it, so this is not prototype pollution — but the key is then persisted to
 * `meta.json` and echoed back to every reader of that vector.
 */
export function stripDangerousKeys<T extends Record<string, unknown>>(input: T): T {
  const output = { ...input };
  for (const key of Object.keys(output)) {
    if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
      delete output[key];
    }
  }
  return output;
}
