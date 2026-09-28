import { parentPort, workerData } from 'node:worker_threads';
import HNSW from './hnsw';
import type { VectorProvider } from '../types';

/**
 * Rebuilds the id key the parent used.
 *
 * The parent serialises a `Map` as a plain object, which stringifies every key.
 * `isNaN(Number(id)) ? id : Number(id)` is what the parent will do to look the
 * id up again, so it has to be mirrored exactly — `'001'` and `'1e3'` both pass
 * the `Number()` check but would not round-trip as numbers.
 */
function normalizeId(id: string): number | string {
  const asNumber = Number(id);
  // `String(asNumber) === id` rejects lossy forms like '001' and '1e3'.
  return Number.isNaN(asNumber) || String(asNumber) !== id ? id : asNumber;
}

class SimpleVectorProvider implements VectorProvider {
  private vectors: Map<number | string, Float32Array>;

  /**
   * @param vectorMap id -> vector. The backing store may be a
   *   `SharedArrayBuffer` (sent as a `Float32Array` view) or a plain array.
   */
  constructor(vectorMap: Record<string, Float32Array | number[]>) {
    this.vectors = new Map();
    for (const [id, vec] of Object.entries(vectorMap)) {
      this.vectors.set(normalizeId(id), vec instanceof Float32Array ? vec : new Float32Array(vec));
    }
  }

  getVector(id: number | string): Float32Array | undefined {
    return this.vectors.get(id);
  }

  getVectorIds(): (number | string)[] {
    return Array.from(this.vectors.keys());
  }
}

if (parentPort) {
  const { vectorMap, options } = workerData;

  const fail = (error: unknown): void => {
    const message = error instanceof Error ? error.message : String(error);
    parentPort?.postMessage({ type: 'error', error: message });
    // Exit explicitly: with no pending work a worker thread would otherwise
    // linger, keeping the whole dataset alive and delaying process exit.
    process.exit(1);
  };

  try {
    const provider = new SimpleVectorProvider(vectorMap);
    const hnsw = new HNSW(provider, options);

    hnsw
      .buildIndex({
        ...options,
        progressCallback: (progress: number) => {
          parentPort?.postMessage({ type: 'progress', progress });
        },
      })
      .then(() => {
        parentPort?.postMessage({ type: 'done', result: hnsw.serialize() });
        // Success path: nothing is left to do, so exit 0 rather than idling.
        process.exit(0);
      })
      .catch(fail);
  } catch (error) {
    fail(error);
  }
}
