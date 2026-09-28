// --- START OF FILE hnsw_filter.test.ts ---

// Regression tests for HNSW returning nothing (or everything wrong) when a
// filter is supplied.
//
// The traversal used to invoke `filter(id)` with the id only, so a metadata
// predicate received `undefined` and rejected every node. The entry point was
// also seeded into the result set unconditionally, so a soft-deleted or
// filtered-out node came back as a top-k hit.

import { describe, it, before, after } from 'mocha';
import { expect } from 'chai';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { ClusteredVectorDB } from '../src/vector/clustered_vector_db';
import HNSW from '../src/ann/hnsw';
import type { VectorProvider, VectorStoreSearchOptions } from '../src/types';

const DIM = 4;
const COUNT = 200;

function makeVector(i: number): number[] {
  // Deterministic, spread-out points so HNSW builds a real graph.
  const v = [];
  for (let d = 0; d < DIM; d++) {
    v.push(Math.sin(i * (d + 1)) + (d / 10) * i);
  }
  return v;
}

describe('HNSW filtering', () => {
  let dir: string;
  let db: ClusteredVectorDB;
  let hnsw: HNSW;

  before(async () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'nbase-hnsw-filter-'));
    db = new ClusteredVectorDB(DIM, path.join(dir, 'data'), { clusterSize: 25 });
    if (db.readyPromise) await db.readyPromise;

    for (let i = 0; i < COUNT; i++) {
      db.addVector(i, makeVector(i));
    }
    // Half even, half odd.
    for (let i = 0; i < COUNT; i++) {
      db.addMetadata(i, { parity: i % 2 === 0 ? 'even' : 'odd' });
    }

    await hnswLoad();
  });

  async function hnswLoad(): Promise<void> {
    const provider: VectorProvider = {
      getVector: (id) => db.getVector(id) ?? null,
      getVectorIds: () => db.getVectorIds(),
      getMetadata: (id) => db.getMetadata(id),
    };
    hnsw = new HNSW(provider, { dimensionAware: false });
    await hnsw.buildIndex();
  }

  after(async () => {
    if (db) await db.close({ save: false });
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('returns results with no filter', () => {
    const results = hnsw.findNearest(makeVector(0), 10);
    expect(results.length).to.be.greaterThan(0);
  });

  it('applies a metadata filter instead of returning nothing', () => {
    const results = hnsw.findNearest(makeVector(0), 20, {
      filter: (_id, metadata) => metadata?.['parity'] === 'even',
    });
    expect(results.length, 'metadata filter must not reject every node').to.be.greaterThan(0);
  });

  it('only returns ids the filter accepts', () => {
    const searchOptions: VectorStoreSearchOptions = {
      filter: (_id, metadata) => metadata?.['parity'] === 'even',
    };
    const results = hnsw.findNearest(makeVector(3), 30, searchOptions);
    for (const r of results) {
      expect(Number(r.id) % 2, `id ${r.id} should be even`).to.equal(0);
    }
  });

  it('does not mutate the caller-supplied options object', () => {
    const original = (_id: number | string, _m?: Record<string, any> | null) => true;
    const options = { k: 5, filter: original };
    hnsw.findNearest(makeVector(0), 5, options as VectorStoreSearchOptions);
    expect(options.filter).to.equal(original);
  });

  it('never returns a soft-deleted node', () => {
    // Delete every even id, then assert none come back.
    for (let i = 0; i < COUNT; i += 2) {
      hnsw.markDelete(i);
    }

    const results = hnsw.findNearest(makeVector(0), 40, {
      filter: (_id, metadata) => metadata?.['parity'] !== undefined,
    });

    for (const r of results) {
      expect(Number(r.id) % 2, `soft-deleted id ${r.id} leaked into results`).to.equal(1);
    }
  });

  it('a filter rejecting everything yields an empty result, not a crash', () => {
    const results = hnsw.findNearest(makeVector(0), 10, {
      filter: () => false,
    });
    expect(results).to.deep.equal([]);
  });

  it('does not throw when a graph node is missing from the backing store', () => {
    // Simulate a dangling edge: the index knows the id, the DB does not.
    const danglingId = 99999;
    (hnsw as any).nodes.set(danglingId, {});
    (hnsw as any).nodeToLevel.set(danglingId, 0);
    const dims = (hnsw as any).dimensionGroups.get(1) ?? (hnsw as any).dimensionGroups.get(DIM);
    if (dims && typeof dims.add === 'function') dims.add(danglingId);
    if (dims && typeof dims.connect === 'function') {
      try {
        dims.connect(danglingId, 0, 0);
      } catch {
        /* graph shape differs; the key assertion is that search does not throw */
      }
    }

    expect(() => hnsw.findNearest(makeVector(0), 10)).to.not.throw();
  });
});
// --- END OF FILE hnsw_filter.test.ts ---
