import { expect } from 'chai';
import HNSW from '../../src/ann/hnsw';
import { ClusteredVectorDB } from '../../src/vector/clustered_vector_db';
import { bruteForceNearest, expectRecall, recallAt, seededVectors, toVectorMap } from '../test-helpers/search-oracle';

/**
 * HNSW-specific correctness, measured against the same brute-force oracle the
 * other search tests use.
 *
 * Four options on `VectorStoreSearchOptions` were accepted and ignored. These
 * tests pin the behaviour that is now real: the accuracy dial (`efSearch`),
 * the metric override, the per-dimension entry point, and the interaction
 * between `exactDimensions` and `buildIndex()`.
 */
describe('HNSW search options', () => {
  describe('efSearch', () => {
    // A dataset where the beam width visibly changes the ranking. On others the
    // graph is navigable enough that every setting finds the same 10 — which is
    // why the assertion is "the result set differs", not "recall is high".
    const vectors = seededVectors(800, 64, 818);
    const query = vectors[3]!;
    const vectorsById = toVectorMap(vectors);

    let hnsw: HNSW;
    let built = false;

    before(async function () {
      this.timeout(30000);
      const db = new ClusteredVectorDB(64, null, {});
      for (const [index, vector] of vectors.entries()) {
        db.addVector(`v${index}`, vector);
      }
      hnsw = new HNSW(db, {});
      await hnsw.buildIndex({});
      built = true;
    });

    it('changes the beam width instead of being ignored', () => {
      expect(built).to.be.true;
      const greedy = hnsw.findNearest(query, 10, { efSearch: 1 }).map((result) => String(result.id));
      const thorough = hnsw.findNearest(query, 10, { efSearch: 300 }).map((result) => String(result.id));

      // `ef` used to be `max(k, this.efSearch)` regardless of the request, so
      // these two lists were always identical.
      expect(greedy.join(',')).to.not.equal(thorough.join(','));

      const expected = bruteForceNearest([...vectorsById.keys()], vectorsById, query, 10);
      // A wider beam must not be worse than a greedy one.
      const thoroughHits = thorough.map((id) => ({ id }));
      const greedyHits = greedy.map((id) => ({ id }));
      expect(recallAt(thoroughHits, expected, 10)).to.be.at.least(recallAt(greedyHits, expected, 10));
      expectRecall(thoroughHits, expected, 10, 0.95, 'efSearch: 300');
    });

    it('falls back to the constructor default when no override is given', () => {
      const withDefault = hnsw.findNearest(query, 10).map((result) => String(result.id));
      expect(withDefault.length).to.equal(10);
    });
  });

  describe('distanceMetric', () => {
    const vectors = seededVectors(200, 8, 77);
    const vectorsById = toVectorMap(vectors);
    const query = vectors[3]!;

    let hnsw: HNSW;

    before(async () => {
      const db = new ClusteredVectorDB(8, null, {});
      for (const [index, vector] of vectors.entries()) {
        db.addVector(`v${index}`, vector);
      }
      hnsw = new HNSW(db, {});
      await hnsw.buildIndex({});
    });

    it('honours a cosine request instead of serving Euclidean distances', () => {
      const euclidean = hnsw.findNearest(query, 10, { distanceMetric: 'euclidean' }).map((result) => String(result.id));
      const cosine = hnsw.findNearest(query, 10, { distanceMetric: 'cosine' }).map((result) => String(result.id));

      // The distance function is fixed at construction, so both used to be the
      // Euclidean ranking.
      expect(cosine.join(',')).to.not.equal(euclidean.join(','));

      const cosineExpected = bruteForceNearest([...vectorsById.keys()], vectorsById, query, 10, 'cosine');
      expectRecall(
        cosine.map((id) => ({ id })),
        cosineExpected,
        10,
        0.95,
        'cosine request',
      );
    });

    it('accepts the `metric` spelling too', () => {
      const viaMetric = hnsw.findNearest(query, 10, { metric: 'cosine' }).map((result) => String(result.id));
      const viaDistanceMetric = hnsw.findNearest(query, 10, { distanceMetric: 'cosine' }).map((result) => String(result.id));
      expect(viaMetric.join(',')).to.equal(viaDistanceMetric.join(','));
    });
  });

  describe('dimension handling', () => {
    let hnsw: HNSW;

    before(async () => {
      const db = new ClusteredVectorDB(8, null, {});
      // Two dimensions in one store, which is what `dimensionAware` is for.
      for (let i = 0; i < 60; i++) {
        db.addVector(`d3_${i}`, new Float32Array([i / 10, i / 20, i / 30]));
      }
      for (let i = 0; i < 60; i++) {
        db.addVector(`d5_${i}`, new Float32Array([i / 10, i / 20, i / 30, i / 40, i / 50]));
      }
      hnsw = new HNSW(db, {});
      await hnsw.buildIndex({});
    });

    it('keeps a query inside its own dimension', () => {
      const results = hnsw.findNearest(new Float32Array([3, 2, 1]), 10);

      expect(results.length).to.be.greaterThan(0);
      // The descent used to start at the global entry point, whichever
      // dimension happened to be indexed first.
      for (const result of results) {
        expect(String(result.id)).to.match(/^d3_/);
      }
    });

    it('returns results for exactDimensions after buildIndex', () => {
      // `dimensionEntryPoints` was only ever written by the `dimensionSet.size
      // === 1` branch in `addVector`, and `buildIndex`'s pre-scan had already
      // filled every group — so the map stayed empty and this option answered
      // `[]` for every query on a freshly built index.
      const results = hnsw.findNearest(new Float32Array([3, 2, 1]), 10, { exactDimensions: true });
      expect(results.length).to.equal(10);
      for (const result of results) {
        expect(String(result.id)).to.match(/^d3_/);
      }
    });

    it('still answers [] for a dimension the index never saw', () => {
      const results = hnsw.findNearest(new Float32Array([1, 1, 1, 1, 1, 1, 1]), 10, { exactDimensions: true });
      expect(results).to.deep.equal([]);
    });
  });

  describe('tombstones', () => {
    it('survives a serialize/deserialize round trip', async () => {
      const vectors = seededVectors(120, 8, 1357);
      const db = new ClusteredVectorDB(8, null, {});
      for (const [index, vector] of vectors.entries()) {
        db.addVector(`v${index}`, vector);
      }
      const hnsw = new HNSW(db, {});
      await hnsw.buildIndex({});

      const deleted = 'v3';
      expect(hnsw.markDelete(deleted)).to.be.true;
      expect(hnsw.getStats().deletedNodesCount).to.equal(1);

      const restored = HNSW.deserialize(hnsw.serialize(), db);
      expect(restored.getStats().deletedNodesCount).to.equal(1);

      // And the reloaded index still refuses to return it.
      const results = restored.findNearest(vectors[3]!, 10);
      expect(results.map((result) => String(result.id))).to.not.include(deleted);
    });
  });

  describe('zero-valued ids', () => {
    it('indexes and searches when the first id is 0', async () => {
      const db = new ClusteredVectorDB(4, null, {});
      for (let i = 0; i < 40; i++) {
        db.addVector(i, new Float32Array([i, i + 1, i + 2, i + 3]));
      }
      const hnsw = new HNSW(db, {});
      await hnsw.buildIndex({});

      // `if (!this.entryPointId)` and `if (dimEntryPoint)` are truthiness
      // tests, and `0` is a valid id: the index silently skipped itself and
      // answered from the linear fallback.
      const results = hnsw.findNearest(new Float32Array([0, 1, 2, 3]), 5);
      expect(results.length).to.equal(5);
      expect(String(results[0]?.id)).to.equal('0');
    });
  });
});
