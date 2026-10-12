import { expect } from 'chai';
import * as os from 'os';
import * as path from 'path';
import { promises as fs } from 'fs';
import { VectorDB } from '../../src/vector/vector_db';
import { ClusteredVectorDB } from '../../src/vector/clustered_vector_db';
import { PartitionedVectorDB } from '../../src/vector/partitioned_vector_db';
import { DistanceMetric } from '../../src/types';
import { getDistanceFunction } from '../../src/utils/distance_metrics';
import { bruteForceNearest, exactIdSequence, expectRecall, referenceDistance, seededVectors, toVectorMap } from '../test-helpers/search-oracle';

/**
 * Correctness oracle for vector search.
 *
 * Before this file every search assertion in the suite was `length > 0` or a
 * property check, so an implementation that returned `k` arbitrary vectors in
 * arbitrary order passed. These tests compare against a naive float64 reference
 * on a seeded dataset: the exact paths must match it id for id, the approximate
 * one must reach a stated recall, and the library's distance kernels must agree
 * with the reference.
 */
describe('search correctness oracle', () => {
  const DIMENSION = 16;
  const COUNT = 500;
  const K = 10;
  const RECALL_FLOOR = 0.9;

  const vectors = seededVectors(COUNT, DIMENSION, 2024);
  const query = vectors[7]!;
  const vectorsById = toVectorMap(vectors);
  const expected = bruteForceNearest([...vectorsById.keys()], vectorsById, query, K);

  describe('distance functions', () => {
    it('agrees with the float64 reference', () => {
      for (const metric of ['euclidean', 'cosine'] as DistanceMetric[]) {
        for (const index of [0, 1, 123, 499]) {
          const library = getDistanceFunction(metric)(vectors[index]!, query);
          const reference = referenceDistance(metric, vectors[index]!, query);
          expect(library, `${metric} distance for vector ${index}`).to.be.closeTo(reference, 1e-5);
        }
      }
    });

    it('is zero on the diagonal and symmetric', () => {
      const metric = getDistanceFunction('euclidean');
      expect(metric(vectors[3]!, vectors[3]!)).to.be.closeTo(0, 1e-6);
      expect(metric(vectors[4]!, vectors[9]!)).to.be.closeTo(metric(vectors[9]!, vectors[4]!), 1e-6);
    });
  });

  describe('VectorDB linear search', () => {
    it('returns the exact nearest neighbours, in order', () => {
      const db = new VectorDB();
      for (const [index, vector] of vectors.entries()) {
        db.addVector(`v${index}`, vector);
      }

      const results = db.findNearest(query, K);
      expect(results.length).to.equal(K);
      // Exact search: the id sequence must equal the reference exactly.
      expect(exactIdSequence(results)).to.equal(exactIdSequence(expected));
      for (const result of results) {
        const reference = expected.find((hit) => String(hit.id) === String(result.id));
        expect(result.dist).to.be.closeTo(reference!.dist, 1e-4);
      }
    });

    it('excludes a filtered-out neighbour without disturbing the rest', () => {
      const db = new VectorDB();
      for (const [index, vector] of vectors.entries()) {
        db.addVector(`v${index}`, vector, { group: index % 2 });
      }

      const results = db.findNearest(query, K, {
        filter: (_id, metadata) => metadata?.['group'] === 0,
      });

      expect(results.length).to.equal(K);
      for (const result of results) {
        const index = Number(String(result.id).slice(1));
        expect(index % 2).to.equal(0);
      }
    });
  });

  describe('ClusteredVectorDB', () => {
    it('reaches the exact neighbours through the clusters', async () => {
      // `nprobe: null` (the default) probes every cluster, so this exercises the
      // clustered code path without the probe limit hiding a correctness bug.
      const db = new ClusteredVectorDB(DIMENSION, null, {});
      for (const [index, vector] of vectors.entries()) {
        db.addVector(`v${index}`, vector);
      }
      await db.runKMeans();

      const results = db.findNearest(query, K);
      expect(results.length).to.equal(K);
      expectRecall(results, expected, K, RECALL_FLOOR, 'ClusteredVectorDB');
    });
  });

  describe('PartitionedVectorDB', () => {
    let tempDir: string;
    let partitionsDir: string;

    beforeEach(async () => {
      tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'nbase-oracle-'));
      partitionsDir = path.join(tempDir, 'partitions');
    });

    afterEach(async () => {
      await fs.rm(tempDir, { recursive: true, force: true });
    });

    it('finds the global nearest neighbours across partitions (linear)', async () => {
      const db = new PartitionedVectorDB({
        partitionsDir,
        vectorSize: DIMENSION,
        partitionCapacity: 50, // ten partitions
        maxActivePartitions: 10, // all resident
        autoCreatePartitions: true,
      });
      await db.initializationPromise;

      for (const [index, vector] of vectors.entries()) {
        await db.addVector(`v${index}`, vector);
      }

      const results = await db.findNearest(query, K);
      expect(results.length).to.equal(K);
      expectRecall(results, expected, K, RECALL_FLOOR, 'PartitionedVectorDB (linear)');
      await db.close();
    });

    it('finds the global nearest neighbours with the HNSW index', async () => {
      const db = new PartitionedVectorDB({
        partitionsDir,
        vectorSize: DIMENSION,
        partitionCapacity: COUNT, // a single partition
        autoCreatePartitions: true,
      });
      await db.initializationPromise;

      for (const [index, vector] of vectors.entries()) {
        await db.addVector(`v${index}`, vector);
      }
      await db.buildIndexHNSW();

      const results = await db.findNearestHNSW(query, K);
      expect(results.length).to.equal(K);
      expectRecall(results, expected, K, RECALL_FLOOR, 'PartitionedVectorDB (HNSW)');
      await db.close();
    });

    it('never returns a deleted vector', async () => {
      const db = new PartitionedVectorDB({
        partitionsDir,
        vectorSize: DIMENSION,
        partitionCapacity: COUNT,
        autoCreatePartitions: true,
      });
      await db.initializationPromise;

      for (const [index, vector] of vectors.entries()) {
        await db.addVector(`v${index}`, vector);
      }
      await db.buildIndexHNSW();

      const nearestId = String(expected[0]!.id);
      expect(await db.deleteVector(nearestId)).to.be.true;

      for (let attempt = 0; attempt < 5; attempt++) {
        const results = await db.findNearestHNSW(query, K);
        expect(results.map((result) => String(result.id))).to.not.include(nearestId);
      }

      await db.close();
    });
  });
});
