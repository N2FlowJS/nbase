import { expect } from "chai";
import { promises as fs } from "fs";
import * as path from "path";
import { ClusteredVectorDB } from "../src/vector/clustered_vector_db";
import { SearchResult, Vector } from "../src/types"; // Corrected import path
import * as os from "os";

describe("ClusteredVectorDB", () => {
  const tempDir = path.join(
    os.tmpdir(),
    `test-clustered-vector-db-${Date.now()}`
  );
  const vectorDimension = 4; // Small dimension for tests

  // Helper to create test vectors
  const createTestVector = (values: number[]): Vector => {
    return new Float32Array(values);
  };

  beforeEach(async () => {
    await fs.mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    try {
      await fs.rm(tempDir, { recursive: true, force: true });
    } catch (err) {
      console.error("Error cleaning up temp directory:", err);
    }
  });

  describe("Construction and Configuration", () => {
    it("should initialize with default parameters", () => {
      const db = new ClusteredVectorDB(vectorDimension);
      expect(db.targetClusterSize).to.exist;
      expect(db.getDistanceMetric()).to.equal("euclidean");
    });

    it("should initialize with custom parameters", () => {
      const db = new ClusteredVectorDB(vectorDimension, null, {
        clusterSize: 50,
        distanceMetric: "cosine",
        maxClusters: 100,
      });
      expect(db.targetClusterSize).to.equal(50);
    });
  });

  describe("Vector Operations", () => {
    it("should add vectors and assign to clusters", () => {
      const db = new ClusteredVectorDB(vectorDimension);

      // Add multiple vectors
      const id1 = db.addVector(1, createTestVector([1, 2, 3, 4]));
      const id2 = db.addVector(2, createTestVector([1.1, 2.1, 3.1, 4.1]));
      const id3 = db.addVector(3, createTestVector([5, 6, 7, 8]));

      // Get cluster info to check assignments
      const clusters = db.getClusterInfo();

      // Should have created at least 1 cluster (could be 1 or 2 depending on distance threshold)
      expect(clusters.length).to.be.at.least(1);

      // Check total vectors across all clusters
      let totalVectors = 0;
      clusters.forEach((cluster) => {
        totalVectors += cluster.size;
      });
      expect(totalVectors).to.equal(3);
    });

    it("should delete vectors and update clusters", () => {
      const db = new ClusteredVectorDB(vectorDimension);

      // Add vectors
      const id1 = db.addVector(1, createTestVector([1, 2, 3, 4]));
      const id2 = db.addVector(2, createTestVector([1.1, 2.1, 3.1, 4.1]));

      // Get initial stats
      const initialClusters = db.getClusterInfo();
      const initialTotalSize = initialClusters.reduce(
        (sum, c) => sum + c.size,
        0
      );
      expect(initialTotalSize).to.equal(2);

      // Delete one vector
      const deleted = db.deleteVector(id1);
      expect(deleted).to.be.true;

      // Get updated stats
      const updatedClusters = db.getStats();

      const updatedTotalSize = updatedClusters.vectorCount;
      expect(updatedTotalSize).to.equal(1);
    });

    it("should find nearest vectors using clusters", () => {
      const db = new ClusteredVectorDB(vectorDimension);

      // Add some vectors forming distinct clusters
      const closerVectors = [
        [1, 2, 3, 4],
        [1.1, 2.1, 3.1, 4.1],
        [1.2, 2.2, 3.2, 4.2],
      ];

      const fartherVectors = [
        [10, 20, 30, 40],
        [11, 21, 31, 41],
        [12, 22, 32, 42],
      ];

      closerVectors.forEach((v, i) =>
        db.addVector(`close-${i}`, createTestVector(v))
      );
      fartherVectors.forEach((v, i) =>
        db.addVector(`far-${i}`, createTestVector(v))
      );

      // Search for vector close to the first cluster
      const query = createTestVector([1.5, 2.5, 3.5, 4.5]);
      const results = db.findNearest(query, 3);

      // Should find vectors from the closer cluster
      expect(results.length).to.equal(3);
      results.forEach((result) => {
        expect(result.id.toString()).to.include("close-");
      });
    });
  });

  describe("Cluster Management", () => {
    it("should create multiple clusters for distant vectors", () => {
      const db = new ClusteredVectorDB(vectorDimension, null, {
        clusterSize: 2, // Small size to force new clusters
        newClusterDistanceThreshold: 0.1, // Low threshold to force clustering
      });

      // Add vectors from two distinct regions
      db.addVector(1, createTestVector([1, 2, 3, 4]));
      db.addVector(2, createTestVector([1.01, 2.01, 3.01, 4.01]));
      db.addVector(3, createTestVector([10, 20, 30, 40]));
      db.addVector(4, createTestVector([10.01, 20.01, 30.01, 40.01]));

      const clusters = db.getClusterInfo();
      expect(clusters.length).to.be.at.least(2);
    });

    it("should update cluster centroids when adding vectors", () => {
      // Create DB with higher cluster threshold to ensure vectors go in same cluster
      const db = new ClusteredVectorDB(vectorDimension, null, {
        newClusterDistanceThreshold: 10, // Much higher threshold to force vectors into same cluster
        clusterSize: 10, // Large enough size to not trigger new cluster creation
      });

      // Add a vector and capture the initial centroid
      db.addVector(1, createTestVector([1, 2, 3, 4]));
      const initialClusters = db.getClusterInfo();
      const initialCentroid = Array.from(initialClusters[0].centroid);

      // Add a second vector that's very similar to ensure it goes in the same cluster
      db.addVector(2, createTestVector([1.1, 2.1, 3.1, 4.1]));
      const updatedClusters = db.getClusterInfo();
      const updatedCentroid = Array.from(updatedClusters[0].centroid);

      // Verify clusters are functioning correctly
      expect(updatedClusters.length).to.equal(
        1,
        "Should have exactly one cluster"
      );
      expect(updatedClusters[0].size).to.equal(
        2,
        "Cluster should contain both vectors"
      );

      // Centroid should have changed (it's the average)
      expect(updatedCentroid).to.not.deep.equal(initialCentroid);

      // Verify centroid calculation: (vector1 + vector2) / 2
      const expectedCentroid = [1.05, 2.05, 3.05, 4.05]; // Average of [1,2,3,4] and [1.1,2.1,3.1,4.1]
      updatedCentroid.forEach((value, i) => {
        expect(value).to.be.closeTo(expectedCentroid[i], 0.001);
      });
    });
  });

  describe("Persistence", () => {
    it("should save and load cluster state", async () => {
      // Create and populate DB
      const dbPath = path.join(tempDir, "cluster-save-test");
      const db = new ClusteredVectorDB(vectorDimension, dbPath);

      db.addVector(1, createTestVector([1, 2, 3, 4]));
      db.addVector(2, createTestVector([5, 6, 7, 8]));
      db.addVector(3, createTestVector([9, 10, 11, 12]));

      const originalClusters = db.getClusterInfo();
      await db.save();
      await db.close();

      // Load into a new DB instance
      const loadedDb = new ClusteredVectorDB(null, dbPath);
      await loadedDb.load();

      const loadedClusters = loadedDb.getClusterInfo();

      // Basic validation of loaded state
      expect(loadedClusters.length).to.equal(originalClusters.length);
      expect(loadedDb.getStats().clusters.count).to.equal(3);

      await loadedDb.close();
    });

    it("should rebuild clusters if cluster state is missing", async () => {
      // Create and populate DB
      const dbPath = path.join(tempDir, "rebuild-test");
      const db = new ClusteredVectorDB(vectorDimension, dbPath);

      db.addVector(1, createTestVector([1, 2, 3, 4]));
      db.addVector(2, createTestVector([5, 6, 7, 8]));

      // Save only the base data by manipulating the save method
      await fs.mkdir(dbPath, { recursive: true });
      await db.save();
      await db.close();

      // Delete cluster state file but keep vector data
      const clusterFile = path.join(dbPath, "cluster.json");
      try {
        await fs.unlink(clusterFile);
      } catch (err) {
        // Might be compressed or another format, try alternative
        try {
          await fs.unlink(`${clusterFile}.gz`);
        } catch (e) {
          console.warn("Could not find cluster file to delete:", e);
        }
      }

      // Load into a new DB instance, should rebuild clusters
      const loadedDb = new ClusteredVectorDB(null, dbPath);
      await loadedDb.load();

      // Should have rebuilt clusters
      const loadedClusters = loadedDb.getClusterInfo();
      expect(loadedClusters.length).to.be.at.least(1);

      // Search should still work
      const results = loadedDb.findNearest(createTestVector([1, 2, 3, 4]), 1);
      expect(results.length).to.equal(1);

      await loadedDb.close();
    });
  });

  describe("Stats and Info", () => {
    it("should provide accurate cluster statistics", () => {
      const db = new ClusteredVectorDB(vectorDimension);

      // Add vectors
      db.addVector(1, createTestVector([1, 2, 3, 4]));
      db.addVector(2, createTestVector([1.1, 2.1, 3.1, 4.1]));
      db.addVector(3, createTestVector([10, 20, 30, 40]));

      const stats = db.getStats();

      expect(stats.vectorCount).to.equal(3); // Total vectors
      expect(stats.clusters).to.exist;
      expect(stats.clusters!.count).to.be.at.least(1); // At least one cluster

      // Check distribution details
      stats.clusters!.distribution.forEach((cluster) => {

        expect(cluster.id).to.be.a("number");
        expect(cluster.size).to.be.a("number");
        expect(cluster.dimension).to.equal(vectorDimension);
      });
    });

    it("should return cluster information", () => {
      const db = new ClusteredVectorDB(vectorDimension);

      db.addVector(1, createTestVector([1, 2, 3, 4]));
      db.addVector(2, createTestVector([1.1, 2.1, 3.1, 4.1]));

      const clusterInfo = db.getClusterInfo();

      expect(clusterInfo.length).to.be.at.least(1);
      clusterInfo.forEach((cluster) => {
        expect(cluster.id).to.be.a("number");
        expect(cluster.size).to.be.a("number");
        expect(cluster.dimension).to.equal(vectorDimension);
        expect(cluster.centroid).to.be.instanceof(Float32Array);
        expect(cluster.centroid.length).to.equal(vectorDimension);
      });
    });
  });

  describe("K-Means Clustering", () => {
    it("should run K-Means without errors on a populated DB", async () => {
      const db = new ClusteredVectorDB(vectorDimension);
      db.addVector(1, createTestVector([1, 2, 3, 4]));
      db.addVector(2, createTestVector([1.1, 2.1, 3.1, 4.1]));
      db.addVector(3, createTestVector([10, 20, 30, 40]));
      db.addVector(4, createTestVector([10.1, 20.1, 30.1, 40.1]));

      const initialClusters = db.getClusterInfo();
      await db.runKMeans(); // Run with default k (current cluster count)
      const finalClusters = db.getClusterInfo();

      // K-Means might or might not change the number of clusters depending on initialization and data
      expect(finalClusters.length).to.be.at.least(1);
      // Check if centroids are valid
      finalClusters.forEach(c => expect(c.centroid).to.be.instanceof(Float32Array));
    });

    it("should run K-Means with a specific k value", async () => {
      const db = new ClusteredVectorDB(vectorDimension);
      // Add vectors that should ideally form 2 clusters
      db.addVector(1, createTestVector([1, 2, 3, 4]));
      db.addVector(2, createTestVector([1.1, 2.1, 3.1, 4.1]));
      db.addVector(3, createTestVector([10, 20, 30, 40]));
      db.addVector(4, createTestVector([10.1, 20.1, 30.1, 40.1]));
      db.addVector(5, createTestVector([0.9, 1.9, 2.9, 3.9]));
      db.addVector(6, createTestVector([9.9, 19.9, 29.9, 39.9]));

      await db.runKMeans(2); // Request exactly 2 clusters
      const finalClusters = db.getClusterInfo();
      const finalStats = db.getStats(); // Get stats after K-Means

      // Depending on random initialization, it might not always achieve exactly k,
      // but it should attempt to create k clusters. Check if it's close or equal.
      expect(finalClusters.length).to.be.at.most(2); // Should not exceed k
      // A more robust test might check the distribution of points if k is achieved.
      if (finalClusters.length === 2 && finalStats.clusters?.distribution) {
        // Check if vectors are roughly assigned correctly (this is probabilistic)
        const cluster1Info = finalStats.clusters.distribution.find(c => c.id === finalClusters[0].id);
        const cluster2Info = finalStats.clusters.distribution.find(c => c.id === finalClusters[1].id);

        const cluster1Members = new Set(cluster1Info?.members?.map(m => m.id) ?? []);
        const cluster2Members = new Set(cluster2Info?.members?.map(m => m.id) ?? []);

        // This check is simplified and might fail due to randomness
        // Verify that the total number of members across clusters matches the total vectors
        expect(cluster1Members.size + cluster2Members.size).to.equal(6);

        // Example check (might need adjustment based on K-Means behavior):
        // Check if vector 1 is in one of the clusters
        // expect(cluster1Members.has(1) || cluster2Members.has(1)).to.be.true;
      }
    });

    it("should handle K-Means on an empty database", async () => {
      const db = new ClusteredVectorDB(vectorDimension);
      await db.runKMeans(); // Should not throw error
      expect(db.getClusterInfo().length).to.equal(0);
    });

    it("should handle K-Means when k is larger than the number of vectors", async () => {
      const db = new ClusteredVectorDB(vectorDimension);
      db.addVector(1, createTestVector([1, 2, 3, 4]));
      db.addVector(2, createTestVector([5, 6, 7, 8]));

      await db.runKMeans(5); // Request more clusters than vectors
      const clusters = db.getClusterInfo();

      // K-Means should create at most as many clusters as there are distinct vectors
      expect(clusters.length).to.be.at.most(2);
    });

    // Note: Testing convergence precisely is difficult due to randomness.
    // We mainly test that it runs and produces a plausible result.
  });
});
