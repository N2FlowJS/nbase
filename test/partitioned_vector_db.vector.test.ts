import { expect } from "chai";
import { describe, it, before, beforeEach, after, afterEach } from "mocha";
import { PartitionedVectorDB } from "../src/vector/partitioned_vector_db";
import {
  TEST_DIR,
  PARTITIONS_DIR,
  generateRandomVector,
  createTestVectors,
  setupTestDirectory,
  cleanupPartitionsDir,
  cleanupTestDirectory,
} from "./test-helpers/vector-db-test-utils";

describe("PartitionedVectorDB - Vector Operations", () => {
  const VECTOR_SIZE = 10;
  let db: PartitionedVectorDB;

  before(async () => {
    await setupTestDirectory();
  });

  beforeEach(async () => {
    db = new PartitionedVectorDB({
      partitionsDir: PARTITIONS_DIR,
      vectorSize: VECTOR_SIZE,
      partitionCapacity: 1000,
    });
    await db.initializationPromise;
    await db.createPartition("test-partition", "Test Partition", {
      setActive: true,
    });
  });

  afterEach(async () => {
    if (db) {
      await db.close();
    }
    await cleanupPartitionsDir();
  });

  after(async () => {
    await cleanupTestDirectory();
  });

  it("should add a vector and retrieve it", async () => {
    const vector = generateRandomVector(VECTOR_SIZE);
    const metadata = { label: "test-item", category: "test" };

    const result = await db.addVector("test-id", vector, metadata);
    expect(result.partitionId).to.equal("test-partition");
    expect(result.vectorId).to.equal("test-id");

    const retrieved = await db.getVector("test-id");
    expect(retrieved).to.not.be.null;
    expect(retrieved?.partitionId).to.equal("test-partition");

    const retrievedMetadata = await db.getMetadata("test-id");
    expect(retrievedMetadata?.metadata).to.deep.include(metadata);
  });

  it("should bulk add vectors", async () => {
    const vectors = createTestVectors(50, VECTOR_SIZE);

    const result = await db.bulkAdd(vectors);
    expect(result.count).to.equal(50);

    const stats = await db.getStats();
    const partitionConfig = db.getPartitionConfigs();
    let vectorCount = 0;
    partitionConfig.forEach((partition) => {
      vectorCount += partition.vectorCount;
    });

    expect(vectorCount).to.equal(50);
    expect(partitionConfig.length).to.equal(2);
  });

  it("should delete a vector", async () => {
    const vector = generateRandomVector(VECTOR_SIZE);
    await db.addVector("delete-test", vector, { test: true });

    const deleted = await db.deleteVector("delete-test");
    expect(deleted).to.be.true;

    const retrieved = await db.getVector("delete-test");
    expect(retrieved).to.be.null;
  });

  it("should update vector metadata", async () => {
    const vector = generateRandomVector(VECTOR_SIZE);
    await db.addVector("update-test", vector, { initial: true });

    const updated = await db.updateMetadata("update-test", { updated: true });
    expect(updated).to.be.true;

    const metadata = await db.getMetadata("update-test");
    expect(metadata?.metadata).to.have.property("updated", true);
  });
});
