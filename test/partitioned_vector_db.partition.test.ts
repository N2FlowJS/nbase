import { expect } from "chai";
import { describe, it, before, beforeEach, after, afterEach } from "mocha";
import { PartitionedVectorDB } from "../src/vector/partitioned_vector_db";
import { ClusteredVectorDB } from "../src/vector/clustered_vector_db";
import {
  TEST_DIR,
  PARTITIONS_DIR,
  setupTestDirectory,
  cleanupPartitionsDir,
  cleanupTestDirectory,
} from "./test-helpers/vector-db-test-utils";

describe("PartitionedVectorDB - Partition Management", () => {
  let db: PartitionedVectorDB;

  before(async () => {
    await setupTestDirectory();
  });

  beforeEach(async () => {
    db = new PartitionedVectorDB({
      partitionsDir: PARTITIONS_DIR,
      vectorSize: 10,
      partitionCapacity: 1000,
      maxActivePartitions: 2,
    });
    await db.initializationPromise;
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

  it("should create a new partition", async () => {
    const partitionId = await db.createPartition(
      "test-partition",
      "Test Partition",
      {
        description: "A test partition",
        setActive: true,
      }
    );

    expect(partitionId).to.equal("test-partition");

    const configs = db.getPartitionConfigs();

    expect(configs.length).to.equal(2);
    expect(configs[0].id).to.include("p-");
    expect(configs[0].name).to.equal("Initial Partition");
    expect(configs[0].active).to.be.false;

    expect(configs[1].id).to.equal("test-partition");
    expect(configs[1].name).to.equal("Test Partition");
    expect(configs[1].active).to.be.true;

    const activePartition = await db.getActivePartition();
    expect(activePartition).to.be.an.instanceOf(ClusteredVectorDB);
  });

  it("should set active partition", async () => {
    await db.createPartition("partition1", "Partition 1");
    await db.createPartition("partition2", "Partition 2");

    await db.setActivePartition("partition2");

    const configs = db.getPartitionConfigs();
    const activeConfig = configs.find((config) => config.active);

    expect(activeConfig).to.exist;
    expect(activeConfig?.id).to.equal("partition2");

    const activePartition = await db.getActivePartition();
    expect(activePartition).to.not.be.null;
  });

  it("should load a specific partition", async () => {
    await db.createPartition("partition1", "Partition 1");
    await db.createPartition("partition2", "Partition 2");

    const partition = await db.getPartition("partition1");
    expect(partition).to.not.be.null;

    const stats = await db.getStats();
    expect(stats.partitions.loadedCount).to.be.at.least(1);
  });
});
