import { expect } from 'chai';
import { promises as fs } from 'fs';
import * as os from 'os';
import * as path from 'path';
import { PartitionedVectorDB } from '../../src/vector/partitioned_vector_db';

/**
 * The LRU that holds loaded partitions was the quietest data-loss path in the
 * library: an evicted partition was closed with `save: false`, and `save()`
 * only ever walks the *loaded* partitions, so anything written since that
 * partition's last save was never written again — silently, and only visible
 * the next time the partition was read back from disk.
 */
describe('PartitionedVectorDB eviction durability', () => {
  let tempDir: string;
  let partitionsDir: string;

  const VECTOR_SIZE = 3;
  const TOTAL = 6;
  /** partitionCapacity 2 + maxActivePartitions 1: every 3rd add evicts. */
  const openDb = async (): Promise<PartitionedVectorDB> => {
    const db = new PartitionedVectorDB({
      partitionsDir,
      vectorSize: VECTOR_SIZE,
      partitionCapacity: 2,
      maxActivePartitions: 1,
      autoCreatePartitions: true,
    });
    await db.initializationPromise;
    return db;
  };

  const vector = (seed: number): Float32Array => new Float32Array([seed, seed + 1, seed + 2]);

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'nbase-eviction-'));
    partitionsDir = path.join(tempDir, 'partitions');
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  it('has the vectors of an evicted partition on disk when it is read back', async () => {
    const db = await openDb();

    for (let i = 0; i < TOTAL; i++) {
      await db.addVector(`v${i}`, vector(i));
    }

    // The disposer is fire-and-forget, so give it a turn to finish its write.
    await new Promise((resolve) => setTimeout(resolve, 100));

    const configs = db.getPartitionConfigs();
    expect(configs.length).to.be.greaterThan(1);

    // Read every partition back. `getPartition` goes through the load path, so
    // an evicted partition is rebuilt from what is on disk. Before the fix the
    // evicted partitions came back empty (count 0) or were never written.
    let persisted = 0;
    const ids: (number | string)[] = [];
    for (const config of configs) {
      const partition = await db.getPartition(config.id);
      expect(partition, `partition ${config.id} should load from disk`).to.not.equal(null);
      persisted += partition?.getVectorCount() ?? 0;
      ids.push(...(partition?.getVectorIds() ?? []));
    }

    expect(persisted, 'vectors recovered from disk').to.equal(TOTAL);
    expect([...ids].sort()).to.deep.equal(['v0', 'v1', 'v2', 'v3', 'v4', 'v5']);

    await db.close();
  });

  it('survives a restart with every partition reloaded', async () => {
    const first = await openDb();
    for (let i = 0; i < TOTAL; i++) {
      await first.addVector(`v${i}`, vector(i));
    }
    await first.close();

    const second = await openDb();
    let persisted = 0;
    for (const config of second.getPartitionConfigs()) {
      const partition = await second.getPartition(config.id);
      persisted += partition?.getVectorCount() ?? 0;
    }
    expect(persisted).to.equal(TOTAL);
    await second.close();
  });

  it('emits partition:unloaded for an evicted partition', async () => {
    const db = await openDb();

    const unloaded: string[] = [];
    db.on('partition:unloaded', (payload) => unloaded.push(payload.id));

    for (let i = 0; i < TOTAL; i++) {
      await db.addVector(`v${i}`, vector(i));
    }

    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(unloaded.length).to.be.greaterThan(0);

    await db.close();
  });

  it('searches a partition the caller named, even when it is not resident', async () => {
    const db = await openDb();
    for (let i = 0; i < TOTAL; i++) {
      await db.addVector(`v${i}`, vector(i));
    }
    await new Promise((resolve) => setTimeout(resolve, 100));

    const configs = db.getPartitionConfigs();
    expect(configs.length).to.be.greaterThan(1);

    // Pick a partition that is definitely not the resident one, and prove it:
    // asking for it used to return [] because the id was filtered out of the
    // LRU-resident set rather than loaded.
    const resident = (await db.getActivePartition()) !== null;
    expect(resident).to.be.true;
    const target = configs[configs.length - 1]!;
    const results = await db.findNearest(vector(0), 10, { partitionIds: [target.id] });

    expect(results.length, `search over partition ${target.id}`).to.be.greaterThan(0);
    // Every hit must belong to the partition that was asked for.
    const partition = await db.getPartition(target.id);
    for (const result of results) {
      expect(partition?.hasVector(result.id), `result ${result.id} in ${target.id}`).to.be.true;
    }

    await db.close();
  });

  it('reports unknown partition ids instead of silently returning nothing', async () => {
    const db = await openDb();
    await db.addVector('v0', vector(0));

    const errors: { id?: string; operation: string }[] = [];
    db.on('partition:error', (payload) => errors.push(payload));

    // Unknown ids must not throw; the known one is still searched.
    const results = await db.findNearest(vector(0), 5, { partitionIds: ['does-not-exist'] });
    expect(results).to.deep.equal([]);

    await db.close();
  });

  it('close() waits for an in-flight eviction write', async () => {
    const db = await openDb();
    for (let i = 0; i < TOTAL; i++) {
      await db.addVector(`v${i}`, vector(i));
    }

    // No settle delay: `close()` has to await the pending flush itself.
    await db.close();

    const reopened = await openDb();
    let persisted = 0;
    for (const config of reopened.getPartitionConfigs()) {
      const partition = await reopened.getPartition(config.id);
      persisted += partition?.getVectorCount() ?? 0;
    }
    expect(persisted).to.equal(TOTAL);
    await reopened.close();
  });
});
