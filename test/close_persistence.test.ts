// --- START OF FILE close_persistence.test.ts ---

// Regression tests for the "close() silently discards everything" defect.
//
// `VectorDB.close()` and `PartitionedVectorDB.close()` used to set their
// `isClosed` / `isClosing` flag *before* awaiting the final `save()`. The
// `save()` guard then bailed out on that same flag, so the "final save" never
// wrote a single byte. `PartitionedVectorDB.close()` additionally had an
// inverted `if (this.isInitialized) return;` guard, making close a no-op for a
// healthy database.

import { describe, it } from 'mocha';
import { expect } from 'chai';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { VectorDB } from '../src/vector/vector_db';
import { ClusteredVectorDB } from '../src/vector/clustered_vector_db';
import { PartitionedVectorDB } from '../src/vector/partitioned_vector_db';

function makeTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), `nbase-${prefix}-`));
}

describe('close() persistence', () => {
  describe('VectorDB', () => {
    it('flushes pending vectors to disk on close()', async () => {
      const dir = makeTempDir('vectordb-close');
      const dbPath = path.join(dir, 'data');

      const db = new VectorDB(3, dbPath);
      await db.load();
      db.addVector(1, [1, 0, 0]);
      db.addVector(2, [0, 1, 0]);
      db.addVector(3, [0, 0, 1]);

      // No explicit save() — close() alone must persist.
      await db.close();

      const reopened = new VectorDB(3, dbPath);
      await reopened.load();
      expect(reopened.getVectorCount()).to.equal(3);

      reopened.close({ save: false });
      fs.rmSync(dir, { recursive: true, force: true });
    });

    it('is idempotent and safe to call twice', async () => {
      const dir = makeTempDir('vectordb-close-twice');
      const db = new VectorDB(3, path.join(dir, 'data'));
      await db.load();
      db.addVector(1, [1, 1, 1]);

      await db.close();
      await db.close(); // must not throw

      const reopened = new VectorDB(3, path.join(dir, 'data'));
      await reopened.load();
      expect(reopened.getVectorCount()).to.equal(1);

      reopened.close({ save: false });
      fs.rmSync(dir, { recursive: true, force: true });
    });

    it('survives concurrent close() calls without throwing', async () => {
      const dir = makeTempDir('vectordb-close-concurrent');
      const db = new VectorDB(3, path.join(dir, 'data'));
      await db.load();
      db.addVector(1, [2, 2, 2]);

      await Promise.allSettled([db.close(), db.close(), db.close()]);

      const reopened = new VectorDB(3, path.join(dir, 'data'));
      await reopened.load();
      expect(reopened.getVectorCount()).to.equal(1);

      reopened.close({ save: false });
      fs.rmSync(dir, { recursive: true, force: true });
    });

    it('close({ save: false }) does not write to disk', async () => {
      const dir = makeTempDir('vectordb-close-nosave');
      const db = new VectorDB(3, path.join(dir, 'data'));
      await db.load();
      db.addVector(1, [9, 9, 9]);

      await db.close({ save: false });

      const reopened = new VectorDB(3, path.join(dir, 'data'));
      await reopened.load();
      expect(reopened.getVectorCount()).to.equal(0);

      reopened.close({ save: false });
      fs.rmSync(dir, { recursive: true, force: true });
    });
  });

  describe('ClusteredVectorDB', () => {
    it('persists vectors and cluster state on close()', async () => {
      const dir = makeTempDir('clustered-close');
      const dbPath = path.join(dir, 'data');

      const db = new ClusteredVectorDB(3, dbPath, { clusterSize: 2 });
      if (db.readyPromise) await db.readyPromise;
      for (let i = 0; i < 6; i++) {
        db.addVector(i, [i, i + 1, i + 2]);
      }

      await db.close();

      const reopened = new ClusteredVectorDB(3, dbPath, { clusterSize: 2 });
      if (reopened.readyPromise) await reopened.readyPromise;
      expect(reopened.getVectorCount()).to.equal(6);

      reopened.close({ save: false });
      fs.rmSync(dir, { recursive: true, force: true });
    });

    it('close({ save: false }) leaves the previous snapshot intact', async () => {
      const dir = makeTempDir('clustered-close-nosave');
      const dbPath = path.join(dir, 'data');

      const db = new ClusteredVectorDB(3, dbPath, { clusterSize: 2 });
      if (db.readyPromise) await db.readyPromise;
      db.addVector(1, [1, 1, 1]);
      await db.close(); // persists 1 vector

      const second = new ClusteredVectorDB(3, dbPath, { clusterSize: 2 });
      if (second.readyPromise) await second.readyPromise;
      second.addVector(2, [2, 2, 2]);
      await second.close({ save: false }); // must not overwrite

      const reopened = new ClusteredVectorDB(3, dbPath, { clusterSize: 2 });
      if (reopened.readyPromise) await reopened.readyPromise;
      expect(reopened.getVectorCount()).to.equal(1);

      reopened.close({ save: false });
      fs.rmSync(dir, { recursive: true, force: true });
    });
  });

  describe('PartitionedVectorDB', () => {
    it('close() on a healthy database actually closes and persists', async () => {
      const dir = makeTempDir('partdb-close');
      const db = new PartitionedVectorDB({ partitionsDir: dir, vectorSize: 3, autoCreatePartitions: false });
      await db.initializationPromise;

      await db.createPartition('p1', 'P1', { clusterSize: 2 });
      await db.addVector(1, [1, 0, 0]);
      await db.addVector(2, [0, 1, 0]);

      expect(db.IsReady()).to.equal(true);
      await db.close();
      // The inverted guard used to leave this true.
      expect(db.IsReady()).to.equal(false);

      const reopened = new PartitionedVectorDB({ partitionsDir: dir, vectorSize: 3, autoCreatePartitions: false });
      await reopened.initializationPromise;
      // Assert against the reloaded partition's own data, which is what the
      // final save is actually responsible for persisting.
      const partition = await reopened.getPartition('p1');
      expect(partition).to.not.equal(null);
      expect(partition!.getVectorCount()).to.equal(2);
      expect(await reopened.getVector(1)).to.not.equal(null);
      expect(await reopened.getVector(2)).to.not.equal(null);
      await reopened.close();

      fs.rmSync(dir, { recursive: true, force: true });
    });

    it('close() before initialization is a no-op that does not throw', async () => {
      const dir = makeTempDir('partdb-close-uninit');
      const db = new PartitionedVectorDB({ partitionsDir: dir, vectorSize: 3, autoCreatePartitions: false });
      // Intentionally not awaiting initialization.
      await db.close();
      fs.rmSync(dir, { recursive: true, force: true });
    });
  });
});
// --- END OF FILE close_persistence.test.ts ---
