import { expect } from 'chai';
import { promises as fs } from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Database } from '../../src/database/database';
import { DatabaseOptions } from '../../src/types';
import { VectorDBMonitor } from '../../src/utils/vector_monitoring';
import { MAX_BATCH_QUERIES } from '../../src/server/utils/validation';
import configDefaults from '../../src/config';

/**
 * The `Database` facade is the entry point every consumer uses, and it had no
 * tests at all: nothing in the suite constructed one. These tests reach the
 * internals that only exist behind the facade, so they go through a cast.
 */
describe('Database', () => {
  let tempDir: string;

  const options = (): DatabaseOptions => ({
    vectorSize: 4,
    persistence: { ...configDefaults.persistence, dbPath: tempDir },
    indexing: { ...configDefaults.indexing, buildOnStart: false },
    clustering: { ...configDefaults.clustering },
    partitioning: { ...configDefaults.partitioning },
  });

  /** The monitor is private; the tests below assert on its wiring. */
  const monitorOf = (db: Database): VectorDBMonitor => {
    const monitor = (db as unknown as { monitor: VectorDBMonitor | null }).monitor;
    if (!monitor) throw new Error('monitor is disabled');
    return monitor;
  };

  const recordEventInMonitor = (db: Database, eventName: string, data: unknown): void => {
    (db as unknown as { recordEventInMonitor(e: string, d: unknown): void }).recordEventInMonitor(eventName, data);
  };

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'nbase-database-'));
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  describe('search cache', () => {
    it('does not serve a metadata-free entry to an includeMetadata query', async () => {
      const db = new Database(options());
      await db.initializationPromise;
      await db.addVector('a', [1, 0, 0, 0], { label: 'first' });

      const query = [1, 0, 0, 0];
      // Prime the cache without metadata.
      const bare = await db.search(query, { k: 5, includeMetadata: false });
      expect(bare[0]?.metadata).to.equal(undefined);

      // Same query, same k, same vector: only `includeMetadata` differs. This
      // used to hit the cache entry above and return results with no metadata,
      // because the key omitted the flag.
      const withMetadata = await db.search(query, { k: 5, includeMetadata: true });
      expect(withMetadata[0]?.metadata).to.deep.equal({ label: 'first' });

      // And the reverse order, so the metadata-bearing entry cannot satisfy a
      // request that did not ask for metadata.
      const db2 = new Database(options());
      await db2.initializationPromise;
      await db2.addVector('a', [1, 0, 0, 0], { label: 'first' });
      await db2.search([1, 0, 0, 0], { k: 5, includeMetadata: true });
      const stillBare = await db2.search([1, 0, 0, 0], { k: 5, includeMetadata: false });
      expect(stillBare[0]?.metadata).to.equal(undefined);

      await db.close();
      await db2.close();
    });

    it('does not let a cached result change under a later reader', async () => {
      const db = new Database(options());
      await db.initializationPromise;
      await db.addVector('a', [1, 0, 0, 0], { label: 'first' });

      const query = [1, 0, 0, 0];
      await db.search(query, { k: 5, includeMetadata: true });

      // Prime it again, then mutate what we were handed. The cached entry is
      // the same payload the next reader gets, so this must not leak into it.
      const first = await db.search(query, { k: 5, includeMetadata: true });
      first[0]!.metadata = { label: 'tampered' };

      const second = await db.search(query, { k: 5, includeMetadata: true });
      expect(second[0]?.metadata).to.deep.equal({ label: 'first' });

      await db.close();
    });

    it('gives a distinct key per filter', async () => {
      const db = new Database(options());
      await db.initializationPromise;
      await db.addVector('a', [1, 0, 0, 0], { label: 'keep' });
      await db.addVector('b', [0, 1, 0, 0], { label: 'drop' });

      const query = [1, 0, 0, 0];
      const onlyKeep = await db.search(query, { k: 5, filter: (_id, meta) => meta?.['label'] === 'keep' });
      const everything = await db.search(query, { k: 5 });

      expect(onlyKeep.map((r) => r.id)).to.deep.equal(['a']);
      expect(everything.map((r) => r.id)).to.have.length(2);

      await db.close();
    });
  });

  describe('batchSearch', () => {
    it('returns one result list per query, in order', async () => {
      const db = new Database(options());
      await db.initializationPromise;
      await db.addVector('a', [1, 0, 0, 0], { label: 'a' });
      await db.addVector('b', [0, 1, 0, 0], { label: 'b' });
      await db.addVector('c', [0, 0, 1, 0], { label: 'c' });

      const { results, errors } = await db.batchSearch([
        { query: [1, 0, 0, 0], k: 2, options: { includeMetadata: true } },
        { query: [0, 1, 0, 0], k: 1 },
        { query: [0, 0, 1, 0], k: 3 },
      ]);

      expect(errors).to.deep.equal([]);
      expect(results.length).to.equal(3);
      expect(results[0]?.[0]?.id).to.equal('a');
      expect(results[1]?.length).to.equal(1);
      expect(results[1]?.[0]?.id).to.equal('b');
      // Order is the caller's, not the completion order.
      expect(results[2]?.[0]?.id).to.equal('c');
      expect(results[2]?.length).to.equal(3);

      await db.close();
    });

    it('reports a failed query with its index instead of an empty list', async () => {
      const db = new Database(options());
      await db.initializationPromise;
      await db.addVector('a', [1, 0, 0, 0]);

      const original = db.findNearest.bind(db);
      let calls = 0;
      (db as unknown as { findNearest: unknown }).findNearest = (...args: unknown[]) => {
        calls++;
        // Fail only the second query, so a caller can see which one.
        if (calls === 2) return Promise.reject(new Error('query blew up'));
        return original(...(args as Parameters<typeof original>));
      };

      const { results, errors } = await db.batchSearch([
        { query: [1, 0, 0, 0], k: 1 },
        { query: [1, 0, 0, 0], k: 1 },
        { query: [1, 0, 0, 0], k: 1 },
      ]);
      (db as unknown as { findNearest: unknown }).findNearest = original;

      expect(errors.length).to.equal(1);
      expect(errors[0]?.index).to.equal(1);
      expect(errors[0]?.message).to.include('blew up');
      expect(results.length).to.equal(3);
      expect(results[1]).to.deep.equal([]);
      expect(results[0]?.length).to.equal(1);
      expect(results[2]?.length).to.equal(1);

      await db.close();
    });

    it('rejects an oversized batch before running any query', async () => {
      const db = new Database(options());
      await db.initializationPromise;

      let ran = 0;
      const original = db.findNearest.bind(db);
      (db as unknown as { findNearest: unknown }).findNearest = (...args: unknown[]) => {
        ran++;
        return original(...(args as Parameters<typeof original>));
      };

      const queries = Array.from({ length: MAX_BATCH_QUERIES + 1 }, () => ({ query: [1, 0, 0, 0], k: 1 }));
      let thrown: unknown = null;
      try {
        await db.batchSearch(queries);
      } catch (error) {
        thrown = error;
      }
      (db as unknown as { findNearest: unknown }).findNearest = original;

      expect(thrown).to.be.instanceOf(Error);
      expect((thrown as Error).message).to.match(/at most/);
      expect(ran).to.equal(0);

      await db.close();
    });

    it('goes through the search cache like a single search', async () => {
      const db = new Database(options());
      await db.initializationPromise;
      await db.addVector('a', [1, 0, 0, 0], { label: 'a' });

      const original = db.findNearest.bind(db);
      let calls = 0;
      (db as unknown as { findNearest: unknown }).findNearest = (...args: unknown[]) => {
        calls++;
        return original(...(args as Parameters<typeof original>));
      };

      await db.batchSearch([
        { query: [1, 0, 0, 0], k: 2 },
        { query: [1, 0, 0, 0], k: 2 },
      ]);
      (db as unknown as { findNearest: unknown }).findNearest = original;

      // Both queries reached findNearest (and therefore the cache) rather than a
      // separate engine that would bypass it.
      expect(calls).to.equal(2);
      const stats = await db.getStats();
      expect(stats.search?.calls).to.be.greaterThan(0);

      await db.close();
    });
  });

  describe('monitor error forwarding', () => {
    it('emits error on the Database instead of throwing', async () => {
      const db = new Database(options());
      await db.initializationPromise;

      const errors: { message: string; context?: string }[] = [];
      db.on('error', (payload) => errors.push(payload));

      // `recordEventInMonitor` routes every `partition:error` into
      // `monitor.recordError`. The monitor emits `error`, which Node treats
      // specially: with no listener attached the emit throws
      // `ERR_UNHANDLED_ERROR`, so the first partition error used to take down
      // whatever code path raised it.
      let threw: unknown = null;
      try {
        recordEventInMonitor(db, 'partition:error', new Error('partition exploded'));
      } catch (error) {
        threw = error;
      }

      expect(threw).to.equal(null);
      expect(errors.length).to.be.greaterThan(0);
      expect(errors[0]!.message).to.include('partition exploded');

      await db.close();
    });

    it('keeps the monitor usable after an error', async () => {
      const db = new Database(options());
      await db.initializationPromise;
      db.on('error', () => undefined);

      recordEventInMonitor(db, 'partition:error', new Error('first'));
      recordEventInMonitor(db, 'partition:error', new Error('second'));

      // A second emit would throw if the first had left the emitter in a bad
      // state (or if the listener had been removed).
      expect(monitorOf(db).listenerCount('error')).to.be.greaterThan(0);

      await db.close();
    });
  });
});