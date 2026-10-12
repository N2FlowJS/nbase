import { expect } from 'chai';
import { promises as fs } from 'fs';
import * as os from 'os';
import * as path from 'path';
import request from 'supertest';
import createServer from '../../src/server/index';
import { IServerInstance } from '../../src/types';
import configDefaults from '../../src/config';

/**
 * `createServer` is the only code that assembles the real Express app: helmet,
 * CORS, the JSON limit, the rate limiter, the auth middleware, the route
 * mounting order and the error handler. The existing API tests hand-build an
 * `express()` and mount a single router, so none of that was covered.
 */
describe('createServer defaults', () => {
  let instance: IServerInstance | null = null;
  let tempDir: string;

  const database = (dbPath: string) => ({
    ...configDefaults,
    vectorSize: 4,
    persistence: { ...configDefaults.persistence, dbPath },
    indexing: { ...configDefaults.indexing, buildOnStart: false },
  });

  const start = async (options: Partial<Parameters<typeof createServer>[0]> = {}): Promise<IServerInstance> => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'nbase-server-'));
    const created: IServerInstance = createServer({
      // Rate limiting off by default here: the tests assert headers, not limits.
      rateLimit: { enable: false },
      database: database(tempDir),
      ...options,
    });
    instance = created;
    await created.database.initializationPromise;
    return created;
  };

  afterEach(async () => {
    await instance?.gracefulShutdown().catch(() => undefined);
    instance = null;
    if (tempDir) {
      await fs.rm(tempDir, { recursive: true, force: true });
    }
  });

  describe('security headers and CORS', () => {
    it('sends helmet headers', async () => {
      const { app } = await start();
      const res = await request(app).get('/api/vectors/does-not-exist');

      expect(res.headers['x-content-type-options']).to.equal('nosniff');
      expect(res.headers['x-powered-by']).to.equal(undefined);
    });

    it('sends no Access-Control-Allow-Origin by default', async () => {
      const { app } = await start();
      const res = await request(app).get('/api/vectors/does-not-exist').set('Origin', 'https://evil.example');

      // `cors()` used to be mounted with no arguments, which reflected any
      // origin. Nothing is added unless `cors` is configured now.
      expect(res.headers['access-control-allow-origin']).to.equal(undefined);
    });

    it('echoes only a configured origin', async () => {
      const { app } = await start({ cors: ['https://app.example'] });

      const allowed = await request(app).get('/api/vectors/does-not-exist').set('Origin', 'https://app.example');
      expect(allowed.headers['access-control-allow-origin']).to.equal('https://app.example');

      const denied = await request(app).get('/api/vectors/does-not-exist').set('Origin', 'https://evil.example');
      expect(denied.headers['access-control-allow-origin']).to.equal(undefined);
    });
  });

  describe('authentication', () => {
    it('rejects requests without a key when keys are configured', async () => {
      const { app } = await start({ auth: { apiKeys: ['secret-key'] } });

      expect((await request(app).get('/api/vectors/x')).status).to.equal(401);
      expect((await request(app).get('/api/vectors/x').set('x-api-key', 'wrong')).status).to.equal(403);

      const withKey = await request(app).get('/api/vectors/x').set('Authorization', 'Bearer secret-key');
      expect(withKey.status).to.not.equal(401);
      expect(withKey.status).to.not.equal(403);
    });

    it('leaves the API open when no keys are configured', async () => {
      const { app } = await start();
      expect((await request(app).get('/api/vectors/x')).status).to.equal(404);
    });

    it('refuses to start when requireAuth is set without keys', async () => {
      let thrown: unknown = null;
      try {
        await start({ auth: { requireAuth: true } });
      } catch (error) {
        thrown = error;
      }
      expect(thrown).to.be.instanceOf(Error);
      expect((thrown as Error).message).to.match(/requireAuth/);
    });
  });

  describe('error handling', () => {
    it('answers an unknown route with a 404 and no stack', async () => {
      const { app } = await start();
      const res = await request(app).get('/definitely-not-a-route');

      expect(res.status).to.equal(404);
      expect(res.body.error).to.equal('Route not found');
      expect(res.body.stack).to.equal(undefined);
    });

    it('reports a malformed JSON body as 400, not 500', async () => {
      const { app } = await start();
      const res = await request(app).post('/api/vectors').set('Content-Type', 'application/json').send('{"vector": [1,2,');

      expect(res.status).to.equal(400);
    });

    it('does not leak internal messages for a 500', async () => {
      const { app, database: db } = await start();

      // Force a failure whose message carries an internal path.
      const original = db.addVector.bind(db);
      (db as unknown as { addVector: unknown }).addVector = () => {
        throw new Error('ENOENT: /home/secret/partitions/vec.bin');
      };
      const res = await request(app)
        .post('/api/vectors')
        .send({ id: 'x', vector: [1, 0, 0, 0] });
      (db as unknown as { addVector: unknown }).addVector = original;

      expect(res.status).to.equal(500);
      expect(res.body.error).to.not.include('/home/secret');
    });
  });

  describe('payload validation', () => {
    it('rejects a non-numeric k with 400 instead of scanning the index', async () => {
      const { app } = await start();
      await request(app)
        .post('/api/vectors')
        .send({ id: 'a', vector: [1, 0, 0, 0] });

      // `Math.max(NaN, efSearch)` is NaN and `results.size >= NaN` is never true,
      // so an unchecked `k` made every request a full-index scan.
      const res = await request(app)
        .post('/api/search')
        .send({ query: [1, 0, 0, 0], k: 'abc' });
      expect(res.status).to.equal(400);
      expect(res.body.error).to.match(/k/);

      expect(
        (
          await request(app)
            .post('/api/search')
            .send({ query: [1, 0, 0, 0], k: 1e9 })
        ).status,
      ).to.equal(400);
      expect(
        (
          await request(app)
            .post('/api/search')
            .send({ query: [1, 0, 0, 0], k: 5 })
        ).status,
      ).to.equal(200);
    });

    it('rejects an object id', async () => {
      const { app } = await start();
      const res = await request(app)
        .post('/api/vectors')
        .send({ id: { a: 1 }, vector: [1, 0, 0, 0] });
      expect(res.status).to.equal(400);
    });

    it('rejects vector components that would be stored as NaN', async () => {
      const { app } = await start();
      const res = await request(app)
        .post('/api/vectors')
        .send({ id: 'a', vector: [1, '2', 0, 0] });
      expect(res.status).to.equal(400);
    });

    it('caps the graph threshold', async () => {
      const { app } = await start();
      const res = await request(app).post('/api/search/relationships').send({ threshold: 1e12 });
      expect(res.status).to.equal(400);
      expect(res.body.error).to.match(/threshold/);
    });

    it('rejects a non-numeric k on /similar', async () => {
      const { app } = await start();
      await request(app)
        .post('/api/vectors')
        .send({ id: 'a', vector: [1, 0, 0, 0] });

      expect((await request(app).get('/api/vectors/a/similar?k=abc')).status).to.equal(400);
      expect((await request(app).get('/api/vectors/a/similar?k=3')).status).to.equal(200);
    });

    it('honours limit in the metadata search', async () => {
      const { app } = await start();
      for (let i = 0; i < 5; i++) {
        await request(app)
          .post('/api/vectors')
          .send({ id: `v${i}`, vector: [1, 0, 0, 0], metadata: { type: 'book' } });
      }

      const all = await request(app).post('/api/search/metadata').send({ criteria: 'type' });
      expect(all.body.count).to.equal(5);

      // The legacy string-criteria branch ignored `limit` entirely and returned
      // every match from every loaded partition.
      const limited = await request(app).post('/api/search/metadata').send({ criteria: 'type', limit: 2 });
      expect(limited.body.count).to.equal(2);

      expect((await request(app).post('/api/search/metadata').send({ criteria: 'type', limit: 'abc' })).status).to.equal(400);
    });

    it('does not match inherited keys in a metadata criteria', async () => {
      const { app } = await start();
      await request(app)
        .post('/api/vectors')
        .send({ id: 'a', vector: [1, 0, 0, 0], metadata: { label: 'x' } });

      // `field in meta` walked the prototype chain, so `constructor` or
      // `toString` matched every stored record and dumped the whole store.
      const res = await request(app).post('/api/search/metadata').send({ criteria: 'constructor' });
      expect(res.body.count ?? 0).to.equal(0);

      const real = await request(app).post('/api/search/metadata').send({ criteria: 'label' });
      expect(real.body.count).to.equal(1);
    });

    it('strips __proto__ from metadata', async () => {
      const { app } = await start();
      const res = await request(app).post('/api/vectors').send(JSON.parse('{"id":"a","vector":[1,0,0,0],"metadata":{"label":"x","__proto__":{"admin":true}}}'));
      expect(res.status).to.be.oneOf([200, 201]);

      const stored = await request(app).get('/api/vectors/a?includeMetadata=true');
      expect(JSON.stringify(stored.body)).to.not.include('__proto__');
      const probe: Record<string, unknown> = {};
      expect(probe['admin']).to.equal(undefined);
    });
  });

  describe('rate limiting scope', () => {
    it('rate-limits /stats as well as /api', async () => {
      const { app } = await start({
        rateLimit: { enable: true, maxRequestsPerMinute: 2, windowMs: 60_000 },
      });

      // `/stats` is mounted at `/` and used to sit outside the limiter.
      const first = await request(app).get('/stats');
      const second = await request(app).get('/stats');
      const third = await request(app).get('/stats');

      expect(first.status).to.equal(200);
      expect(third.status).to.equal(429);
      expect(third.headers['ratelimit-limit'] ?? third.headers['x-ratelimit-limit']).to.be.a('string');
      expect(second.status).to.be.oneOf([200, 429]);
    });
  });
});
