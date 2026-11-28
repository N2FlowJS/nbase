# `@n2flowjs/nbase` Node.js SDK

Node.js client for the nbase HTTP API served by `nbased`. This package replaces the former embedded TypeScript database API with an SDK for the current C11 nbase engine. It requires Node.js 18 or newer and has no runtime dependencies.

```bash
npm install @n2flowjs/nbase
```

```js
import { NBaseClient } from "@n2flowjs/nbase";

const db = new NBaseClient({
  url: "http://localhost:1307",
  apiKey: process.env.NBASE_API_KEY,
});

await db.add(1, [0.1, 0.2, 0.3], {
  metadata: { category: "demo" },
  text: "A sample document",
});

const hits = await db.search([0.1, 0.2, 0.3], 5, {
  includeMetadata: true,
});
console.log(hits);
```

## Client API

- `health()` and `stats()`
- `add`, `upsert`, `addMany`, `get`, `getMany`, `delete`, `deleteMany`, `count`
- `setMetadata`, `setText`
- `search`, `searchBatch`, `searchText`, `searchHybrid`
- `request(method, path, body)` for other documented REST endpoints

Search options are forwarded to the server, including `filters`, `metric`, `mode`, `efSearch`, `maxDistance`, `skipCache`, and `includeMetadata`/`includeVectors`/`includeText`. Errors reject with `NBaseError`, with `status` and `payload` populated for HTTP errors.

Start a server with the nbase daemon, for example:

```bash
nbased --host 127.0.0.1 --port 1307 --dim 3
```

See the [REST API reference](https://github.com/N2FlowJS/nbase/blob/main/docs/server.md) for filters and endpoint details.
