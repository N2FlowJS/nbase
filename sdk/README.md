# nbase SDKs

Both SDKs connect to the `nbased` REST server. The Node.js SDK uses the existing npm package name `@n2flowjs/nbase` and replaces its former embedded TypeScript implementation with a client for the current C11 engine.

- [Node.js SDK](nodejs/README.md): `@n2flowjs/nbase`, Node.js 18+, no runtime dependencies.
- [Python SDK](python/README.md): Python 3.8+, standard library only.

The Python client accepts the server URL and optional bearer API key, returns decoded JSON, and exposes common vector CRUD, metadata, text, and search methods. See [server API documentation](../docs/server.md) for endpoint behavior and filter formats.
