export class NBaseError extends Error {
  constructor(message, { status = null, payload = null, cause } = {}) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = "NBaseError";
    this.status = status;
    this.payload = payload;
  }
}

/** Client for an nbase `nbased` HTTP server. Requires Node.js 18 or newer. */
export class NBaseClient {
  constructor({ url = "http://127.0.0.1:3000", apiKey, timeout = 30_000, fetchImpl = globalThis.fetch } = {}) {
    if (typeof fetchImpl !== "function") {
      throw new TypeError("A fetch implementation is required (Node.js 18+)");
    }
    this.url = url.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.timeout = timeout;
    this.fetch = fetchImpl;
  }

  async request(method, path, body) {
    const headers = { Accept: "application/json" };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    let response;
    try {
      response = await this.fetch(`${this.url}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeout),
      });
    } catch (cause) {
      throw new NBaseError(cause.message || "Request failed", { cause });
    }
    const raw = await response.text();
    let payload = null;
    if (raw) {
      try {
        payload = JSON.parse(raw);
      } catch (cause) {
        throw new NBaseError("Server returned invalid JSON", { status: response.status, cause });
      }
    }
    if (!response.ok) {
      const message = payload?.error ?? payload?.message ?? `HTTP ${response.status}`;
      throw new NBaseError(message, { status: response.status, payload });
    }
    return payload;
  }

  health() { return this.request("GET", "/health"); }
  stats() { return this.request("GET", "/stats"); }

  add(id, vector, { metadata, text } = {}) {
    return this.request("POST", "/api/vectors", {
      id, vector,
      ...(metadata === undefined ? {} : { metadata }),
      ...(text === undefined ? {} : { text }),
    });
  }

  upsert(id, vector, { metadata, text } = {}) {
    return this.request("POST", "/api/vectors/upsert", {
      id, vector,
      ...(metadata === undefined ? {} : { metadata }),
      ...(text === undefined ? {} : { text }),
    });
  }

  addMany(vectors, { upsert = false, threads } = {}) {
    return this.request("POST", upsert ? "/api/vectors/upsert" : "/api/vectors", {
      vectors,
      ...(threads === undefined ? {} : { threads }),
    });
  }

  get(id, { includeVector = true, includeMetadata = true, includeText = false } = {}) {
    const query = new URLSearchParams({
      includeVector: String(includeVector),
      includeMetadata: String(includeMetadata),
      includeText: String(includeText),
    });
    return this.request("GET", `/api/vectors/${encodeURIComponent(id)}?${query}`);
  }

  getMany(ids, { includeVectors = true, includeMetadata = true, includeText = false } = {}) {
    return this.request("POST", "/api/vectors/get", { ids, includeVectors, includeMetadata, includeText });
  }

  delete(id) { return this.request("DELETE", `/api/vectors/${encodeURIComponent(id)}`); }
  deleteMany(ids) { return this.request("POST", "/api/vectors/delete", { ids }); }

  count(filters) {
    return filters === undefined
      ? this.request("GET", "/api/vectors/count")
      : this.request("POST", "/api/vectors/count", { filters });
  }

  search(query, k = 10, options = {}) {
    return this.request("POST", "/api/search", { query, k, ...options });
  }

  searchBatch(queries, k = 10, options = {}) {
    return this.request("POST", "/api/search/batch", { queries, k, ...options });
  }

  searchText(query, k = 10, options = {}) {
    return this.request("POST", "/api/search/text", { query, k, ...options });
  }

  searchHybrid(query, text, k = 10, options = {}) {
    return this.request("POST", "/api/search/hybrid", { query, text, k, ...options });
  }

  setMetadata(id, metadata, { update = false } = {}) {
    return this.request(update ? "PATCH" : "POST", "/api/vectors/metadata", {
      items: [{ id, metadata }],
    });
  }

  setText(id, text) {
    return this.request("POST", "/api/vectors/text", { items: [{ id, text }] });
  }
}

export default NBaseClient;
