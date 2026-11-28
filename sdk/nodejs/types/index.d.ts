export type MetadataFilter = Array<{
  field: string;
  op: "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "exists" | "contains";
  value?: string | number | boolean;
}>;

export interface RequestOptions {
  filters?: MetadataFilter;
  metric?: "euclidean" | "squared_euclidean" | "cosine" | "inner_product" | "manhattan" | "chebyshev" | "hamming";
  mode?: "auto" | "exact" | "hnsw";
  efSearch?: number;
  maxDistance?: number;
  skipCache?: boolean;
  includeVectors?: boolean;
  includeMetadata?: boolean;
  includeText?: boolean;
  [option: string]: unknown;
}

export interface NBaseClientOptions {
  url?: string;
  apiKey?: string;
  timeout?: number;
  fetchImpl?: typeof fetch;
}

export class NBaseError extends Error {
  status: number | null;
  payload: unknown;
}

export class NBaseClient {
  constructor(options?: NBaseClientOptions);
  request(method: string, path: string, body?: unknown): Promise<unknown>;
  health(): Promise<Record<string, unknown>>;
  stats(): Promise<Record<string, unknown>>;
  add(id: number | string, vector: number[], options?: { metadata?: Record<string, unknown>; text?: string }): Promise<unknown>;
  upsert(id: number | string, vector: number[], options?: { metadata?: Record<string, unknown>; text?: string }): Promise<unknown>;
  addMany(vectors: Array<Record<string, unknown>>, options?: { upsert?: boolean; threads?: number }): Promise<unknown>;
  get(id: number | string, options?: { includeVector?: boolean; includeMetadata?: boolean; includeText?: boolean }): Promise<unknown>;
  getMany(ids: Array<number | string>, options?: { includeVectors?: boolean; includeMetadata?: boolean; includeText?: boolean }): Promise<unknown>;
  delete(id: number | string): Promise<unknown>;
  deleteMany(ids: Array<number | string>): Promise<unknown>;
  count(filters?: MetadataFilter): Promise<unknown>;
  search(query: number[], k?: number, options?: RequestOptions): Promise<unknown>;
  searchBatch(queries: number[][], k?: number, options?: RequestOptions): Promise<unknown>;
  searchText(query: string, k?: number, options?: RequestOptions): Promise<unknown>;
  searchHybrid(query: number[], text: string, k?: number, options?: RequestOptions): Promise<unknown>;
  setMetadata(id: number | string, metadata: Record<string, unknown>, options?: { update?: boolean }): Promise<unknown>;
  setText(id: number | string, text: string): Promise<unknown>;
}

export default NBaseClient;
