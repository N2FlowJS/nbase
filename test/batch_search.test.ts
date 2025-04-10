import { expect } from "chai";
import sinon from "sinon";
import { BatchEngineSearch } from "../src/search/batch_search";
import { describe, it, beforeEach } from "mocha";

import {
  PartitionedVectorDBInterface,
  BatchSearchConfiguration,
  BatchSearchQuery,
  SearchResult,
} from "../src/types";

describe("BatchEngineSearch", () => {
  let mockSearchEngine: sinon.SinonStubbedInstance<PartitionedVectorDBInterface>;
  let batchEngineSearch: BatchEngineSearch;

  beforeEach(() => {
    mockSearchEngine = {
      findNearest: sinon.stub(),
      findNearestHNSW: sinon.stub(),
    } as unknown as sinon.SinonStubbedInstance<PartitionedVectorDBInterface>;
    const config: BatchSearchConfiguration = {
      maxBatchSize: 2,
      prioritizeOrder: true,
      groupSimilarQueries: false,
      defaultSearchTimeoutMs: 5000,
    };
    batchEngineSearch = new BatchEngineSearch(mockSearchEngine, config);
  });

  afterEach(() => {
    sinon.restore();
  });

  it("should process a batch of queries using findNearest", async () => {
    const queries: BatchSearchQuery[] = [
      { query: [1, 2, 3], k: 5, options: {} },
      { query: [4, 5, 6], k: 3, options: {} },
    ];

    const mockResults: SearchResult[][] = [
      [{ id: "a", score: 0.9, dist: 0.1 }],
      [{ id: "b", score: 0.8, dist: 0.2 }],
    ];

    mockSearchEngine.findNearest.resolves(mockResults[0]);

    const results = await batchEngineSearch.searchBatch(queries);

    expect(results.length).to.equal(2);
    expect(mockSearchEngine.findNearest.callCount).to.equal(2);
  });

  it("should process a batch of queries using findNearestHNSW when useHNSW is true", async () => {
    const queries: BatchSearchQuery[] = [
      { query: [1, 2, 3], k: 5, options: { useHNSW: true } },
      { query: [4, 5, 6], k: 3, options: { useHNSW: true } },
    ];

    const mockResults: SearchResult[][] = [
      [{ id: "a", score: 0.9, dist: 0.1 }],
      [{ id: "b", score: 0.8, dist: 0.2 }],
    ];

    mockSearchEngine.findNearestHNSW.resolves(mockResults[0]);

    const results = await batchEngineSearch.searchBatch(queries);

    expect(results.length).to.equal(2);
    expect(mockSearchEngine.findNearestHNSW.callCount).to.equal(2);
  });

  it("should split large batches into smaller chunks", async () => {
    const queries: BatchSearchQuery[] = [
      { query: [1, 2, 3], k: 5, options: {} },
      { query: [4, 5, 6], k: 3, options: {} },
      { query: [7, 8, 9], k: 2, options: {} },
    ];

    const mockResults: SearchResult[][] = [
      [{ id: "a", score: 0.9, dist: 0.1 }],
      [{ id: "b", score: 0.8, dist: 0.2 }],
      [{ id: "c", score: 0.7, dist: 0.3 }],
    ];

    mockSearchEngine.findNearest.onCall(0).resolves(mockResults[0]);
    mockSearchEngine.findNearest.onCall(1).resolves(mockResults[1]);
    mockSearchEngine.findNearest.onCall(2).resolves(mockResults[2]);

    const results = await batchEngineSearch.searchBatch(queries);

    expect(results).to.deep.equal(mockResults);
    expect(mockSearchEngine.findNearest.callCount).to.equal(3);
  });

  it("should handle query timeouts gracefully", async function() { // Sử dụng function để có 'this'
    // Tăng timeout của Mocha đủ lớn để chứa timeout nội bộ + buffer
    this.timeout(6000); // 5000ms timeout + 1000ms buffer

    const queries: BatchSearchQuery[] = [
        { query: [1, 2, 3], k: 5, options: {} },
    ];

    // Mock search engine để trả về một promise KHÔNG BAO GIỜ resolve
    // Điều này đảm bảo rằng timeout nội bộ của BatchEngineSearch sẽ luôn thắng cuộc đua.
    mockSearchEngine.findNearest.callsFake(
        () => new Promise(() => {}) // Promise này sẽ mãi mãi ở trạng thái pending
    );

    // Gọi hàm cần test
    const results = await batchEngineSearch.searchBatch(queries);

    // Kiểm tra kết quả mong đợi:
    // - Logic timeout nội bộ phải được kích hoạt.
    // - Khối .catch trong _processBatch phải xử lý lỗi timeout.
    // - Kết quả trả về cho query đó phải là mảng rỗng.
    expect(results).to.deep.equal([[]]);

    // Đảm bảo mock đã được gọi
    expect(mockSearchEngine.findNearest.callCount).to.equal(1);

    // Có thể thêm một log nhỏ ở đây để xác nhận test đã đi đến cuối cùng
    // console.log("Timeout test finished assertions.");
});

  it("should prioritize order in results", async () => {
    const queries: BatchSearchQuery[] = [
      { query: [1, 2, 3], k: 5, options: {} },
      { query: [4, 5, 6], k: 3, options: {} },
    ];

    const mockResults: SearchResult[][] = [
      [{ id: "b", score: 0.8, dist: 0.2 }],
      [{ id: "a", score: 0.9, dist: 0.1 }],
    ];

    mockSearchEngine.findNearest.onCall(0).resolves(mockResults[1]);
    mockSearchEngine.findNearest.onCall(1).resolves(mockResults[0]);

    const results = await batchEngineSearch.searchBatch(queries);

    expect(results).to.deep.equal(mockResults.reverse());
    expect(mockSearchEngine.findNearest.callCount).to.equal(2);
  });

  it("should handle errors during query processing", async () => {
    const queries: BatchSearchQuery[] = [
      { query: [1, 2, 3], k: 5, options: {} },
    ];

    mockSearchEngine.findNearest.rejects(new Error("Search failed"));

    const results = await batchEngineSearch.searchBatch(queries);

    expect(results).to.deep.equal([[]]);
    expect(mockSearchEngine.findNearest.callCount).to.equal(1);
  });

  it("should shutdown gracefully", () => {
    const consoleSpy = sinon.spy(console, "log");
    batchEngineSearch.shutdown();
    expect(consoleSpy.calledWith("PartitionedBatchSearch shutdown.")).to.be
      .true;
  });
});
