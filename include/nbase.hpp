#ifndef NBASE_HPP
#define NBASE_HPP

#include "nbase.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nbase {

class Error : public std::runtime_error {
 public:
  explicit Error(nbase_status_t st)
      : std::runtime_error(nbase_strerror(st)), status_(st) {}

  nbase_status_t status() const noexcept { return status_; }

 private:
  nbase_status_t status_;
};

namespace detail {
inline void check(nbase_status_t st) {
  if (st != NBASE_OK) {
    throw Error(st);
  }
}
}  // namespace detail

inline std::vector<float> distance_batch(
    nbase_metric_t metric,
    const std::vector<float> &query,
    const std::vector<float> &vectors) {
  if (query.empty()) {
    throw Error(NBASE_ERR_INVALID_ARG);
  }
  uint32_t dim = static_cast<uint32_t>(query.size());
  if (vectors.size() % static_cast<size_t>(dim) != 0U) {
    throw Error(NBASE_ERR_DIM_MISMATCH);
  }
  uint32_t count = static_cast<uint32_t>(vectors.size() / static_cast<size_t>(dim));
  std::vector<float> out(count);
  detail::check(nbase_distance_batch(
      metric,
      query.data(),
      vectors.empty() ? nullptr : vectors.data(),
      count,
      dim,
      out.empty() ? nullptr : out.data()));
  return out;
}

class Database {
 public:
  Database() noexcept = default;
  explicit Database(nbase_db_t *raw_db) noexcept : db_(raw_db) {}

  Database(const Database &) = delete;
  Database &operator=(const Database &) = delete;

  Database(Database &&other) noexcept : db_(other.db_) { other.db_ = nullptr; }

  Database &operator=(Database &&other) noexcept {
    if (this != &other) {
      reset();
      db_ = other.db_;
      other.db_ = nullptr;
    }
    return *this;
  }

  ~Database() { reset(); }

  void reset() noexcept {
    if (db_ != nullptr) {
      nbase_close(db_);
      db_ = nullptr;
    }
  }

  static Database create(const nbase_config_t &cfg) {
    nbase_db_t *db = nullptr;
    detail::check(nbase_create(&cfg, &db));
    return Database(db);
  }

  static Database create(uint32_t dim) {
    return create(nbase_config_default(dim));
  }

  static Database open(const std::string &dir_path) {
    nbase_db_t *db = nullptr;
    detail::check(nbase_open(dir_path.c_str(), &db));
    return Database(db);
  }

  static Database open_mmap(const std::string &dir_path) {
    nbase_db_t *db = nullptr;
    detail::check(nbase_open_mmap(dir_path.c_str(), &db));
    return Database(db);
  }

  static Database open_wal(const std::string &dir_path) {
    nbase_db_t *db = nullptr;
    detail::check(nbase_open_wal(dir_path.c_str(), &db));
    return Database(db);
  }

  void wal_enable(const std::string &dir_path) {
    detail::check(nbase_wal_enable(db_, dir_path.c_str()));
  }

  void wal_checkpoint() { detail::check(nbase_wal_checkpoint(db_)); }

  void save(const std::string &dir_path) const {
    detail::check(nbase_save(db_, dir_path.c_str()));
  }

  void add(uint64_t id, const float *vec, uint32_t dim) {
    detail::check(nbase_add(db_, id, vec, dim));
  }

  void add(uint64_t id, const std::vector<float> &vec) {
    add(id, vec.data(), static_cast<uint32_t>(vec.size()));
  }

  void add_batch(
      const std::vector<uint64_t> &ids,
      const std::vector<float> &vectors,
      uint32_t dim,
      uint32_t num_threads = 0U) {
    detail::check(nbase_add_batch(
        db_,
        ids.data(),
        vectors.data(),
        static_cast<uint32_t>(ids.size()),
        dim,
        num_threads));
  }

  void upsert(uint64_t id, const float *vec, uint32_t dim) {
    detail::check(nbase_upsert(db_, id, vec, dim));
  }

  void upsert(uint64_t id, const std::vector<float> &vec) {
    upsert(id, vec.data(), static_cast<uint32_t>(vec.size()));
  }

  void upsert_batch(
      const std::vector<uint64_t> &ids,
      const std::vector<float> &vectors,
      uint32_t dim,
      uint32_t num_threads = 0U) {
    detail::check(nbase_upsert_batch(
        db_,
        ids.empty() ? nullptr : ids.data(),
        vectors.empty() ? nullptr : vectors.data(),
        static_cast<uint32_t>(ids.size()),
        dim,
        num_threads));
  }

  void remove(uint64_t id) { detail::check(nbase_delete(db_, id)); }

  uint32_t delete_batch(const std::vector<uint64_t> &ids) {
    uint32_t deleted = 0U;
    detail::check(nbase_delete_batch(
        db_, ids.empty() ? nullptr : ids.data(), static_cast<uint32_t>(ids.size()), &deleted));
    return deleted;
  }

  uint32_t delete_by_filter(const nbase_meta_filter_t &filter) {
    uint32_t deleted = 0U;
    detail::check(nbase_delete_by_filter(db_, &filter, &deleted));
    return deleted;
  }

  std::vector<float> get(uint64_t id, uint32_t dim) const {
    std::vector<float> out(dim);
    detail::check(nbase_get(db_, id, out.data(), dim));
    return out;
  }

  void get_batch(
      const std::vector<uint64_t> &ids,
      uint32_t dim,
      std::vector<float> &out_vectors,
      std::vector<bool> &out_found) const {
    const size_t n = ids.size();
    out_vectors.resize(n * static_cast<size_t>(dim));
    out_found.assign(n, false);
    if (n == 0U) {
      return;
    }
    std::unique_ptr<bool[]> found_buf(new bool[n]);
    detail::check(nbase_get_batch(
        db_, ids.data(), static_cast<uint32_t>(n), dim, out_vectors.data(), found_buf.get()));
    for (size_t i = 0U; i < n; ++i) {
      out_found[i] = found_buf[i];
    }
  }

  bool contains(uint64_t id) const noexcept { return nbase_contains(db_, id); }

  std::vector<uint64_t> list_ids(
      uint64_t offset, uint32_t limit, uint64_t *out_total = nullptr) const {
    std::vector<uint64_t> ids(limit);
    uint32_t count = 0U;
    detail::check(nbase_list_ids(
        db_, offset, limit, limit > 0U ? ids.data() : nullptr, &count, out_total));
    ids.resize(count);
    return ids;
  }

  uint64_t count(const nbase_meta_filter_t *filter = nullptr) const {
    uint64_t out = 0ULL;
    detail::check(nbase_count(db_, filter, &out));
    return out;
  }

  std::vector<nbase_result_t> search(
      const std::vector<float> &query,
      uint32_t k,
      const nbase_search_opts_t *opts = nullptr) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_search(
        db_,
        query.data(),
        static_cast<uint32_t>(query.size()),
        k,
        opts,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_result_t> search_radius(
      const std::vector<float> &query,
      float max_distance,
      uint32_t max_results,
      const nbase_search_opts_t *opts = nullptr) const {
    std::vector<nbase_result_t> out(max_results);
    uint32_t count = 0U;
    detail::check(nbase_search_radius(
        db_,
        query.data(),
        static_cast<uint32_t>(query.size()),
        max_distance,
        max_results,
        opts,
        max_results > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_result_t> find_similar(
      uint64_t id,
      uint32_t k,
      const nbase_search_opts_t *opts = nullptr) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_find_similar(
        db_, id, k, opts, k > 0U ? out.data() : nullptr, &count));
    out.resize(count);
    return out;
  }

  std::vector<uint64_t> query_meta(
      const nbase_meta_filter_t *filter, uint32_t limit) const {
    std::vector<uint64_t> ids(limit);
    uint32_t count = 0U;
    detail::check(nbase_query_meta(
        db_, filter, limit, limit > 0U ? ids.data() : nullptr, &count));
    ids.resize(count);
    return ids;
  }

  void set_meta(uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
    detail::check(nbase_set_meta(db_, id, fields, count));
  }

  void set_meta_batch(const std::vector<nbase_meta_batch_item_t> &items) {
    detail::check(nbase_set_meta_batch(
        db_, items.empty() ? nullptr : items.data(), static_cast<uint32_t>(items.size())));
  }

  void update_meta(uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
    detail::check(nbase_update_meta(db_, id, fields, count));
  }

  void update_meta_batch(const std::vector<nbase_meta_batch_item_t> &items) {
    detail::check(nbase_update_meta_batch(
        db_, items.empty() ? nullptr : items.data(), static_cast<uint32_t>(items.size())));
  }

  void delete_meta(uint64_t id) { detail::check(nbase_delete_meta(db_, id)); }

  void set_text(uint64_t id, const std::string &text) {
    detail::check(nbase_set_text(db_, id, text.c_str()));
  }

  void set_text_batch(
      const std::vector<uint64_t> &ids,
      const std::vector<std::string> &texts) {
    if (ids.size() != texts.size()) {
      throw Error(NBASE_ERR_INVALID_ARG);
    }
    std::vector<const char *> ptrs;
    ptrs.reserve(texts.size());
    for (const std::string &t : texts) {
      ptrs.push_back(t.c_str());
    }
    detail::check(nbase_set_text_batch(
        db_,
        ids.empty() ? nullptr : ids.data(),
        ptrs.empty() ? nullptr : ptrs.data(),
        static_cast<uint32_t>(ids.size())));
  }

  std::string get_text(uint64_t id) const {
    size_t len = 0U;
    detail::check(nbase_get_text(db_, id, nullptr, 0U, &len));
    std::string out(len, '\0');
    detail::check(nbase_get_text(db_, id, &out[0], len + 1U, &len));
    return out;
  }

  void delete_text(uint64_t id) {
    detail::check(nbase_delete_text(db_, id));
  }

  std::vector<nbase_scored_id_t> search_text(
      const std::string &query,
      uint32_t k,
      const nbase_meta_filter_t *meta_filter = nullptr) const {
    std::vector<nbase_scored_id_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_search_text(
        db_,
        query.c_str(),
        k,
        meta_filter,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_scored_id_t> search_hybrid(
      const std::vector<float> &query_vec,
      const std::string &query_text,
      uint32_t k,
      float alpha = 0.5f,
      float rrf_k = 60.0f,
      const nbase_search_opts_t *opts = nullptr) const {
    std::vector<nbase_scored_id_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_search_hybrid(
        db_,
        query_vec.empty() ? nullptr : query_vec.data(),
        static_cast<uint32_t>(query_vec.size()),
        query_text.empty() ? nullptr : query_text.c_str(),
        k,
        alpha,
        rrf_k,
        opts,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  void compact() { detail::check(nbase_compact(db_)); }

  void rebuild_hnsw_mt(uint32_t num_threads = 0U) {
    detail::check(nbase_rebuild_hnsw_mt(db_, num_threads));
  }

  void cache_configure(uint32_t max_entries) {
    detail::check(nbase_cache_configure(db_, max_entries));
  }

  void cache_clear() { detail::check(nbase_cache_clear(db_)); }

  std::vector<nbase_result_t> rerank_mmr(
      const std::vector<nbase_result_t> &candidates,
      uint32_t k,
      float lambda) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_rerank_mmr(
        db_,
        candidates.empty() ? nullptr : candidates.data(),
        static_cast<uint32_t>(candidates.size()),
        k,
        lambda,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_result_t> rerank_weighted(
      const std::vector<nbase_result_t> &candidates,
      const std::vector<nbase_rerank_weight_t> &weights,
      uint32_t k) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_rerank_weighted(
        db_,
        candidates.empty() ? nullptr : candidates.data(),
        static_cast<uint32_t>(candidates.size()),
        weights.empty() ? nullptr : weights.data(),
        static_cast<uint32_t>(weights.size()),
        k,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_edge_t> extract_relationships(
      uint32_t k_neighbors,
      float max_distance,
      const nbase_meta_filter_t *meta_filter = nullptr) const {
    struct EdgeGuard {
      nbase_edge_t *ptr = nullptr;
      ~EdgeGuard() { nbase_free_edges(ptr); }
    } guard;
    uint64_t edge_cnt = 0ULL;
    detail::check(nbase_extract_relationships_filtered(
        db_, k_neighbors, max_distance, meta_filter, &guard.ptr, &edge_cnt));
    std::vector<nbase_edge_t> out;
    if (guard.ptr != nullptr && edge_cnt > 0ULL) {
      out.assign(guard.ptr, guard.ptr + edge_cnt);
    }
    return out;
  }

  static std::vector<uint32_t> extract_communities(
      const std::vector<uint64_t> &ids,
      const std::vector<nbase_edge_t> &edges,
      uint32_t *out_community_count = nullptr) {
    std::vector<uint32_t> labels(ids.size());
    uint32_t comm_cnt = 0U;
    detail::check(nbase_extract_communities(
        ids.empty() ? nullptr : ids.data(),
        static_cast<uint32_t>(ids.size()),
        edges.empty() ? nullptr : edges.data(),
        static_cast<uint64_t>(edges.size()),
        labels.empty() ? nullptr : labels.data(),
        &comm_cnt));
    if (out_community_count != nullptr) {
      *out_community_count = comm_cnt;
    }
    return labels;
  }

  nbase_stats_t stats() const {
    nbase_stats_t s{};
    detail::check(nbase_stats(db_, &s));
    return s;
  }

  nbase_db_t *raw() noexcept { return db_; }
  const nbase_db_t *raw() const noexcept { return db_; }

 private:
  nbase_db_t *db_ = nullptr;
};

class Sq8 {
 public:
  Sq8() noexcept = default;
  explicit Sq8(nbase_sq8_t *raw_sq8) noexcept : ptr_(raw_sq8) {}
  Sq8(const Sq8 &) = delete;
  Sq8 &operator=(const Sq8 &) = delete;
  Sq8(Sq8 &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Sq8 &operator=(Sq8 &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  ~Sq8() { reset(); }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      nbase_sq8_destroy(ptr_);
      ptr_ = nullptr;
    }
  }

  static Sq8 train(const float *vectors, uint32_t count, uint32_t dim) {
    nbase_sq8_t *p = nullptr;
    detail::check(nbase_sq8_train(vectors, count, dim, &p));
    return Sq8(p);
  }

  static Sq8 load(const std::string &file_path) {
    nbase_sq8_t *p = nullptr;
    detail::check(nbase_sq8_load(file_path.c_str(), &p));
    return Sq8(p);
  }

  uint32_t dim() const noexcept { return nbase_sq8_dim(ptr_); }

  std::vector<uint8_t> encode(const float *vectors, uint32_t count) const {
    std::vector<uint8_t> out(static_cast<size_t>(count) * dim());
    detail::check(nbase_sq8_encode(ptr_, vectors, count, out.data()));
    return out;
  }

  std::vector<float> decode(const uint8_t *codes, uint32_t count) const {
    std::vector<float> out(static_cast<size_t>(count) * dim());
    detail::check(nbase_sq8_decode(ptr_, codes, count, out.data()));
    return out;
  }

  float l2_sq_asymmetric(const float *query, const uint8_t *code) const noexcept {
    return nbase_sq8_l2_sq_asymmetric(ptr_, query, code);
  }

  float l2_sq_symmetric(const uint8_t *code_a, const uint8_t *code_b) const noexcept {
    return nbase_sq8_l2_sq_symmetric(ptr_, code_a, code_b);
  }

  std::vector<float> l2_sq_asymmetric_batch(
      const std::vector<float> &query,
      const uint8_t *codes,
      uint32_t count) const {
    if (query.size() != dim()) {
      throw Error(NBASE_ERR_DIM_MISMATCH);
    }
    std::vector<float> out(count);
    detail::check(nbase_sq8_l2_sq_asymmetric_batch(
        ptr_, query.data(), codes, count, out.empty() ? nullptr : out.data()));
    return out;
  }

  std::vector<float> l2_sq_asymmetric_batch(
      const std::vector<float> &query,
      const std::vector<uint8_t> &codes) const {
    uint32_t d = dim();
    if (d == 0U || (codes.size() % d) != 0U) {
      throw Error(NBASE_ERR_INVALID_ARG);
    }
    uint32_t count = static_cast<uint32_t>(codes.size() / d);
    return l2_sq_asymmetric_batch(query, codes.data(), count);
  }

  std::vector<float> l2_sq_symmetric_batch(
      const uint8_t *query_code,
      const uint8_t *codes,
      uint32_t count) const {
    std::vector<float> out(count);
    detail::check(nbase_sq8_l2_sq_symmetric_batch(
        ptr_, query_code, codes, count, out.empty() ? nullptr : out.data()));
    return out;
  }

  std::vector<float> l2_sq_symmetric_batch(
      const std::vector<uint8_t> &query_code,
      const std::vector<uint8_t> &codes) const {
    uint32_t d = dim();
    if (d == 0U || query_code.size() != d || (codes.size() % d) != 0U) {
      throw Error(NBASE_ERR_INVALID_ARG);
    }
    uint32_t count = static_cast<uint32_t>(codes.size() / d);
    return l2_sq_symmetric_batch(query_code.data(), codes.data(), count);
  }

  void save(const std::string &file_path) const {
    detail::check(nbase_sq8_save(ptr_, file_path.c_str()));
  }

  nbase_sq8_t *raw() noexcept { return ptr_; }

 private:
  nbase_sq8_t *ptr_ = nullptr;
};

class Bq {
 public:
  Bq() noexcept = default;
  explicit Bq(nbase_bq_t *raw_bq) noexcept : ptr_(raw_bq) {}
  Bq(const Bq &) = delete;
  Bq &operator=(const Bq &) = delete;
  Bq(Bq &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Bq &operator=(Bq &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  ~Bq() { reset(); }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      nbase_bq_destroy(ptr_);
      ptr_ = nullptr;
    }
  }

  static Bq train(const float *vectors, uint32_t count, uint32_t dim) {
    nbase_bq_t *p = nullptr;
    detail::check(nbase_bq_train(vectors, count, dim, &p));
    return Bq(p);
  }

  static Bq load(const std::string &file_path) {
    nbase_bq_t *p = nullptr;
    detail::check(nbase_bq_load(file_path.c_str(), &p));
    return Bq(p);
  }

  uint32_t dim() const noexcept { return nbase_bq_dim(ptr_); }
  uint32_t words_per_vec() const noexcept { return nbase_bq_words_per_vec(ptr_); }

  std::vector<uint64_t> encode(const float *vectors, uint32_t count) const {
    std::vector<uint64_t> out(static_cast<size_t>(count) * words_per_vec());
    detail::check(nbase_bq_encode(ptr_, vectors, count, out.data()));
    return out;
  }

  uint32_t hamming(const uint64_t *a, const uint64_t *b) const noexcept {
    return nbase_bq_hamming(a, b, words_per_vec());
  }

  std::vector<uint32_t> hamming_batch(
      const uint64_t *query_words,
      const uint64_t *codes,
      uint32_t count) const {
    std::vector<uint32_t> out(count);
    detail::check(nbase_bq_hamming_batch(
        query_words,
        codes,
        count,
        words_per_vec(),
        out.empty() ? nullptr : out.data()));
    return out;
  }

  std::vector<uint32_t> hamming_batch(
      const std::vector<uint64_t> &query_words,
      const std::vector<uint64_t> &codes) const {
    uint32_t wpv = words_per_vec();
    if (wpv == 0U || query_words.size() < wpv || (codes.size() % wpv) != 0U) {
      throw Error(NBASE_ERR_INVALID_ARG);
    }
    uint32_t count = static_cast<uint32_t>(codes.size() / wpv);
    return hamming_batch(query_words.data(), codes.data(), count);
  }

  void save(const std::string &file_path) const {
    detail::check(nbase_bq_save(ptr_, file_path.c_str()));
  }

  nbase_bq_t *raw() noexcept { return ptr_; }

 private:
  nbase_bq_t *ptr_ = nullptr;
};

class Pq {
 public:
  Pq() noexcept = default;
  explicit Pq(nbase_pq_t *raw_pq) noexcept : ptr_(raw_pq) {}
  Pq(const Pq &) = delete;
  Pq &operator=(const Pq &) = delete;
  Pq(Pq &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Pq &operator=(Pq &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  ~Pq() { reset(); }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      nbase_pq_destroy(ptr_);
      ptr_ = nullptr;
    }
  }

  static Pq train(
      const float *vectors,
      uint32_t count,
      uint32_t dim,
      uint32_t num_subvectors,
      uint32_t num_centroids = 256U,
      uint32_t max_iters = 20U,
      uint64_t seed = 42ULL) {
    nbase_pq_t *p = nullptr;
    detail::check(nbase_pq_train(
        vectors, count, dim, num_subvectors, num_centroids, max_iters, seed, &p));
    return Pq(p);
  }

  static Pq load(const std::string &file_path) {
    nbase_pq_t *p = nullptr;
    detail::check(nbase_pq_load(file_path.c_str(), &p));
    return Pq(p);
  }

  uint32_t dim() const noexcept { return nbase_pq_dim(ptr_); }
  uint32_t num_subvectors() const noexcept { return nbase_pq_num_subvectors(ptr_); }
  uint32_t num_centroids() const noexcept { return nbase_pq_num_centroids(ptr_); }

  std::vector<uint8_t> encode(const float *vectors, uint32_t count) const {
    std::vector<uint8_t> out(static_cast<size_t>(count) * num_subvectors());
    detail::check(nbase_pq_encode(ptr_, vectors, count, out.data()));
    return out;
  }

  std::vector<float> decode(const uint8_t *codes, uint32_t count) const {
    std::vector<float> out(static_cast<size_t>(count) * dim());
    detail::check(nbase_pq_decode(ptr_, codes, count, out.data()));
    return out;
  }

  std::vector<float> build_lut(const std::vector<float> &query) const {
    if (query.size() != dim()) {
      throw Error(NBASE_ERR_DIM_MISMATCH);
    }
    std::vector<float> lut(
        static_cast<size_t>(num_subvectors()) * static_cast<size_t>(num_centroids()));
    detail::check(nbase_pq_build_lut(ptr_, query.data(), lut.data()));
    return lut;
  }

  float adc_distance(const std::vector<float> &lut, const uint8_t *code) const {
    size_t req = static_cast<size_t>(num_subvectors()) * static_cast<size_t>(num_centroids());
    if (lut.size() < req || code == nullptr) {
      throw Error(NBASE_ERR_INVALID_ARG);
    }
    return nbase_pq_adc_distance(ptr_, lut.data(), code);
  }

  std::vector<float> adc_distance_batch(
      const std::vector<float> &lut,
      const uint8_t *codes,
      uint32_t count) const {
    size_t req = static_cast<size_t>(num_subvectors()) * static_cast<size_t>(num_centroids());
    if (lut.size() < req) {
      throw Error(NBASE_ERR_INVALID_ARG);
    }
    std::vector<float> out(count);
    detail::check(nbase_pq_adc_distance_batch(
        ptr_, lut.data(), codes, count, out.empty() ? nullptr : out.data()));
    return out;
  }

  std::vector<float> adc_distance_batch(
      const std::vector<float> &lut,
      const std::vector<uint8_t> &codes) const {
    uint32_t m = num_subvectors();
    if (m == 0U || (codes.size() % m) != 0U) {
      throw Error(NBASE_ERR_INVALID_ARG);
    }
    uint32_t count = static_cast<uint32_t>(codes.size() / m);
    return adc_distance_batch(lut, codes.data(), count);
  }

  void save(const std::string &file_path) const {
    detail::check(nbase_pq_save(ptr_, file_path.c_str()));
  }

  nbase_pq_t *raw() noexcept { return ptr_; }
  const nbase_pq_t *raw() const noexcept { return ptr_; }

 private:
  nbase_pq_t *ptr_ = nullptr;
};

class Bm25 {
 public:
  Bm25() noexcept = default;
  explicit Bm25(nbase_bm25_t *raw_bm25) noexcept : ptr_(raw_bm25) {}
  Bm25(const Bm25 &) = delete;
  Bm25 &operator=(const Bm25 &) = delete;
  Bm25(Bm25 &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Bm25 &operator=(Bm25 &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  ~Bm25() { reset(); }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      nbase_bm25_destroy(ptr_);
      ptr_ = nullptr;
    }
  }

  static Bm25 create(float k1 = 1.5f, float b = 0.75f) {
    nbase_bm25_t *p = nullptr;
    detail::check(nbase_bm25_create(k1, b, &p));
    return Bm25(p);
  }

  static Bm25 load(const std::string &file_path) {
    nbase_bm25_t *p = nullptr;
    detail::check(nbase_bm25_load(file_path.c_str(), &p));
    return Bm25(p);
  }

  uint64_t doc_count() const noexcept { return nbase_bm25_doc_count(ptr_); }

  void index(uint64_t id, const std::string &text) {
    detail::check(nbase_bm25_index(ptr_, id, text.c_str()));
  }

  std::string get(uint64_t id) const {
    size_t len = 0U;
    detail::check(nbase_bm25_get(ptr_, id, nullptr, 0U, &len));
    std::string out(len, '\0');
    detail::check(nbase_bm25_get(ptr_, id, &out[0], len + 1U, &len));
    return out;
  }

  void remove(uint64_t id) { detail::check(nbase_bm25_delete(ptr_, id)); }

  std::vector<nbase_scored_id_t> search(const std::string &query, uint32_t k) const {
    std::vector<nbase_scored_id_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_bm25_search(
        ptr_, query.c_str(), k, k > 0U ? out.data() : nullptr, &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_scored_id_t> search_filtered(
      const std::string &query,
      uint32_t k,
      nbase_filter_fn filter,
      void *filter_ctx = nullptr) const {
    std::vector<nbase_scored_id_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_bm25_search_filtered(
        ptr_,
        query.c_str(),
        k,
        filter,
        filter_ctx,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  void save(const std::string &file_path) const {
    detail::check(nbase_bm25_save(ptr_, file_path.c_str()));
  }

  nbase_bm25_t *raw() noexcept { return ptr_; }

 private:
  nbase_bm25_t *ptr_ = nullptr;
};

class Lsh {
 public:
  Lsh() noexcept = default;
  explicit Lsh(nbase_lsh_t *raw_lsh) noexcept : ptr_(raw_lsh) {}
  Lsh(const Lsh &) = delete;
  Lsh &operator=(const Lsh &) = delete;
  Lsh(Lsh &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Lsh &operator=(Lsh &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  ~Lsh() { reset(); }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      nbase_lsh_destroy(ptr_);
      ptr_ = nullptr;
    }
  }

  static Lsh create(const nbase_lsh_config_t &cfg) {
    nbase_lsh_t *p = nullptr;
    detail::check(nbase_lsh_create(&cfg, &p));
    return Lsh(p);
  }

  static Lsh load(const std::string &file_path) {
    nbase_lsh_t *p = nullptr;
    detail::check(nbase_lsh_load(file_path.c_str(), &p));
    return Lsh(p);
  }

  void add(uint64_t id, const float *vec, uint32_t dim) {
    detail::check(nbase_lsh_add(ptr_, id, vec, dim));
  }

  void add(uint64_t id, const std::vector<float> &vec) {
    add(id, vec.data(), static_cast<uint32_t>(vec.size()));
  }

  void add_batch(
      const std::vector<uint64_t> &ids,
      const std::vector<float> &vectors,
      uint32_t dim) {
    if (dim == 0U || vectors.size() != ids.size() * static_cast<size_t>(dim)) {
      throw Error(NBASE_ERR_INVALID_ARG);
    }
    detail::check(nbase_lsh_add_batch(
        ptr_,
        ids.empty() ? nullptr : ids.data(),
        vectors.empty() ? nullptr : vectors.data(),
        static_cast<uint32_t>(ids.size()),
        dim));
  }

  void remove(uint64_t id) { detail::check(nbase_lsh_delete(ptr_, id)); }

  void upsert(uint64_t id, const float *vec, uint32_t dim) {
    detail::check(nbase_lsh_upsert(ptr_, id, vec, dim));
  }

  void upsert(uint64_t id, const std::vector<float> &vec) {
    upsert(id, vec.data(), static_cast<uint32_t>(vec.size()));
  }

  std::vector<float> get(uint64_t id, uint32_t dim) const {
    std::vector<float> out(dim);
    detail::check(nbase_lsh_get(ptr_, id, out.data(), dim));
    return out;
  }

  bool contains(uint64_t id) const { return nbase_lsh_contains(ptr_, id); }

  std::vector<uint64_t> list_ids(
      uint64_t offset, uint32_t limit, uint64_t *out_total = nullptr) const {
    std::vector<uint64_t> ids(limit);
    uint32_t got = 0U;
    detail::check(nbase_lsh_list_ids(
        ptr_, offset, limit, limit > 0U ? ids.data() : nullptr, &got, out_total));
    ids.resize(got);
    return ids;
  }

  void compact() { detail::check(nbase_lsh_compact(ptr_)); }

  nbase_lsh_stats_t stats() const {
    nbase_lsh_stats_t s{};
    detail::check(nbase_lsh_stats(ptr_, &s));
    return s;
  }

  std::vector<nbase_result_t> search(
      const std::vector<float> &query,
      uint32_t k,
      uint32_t multi_probe = 2U) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_lsh_search(
        ptr_,
        query.data(),
        static_cast<uint32_t>(query.size()),
        k,
        multi_probe,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_result_t> search_filtered(
      const std::vector<float> &query,
      uint32_t k,
      uint32_t multi_probe,
      nbase_filter_fn filter,
      void *filter_ctx = nullptr) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_lsh_search_filtered(
        ptr_,
        query.data(),
        static_cast<uint32_t>(query.size()),
        k,
        multi_probe,
        filter,
        filter_ctx,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<std::vector<nbase_result_t>> search_batch(
      const std::vector<float> &queries,
      uint32_t query_count,
      uint32_t dim,
      uint32_t k,
      uint32_t multi_probe = 2U,
      uint32_t num_threads = 0U,
      std::vector<uint32_t> *out_counts = nullptr) const {
    if (dim == 0U || queries.size() != static_cast<size_t>(query_count) * static_cast<size_t>(dim)) {
      throw Error(dim == 0U ? NBASE_ERR_INVALID_ARG : NBASE_ERR_DIM_MISMATCH);
    }
    if (query_count == 0U) {
      detail::check(nbase_lsh_search_batch(
          ptr_, nullptr, 0U, dim, k, multi_probe, num_threads, nullptr, nullptr));
      if (out_counts != nullptr) {
        out_counts->clear();
      }
      return {};
    }
    std::vector<nbase_result_t> flat(static_cast<size_t>(query_count) * static_cast<size_t>(k));
    std::vector<uint32_t> counts(query_count, 0U);
    detail::check(nbase_lsh_search_batch(
        ptr_,
        queries.data(),
        query_count,
        dim,
        k,
        multi_probe,
        num_threads,
        k > 0U ? flat.data() : nullptr,
        counts.data()));
    std::vector<std::vector<nbase_result_t>> res(query_count);
    for (uint32_t q = 0U; q < query_count; q++) {
      const nbase_result_t *row = flat.data() + static_cast<size_t>(q) * static_cast<size_t>(k);
      res[q].assign(row, row + counts[q]);
    }
    if (out_counts != nullptr) {
      *out_counts = counts;
    }
    return res;
  }

  void save(const std::string &file_path) const {
    detail::check(nbase_lsh_save(ptr_, file_path.c_str()));
  }

  nbase_lsh_t *raw() noexcept { return ptr_; }

 private:
  nbase_lsh_t *ptr_ = nullptr;
};

class PartitionedDatabase {
 public:
  PartitionedDatabase() noexcept = default;
  explicit PartitionedDatabase(nbase_pdb_t *raw_pdb) noexcept : pdb_(raw_pdb) {}
  PartitionedDatabase(const PartitionedDatabase &) = delete;
  PartitionedDatabase &operator=(const PartitionedDatabase &) = delete;

  PartitionedDatabase(PartitionedDatabase &&other) noexcept : pdb_(other.pdb_) {
    other.pdb_ = nullptr;
  }

  PartitionedDatabase &operator=(PartitionedDatabase &&other) noexcept {
    if (this != &other) {
      reset();
      pdb_ = other.pdb_;
      other.pdb_ = nullptr;
    }
    return *this;
  }

  ~PartitionedDatabase() { reset(); }

  void reset() noexcept {
    if (pdb_ != nullptr) {
      (void)nbase_pdb_close(pdb_);
      pdb_ = nullptr;
    }
  }

  static PartitionedDatabase open(
      const std::string &dir_path, const nbase_pdb_config_t &cfg) {
    nbase_pdb_t *p = nullptr;
    detail::check(nbase_pdb_open(dir_path.c_str(), &cfg, &p));
    return PartitionedDatabase(p);
  }

  void save() { detail::check(nbase_pdb_save(pdb_)); }

  void add(uint64_t id, const float *vec, uint32_t dim) {
    detail::check(nbase_pdb_add(pdb_, id, vec, dim));
  }

  void add(uint64_t id, const std::vector<float> &vec) {
    add(id, vec.data(), static_cast<uint32_t>(vec.size()));
  }

  void add_batch(
      const std::vector<uint64_t> &ids,
      const std::vector<float> &vectors,
      uint32_t dim,
      uint32_t num_threads = 0U) {
    detail::check(nbase_pdb_add_batch(
        pdb_,
        ids.empty() ? nullptr : ids.data(),
        vectors.empty() ? nullptr : vectors.data(),
        static_cast<uint32_t>(ids.size()),
        dim,
        num_threads));
  }

  void upsert(uint64_t id, const float *vec, uint32_t dim) {
    detail::check(nbase_pdb_upsert(pdb_, id, vec, dim));
  }

  void upsert(uint64_t id, const std::vector<float> &vec) {
    upsert(id, vec.data(), static_cast<uint32_t>(vec.size()));
  }

  void upsert_batch(
      const std::vector<uint64_t> &ids,
      const std::vector<float> &vectors,
      uint32_t dim,
      uint32_t num_threads = 0U) {
    detail::check(nbase_pdb_upsert_batch(
        pdb_,
        ids.empty() ? nullptr : ids.data(),
        vectors.empty() ? nullptr : vectors.data(),
        static_cast<uint32_t>(ids.size()),
        dim,
        num_threads));
  }

  void remove(uint64_t id) { detail::check(nbase_pdb_delete(pdb_, id)); }

  uint32_t delete_batch(const std::vector<uint64_t> &ids) {
    uint32_t deleted = 0U;
    detail::check(nbase_pdb_delete_batch(
        pdb_, ids.empty() ? nullptr : ids.data(), static_cast<uint32_t>(ids.size()), &deleted));
    return deleted;
  }

  uint32_t delete_by_filter(const nbase_meta_filter_t &filter) {
    uint32_t deleted = 0U;
    detail::check(nbase_pdb_delete_by_filter(pdb_, &filter, &deleted));
    return deleted;
  }

  std::vector<float> get(uint64_t id, uint32_t dim) const {
    std::vector<float> out(dim);
    detail::check(nbase_pdb_get(pdb_, id, out.data(), dim));
    return out;
  }

  void get_batch(
      const std::vector<uint64_t> &ids,
      uint32_t dim,
      std::vector<float> &out_vectors,
      std::vector<bool> &out_found) const {
    const size_t n = ids.size();
    out_vectors.resize(n * static_cast<size_t>(dim));
    out_found.assign(n, false);
    if (n == 0U) {
      return;
    }
    std::unique_ptr<bool[]> found_buf(new bool[n]);
    detail::check(nbase_pdb_get_batch(
        pdb_, ids.data(), static_cast<uint32_t>(n), dim, out_vectors.data(), found_buf.get()));
    for (size_t i = 0U; i < n; ++i) {
      out_found[i] = found_buf[i];
    }
  }

  bool contains(uint64_t id) const { return nbase_pdb_contains(pdb_, id); }

  std::vector<uint64_t> list_ids(
      uint64_t offset, uint32_t limit, uint64_t *out_total = nullptr) const {
    std::vector<uint64_t> ids(limit);
    uint32_t count = 0U;
    detail::check(nbase_pdb_list_ids(
        pdb_, offset, limit, limit > 0U ? ids.data() : nullptr, &count, out_total));
    ids.resize(count);
    return ids;
  }

  uint64_t count(const nbase_meta_filter_t *filter = nullptr) const {
    uint64_t out = 0ULL;
    detail::check(nbase_pdb_count(pdb_, filter, &out));
    return out;
  }

  std::vector<nbase_result_t> search_radius(
      const std::vector<float> &query,
      float max_distance,
      uint32_t max_results,
      const nbase_pdb_search_opts_t *opts = nullptr) const {
    std::vector<nbase_result_t> out(max_results);
    uint32_t count = 0U;
    detail::check(nbase_pdb_search_radius(
        pdb_,
        query.data(),
        static_cast<uint32_t>(query.size()),
        max_distance,
        max_results,
        opts,
        max_results > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  void set_meta(uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
    detail::check(nbase_pdb_set_meta(pdb_, id, fields, count));
  }

  void set_meta_batch(const std::vector<nbase_meta_batch_item_t> &items) {
    detail::check(nbase_pdb_set_meta_batch(
        pdb_, items.empty() ? nullptr : items.data(), static_cast<uint32_t>(items.size())));
  }

  void update_meta(uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
    detail::check(nbase_pdb_update_meta(pdb_, id, fields, count));
  }

  void update_meta_batch(const std::vector<nbase_meta_batch_item_t> &items) {
    detail::check(nbase_pdb_update_meta_batch(
        pdb_, items.empty() ? nullptr : items.data(), static_cast<uint32_t>(items.size())));
  }

  void delete_meta(uint64_t id) { detail::check(nbase_pdb_delete_meta(pdb_, id)); }

  std::vector<uint64_t> query_meta(
      const nbase_meta_filter_t *filter, uint32_t limit) const {
    std::vector<uint64_t> ids(limit);
    uint32_t count = 0U;
    detail::check(nbase_pdb_query_meta(
        pdb_, filter, limit, limit > 0U ? ids.data() : nullptr, &count));
    ids.resize(count);
    return ids;
  }

  void set_text(uint64_t id, const std::string &text) {
    detail::check(nbase_pdb_set_text(pdb_, id, text.c_str()));
  }

  void set_text_batch(
      const std::vector<uint64_t> &ids,
      const std::vector<std::string> &texts) {
    if (ids.size() != texts.size()) {
      throw Error(NBASE_ERR_INVALID_ARG);
    }
    std::vector<const char *> ptrs;
    ptrs.reserve(texts.size());
    for (const std::string &t : texts) {
      ptrs.push_back(t.c_str());
    }
    detail::check(nbase_pdb_set_text_batch(
        pdb_,
        ids.empty() ? nullptr : ids.data(),
        ptrs.empty() ? nullptr : ptrs.data(),
        static_cast<uint32_t>(ids.size())));
  }

  std::string get_text(uint64_t id) const {
    size_t len = 0U;
    detail::check(nbase_pdb_get_text(pdb_, id, nullptr, 0U, &len));
    std::string out(len, '\0');
    detail::check(nbase_pdb_get_text(pdb_, id, &out[0], len + 1U, &len));
    return out;
  }

  void delete_text(uint64_t id) { detail::check(nbase_pdb_delete_text(pdb_, id)); }

  std::vector<nbase_result_t> search(
      const std::vector<float> &query,
      uint32_t k,
      const nbase_pdb_search_opts_t *opts = nullptr) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_pdb_search(
        pdb_,
        query.data(),
        static_cast<uint32_t>(query.size()),
        k,
        opts,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_scored_id_t> search_text(
      const std::string &query,
      uint32_t k,
      const nbase_pdb_search_opts_t *opts = nullptr) const {
    std::vector<nbase_scored_id_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_pdb_search_text(
        pdb_, query.c_str(), k, opts, k > 0U ? out.data() : nullptr, &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_scored_id_t> search_hybrid(
      const std::vector<float> &query_vec,
      const std::string &query_text,
      uint32_t k,
      float alpha = 0.5f,
      float rrf_k = 60.0f,
      const nbase_pdb_search_opts_t *opts = nullptr) const {
    std::vector<nbase_scored_id_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_pdb_search_hybrid(
        pdb_,
        query_vec.empty() ? nullptr : query_vec.data(),
        static_cast<uint32_t>(query_vec.size()),
        query_text.empty() ? nullptr : query_text.c_str(),
        k,
        alpha,
        rrf_k,
        opts,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_result_t> find_similar(
      uint64_t id,
      uint32_t k,
      const nbase_pdb_search_opts_t *opts = nullptr) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_pdb_find_similar(
        pdb_, id, k, opts, k > 0U ? out.data() : nullptr, &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_result_t> rerank_mmr(
      const std::vector<nbase_result_t> &candidates,
      uint32_t k,
      float lambda) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_pdb_rerank_mmr(
        pdb_,
        candidates.empty() ? nullptr : candidates.data(),
        static_cast<uint32_t>(candidates.size()),
        k,
        lambda,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_result_t> rerank_weighted(
      const std::vector<nbase_result_t> &candidates,
      const std::vector<nbase_rerank_weight_t> &weights,
      uint32_t k) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_pdb_rerank_weighted(
        pdb_,
        candidates.empty() ? nullptr : candidates.data(),
        static_cast<uint32_t>(candidates.size()),
        weights.empty() ? nullptr : weights.data(),
        static_cast<uint32_t>(weights.size()),
        k,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<std::vector<nbase_result_t>> search_batch(
      const std::vector<float> &queries,
      uint32_t query_count,
      uint32_t dim,
      uint32_t k,
      const nbase_pdb_search_opts_t *opts = nullptr) const {
    std::vector<nbase_result_t> flat(static_cast<size_t>(query_count) * k);
    std::vector<uint32_t> counts(query_count, 0U);
    detail::check(nbase_pdb_search_batch(
        pdb_,
        queries.empty() ? nullptr : queries.data(),
        query_count,
        dim,
        k,
        opts,
        flat.empty() ? nullptr : flat.data(),
        counts.empty() ? nullptr : counts.data()));
    std::vector<std::vector<nbase_result_t>> out(query_count);
    for (uint32_t i = 0U; i < query_count; ++i) {
      size_t off = static_cast<size_t>(i) * k;
      out[i].assign(flat.begin() + static_cast<std::ptrdiff_t>(off),
                    flat.begin() + static_cast<std::ptrdiff_t>(off + counts[i]));
    }
    return out;
  }

  std::vector<nbase_edge_t> extract_relationships(
      uint32_t k_neighbors,
      float max_distance,
      const nbase_pdb_search_opts_t *opts = nullptr) const {
    struct EdgeGuard {
      nbase_edge_t *ptr = nullptr;
      ~EdgeGuard() { nbase_free_edges(ptr); }
    } guard;
    uint64_t edge_cnt = 0ULL;
    detail::check(nbase_pdb_extract_relationships(
        pdb_, k_neighbors, max_distance, opts, &guard.ptr, &edge_cnt));
    std::vector<nbase_edge_t> out;
    if (guard.ptr != nullptr && edge_cnt > 0ULL) {
      out.assign(guard.ptr, guard.ptr + edge_cnt);
    }
    return out;
  }

  void compact() { detail::check(nbase_pdb_compact(pdb_)); }

  void rebuild_hnsw_mt(uint32_t num_threads = 0U) {
    detail::check(nbase_pdb_rebuild_hnsw_mt(pdb_, num_threads));
  }

  nbase_pdb_stats_t stats() const {
    nbase_pdb_stats_t s{};
    detail::check(nbase_pdb_stats(pdb_, &s));
    return s;
  }

  nbase_pdb_t *raw() noexcept { return pdb_; }

 private:
  nbase_pdb_t *pdb_ = nullptr;
};

class IvfPq {
 public:
  IvfPq() noexcept = default;
  explicit IvfPq(nbase_ivfpq_t *raw_idx) noexcept : ptr_(raw_idx) {}
  IvfPq(const IvfPq &) = delete;
  IvfPq &operator=(const IvfPq &) = delete;

  IvfPq(IvfPq &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

  IvfPq &operator=(IvfPq &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  ~IvfPq() { reset(); }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      nbase_ivfpq_destroy(ptr_);
      ptr_ = nullptr;
    }
  }

  static IvfPq train(
      const nbase_ivfpq_config_t &cfg,
      const float *train_vectors,
      uint32_t train_count) {
    nbase_ivfpq_t *p = nullptr;
    detail::check(nbase_ivfpq_train(&cfg, train_vectors, train_count, &p));
    return IvfPq(p);
  }

  static IvfPq load(const std::string &file_path) {
    nbase_ivfpq_t *p = nullptr;
    detail::check(nbase_ivfpq_load(file_path.c_str(), &p));
    return IvfPq(p);
  }

  void add(const std::vector<uint64_t> &ids, const std::vector<float> &vectors) {
    detail::check(nbase_ivfpq_add(
        ptr_, ids.data(), vectors.data(), static_cast<uint32_t>(ids.size())));
  }

  void upsert(uint64_t id, const float *vec, uint32_t dim) {
    detail::check(nbase_ivfpq_upsert(ptr_, id, vec, dim));
  }

  void upsert(uint64_t id, const std::vector<float> &vec) {
    upsert(id, vec.data(), static_cast<uint32_t>(vec.size()));
  }

  void remove(uint64_t id) { detail::check(nbase_ivfpq_delete(ptr_, id)); }

  bool contains(uint64_t id) const { return nbase_ivfpq_contains(ptr_, id); }

  std::vector<uint64_t> list_ids(
      uint64_t offset, uint32_t limit, uint64_t *out_total = nullptr) const {
    std::vector<uint64_t> ids(limit);
    uint32_t got = 0U;
    detail::check(nbase_ivfpq_list_ids(
        ptr_, offset, limit, limit > 0U ? ids.data() : nullptr, &got, out_total));
    ids.resize(got);
    return ids;
  }

  void compact() { detail::check(nbase_ivfpq_compact(ptr_)); }

  std::vector<nbase_result_t> search(
      const std::vector<float> &query, uint32_t k, uint32_t nprobe = 8U) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_ivfpq_search(
        ptr_,
        query.data(),
        k,
        nprobe,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_result_t> search_filtered(
      const std::vector<float> &query,
      uint32_t k,
      uint32_t nprobe,
      nbase_filter_fn filter,
      void *filter_ctx = nullptr) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_ivfpq_search_filtered(
        ptr_,
        query.data(),
        k,
        nprobe,
        filter,
        filter_ctx,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<std::vector<nbase_result_t>> search_batch(
      const std::vector<float> &queries,
      uint32_t query_count,
      uint32_t k,
      uint32_t nprobe = 8U,
      uint32_t num_threads = 0U) const {
    if (query_count == 0U) {
      return {};
    }
    uint32_t d = stats().dim;
    if (d == 0U || queries.size() != static_cast<size_t>(query_count) * static_cast<size_t>(d)) {
      throw Error(NBASE_ERR_DIM_MISMATCH);
    }
    std::vector<nbase_result_t> flat(static_cast<size_t>(query_count) * static_cast<size_t>(k));
    std::vector<uint32_t> counts(query_count, 0U);
    detail::check(nbase_ivfpq_search_batch(
        ptr_,
        queries.data(),
        query_count,
        k,
        nprobe,
        num_threads,
        k > 0U ? flat.data() : nullptr,
        counts.data()));
    std::vector<std::vector<nbase_result_t>> res(query_count);
    for (uint32_t q = 0U; q < query_count; q++) {
      const nbase_result_t *row = flat.data() + static_cast<size_t>(q) * static_cast<size_t>(k);
      res[q].assign(row, row + counts[q]);
    }
    return res;
  }

  void save(const std::string &file_path) const {
    detail::check(nbase_ivfpq_save(ptr_, file_path.c_str()));
  }

  nbase_ivfpq_stats_t stats() const {
    nbase_ivfpq_stats_t s{};
    detail::check(nbase_ivfpq_stats(ptr_, &s));
    return s;
  }

  nbase_ivfpq_t *raw() noexcept { return ptr_; }

 private:
  nbase_ivfpq_t *ptr_ = nullptr;
};

class HttpServer {
 public:
  HttpServer() noexcept = default;
  explicit HttpServer(nbase_http_server_t *raw_srv) noexcept : srv_(raw_srv) {}
  HttpServer(const HttpServer &) = delete;
  HttpServer &operator=(const HttpServer &) = delete;

  HttpServer(HttpServer &&other) noexcept : srv_(other.srv_) {
    other.srv_ = nullptr;
  }

  HttpServer &operator=(HttpServer &&other) noexcept {
    if (this != &other) {
      stop();
      srv_ = other.srv_;
      other.srv_ = nullptr;
    }
    return *this;
  }

  ~HttpServer() { stop(); }

  void stop() noexcept {
    if (srv_ != nullptr) {
      nbase_http_server_stop(srv_);
      srv_ = nullptr;
    }
  }

  static HttpServer start(Database &db, const nbase_http_config_t &cfg) {
    nbase_http_server_t *s = nullptr;
    detail::check(nbase_http_server_start(db.raw(), &cfg, &s));
    return HttpServer(s);
  }

  uint16_t port() const noexcept { return nbase_http_server_port(srv_); }
  nbase_http_server_t *raw() noexcept { return srv_; }

 private:
  nbase_http_server_t *srv_ = nullptr;
};

class ClusteredDatabase {
 public:
  ClusteredDatabase() noexcept = default;
  explicit ClusteredDatabase(nbase_clustered_t *raw_cl) noexcept : ptr_(raw_cl) {}
  ClusteredDatabase(const ClusteredDatabase &) = delete;
  ClusteredDatabase &operator=(const ClusteredDatabase &) = delete;

  ClusteredDatabase(ClusteredDatabase &&other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  ClusteredDatabase &operator=(ClusteredDatabase &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  ~ClusteredDatabase() { reset(); }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      nbase_clustered_destroy(ptr_);
      ptr_ = nullptr;
    }
  }

  static ClusteredDatabase create(const nbase_clustered_config_t &cfg) {
    nbase_clustered_t *p = nullptr;
    detail::check(nbase_clustered_create(&cfg, &p));
    return ClusteredDatabase(p);
  }

  static ClusteredDatabase create(uint32_t dim) {
    return create(nbase_clustered_config_default(dim));
  }

  static ClusteredDatabase load(const std::string &file_path) {
    nbase_clustered_t *p = nullptr;
    detail::check(nbase_clustered_load(file_path.c_str(), &p));
    return ClusteredDatabase(p);
  }

  void add(uint64_t id, const float *vec, uint32_t dim) {
    detail::check(nbase_clustered_add(ptr_, id, vec, dim));
  }

  void add(uint64_t id, const std::vector<float> &vec) {
    add(id, vec.data(), static_cast<uint32_t>(vec.size()));
  }

  void add_batch(
      const std::vector<uint64_t> &ids,
      const std::vector<float> &vectors,
      uint32_t dim) {
    detail::check(nbase_clustered_add_batch(
        ptr_,
        ids.empty() ? nullptr : ids.data(),
        vectors.empty() ? nullptr : vectors.data(),
        static_cast<uint32_t>(ids.size()),
        dim));
  }

  void upsert(uint64_t id, const float *vec, uint32_t dim) {
    detail::check(nbase_clustered_upsert(ptr_, id, vec, dim));
  }

  void upsert(uint64_t id, const std::vector<float> &vec) {
    upsert(id, vec.data(), static_cast<uint32_t>(vec.size()));
  }

  void upsert_batch(
      const std::vector<uint64_t> &ids,
      const std::vector<float> &vectors,
      uint32_t dim) {
    detail::check(nbase_clustered_upsert_batch(
        ptr_,
        ids.empty() ? nullptr : ids.data(),
        vectors.empty() ? nullptr : vectors.data(),
        static_cast<uint32_t>(ids.size()),
        dim));
  }

  void remove(uint64_t id) { detail::check(nbase_clustered_delete(ptr_, id)); }

  uint32_t delete_batch(const std::vector<uint64_t> &ids) {
    uint32_t deleted = 0U;
    detail::check(nbase_clustered_delete_batch(
        ptr_, ids.empty() ? nullptr : ids.data(), static_cast<uint32_t>(ids.size()), &deleted));
    return deleted;
  }

  std::vector<float> get(uint64_t id, uint32_t dim) const {
    std::vector<float> out(dim);
    detail::check(nbase_clustered_get(ptr_, id, out.data(), dim));
    return out;
  }

  void get_batch(
      const std::vector<uint64_t> &ids,
      uint32_t dim,
      std::vector<float> &out_vectors,
      std::vector<bool> &out_found) const {
    const size_t n = ids.size();
    out_vectors.resize(n * static_cast<size_t>(dim));
    out_found.assign(n, false);
    if (n == 0U) {
      return;
    }
    std::unique_ptr<bool[]> found_buf(new bool[n]);
    detail::check(nbase_clustered_get_batch(
        ptr_, ids.data(), static_cast<uint32_t>(n), dim, out_vectors.data(), found_buf.get()));
    for (size_t i = 0U; i < n; ++i) {
      out_found[i] = found_buf[i];
    }
  }

  std::vector<float> get_batch(
      const std::vector<uint64_t> &ids,
      uint32_t dim,
      std::vector<bool> *out_found = nullptr) const {
    std::vector<float> out_vectors;
    std::vector<bool> found;
    get_batch(ids, dim, out_vectors, found);
    if (out_found != nullptr) {
      *out_found = std::move(found);
    }
    return out_vectors;
  }

  bool contains(uint64_t id) const noexcept {
    return nbase_clustered_contains(ptr_, id);
  }

  std::vector<uint64_t> list_ids(
      uint64_t offset, uint32_t limit, uint64_t *out_total = nullptr) const {
    std::vector<uint64_t> out(limit);
    uint32_t got = 0U;
    detail::check(nbase_clustered_list_ids(
        ptr_, offset, limit, limit > 0U ? out.data() : nullptr, &got, out_total));
    out.resize(got);
    return out;
  }

  void recluster(uint32_t num_clusters = 0U) {
    detail::check(nbase_clustered_recluster(ptr_, num_clusters));
  }

  void compact() { detail::check(nbase_clustered_compact(ptr_)); }

  std::vector<nbase_result_t> search(
      const std::vector<float> &query, uint32_t k, uint32_t nprobe = 0U) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_clustered_search(
        ptr_,
        query.data(),
        static_cast<uint32_t>(query.size()),
        k,
        nprobe,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_result_t> search_filtered(
      const std::vector<float> &query,
      uint32_t k,
      uint32_t nprobe,
      nbase_filter_fn filter,
      void *filter_ctx = nullptr) const {
    std::vector<nbase_result_t> out(k);
    uint32_t count = 0U;
    detail::check(nbase_clustered_search_filtered(
        ptr_,
        query.data(),
        static_cast<uint32_t>(query.size()),
        k,
        nprobe,
        filter,
        filter_ctx,
        k > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<nbase_result_t> search_radius(
      const std::vector<float> &query,
      float max_distance,
      uint32_t max_results,
      uint32_t nprobe = 0U,
      nbase_filter_fn filter = nullptr,
      void *filter_ctx = nullptr) const {
    std::vector<nbase_result_t> out(max_results);
    uint32_t count = 0U;
    detail::check(nbase_clustered_search_radius(
        ptr_,
        query.data(),
        static_cast<uint32_t>(query.size()),
        max_distance,
        max_results,
        nprobe,
        filter,
        filter_ctx,
        max_results > 0U ? out.data() : nullptr,
        &count));
    out.resize(count);
    return out;
  }

  std::vector<std::vector<nbase_result_t>> search_batch(
      const std::vector<float> &queries,
      uint32_t query_count,
      uint32_t dim,
      uint32_t k,
      uint32_t nprobe = 0U,
      uint32_t num_threads = 0U,
      std::vector<uint32_t> *out_counts = nullptr) const {
    if (dim == 0U || queries.size() != static_cast<size_t>(query_count) * static_cast<size_t>(dim)) {
      throw Error(dim == 0U ? NBASE_ERR_INVALID_ARG : NBASE_ERR_DIM_MISMATCH);
    }
    if (query_count == 0U) {
      detail::check(nbase_clustered_search_batch(
          ptr_, nullptr, 0U, dim, k, nprobe, num_threads, nullptr, nullptr));
      if (out_counts != nullptr) {
        out_counts->clear();
      }
      return {};
    }
    std::vector<nbase_result_t> flat(static_cast<size_t>(query_count) * static_cast<size_t>(k));
    std::vector<uint32_t> counts(query_count, 0U);
    detail::check(nbase_clustered_search_batch(
        ptr_,
        queries.data(),
        query_count,
        dim,
        k,
        nprobe,
        num_threads,
        k > 0U ? flat.data() : nullptr,
        counts.data()));
    std::vector<std::vector<nbase_result_t>> res(query_count);
    for (uint32_t q = 0U; q < query_count; q++) {
      const nbase_result_t *row = flat.data() + static_cast<size_t>(q) * static_cast<size_t>(k);
      res[q].assign(row, row + counts[q]);
    }
    if (out_counts != nullptr) {
      *out_counts = counts;
    }
    return res;
  }

  nbase_clustered_stats_t stats() const {
    nbase_clustered_stats_t s{};
    detail::check(nbase_clustered_stats(ptr_, &s));
    return s;
  }

  void save(const std::string &file_path) const {
    detail::check(nbase_clustered_save(ptr_, file_path.c_str()));
  }

  nbase_clustered_t *raw() noexcept { return ptr_; }

 private:
  nbase_clustered_t *ptr_ = nullptr;
};

}  // namespace nbase

#endif /* NBASE_HPP */
