#include "nbase_internal.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define BM25_MAX_TOKEN 64U
#define BM25_EMPTY 0U
#define BM25_OCCUPIED 1U
#define BM25_TOMBSTONE 2U

typedef struct bm25_doc {
  char *text;
  uint32_t text_len;
  uint32_t doc_len;
} bm25_doc_t;

typedef struct bm25_posting {
  uint64_t id;
  uint32_t tf;
  uint32_t doc_len;
} bm25_posting_t;

typedef struct bm25_postings {
  bm25_posting_t *items;
  uint32_t len;
  uint32_t cap;
} bm25_postings_t;

struct nbase_bm25 {
  float k1;
  float b;
  uint64_t doc_count;
  uint64_t total_doc_len;

  size_t doc_cap;
  size_t doc_tombstones;
  uint64_t *doc_keys;
  bm25_doc_t *doc_vals;
  uint8_t *doc_states;

  size_t term_cap;
  size_t term_count;
  char **term_keys;
  bm25_postings_t *term_vals;

  pthread_rwlock_t rwlock;
  bool rwlock_init;
};

typedef struct tok_word {
  char s[BM25_MAX_TOKEN];
} tok_word_t;

typedef struct tok_count {
  char term[BM25_MAX_TOKEN];
  uint32_t tf;
} tok_count_t;

static inline uint64_t hash_u64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static inline uint64_t hash_cstr(const char *s) {
  uint64_t h = 14695981039346656037ULL;
  while (*s != '\0') {
    h ^= (uint64_t)(unsigned char)(*s++);
    h *= 1099511628211ULL;
  }
  return h;
}

static int cmp_tok_word(const void *pa, const void *pb) {
  const tok_word_t *a = (const tok_word_t *)pa;
  const tok_word_t *b = (const tok_word_t *)pb;
  return strcmp(a->s, b->s);
}

static int cmp_bm25_u64_asc(const void *pa, const void *pb) {
  uint64_t a = *(const uint64_t *)pa;
  uint64_t b = *(const uint64_t *)pb;
  return (a > b) - (a < b);
}

static bool scored_worse(nbase_scored_id_t a, nbase_scored_id_t b) {
  return a.score < b.score || (a.score == b.score && a.id > b.id);
}

static void scored_heap_push(
    nbase_scored_id_t *heap,
    uint32_t *len,
    uint32_t cap,
    nbase_scored_id_t item) {
  if (*len < cap) {
    uint32_t idx = (*len)++;
    while (idx > 0U) {
      uint32_t parent = (idx - 1U) / 2U;
      if (!scored_worse(item, heap[parent])) {
        break;
      }
      heap[idx] = heap[parent];
      idx = parent;
    }
    heap[idx] = item;
  } else if (*len > 0U && scored_worse(heap[0], item)) {
    uint32_t idx = 0U;
    uint32_t n = *len;
    while (true) {
      uint32_t left = 2U * idx + 1U;
      if (left >= n) {
        break;
      }
      uint32_t worst = left;
      uint32_t right = left + 1U;
      if (right < n && scored_worse(heap[right], heap[left])) {
        worst = right;
      }
      if (!scored_worse(heap[worst], item)) {
        break;
      }
      heap[idx] = heap[worst];
      idx = worst;
    }
    heap[idx] = item;
  }
}

static int cmp_scored_desc(const void *pa, const void *pb) {
  const nbase_scored_id_t *a = (const nbase_scored_id_t *)pa;
  const nbase_scored_id_t *b = (const nbase_scored_id_t *)pb;
  if (a->score > b->score) {
    return -1;
  }
  if (a->score < b->score) {
    return 1;
  }
  if (a->id < b->id) {
    return -1;
  }
  if (a->id > b->id) {
    return 1;
  }
  return 0;
}

static nbase_status_t tokenize_counts(
    const char *text,
    tok_count_t **out_counts,
    uint32_t *out_unique,
    uint32_t *out_total_tokens) {
  *out_counts = NULL;
  *out_unique = 0U;
  *out_total_tokens = 0U;
  if (!text) {
    return NBASE_ERR_INVALID_ARG;
  }

  tok_word_t *words = NULL;
  size_t len = 0U;
  size_t cap = 0U;

  const char *p = text;
  while (*p != '\0') {
    while (*p != '\0' && !isalnum((unsigned char)*p)) {
      p++;
    }
    if (*p == '\0') {
      break;
    }
    tok_word_t w;
    size_t wi = 0U;
    while (*p != '\0' && isalnum((unsigned char)*p)) {
      if (wi + 1U < BM25_MAX_TOKEN) {
        w.s[wi++] = (char)tolower((unsigned char)*p);
      }
      p++;
    }
    w.s[wi] = '\0';

    if (len == cap) {
      size_t new_cap = (cap == 0U) ? 32U : cap * 2U;
      tok_word_t *next = realloc(words, new_cap * sizeof(*next));
      if (!next) {
        free(words);
        return NBASE_ERR_NOMEM;
      }
      words = next;
      cap = new_cap;
    }
    words[len++] = w;
  }

  if (len == 0U) {
    free(words);
    return NBASE_OK;
  }

  qsort(words, len, sizeof(tok_word_t), cmp_tok_word);

  tok_count_t *counts = malloc(len * sizeof(tok_count_t));
  if (!counts) {
    free(words);
    return NBASE_ERR_NOMEM;
  }

  uint32_t uniq = 0U;
  for (size_t i = 0U; i < len; i++) {
    if (uniq == 0U || strcmp(counts[uniq - 1U].term, words[i].s) != 0) {
      memcpy(counts[uniq].term, words[i].s, BM25_MAX_TOKEN);
      counts[uniq].tf = 1U;
      uniq++;
    } else {
      counts[uniq - 1U].tf++;
    }
  }

  free(words);
  *out_counts = counts;
  *out_unique = uniq;
  *out_total_tokens = (uint32_t)len;
  return NBASE_OK;
}

static nbase_status_t doc_map_rehash(nbase_bm25_t *bm25, size_t new_cap) {
  uint64_t *nkeys = calloc(new_cap, sizeof(uint64_t));
  bm25_doc_t *nvals = calloc(new_cap, sizeof(bm25_doc_t));
  uint8_t *nstates = calloc(new_cap, sizeof(uint8_t));
  if (!nkeys || !nvals || !nstates) {
    free(nkeys);
    free(nvals);
    free(nstates);
    return NBASE_ERR_NOMEM;
  }

  size_t mask = new_cap - 1U;
  for (size_t i = 0U; i < bm25->doc_cap; i++) {
    if (bm25->doc_states[i] != BM25_OCCUPIED) {
      continue;
    }
    size_t pos = (size_t)hash_u64(bm25->doc_keys[i]) & mask;
    while (nstates[pos] == BM25_OCCUPIED) {
      pos = (pos + 1U) & mask;
    }
    nstates[pos] = BM25_OCCUPIED;
    nkeys[pos] = bm25->doc_keys[i];
    nvals[pos] = bm25->doc_vals[i];
  }

  free(bm25->doc_keys);
  free(bm25->doc_vals);
  free(bm25->doc_states);
  bm25->doc_keys = nkeys;
  bm25->doc_vals = nvals;
  bm25->doc_states = nstates;
  bm25->doc_cap = new_cap;
  bm25->doc_tombstones = 0U;
  return NBASE_OK;
}

static ssize_t doc_map_find(const nbase_bm25_t *bm25, uint64_t id) {
  if (bm25->doc_cap == 0U) {
    return -1;
  }
  size_t mask = bm25->doc_cap - 1U;
  size_t pos = (size_t)hash_u64(id) & mask;
  for (size_t probe = 0U; probe < bm25->doc_cap; probe++) {
    uint8_t st = bm25->doc_states[pos];
    if (st == BM25_EMPTY) {
      return -1;
    }
    if (st == BM25_OCCUPIED && bm25->doc_keys[pos] == id) {
      return (ssize_t)pos;
    }
    pos = (pos + 1U) & mask;
  }
  return -1;
}

static nbase_status_t term_map_rehash(nbase_bm25_t *bm25, size_t new_cap) {
  char **nkeys = calloc(new_cap, sizeof(char *));
  bm25_postings_t *nvals = calloc(new_cap, sizeof(bm25_postings_t));
  if (!nkeys || !nvals) {
    free(nkeys);
    free(nvals);
    return NBASE_ERR_NOMEM;
  }
  size_t mask = new_cap - 1U;
  for (size_t i = 0U; i < bm25->term_cap; i++) {
    if (bm25->term_keys[i] == NULL) {
      continue;
    }
    size_t pos = (size_t)hash_cstr(bm25->term_keys[i]) & mask;
    while (nkeys[pos] != NULL) {
      pos = (pos + 1U) & mask;
    }
    nkeys[pos] = bm25->term_keys[i];
    nvals[pos] = bm25->term_vals[i];
  }
  free(bm25->term_keys);
  free(bm25->term_vals);
  bm25->term_keys = nkeys;
  bm25->term_vals = nvals;
  bm25->term_cap = new_cap;
  return NBASE_OK;
}

static bm25_postings_t *term_map_find(const nbase_bm25_t *bm25, const char *term) {
  if (bm25->term_cap == 0U) {
    return NULL;
  }
  size_t mask = bm25->term_cap - 1U;
  size_t pos = (size_t)hash_cstr(term) & mask;
  while (bm25->term_keys[pos] != NULL) {
    if (strcmp(bm25->term_keys[pos], term) == 0) {
      return &bm25->term_vals[pos];
    }
    pos = (pos + 1U) & mask;
  }
  return NULL;
}

static nbase_status_t term_map_get_or_insert(
    nbase_bm25_t *bm25,
    const char *term,
    bm25_postings_t **out_post) {
  if ((bm25->term_count + 1U) * 10U >= bm25->term_cap * 7U) {
    size_t next_cap = (bm25->term_cap == 0U) ? 64U : bm25->term_cap * 2U;
    nbase_status_t st = term_map_rehash(bm25, next_cap);
    if (st != NBASE_OK) {
      return st;
    }
  }
  size_t mask = bm25->term_cap - 1U;
  size_t pos = (size_t)hash_cstr(term) & mask;
  while (bm25->term_keys[pos] != NULL) {
    if (strcmp(bm25->term_keys[pos], term) == 0) {
      *out_post = &bm25->term_vals[pos];
      return NBASE_OK;
    }
    pos = (pos + 1U) & mask;
  }
  size_t tlen = strlen(term);
  char *copy = malloc(tlen + 1U);
  if (!copy) {
    return NBASE_ERR_NOMEM;
  }
  memcpy(copy, term, tlen + 1U);
  bm25->term_keys[pos] = copy;
  memset(&bm25->term_vals[pos], 0, sizeof(bm25_postings_t));
  bm25->term_count++;
  *out_post = &bm25->term_vals[pos];
  return NBASE_OK;
}

static nbase_status_t postings_push(
    bm25_postings_t *p,
    uint64_t id,
    uint32_t tf,
    uint32_t doc_len) {
  if (p->len == p->cap) {
    uint32_t new_cap = (p->cap == 0U) ? 4U : p->cap * 2U;
    bm25_posting_t *next = realloc(p->items, (size_t)new_cap * sizeof(*next));
    if (!next) {
      return NBASE_ERR_NOMEM;
    }
    p->items = next;
    p->cap = new_cap;
  }
  p->items[p->len].id = id;
  p->items[p->len].tf = tf;
  p->items[p->len].doc_len = doc_len;
  p->len++;
  return NBASE_OK;
}

static void postings_remove(bm25_postings_t *p, uint64_t id) {
  for (uint32_t i = 0U; i < p->len; i++) {
    if (p->items[i].id == id) {
      p->items[i] = p->items[p->len - 1U];
      p->len--;
      return;
    }
  }
}

nbase_status_t nbase_bm25_create(float k1, float b, nbase_bm25_t **out_bm25) {
  if (!out_bm25 || !isfinite(k1) || k1 < 0.0f || !isfinite(b) || b < 0.0f || b > 1.0f) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_bm25 = NULL;

  nbase_bm25_t *bm25 = calloc(1, sizeof(*bm25));
  if (!bm25) {
    return NBASE_ERR_NOMEM;
  }
  bm25->k1 = k1;
  bm25->b = b;
  if (doc_map_rehash(bm25, 64U) != NBASE_OK || term_map_rehash(bm25, 128U) != NBASE_OK ||
      pthread_rwlock_init(&bm25->rwlock, NULL) != 0) {
    nbase_bm25_destroy(bm25);
    return NBASE_ERR_NOMEM;
  }
  bm25->rwlock_init = true;
  *out_bm25 = bm25;
  return NBASE_OK;
}

void nbase_bm25_destroy(nbase_bm25_t *bm25) {
  if (!bm25) {
    return;
  }
  if (bm25->rwlock_init) {
    pthread_rwlock_destroy(&bm25->rwlock);
  }
  for (size_t i = 0U; i < bm25->doc_cap; i++) {
    if (bm25->doc_states && bm25->doc_states[i] == BM25_OCCUPIED) {
      free(bm25->doc_vals[i].text);
    }
  }
  free(bm25->doc_keys);
  free(bm25->doc_vals);
  free(bm25->doc_states);

  for (size_t i = 0U; i < bm25->term_cap; i++) {
    if (bm25->term_keys && bm25->term_keys[i] != NULL) {
      free(bm25->term_keys[i]);
      free(bm25->term_vals[i].items);
    }
  }
  free(bm25->term_keys);
  free(bm25->term_vals);
  free(bm25);
}

uint64_t nbase_bm25_doc_count(const nbase_bm25_t *bm25) {
  if (!bm25) {
    return 0ULL;
  }
  nbase_bm25_t *mutable_bm = (nbase_bm25_t *)(uintptr_t)bm25;
  pthread_rwlock_rdlock(&mutable_bm->rwlock);
  uint64_t cnt = bm25->doc_count;
  pthread_rwlock_unlock(&mutable_bm->rwlock);
  return cnt;
}

nbase_status_t nbase_bm25_get(
    const nbase_bm25_t *bm25,
    uint64_t id,
    char *out_buf,
    size_t buf_len,
    size_t *out_len) {
  if (!bm25 || (!out_buf && !out_len)) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_bm25_t *mutable_bm = (nbase_bm25_t *)(uintptr_t)bm25;
  pthread_rwlock_rdlock(&mutable_bm->rwlock);
  ssize_t idx = doc_map_find(bm25, id);
  if (idx < 0) {
    pthread_rwlock_unlock(&mutable_bm->rwlock);
    return NBASE_ERR_NOT_FOUND;
  }
  size_t tlen = (size_t)bm25->doc_vals[idx].text_len;
  if (out_len != NULL) {
    *out_len = tlen;
  }
  if (out_buf != NULL) {
    if (buf_len <= tlen) {
      pthread_rwlock_unlock(&mutable_bm->rwlock);
      return NBASE_ERR_INVALID_ARG;
    }
    if (tlen > 0U) {
      memcpy(out_buf, bm25->doc_vals[idx].text, tlen);
    }
    out_buf[tlen] = '\0';
  }
  pthread_rwlock_unlock(&mutable_bm->rwlock);
  return NBASE_OK;
}

bool nbase_bm25_contains(const nbase_bm25_t *bm25, uint64_t id) {
  if (!bm25) {
    return false;
  }
  nbase_bm25_t *mutable_bm = (nbase_bm25_t *)(uintptr_t)bm25;
  pthread_rwlock_rdlock(&mutable_bm->rwlock);
  bool found = (doc_map_find(bm25, id) >= 0);
  pthread_rwlock_unlock(&mutable_bm->rwlock);
  return found;
}

static nbase_status_t bm25_delete_unlocked(nbase_bm25_t *bm25, uint64_t id) {
  ssize_t idx = doc_map_find(bm25, id);
  if (idx < 0) {
    return NBASE_ERR_NOT_FOUND;
  }

  tok_count_t *counts = NULL;
  uint32_t uniq = 0U;
  uint32_t total = 0U;
  nbase_status_t st = tokenize_counts(bm25->doc_vals[idx].text, &counts, &uniq, &total);
  if (st != NBASE_OK) {
    return st;
  }
  for (uint32_t i = 0U; i < uniq; i++) {
    bm25_postings_t *p = term_map_find(bm25, counts[i].term);
    if (p) {
      postings_remove(p, id);
    }
  }
  free(counts);

  bm25->total_doc_len -= bm25->doc_vals[idx].doc_len;
  bm25->doc_count--;
  free(bm25->doc_vals[idx].text);
  bm25->doc_vals[idx].text = NULL;
  bm25->doc_vals[idx].text_len = 0U;
  bm25->doc_vals[idx].doc_len = 0U;
  bm25->doc_states[idx] = BM25_TOMBSTONE;
  bm25->doc_tombstones++;
  return NBASE_OK;
}

nbase_status_t nbase_bm25_delete(nbase_bm25_t *bm25, uint64_t id) {
  if (!bm25) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&bm25->rwlock);
  nbase_status_t st = bm25_delete_unlocked(bm25, id);
  pthread_rwlock_unlock(&bm25->rwlock);
  return st;
}

nbase_status_t nbase_bm25_delete_batch(
    nbase_bm25_t *bm25,
    const uint64_t *ids,
    uint32_t count,
    uint32_t *out_deleted) {
  if (!bm25 || (count > 0U && !ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (out_deleted != NULL) {
    *out_deleted = 0U;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_rwlock_wrlock(&bm25->rwlock);
  nbase_status_t st = NBASE_OK;
  uint32_t deleted = 0U;
  for (uint32_t i = 0U; i < count; i++) {
    nbase_status_t ds = bm25_delete_unlocked(bm25, ids[i]);
    if (ds == NBASE_ERR_NOT_FOUND) {
      continue;
    }
    if (ds != NBASE_OK) {
      st = ds;
      break;
    }
    deleted++;
  }
  pthread_rwlock_unlock(&bm25->rwlock);
  if (out_deleted != NULL) {
    *out_deleted = deleted;
  }
  return st;
}

static nbase_status_t bm25_index_unlocked(nbase_bm25_t *bm25, uint64_t id, const char *text) {
  size_t tlen = strlen(text);
  if (tlen > (size_t)UINT32_MAX - 1U) {
    return NBASE_ERR_INVALID_ARG;
  }
  char *text_copy = malloc(tlen + 1U);
  if (!text_copy) {
    return NBASE_ERR_NOMEM;
  }
  memcpy(text_copy, text, tlen + 1U);

  tok_count_t *counts = NULL;
  uint32_t uniq = 0U;
  uint32_t total = 0U;
  nbase_status_t st = tokenize_counts(text, &counts, &uniq, &total);
  if (st != NBASE_OK) {
    free(text_copy);
    return st;
  }

  if (doc_map_find(bm25, id) >= 0) {
    st = bm25_delete_unlocked(bm25, id);
    if (st != NBASE_OK) {
      free(counts);
      free(text_copy);
      return st;
    }
  }

  if ((bm25->doc_count + bm25->doc_tombstones + 1ULL) * 10ULL >= (uint64_t)bm25->doc_cap * 7ULL) {
    st = doc_map_rehash(bm25, bm25->doc_cap * 2U);
    if (st != NBASE_OK) {
      free(counts);
      free(text_copy);
      return st;
    }
  }

  size_t mask = bm25->doc_cap - 1U;
  size_t pos = (size_t)hash_u64(id) & mask;
  size_t first_tomb = SIZE_MAX;
  while (bm25->doc_states[pos] != BM25_EMPTY) {
    if (bm25->doc_states[pos] == BM25_TOMBSTONE && first_tomb == SIZE_MAX) {
      first_tomb = pos;
    }
    pos = (pos + 1U) & mask;
  }
  if (first_tomb != SIZE_MAX) {
    pos = first_tomb;
    bm25->doc_tombstones--;
  }

  for (uint32_t i = 0U; i < uniq; i++) {
    bm25_postings_t *p = NULL;
    st = term_map_get_or_insert(bm25, counts[i].term, &p);
    if (st == NBASE_OK) {
      st = postings_push(p, id, counts[i].tf, total);
    }
    if (st != NBASE_OK) {
      for (uint32_t prev = 0U; prev < i; prev++) {
        bm25_postings_t *pp = term_map_find(bm25, counts[prev].term);
        if (pp) {
          postings_remove(pp, id);
        }
      }
      free(counts);
      free(text_copy);
      return st;
    }
  }
  free(counts);

  bm25->doc_states[pos] = BM25_OCCUPIED;
  bm25->doc_keys[pos] = id;
  bm25->doc_vals[pos].text = text_copy;
  bm25->doc_vals[pos].text_len = (uint32_t)tlen;
  bm25->doc_vals[pos].doc_len = total;
  bm25->doc_count++;
  bm25->total_doc_len += total;
  return NBASE_OK;
}

nbase_status_t nbase_bm25_index(nbase_bm25_t *bm25, uint64_t id, const char *text) {
  if (!bm25 || !text) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&bm25->rwlock);
  nbase_status_t st = bm25_index_unlocked(bm25, id, text);
  pthread_rwlock_unlock(&bm25->rwlock);
  return st;
}

nbase_status_t nbase_bm25_index_batch(
    nbase_bm25_t *bm25,
    const uint64_t *ids,
    const char *const *texts,
    uint32_t count) {
  if (!bm25 || (count > 0U && (!ids || !texts))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!texts[i]) {
      return NBASE_ERR_INVALID_ARG;
    }
  }
  pthread_rwlock_wrlock(&bm25->rwlock);
  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < count && st == NBASE_OK; i++) {
    st = bm25_index_unlocked(bm25, ids[i], texts[i]);
  }
  pthread_rwlock_unlock(&bm25->rwlock);
  return st;
}

nbase_status_t nbase_bm25_list_ids(
    const nbase_bm25_t *bm25,
    uint64_t offset,
    uint32_t limit,
    uint64_t *out_ids,
    uint32_t *out_count,
    uint64_t *out_total) {
  if (!bm25 || !out_count || (limit > 0U && !out_ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;

  nbase_bm25_t *mutable_bm = (nbase_bm25_t *)(uintptr_t)bm25;
  pthread_rwlock_rdlock(&mutable_bm->rwlock);
  uint64_t total = bm25->doc_count;
  if (out_total) {
    *out_total = total;
  }
  if (total == 0ULL || offset >= total || limit == 0U) {
    pthread_rwlock_unlock(&mutable_bm->rwlock);
    return NBASE_OK;
  }
  if (total > (uint64_t)(SIZE_MAX / sizeof(uint64_t))) {
    pthread_rwlock_unlock(&mutable_bm->rwlock);
    return NBASE_ERR_NOMEM;
  }

  uint64_t *all = malloc((size_t)total * sizeof(uint64_t));
  if (!all) {
    pthread_rwlock_unlock(&mutable_bm->rwlock);
    return NBASE_ERR_NOMEM;
  }
  size_t pos = 0U;
  for (size_t i = 0U; i < bm25->doc_cap && pos < (size_t)total; i++) {
    if (bm25->doc_states[i] == BM25_OCCUPIED) {
      all[pos++] = bm25->doc_keys[i];
    }
  }
  pthread_rwlock_unlock(&mutable_bm->rwlock);

  if (pos > 1U) {
    qsort(all, pos, sizeof(uint64_t), cmp_bm25_u64_asc);
  }
  if (offset < (uint64_t)pos) {
    uint64_t avail = (uint64_t)pos - offset;
    uint32_t take = (avail > (uint64_t)limit) ? limit : (uint32_t)avail;
    memcpy(out_ids, all + (size_t)offset, (size_t)take * sizeof(uint64_t));
    *out_count = take;
  }
  free(all);
  return NBASE_OK;
}

nbase_status_t nbase_bm25_compact(nbase_bm25_t *bm25) {
  if (!bm25) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&bm25->rwlock);

  size_t new_doc_cap = 64U;
  while ((bm25->doc_count + 1ULL) * 10ULL >= (uint64_t)new_doc_cap * 7ULL) {
    new_doc_cap *= 2U;
  }
  if (bm25->doc_tombstones > 0U || new_doc_cap != bm25->doc_cap) {
    nbase_status_t st = doc_map_rehash(bm25, new_doc_cap);
    if (st != NBASE_OK) {
      pthread_rwlock_unlock(&bm25->rwlock);
      return st;
    }
  }

  size_t alive_terms = 0U;
  for (size_t i = 0U; i < bm25->term_cap; i++) {
    if (bm25->term_keys[i] != NULL && bm25->term_vals[i].len > 0U) {
      alive_terms++;
    }
  }
  size_t new_term_cap = 128U;
  while ((alive_terms + 1U) * 10U >= new_term_cap * 7U) {
    new_term_cap *= 2U;
  }

  char **nkeys = calloc(new_term_cap, sizeof(char *));
  bm25_postings_t *nvals = calloc(new_term_cap, sizeof(bm25_postings_t));
  if (!nkeys || !nvals) {
    free(nkeys);
    free(nvals);
    pthread_rwlock_unlock(&bm25->rwlock);
    return NBASE_ERR_NOMEM;
  }

  size_t mask = new_term_cap - 1U;
  for (size_t i = 0U; i < bm25->term_cap; i++) {
    if (bm25->term_keys[i] == NULL) {
      continue;
    }
    if (bm25->term_vals[i].len == 0U) {
      free(bm25->term_keys[i]);
      free(bm25->term_vals[i].items);
      continue;
    }
    bm25_postings_t p = bm25->term_vals[i];
    if (p.cap > p.len) {
      bm25_posting_t *shrunk = realloc(p.items, (size_t)p.len * sizeof(*p.items));
      if (shrunk) {
        p.items = shrunk;
        p.cap = p.len;
      }
    }
    size_t pos = (size_t)hash_cstr(bm25->term_keys[i]) & mask;
    while (nkeys[pos] != NULL) {
      pos = (pos + 1U) & mask;
    }
    nkeys[pos] = bm25->term_keys[i];
    nvals[pos] = p;
  }
  free(bm25->term_keys);
  free(bm25->term_vals);
  bm25->term_keys = nkeys;
  bm25->term_vals = nvals;
  bm25->term_cap = new_term_cap;
  bm25->term_count = alive_terms;

  pthread_rwlock_unlock(&bm25->rwlock);
  return NBASE_OK;
}

typedef struct score_map {
  size_t cap;
  size_t count;
  uint64_t *keys;
  double *vals;
  uint8_t *used;
} score_map_t;

static bool score_map_init(score_map_t *m, size_t init_cap) {
  size_t cap = 16U;
  while (cap < init_cap) {
    cap *= 2U;
  }
  m->keys = malloc(cap * sizeof(uint64_t));
  m->vals = malloc(cap * sizeof(double));
  m->used = calloc(cap, sizeof(uint8_t));
  if (!m->keys || !m->vals || !m->used) {
    free(m->keys);
    free(m->vals);
    free(m->used);
    return false;
  }
  m->cap = cap;
  m->count = 0U;
  return true;
}

static void score_map_free(score_map_t *m) {
  free(m->keys);
  free(m->vals);
  free(m->used);
}

static bool score_map_add(score_map_t *m, uint64_t key, double delta) {
  if ((m->count + 1U) * 10U >= m->cap * 7U) {
    size_t ncap = m->cap * 2U;
    uint64_t *nkeys = malloc(ncap * sizeof(uint64_t));
    double *nvals = malloc(ncap * sizeof(double));
    uint8_t *nused = calloc(ncap, sizeof(uint8_t));
    if (!nkeys || !nvals || !nused) {
      free(nkeys);
      free(nvals);
      free(nused);
      return false;
    }
    size_t nmask = ncap - 1U;
    for (size_t i = 0U; i < m->cap; i++) {
      if (m->used[i] == 0U) {
        continue;
      }
      size_t pos = (size_t)hash_u64(m->keys[i]) & nmask;
      while (nused[pos] != 0U) {
        pos = (pos + 1U) & nmask;
      }
      nused[pos] = 1U;
      nkeys[pos] = m->keys[i];
      nvals[pos] = m->vals[i];
    }
    free(m->keys);
    free(m->vals);
    free(m->used);
    m->keys = nkeys;
    m->vals = nvals;
    m->used = nused;
    m->cap = ncap;
  }

  size_t mask = m->cap - 1U;
  size_t pos = (size_t)hash_u64(key) & mask;
  while (m->used[pos] != 0U) {
    if (m->keys[pos] == key) {
      m->vals[pos] += delta;
      return true;
    }
    pos = (pos + 1U) & mask;
  }
  m->used[pos] = 1U;
  m->keys[pos] = key;
  m->vals[pos] = delta;
  m->count++;
  return true;
}

static nbase_status_t bm25_search_filtered_unlocked(
    const nbase_bm25_t *bm25,
    const char *query,
    uint32_t k,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_scored_id_t *out_results,
    uint32_t *out_count) {
  *out_count = 0U;
  if (k == 0U || bm25->doc_count == 0ULL) {
    return NBASE_OK;
  }

  tok_count_t *qterms = NULL;
  uint32_t q_uniq = 0U;
  uint32_t q_tot = 0U;
  nbase_status_t st = tokenize_counts(query, &qterms, &q_uniq, &q_tot);
  if (st != NBASE_OK) {
    return st;
  }
  if (q_uniq == 0U) {
    free(qterms);
    return NBASE_OK;
  }

  size_t max_df = 0U;
  for (uint32_t qi = 0U; qi < q_uniq; qi++) {
    const bm25_postings_t *p = term_map_find(bm25, qterms[qi].term);
    if (p && (size_t)p->len > max_df) {
      max_df = (size_t)p->len;
    }
  }
  if (max_df == 0U) {
    free(qterms);
    return NBASE_OK;
  }

  size_t init_cap = 64U;
  if (max_df * 2U > init_cap) {
    init_cap = max_df * 2U;
  }
  size_t doc_cap_hint = (size_t)bm25->doc_count * 2U;
  if (doc_cap_hint >= 64U && init_cap > doc_cap_hint) {
    init_cap = doc_cap_hint;
  }

  score_map_t scores;
  if (!score_map_init(&scores, init_cap)) {
    free(qterms);
    return NBASE_ERR_NOMEM;
  }

  double avgdl = (bm25->total_doc_len > 0ULL)
                     ? ((double)bm25->total_doc_len / (double)bm25->doc_count)
                     : 1.0;
  double n_docs = (double)bm25->doc_count;
  double k1 = (double)bm25->k1;
  double b = (double)bm25->b;
  double k1_plus_1 = k1 + 1.0;
  double k1_one_minus_b = k1 * (1.0 - b);
  double k1_b_inv_avgdl = (avgdl > 0.0) ? (k1 * b / avgdl) : 0.0;

  for (uint32_t qi = 0U; qi < q_uniq; qi++) {
    const bm25_postings_t *p = term_map_find(bm25, qterms[qi].term);
    if (!p || p->len == 0U) {
      continue;
    }
    double df = (double)p->len;
    double idf = log(1.0 + (n_docs - df + 0.5) / (df + 0.5));
    double idf_k1_p1 = idf * k1_plus_1;
    for (uint32_t pi = 0U; pi < p->len; pi++) {
      uint64_t did = p->items[pi].id;
      if (filter != NULL && !filter(did, filter_ctx)) {
        continue;
      }
      double dl = (double)p->items[pi].doc_len;
      double tf = (double)p->items[pi].tf;
      double denom = tf + k1_one_minus_b + k1_b_inv_avgdl * dl;
      double contrib = (denom > 0.0) ? (idf_k1_p1 * tf / denom) : 0.0;
      if (!score_map_add(&scores, did, contrib)) {
        score_map_free(&scores);
        free(qterms);
        return NBASE_ERR_NOMEM;
      }
    }
  }
  free(qterms);

  if (scores.count == 0U) {
    score_map_free(&scores);
    return NBASE_OK;
  }

  nbase_scored_id_t *heap = malloc((size_t)k * sizeof(nbase_scored_id_t));
  if (!heap) {
    score_map_free(&scores);
    return NBASE_ERR_NOMEM;
  }
  uint32_t heap_len = 0U;
  for (size_t i = 0U; i < scores.cap; i++) {
    if (scores.used[i] != 0U) {
      nbase_scored_id_t cand = {scores.keys[i], (float)scores.vals[i]};
      scored_heap_push(heap, &heap_len, k, cand);
    }
  }
  score_map_free(&scores);

  if (heap_len > 1U) {
    qsort(heap, heap_len, sizeof(nbase_scored_id_t), cmp_scored_desc);
  }
  memcpy(out_results, heap, (size_t)heap_len * sizeof(nbase_scored_id_t));
  *out_count = heap_len;
  free(heap);
  return NBASE_OK;
}

nbase_status_t nbase_bm25_search_filtered(
    const nbase_bm25_t *bm25,
    const char *query,
    uint32_t k,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_scored_id_t *out_results,
    uint32_t *out_count) {
  if (!bm25 || !query || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_bm25_t *mutable_bm = (nbase_bm25_t *)(uintptr_t)bm25;
  pthread_rwlock_rdlock(&mutable_bm->rwlock);
  nbase_status_t st = bm25_search_filtered_unlocked(
      bm25, query, k, filter, filter_ctx, out_results, out_count);
  pthread_rwlock_unlock(&mutable_bm->rwlock);
  return st;
}

nbase_status_t nbase_bm25_search(
    const nbase_bm25_t *bm25,
    const char *query,
    uint32_t k,
    nbase_scored_id_t *out_results,
    uint32_t *out_count) {
  return nbase_bm25_search_filtered(bm25, query, k, NULL, NULL, out_results, out_count);
}

typedef struct bm25_batch_ctx {
  const nbase_bm25_t *bm25;
  const char *const *queries;
  uint32_t query_count;
  uint32_t k;
  nbase_scored_id_t *out_results;
  uint32_t *out_counts;
  _Atomic uint32_t next_q;
  _Atomic int err;
} bm25_batch_ctx_t;

static void *bm25_batch_worker(void *raw) {
  bm25_batch_ctx_t *ctx = (bm25_batch_ctx_t *)raw;
  while (true) {
    uint32_t q = atomic_fetch_add_explicit(&ctx->next_q, 1U, memory_order_relaxed);
    if (q >= ctx->query_count) {
      break;
    }
    uint32_t cnt = 0U;
    nbase_status_t st = bm25_search_filtered_unlocked(
        ctx->bm25,
        ctx->queries[q],
        ctx->k,
        NULL,
        NULL,
        ctx->out_results + (size_t)q * (size_t)ctx->k,
        &cnt);
    if (st != NBASE_OK) {
      int expected = (int)NBASE_OK;
      (void)atomic_compare_exchange_strong_explicit(
          &ctx->err, &expected, (int)st, memory_order_relaxed, memory_order_relaxed);
    } else {
      ctx->out_counts[q] = cnt;
    }
  }
  return NULL;
}

nbase_status_t nbase_bm25_search_batch(
    const nbase_bm25_t *bm25,
    const char *const *queries,
    uint32_t query_count,
    uint32_t k,
    uint32_t num_threads,
    nbase_scored_id_t *out_results,
    uint32_t *out_counts) {
  if (!bm25 || (query_count > 0U && (!queries || !out_counts || (k > 0U && !out_results)))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (query_count == 0U) {
    return NBASE_OK;
  }
  for (uint32_t q = 0U; q < query_count; q++) {
    if (!queries[q]) {
      return NBASE_ERR_INVALID_ARG;
    }
  }
  memset(out_counts, 0, (size_t)query_count * sizeof(uint32_t));

  nbase_bm25_t *mutable_bm = (nbase_bm25_t *)(uintptr_t)bm25;
  pthread_rwlock_rdlock(&mutable_bm->rwlock);
  if (k == 0U || bm25->doc_count == 0ULL) {
    pthread_rwlock_unlock(&mutable_bm->rwlock);
    return NBASE_OK;
  }

  uint32_t eff_threads = (num_threads == 0U) ? 4U : num_threads;
  if (eff_threads > query_count) {
    eff_threads = query_count;
  }
  if (eff_threads > 64U) {
    eff_threads = 64U;
  }

  if (eff_threads <= 1U) {
    nbase_status_t st = NBASE_OK;
    for (uint32_t q = 0U; q < query_count && st == NBASE_OK; q++) {
      st = bm25_search_filtered_unlocked(
          bm25,
          queries[q],
          k,
          NULL,
          NULL,
          out_results + (size_t)q * (size_t)k,
          &out_counts[q]);
    }
    pthread_rwlock_unlock(&mutable_bm->rwlock);
    return st;
  }

  bm25_batch_ctx_t ctx;
  ctx.bm25 = bm25;
  ctx.queries = queries;
  ctx.query_count = query_count;
  ctx.k = k;
  ctx.out_results = out_results;
  ctx.out_counts = out_counts;
  atomic_init(&ctx.next_q, 0U);
  atomic_init(&ctx.err, (int)NBASE_OK);

  pthread_t threads[64];
  uint32_t launched = 0U;
  for (uint32_t t = 0U; t < eff_threads; t++) {
    if (pthread_create(&threads[t], NULL, bm25_batch_worker, &ctx) == 0) {
      launched++;
    } else {
      break;
    }
  }
  if (launched == 0U) {
    (void)bm25_batch_worker(&ctx);
  } else {
    for (uint32_t t = 0U; t < launched; t++) {
      pthread_join(threads[t], NULL);
    }
  }
  pthread_rwlock_unlock(&mutable_bm->rwlock);
  return (nbase_status_t)atomic_load_explicit(&ctx.err, memory_order_relaxed);
}

nbase_status_t nbase_hybrid_rrf(
    const nbase_result_t *vec_results,
    uint32_t vec_count,
    const nbase_scored_id_t *text_results,
    uint32_t text_count,
    float alpha,
    float rrf_k,
    uint32_t k,
    nbase_scored_id_t *out_results,
    uint32_t *out_count) {
  if (!out_count || (vec_count > 0U && !vec_results) || (text_count > 0U && !text_results) ||
      (k > 0U && !out_results) || isnan(alpha) || alpha < 0.0f || alpha > 1.0f ||
      !isfinite(rrf_k) || rrf_k <= 0.0f) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  if (k == 0U || (vec_count == 0U && text_count == 0U)) {
    return NBASE_OK;
  }

  score_map_t map;
  if (!score_map_init(&map, (size_t)vec_count + (size_t)text_count + 8U)) {
    return NBASE_ERR_NOMEM;
  }

  for (uint32_t i = 0U; i < vec_count; i++) {
    double s = (double)alpha / ((double)rrf_k + (double)(i + 1U));
    if (!score_map_add(&map, vec_results[i].id, s)) {
      score_map_free(&map);
      return NBASE_ERR_NOMEM;
    }
  }
  for (uint32_t j = 0U; j < text_count; j++) {
    double s = (1.0 - (double)alpha) / ((double)rrf_k + (double)(j + 1U));
    if (!score_map_add(&map, text_results[j].id, s)) {
      score_map_free(&map);
      return NBASE_ERR_NOMEM;
    }
  }

  nbase_scored_id_t *all = malloc(map.count * sizeof(nbase_scored_id_t));
  if (!all) {
    score_map_free(&map);
    return NBASE_ERR_NOMEM;
  }
  size_t pos = 0U;
  for (size_t i = 0U; i < map.cap; i++) {
    if (map.used[i] != 0U) {
      all[pos].id = map.keys[i];
      all[pos].score = (float)map.vals[i];
      pos++;
    }
  }
  score_map_free(&map);

  qsort(all, pos, sizeof(nbase_scored_id_t), cmp_scored_desc);
  uint32_t ret = (pos < (size_t)k) ? (uint32_t)pos : k;
  memcpy(out_results, all, (size_t)ret * sizeof(nbase_scored_id_t));
  *out_count = ret;
  free(all);
  return NBASE_OK;
}

nbase_status_t nbase_bm25_save(const nbase_bm25_t *bm25, const char *file_path) {
  if (!bm25 || !file_path || file_path[0] == '\0') {
    return NBASE_ERR_INVALID_ARG;
  }

  nbase_bm25_t *mutable_bm = (nbase_bm25_t *)(uintptr_t)bm25;
  pthread_rwlock_rdlock(&mutable_bm->rwlock);

  size_t payload_bytes = 0U;
  for (size_t i = 0U; i < bm25->doc_cap; i++) {
    if (bm25->doc_states[i] == BM25_OCCUPIED) {
      payload_bytes += 12U + (size_t)bm25->doc_vals[i].text_len;
    }
  }

  uint8_t *buf = NULL;
  if (payload_bytes > 0U) {
    buf = malloc(payload_bytes);
    if (!buf) {
      pthread_rwlock_unlock(&mutable_bm->rwlock);
      return NBASE_ERR_NOMEM;
    }
    size_t off = 0U;
    for (size_t i = 0U; i < bm25->doc_cap; i++) {
      if (bm25->doc_states[i] != BM25_OCCUPIED) {
        continue;
      }
      uint64_t id = bm25->doc_keys[i];
      uint32_t tlen = bm25->doc_vals[i].text_len;
      memcpy(buf + off, &id, 8U);
      off += 8U;
      memcpy(buf + off, &tlen, 4U);
      off += 4U;
      if (tlen > 0U) {
        memcpy(buf + off, bm25->doc_vals[i].text, tlen);
        off += tlen;
      }
    }
  }

  nbbm_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBBM_MAGIC, 8);
  hdr.version = NBASE_NBBM_VERSION;
  hdr.k1 = bm25->k1;
  hdr.b = bm25->b;
  hdr.flags = 0U;
  hdr.doc_count = bm25->doc_count;
  pthread_rwlock_unlock(&mutable_bm->rwlock);

  hdr.payload_bytes = (uint64_t)payload_bytes;
  hdr.payload_crc32 = (payload_bytes > 0U) ? nbase_crc32(buf, payload_bytes) : 0U;
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbbm_header_t, header_crc32));

  nbase_io_chunk_t chunk = {buf, payload_bytes};
  nbase_status_t st = nbase_write_file_atomic(file_path, &hdr, sizeof(hdr), &chunk, (payload_bytes > 0U) ? 1U : 0U);
  free(buf);
  return st;
}

nbase_status_t nbase_bm25_load(const char *file_path, nbase_bm25_t **out_bm25) {
  if (!file_path || file_path[0] == '\0' || !out_bm25) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_bm25 = NULL;

  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    return errno == ENOENT ? NBASE_ERR_NOT_FOUND : NBASE_ERR_IO;
  }

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0) {
    close(fd);
    return NBASE_ERR_IO;
  }
  if (st_buf.st_size < (off_t)sizeof(nbbm_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nbbm_header_t hdr;
  nbase_status_t st = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (st != NBASE_OK) {
    close(fd);
    return st;
  }
  if (memcmp(hdr.magic, NBASE_NBBM_MAGIC, 8) != 0 || hdr.version != NBASE_NBBM_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nbbm_header_t, header_crc32))) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (!isfinite(hdr.k1) || hdr.k1 < 0.0f || !isfinite(hdr.b) || hdr.b < 0.0f || hdr.b > 1.0f ||
      hdr.payload_bytes > (uint64_t)SIZE_MAX || file_size != sizeof(nbbm_header_t) + hdr.payload_bytes ||
      hdr.doc_count > hdr.payload_bytes / 12ULL) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint8_t *payload = NULL;
  if (hdr.payload_bytes > 0ULL) {
    payload = malloc((size_t)hdr.payload_bytes);
    if (!payload) {
      close(fd);
      return NBASE_ERR_NOMEM;
    }
    st = nbase_read_exact(fd, payload, (size_t)hdr.payload_bytes);
  }
  close(fd);
  if (st != NBASE_OK) {
    free(payload);
    return st;
  }
  uint32_t crc = (hdr.payload_bytes > 0ULL) ? nbase_crc32(payload, (size_t)hdr.payload_bytes) : 0U;
  if (crc != hdr.payload_crc32) {
    free(payload);
    return NBASE_ERR_CORRUPT;
  }

  nbase_bm25_t *bm25 = NULL;
  st = nbase_bm25_create(hdr.k1, hdr.b, &bm25);
  if (st != NBASE_OK) {
    free(payload);
    return st;
  }

  size_t off = 0U;
  for (uint64_t i = 0ULL; i < hdr.doc_count; i++) {
    if (off + 12U > (size_t)hdr.payload_bytes) {
      free(payload);
      nbase_bm25_destroy(bm25);
      return NBASE_ERR_CORRUPT;
    }
    uint64_t id = 0ULL;
    uint32_t tlen = 0U;
    memcpy(&id, payload + off, 8U);
    off += 8U;
    memcpy(&tlen, payload + off, 4U);
    off += 4U;
    if (tlen > (size_t)hdr.payload_bytes - off) {
      free(payload);
      nbase_bm25_destroy(bm25);
      return NBASE_ERR_CORRUPT;
    }
    char *tbuf = malloc((size_t)tlen + 1U);
    if (!tbuf) {
      free(payload);
      nbase_bm25_destroy(bm25);
      return NBASE_ERR_NOMEM;
    }
    if (tlen > 0U) {
      memcpy(tbuf, payload + off, tlen);
    }
    tbuf[tlen] = '\0';
    off += tlen;
    st = bm25_index_unlocked(bm25, id, tbuf);
    free(tbuf);
    if (st != NBASE_OK) {
      free(payload);
      nbase_bm25_destroy(bm25);
      return st;
    }
  }

  free(payload);
  if (off != (size_t)hdr.payload_bytes) {
    nbase_bm25_destroy(bm25);
    return NBASE_ERR_CORRUPT;
  }

  *out_bm25 = bm25;
  return NBASE_OK;
}

uint64_t nbase_bm25_memory_bytes(const nbase_bm25_t *bm25) {
  if (!bm25) {
    return 0ULL;
  }
  nbase_bm25_t *mutable_bm = (nbase_bm25_t *)(uintptr_t)bm25;
  pthread_rwlock_rdlock(&mutable_bm->rwlock);
  uint64_t total = (uint64_t)sizeof(*bm25);
  total += (uint64_t)bm25->doc_cap *
           ((uint64_t)sizeof(uint64_t) + (uint64_t)sizeof(bm25_doc_t) + (uint64_t)sizeof(uint8_t));
  for (size_t i = 0U; i < bm25->doc_cap; i++) {
    if (bm25->doc_states != NULL && bm25->doc_states[i] == BM25_OCCUPIED) {
      total += (uint64_t)bm25->doc_vals[i].text_len + 1ULL;
    }
  }
  total += (uint64_t)bm25->term_cap *
           ((uint64_t)sizeof(char *) + (uint64_t)sizeof(bm25_postings_t));
  for (size_t i = 0U; i < bm25->term_cap; i++) {
    if (bm25->term_keys != NULL && bm25->term_keys[i] != NULL) {
      total += (uint64_t)strlen(bm25->term_keys[i]) + 1ULL;
      total += (uint64_t)bm25->term_vals[i].cap * (uint64_t)sizeof(bm25_posting_t);
    }
  }
  pthread_rwlock_unlock(&mutable_bm->rwlock);
  return total;
}
