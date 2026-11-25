#include "test_common.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

typedef struct http_resp {
  int status;
  char *body;
  size_t body_len;
} http_resp_t;

static bool http_req(
    uint16_t port,
    const char *method,
    const char *path,
    const char *api_key,
    const char *body,
    http_resp_t *out) {
  memset(out, 0, sizeof(*out));
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return false;
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (connect(fd, (const struct sockaddr *)&addr, sizeof(addr)) != 0) {
    close(fd);
    return false;
  }

  size_t body_len = body ? strlen(body) : 0U;
  char hdr[1024];
  int hlen;
  if (api_key != NULL) {
    hlen = snprintf(
        hdr,
        sizeof(hdr),
        "%s %s HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Authorization: Bearer %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n\r\n",
        method,
        path,
        api_key,
        body_len);
  } else {
    hlen = snprintf(
        hdr,
        sizeof(hdr),
        "%s %s HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n\r\n",
        method,
        path,
        body_len);
  }
  if (hlen <= 0) {
    close(fd);
    return false;
  }

  size_t off = 0U;
  while (off < (size_t)hlen) {
    ssize_t wr = send(fd, hdr + off, (size_t)hlen - off, 0);
    if (wr <= 0) {
      close(fd);
      return false;
    }
    off += (size_t)wr;
  }
  off = 0U;
  while (off < body_len) {
    ssize_t wr = send(fd, body + off, body_len - off, 0);
    if (wr <= 0) {
      close(fd);
      return false;
    }
    off += (size_t)wr;
  }

  size_t cap = 4096U;
  size_t len = 0U;
  char *buf = malloc(cap);
  if (!buf) {
    close(fd);
    return false;
  }
  while (true) {
    if (len + 1024U >= cap) {
      cap *= 2U;
      char *next = realloc(buf, cap);
      if (!next) {
        free(buf);
        close(fd);
        return false;
      }
      buf = next;
    }
    ssize_t n = recv(fd, buf + len, cap - len - 1U, 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      free(buf);
      close(fd);
      return false;
    }
    if (n == 0) {
      break;
    }
    len += (size_t)n;
  }
  close(fd);
  buf[len] = '\0';

  char *sep = strstr(buf, "\r\n\r\n");
  if (!sep) {
    free(buf);
    return false;
  }
  int code = 0;
  if (sscanf(buf, "HTTP/1.1 %d", &code) != 1) {
    free(buf);
    return false;
  }
  out->status = code;
  size_t blen = len - (size_t)(sep + 4 - buf);
  out->body = malloc(blen + 1U);
  if (!out->body) {
    free(buf);
    return false;
  }
  memcpy(out->body, sep + 4, blen);
  out->body[blen] = '\0';
  out->body_len = blen;
  free(buf);
  return true;
}

static void test_http_server_crud_search_auth_and_admin(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_config_t cfg = nbase_config_default(3U);
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  ASSERT_STATUS(nbase_wal_enable(db, dir), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 4U;
  hcfg.max_body_bytes = 4096U;
  hcfg.api_key = "secret-token";
  hcfg.save_dir = dir;

  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);
  ASSERT_TRUE(port != 0U);

  http_resp_t r;

  /* /health is accessible without API key. */
  ASSERT_TRUE(http_req(port, "GET", "/health", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, nbase_version()) != NULL);
  free(r.body);

  /* Protected endpoint requires Bearer token. */
  ASSERT_TRUE(http_req(port, "GET", "/stats", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 401);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/stats", "secret-token", NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"dim\":3") != NULL);
  free(r.body);

  /* Single add + metadata */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors",
      "secret-token",
      "{\"id\":1,\"vector\":[1,0,0],\"metadata\":{\"cat\":\"ml\",\"score\":95}}",
      &r));
  ASSERT_EQ_I64(r.status, 201);
  free(r.body);

  /* Duplicate ID returns 409 Conflict */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors",
      "secret-token",
      "{\"id\":1,\"vector\":[1,0,0]}",
      &r));
  ASSERT_EQ_I64(r.status, 409);
  free(r.body);

  /* Bulk add */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors",
      "secret-token",
      "{\"vectors\":[{\"id\":2,\"vector\":[0.99,0.01,0],\"metadata\":{\"cat\":\"ml\",\"score\":80}},"
      "{\"id\":3,\"vector\":[0,1,0],\"metadata\":{\"cat\":\"db\",\"score\":92}}]}",
      &r));
  ASSERT_EQ_I64(r.status, 201);
  ASSERT_TRUE(strstr(r.body, "\"added\":2") != NULL);
  free(r.body);

  /* Upsert & metadata patch */
  ASSERT_TRUE(http_req(port, "PUT", "/api/vectors/2", "secret-token", "{\"vector\":[0.98,0.02,0]}", &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  ASSERT_TRUE(http_req(port, "PATCH", "/api/vectors/2/metadata", "secret-token", "{\"score\":91}", &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  /* Get + exists */
  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/1/exists", "secret-token", NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"exists\":true") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/1", "secret-token", NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"dimension\":3") != NULL);
  free(r.body);

  /* Search with metadata filter */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search",
      "secret-token",
      "{\"query\":[1,0,0],\"k\":2,\"filters\":{\"cat\":\"db\"}}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":3") != NULL);
  free(r.body);

  /* Search with MMR diversity rerank */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search",
      "secret-token",
      "{\"query\":[1,0,0],\"k\":2,\"rerank\":true,\"rerankLambda\":0.2}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":1") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":3") != NULL);
  free(r.body);

  /* Batch search & relationships */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/batch",
      "secret-token",
      "{\"queries\":[[1,0,0],[0,1,0]],\"k\":1,\"threads\":2}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/relationships",
      "secret-token",
      "{\"kNeighbors\":2,\"maxDistance\":0.5}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"src\":1") != NULL);
  free(r.body);

  /* Delete & admin endpoints */
  ASSERT_TRUE(http_req(port, "DELETE", "/api/vectors/3", "secret-token", NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/3", "secret-token", NULL, &r));
  ASSERT_EQ_I64(r.status, 404);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/admin/compact", "secret-token", NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/admin/checkpoint", "secret-token", NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  /* v0.5.0 parity endpoints: includeMetadata/includeVectors, /similar, /search/metadata, /search/communities, /metrics */
  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/1?includeVector=false&includeMetadata=true", "secret-token", NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"metadata\":") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"cat\":\"ml\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"vector\"") == NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/1/similar?k=1&includeMetadata=true&includeVectors=true", "secret-token", NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"metadata\":") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"vector\":[") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search",
      "secret-token",
      "{\"query\":[1,0,0],\"k\":1,\"includeMetadata\":true,\"includeVectors\":true}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"metadata\":") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"vector\":[") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/metadata",
      "secret-token",
      "{\"filters\":[{\"key\":\"score\",\"op\":\"gte\",\"value\":90}],\"limit\":10,\"includeVectors\":true}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":1") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"vector\":[") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/search/communities", "secret-token", "{\"kNeighbors\":2,\"maxDistance\":0.5}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"community_count\":1") != NULL);
  free(r.body);

  /* Prometheus /metrics is accessible without API key */
  ASSERT_TRUE(http_req(port, "GET", "/metrics", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "nbase_requests_total") != NULL);
  ASSERT_TRUE(strstr(r.body, "nbase_vectors_active 2") != NULL);
  free(r.body);

  /* Oversized body returns 413 */
  char big[5000];
  memset(big, ' ', sizeof(big));
  big[0] = '{';
  big[sizeof(big) - 2U] = '}';
  big[sizeof(big) - 1U] = '\0';
  ASSERT_TRUE(http_req(port, "POST", "/api/search", "secret-token", big, &r));
  ASSERT_EQ_I64(r.status, 413);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
  test_rm_rf(dir);
}

typedef struct http_stress_ctx {
  uint16_t port;
  uint32_t tid;
  atomic_int errors;
} http_stress_ctx_t;

static void *http_stress_worker(void *raw) {
  http_stress_ctx_t *ctx = (http_stress_ctx_t *)raw;
  for (uint32_t i = 0U; i < 20U; i++) {
    http_resp_t r;
    if (i % 2U == 0U) {
      char body[128];
      snprintf(body, sizeof(body), "{\"id\":%u,\"vector\":[0.5,0.25,0.125,0.1]}", 1000U + ctx->tid * 100U + i);
      if (!http_req(ctx->port, "POST", "/api/vectors", NULL, body, &r) || r.status != 201) {
        atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      }
      free(r.body);
    } else {
      if (!http_req(ctx->port, "POST", "/api/search", NULL, "{\"query\":[0.5,0.25,0.125,0.1],\"k\":3}", &r) || r.status != 200) {
        atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      }
      free(r.body);
    }
  }
  return NULL;
}

static void test_http_server_concurrent_clients_tsan(void) {
  nbase_config_t cfg = nbase_config_default(4U);
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float seed_v[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_STATUS(nbase_add(db, 1ULL, seed_v, 4U), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 4U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  pthread_t tids[4];
  http_stress_ctx_t ctxs[4];
  for (uint32_t t = 0U; t < 4U; t++) {
    ctxs[t].port = port;
    ctxs[t].tid = t;
    atomic_init(&ctxs[t].errors, 0);
    ASSERT_TRUE(pthread_create(&tids[t], NULL, http_stress_worker, &ctxs[t]) == 0);
  }
  for (uint32_t t = 0U; t < 4U; t++) {
    pthread_join(tids[t], NULL);
    ASSERT_EQ_I64(atomic_load(&ctxs[t].errors), 0LL);
  }

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_cors_and_ratelimit(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  hcfg.max_requests_per_sec = 2U;
  hcfg.cors_origin = "*";

  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  ASSERT_TRUE(http_req(port, "OPTIONS", "/api/search", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 204);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/vectors", NULL, "{\"id\":1,\"vector\":[1,0,0]}", &r));
  ASSERT_EQ_I64(r.status, 201);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/search", NULL, "{\"query\":[1,0,0],\"k\":1}", &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  /* 3rd API search request in the same second hits 429; /health stays exempt. */
  ASSERT_TRUE(http_req(port, "POST", "/api/search", NULL, "{\"query\":[1,0,0],\"k\":1}", &r));
  ASSERT_EQ_I64(r.status, 429);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/health", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static bool http_req_ka(
    int fd,
    const char *method,
    const char *path,
    const char *body,
    char *rx_buf,
    size_t *rx_len,
    size_t rx_cap,
    http_resp_t *out,
    bool *out_saw_ka) {
  memset(out, 0, sizeof(*out));
  *out_saw_ka = false;

  size_t body_len = body ? strlen(body) : 0U;
  char req[1024];
  int hlen = snprintf(
      req,
      sizeof(req),
      "%s %s HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n"
      "Connection: keep-alive\r\n"
      "Content-Type: application/json\r\n"
      "Content-Length: %zu\r\n\r\n",
      method,
      path,
      body_len);
  if (hlen <= 0 || (size_t)hlen >= sizeof(req)) {
    return false;
  }

  size_t off = 0U;
  while (off < (size_t)hlen) {
    ssize_t wr = send(fd, req + off, (size_t)hlen - off, 0);
    if (wr <= 0) {
      return false;
    }
    off += (size_t)wr;
  }
  off = 0U;
  while (off < body_len) {
    ssize_t wr = send(fd, body + off, body_len - off, 0);
    if (wr <= 0) {
      return false;
    }
    off += (size_t)wr;
  }

  rx_buf[*rx_len] = '\0';
  char *sep = strstr(rx_buf, "\r\n\r\n");
  while (sep == NULL) {
    if (*rx_len + 1U >= rx_cap) {
      return false;
    }
    ssize_t n = recv(fd, rx_buf + *rx_len, rx_cap - *rx_len - 1U, 0);
    if (n <= 0) {
      return false;
    }
    *rx_len += (size_t)n;
    rx_buf[*rx_len] = '\0';
    sep = strstr(rx_buf, "\r\n\r\n");
  }

  int code = 0;
  if (sscanf(rx_buf, "HTTP/1.1 %d", &code) != 1) {
    return false;
  }
  size_t hdr_len = (size_t)(sep - rx_buf) + 4U;

  size_t clen = 0U;
  const char *p = rx_buf;
  while (p < sep) {
    const char *eol = strstr(p, "\r\n");
    if (!eol || eol > sep) {
      eol = sep;
    }
    size_t l = (size_t)(eol - p);
    if (l > 15U && strncasecmp(p, "Content-Length:", 15U) == 0) {
      clen = (size_t)strtoull(p + 15, NULL, 10);
    } else if (l > 11U && strncasecmp(p, "Connection:", 11U) == 0) {
      const char *v = p + 11;
      while (v < eol && (*v == ' ' || *v == '\t')) {
        v++;
      }
      if ((size_t)(eol - v) >= 10U && strncasecmp(v, "keep-alive", 10U) == 0) {
        *out_saw_ka = true;
      }
    }
    if (eol == sep) {
      break;
    }
    p = eol + 2;
  }

  while (*rx_len < hdr_len + clen) {
    if (*rx_len + 1U >= rx_cap) {
      return false;
    }
    ssize_t n = recv(fd, rx_buf + *rx_len, rx_cap - *rx_len - 1U, 0);
    if (n <= 0) {
      return false;
    }
    *rx_len += (size_t)n;
  }

  char *resp_body = malloc(clen + 1U);
  if (!resp_body) {
    return false;
  }
  if (clen > 0U) {
    memcpy(resp_body, rx_buf + hdr_len, clen);
  }
  resp_body[clen] = '\0';

  size_t rem = *rx_len - (hdr_len + clen);
  if (rem > 0U) {
    memmove(rx_buf, rx_buf + hdr_len + clen, rem);
  }
  *rx_len = rem;
  rx_buf[*rx_len] = '\0';

  out->status = code;
  out->body = resp_body;
  out->body_len = clen;
  return true;
}

static void test_http_server_keepalive_and_hybrid(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  hcfg.socket_timeout_ms = 5000U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  int fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_TRUE(fd >= 0);
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ASSERT_TRUE(connect(fd, (const struct sockaddr *)&addr, sizeof(addr)) == 0);

  char *rx = malloc(65536U);
  ASSERT_TRUE(rx != NULL);
  size_t rx_len = 0U;
  http_resp_t r;
  bool ka = false;

  ASSERT_TRUE(http_req_ka(
      fd,
      "POST",
      "/api/vectors",
      "{\"vectors\":["
      "{\"id\":1,\"vector\":[1,0,0],\"metadata\":{\"cat\":\"sys\"},\"text\":\"pure c11 avx2 vector engine\"},"
      "{\"id\":2,\"vector\":[0.9,0.1,0],\"metadata\":{\"cat\":\"ml\"}}"
      "]}",
      rx,
      &rx_len,
      65536U,
      &r,
      &ka));
  ASSERT_EQ_I64(r.status, 201);
  ASSERT_TRUE(ka);
  free(r.body);

  ASSERT_TRUE(http_req_ka(
      fd,
      "PUT",
      "/api/vectors/2/text",
      "{\"text\":\"neural transformer embeddings\"}",
      rx,
      &rx_len,
      65536U,
      &r,
      &ka));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(ka);
  free(r.body);

  ASSERT_TRUE(http_req_ka(
      fd,
      "GET",
      "/api/vectors/1?includeVector=false&includeMetadata=false&includeText=true",
      NULL,
      rx,
      &rx_len,
      65536U,
      &r,
      &ka));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(ka);
  ASSERT_TRUE(strstr(r.body, "pure c11 avx2 vector engine") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req_ka(
      fd,
      "POST",
      "/api/search/text",
      "{\"query\":\"avx2 engine\",\"k\":2,\"includeText\":true}",
      rx,
      &rx_len,
      65536U,
      &r,
      &ka));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(ka);
  ASSERT_TRUE(strstr(r.body, "\"id\":1") != NULL);
  ASSERT_TRUE(strstr(r.body, "pure c11 avx2 vector engine") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req_ka(
      fd,
      "POST",
      "/api/search/hybrid",
      "{\"query\":[0.9,0.1,0],\"text\":\"neural embeddings\",\"k\":2,\"alpha\":0.5,\"rrfK\":60,\"includeMetadata\":true,\"includeText\":true}",
      rx,
      &rx_len,
      65536U,
      &r,
      &ka));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(ka);
  ASSERT_TRUE(strstr(r.body, "\"id\":2") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req_ka(fd, "DELETE", "/api/vectors/2/text", NULL, rx, &rx_len, 65536U, &r, &ka));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(ka);
  free(r.body);

  ASSERT_TRUE(http_req_ka(fd, "GET", "/metrics", NULL, rx, &rx_len, 65536U, &r, &ka));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(ka);
  ASSERT_TRUE(strstr(r.body, "nbase_keepalive_requests_total 6") != NULL);
  ASSERT_TRUE(strstr(r.body, "nbase_text_docs_active 1") != NULL);
  ASSERT_TRUE(strstr(r.body, "nbase_memory_bytes ") != NULL);
  ASSERT_TRUE(strstr(r.body, "nbase_search_latency_p95_microseconds ") != NULL);
  free(r.body);

  /* Stop the server while the idle keep-alive socket is still open: must return in < 1s. */
  struct timespec ts0;
  struct timespec ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  nbase_http_server_stop(srv);
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  double stop_ms = (double)(ts1.tv_sec - ts0.tv_sec) * 1000.0 + (double)(ts1.tv_nsec - ts0.tv_nsec) / 1e6;
  ASSERT_TRUE(stop_ms < 1000.0);

  close(fd);
  free(rx);
  nbase_close(db);
}

static void test_http_server_batch_delete_and_telemetry(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors",
      NULL,
      "{\"vectors\":["
      "{\"id\":1,\"vector\":[1,0,0],\"metadata\":{\"env\":\"prod\"}},"
      "{\"id\":2,\"vector\":[0.9,0.1,0],\"metadata\":{\"env\":\"tmp\"}},"
      "{\"id\":3,\"vector\":[0,1,0],\"metadata\":{\"env\":\"tmp\"}},"
      "{\"id\":4,\"vector\":[0,0,1],\"metadata\":{\"env\":\"prod\"}}"
      "]}",
      &r));
  ASSERT_EQ_I64(r.status, 201);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/search", NULL, "{\"query\":[1,0,0],\"k\":2}", &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/vectors/delete", NULL, "{\"ids\":[1,999]}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"deleted\":1") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "DELETE",
      "/api/vectors",
      NULL,
      "{\"filters\":[{\"key\":\"env\",\"op\":\"eq\",\"value\":\"tmp\"}]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"deleted\":2") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/stats", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"memory_bytes\":") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"search_p95_us\":") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"active_vectors\":1") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/metrics", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "nbase_search_latency_p50_microseconds ") != NULL);
  ASSERT_TRUE(strstr(r.body, "nbase_search_latency_p95_microseconds ") != NULL);
  ASSERT_TRUE(strstr(r.body, "nbase_search_latency_p99_microseconds ") != NULL);
  ASSERT_TRUE(strstr(r.body, "nbase_memory_bytes ") != NULL);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_weighted_rerank_and_cache(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.enable_hnsw = false;
  cfg.cache_capacity = 16U;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors",
      NULL,
      "{\"vectors\":["
      "{\"id\":1,\"vector\":[1,0,0],\"metadata\":{\"boost\":0.1}},"
      "{\"id\":2,\"vector\":[0.8,0.6,0],\"metadata\":{\"boost\":5.0}}"
      "]}",
      &r));
  ASSERT_EQ_I64(r.status, 201);
  free(r.body);

  /* Uncached query twice populates cache on miss then hits on repeat */
  ASSERT_TRUE(http_req(port, "POST", "/api/search", NULL, "{\"query\":[1,0,0],\"k\":2}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":1") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/search", NULL, "{\"query\":[1,0,0],\"k\":2}", &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  /* Weighted reranking promotes id 2 via rerankWeights */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search",
      NULL,
      "{\"query\":[1,0,0],\"k\":2,\"rerank\":true,\"rerankingMethod\":\"weighted\",\"rerankWeights\":{\"boost\":1.0},\"skipCache\":true}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  char *p2 = strstr(r.body, "\"id\":2");
  char *p1 = strstr(r.body, "\"id\":1");
  ASSERT_TRUE(p2 != NULL && p1 != NULL && p2 < p1);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/stats", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"cache_capacity\":16") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"cache_hits\":1") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"cache_entries\":1") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/metrics", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "nbase_cache_hits_total 1") != NULL);
  ASSERT_TRUE(strstr(r.body, "nbase_cache_entries 1") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/admin/cache", NULL, "{\"max_entries\":32,\"clear\":true}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"cache_capacity\":32") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"cache_entries\":0") != NULL);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_upsert_and_get_batch(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;

  /* Single upsert inserts new ID */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/upsert",
      NULL,
      "{\"id\":1,\"vector\":[1,0,0],\"metadata\":{\"tag\":\"v1\"},\"text\":\"alpha doc\"}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"upserted\":1") != NULL);
  free(r.body);

  /* Bulk upsert updates ID 1 and inserts ID 2 */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/upsert",
      NULL,
      "{\"vectors\":["
      "{\"id\":1,\"vector\":[0.5,0.5,0],\"metadata\":{\"tag\":\"v2\"},\"text\":\"beta doc\"},"
      "{\"id\":2,\"vector\":[0,1,0],\"metadata\":{\"tag\":\"new\"},\"text\":\"gamma doc\"}"
      "]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"upserted\":2") != NULL);
  free(r.body);

  /* Batch get returns 2 found and 1 missing */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/get",
      NULL,
      "{\"ids\":[1,999,2],\"includeVectors\":true,\"includeMetadata\":true,\"includeText\":true}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":3") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"foundCount\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"found\":false") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"tag\":\"v2\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"text\":\"gamma doc\"") != NULL);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_list_vectors_and_radius_search(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/upsert",
      NULL,
      "{\"vectors\":["
      "{\"id\":30,\"vector\":[1.2,0,0],\"metadata\":{\"slot\":\"c\"}},"
      "{\"id\":10,\"vector\":[1,0,0],\"metadata\":{\"slot\":\"a\"}},"
      "{\"id\":40,\"vector\":[3,0,0],\"metadata\":{\"slot\":\"d\"}},"
      "{\"id\":20,\"vector\":[1.05,0,0],\"metadata\":{\"slot\":\"b\"}}"
      "]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "GET",
      "/api/vectors?offset=1&limit=2&includeVectors=true&includeMetadata=true",
      NULL,
      NULL,
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"total\":4") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"count\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"ids\":[20,30]") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"slot\":\"b\"") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search",
      NULL,
      "{\"query\":[1,0,0],\"k\":10,\"maxDistance\":0.25}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":3") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":10") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":20") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":30") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":40") == NULL);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_batch_meta_text_and_filtered_graph(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/upsert",
      NULL,
      "{\"vectors\":["
      "{\"id\":1,\"vector\":[1,0,0],\"metadata\":{\"team\":\"a\"},\"text\":\"init one\"},"
      "{\"id\":2,\"vector\":[1,0.05,0],\"metadata\":{\"team\":\"a\"},\"text\":\"init two\"},"
      "{\"id\":3,\"vector\":[1,0.1,0],\"metadata\":{\"team\":\"b\"},\"text\":\"init three\"}"
      "]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  /* POST /api/vectors/metadata (replace mode) */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/metadata",
      NULL,
      "{\"items\":["
      "{\"id\":1,\"metadata\":{\"team\":\"blue\",\"tier\":1}},"
      "{\"id\":2,\"metadata\":{\"team\":\"blue\",\"tier\":2}},"
      "{\"id\":3,\"metadata\":{\"team\":\"red\",\"tier\":3}}"
      "]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"updated\":3") != NULL);
  free(r.body);

  /* PATCH /api/vectors/metadata (partial update preserving tier) */
  ASSERT_TRUE(http_req(
      port,
      "PATCH",
      "/api/vectors/metadata",
      NULL,
      "{\"items\":["
      "{\"id\":1,\"metadata\":{\"active\":true}},"
      "{\"id\":2,\"metadata\":{\"active\":true}}"
      "]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"updated\":2") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/1?includeVector=false&includeMetadata=true", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"team\":\"blue\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"tier\":1") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"active\":true") != NULL);
  free(r.body);

  /* Error cases: invalid mode -> 400, missing ID -> 404. */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/metadata",
      NULL,
      "{\"mode\":\"invalid\",\"items\":[{\"id\":1,\"metadata\":{\"x\":1}}]}",
      &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/metadata",
      NULL,
      "{\"items\":[{\"id\":999,\"metadata\":{\"x\":1}}]}",
      &r));
  ASSERT_EQ_I64(r.status, 404);
  free(r.body);

  /* POST /api/vectors/text */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/text",
      NULL,
      "{\"items\":["
      "{\"id\":1,\"text\":\"fast c11 simd engine\"},"
      "{\"id\":2,\"text\":\"fast c11 partitioned store\"},"
      "{\"id\":3,\"text\":\"unrelated payload\"}"
      "]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"updated\":3") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/search/text", NULL, "{\"query\":\"simd engine\",\"k\":3}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":1") != NULL);
  free(r.body);

  /* Filtered relationship & community extraction. */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/relationships",
      NULL,
      "{\"kNeighbors\":2,\"maxDistance\":0.5,\"filters\":{\"team\":\"blue\"}}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"src\":1") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"dst\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "3") == NULL || strstr(r.body, "\"dst\":3") == NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/communities",
      NULL,
      "{\"kNeighbors\":2,\"maxDistance\":0.5,\"filters\":{\"team\":\"blue\"}}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"community_count\":1") != NULL);
  free(r.body);

  /* Zero-match filter on communities returns clean empty list. */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/communities",
      NULL,
      "{\"kNeighbors\":2,\"maxDistance\":0.5,\"filters\":{\"team\":\"none\"}}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"community_count\":0") != NULL);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_count_and_resource_crud(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/upsert",
      NULL,
      "{\"vectors\":["
      "{\"id\":1,\"vector\":[1,0,0],\"metadata\":{\"env\":\"prod\",\"tier\":1},\"text\":\"one\"},"
      "{\"id\":2,\"vector\":[0,1,0],\"metadata\":{\"env\":\"stage\",\"tier\":2},\"text\":\"two\"},"
      "{\"id\":3,\"vector\":[0,0,1],\"metadata\":{\"env\":\"prod\",\"tier\":3}}"
      "]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  /* GET /api/vectors/count & POST /api/vectors/count */
  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/count", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":3") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/count",
      NULL,
      "{\"filters\":{\"env\":\"prod\"}}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":2") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/vectors/count", NULL, "{\"filters\":123}", &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  ASSERT_TRUE(http_req(port, "DELETE", "/api/vectors/count", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  /* GET/PUT/PATCH/DELETE /api/vectors/:id/metadata */
  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/1/metadata", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"env\":\"prod\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"tier\":1") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "PUT", "/api/vectors/1/metadata", NULL, "{\"env\":\"canary\"}", &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/1/metadata", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"env\":\"canary\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"tier\"") == NULL); /* PUT replaced prior fields cleanly. */
  free(r.body);

  ASSERT_TRUE(http_req(port, "DELETE", "/api/vectors/1/metadata", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/1/metadata", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"metadata\":{}") != NULL);
  free(r.body);

  /* GET /api/vectors/:id/text & 404 checks */
  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/2/text", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"text\":\"two\"") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/999/text", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 404);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_post_similar_and_distance(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/upsert",
      NULL,
      "{\"vectors\":["
      "{\"id\":1,\"vector\":[1,0,0],\"metadata\":{\"team\":\"alpha\"}},"
      "{\"id\":2,\"vector\":[1,0.1,0],\"metadata\":{\"team\":\"beta\"}},"
      "{\"id\":3,\"vector\":[1,0.2,0],\"metadata\":{\"team\":\"alpha\"}},"
      "{\"id\":4,\"vector\":[0,1,0],\"metadata\":{\"team\":\"alpha\"}}"
      "]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  /* POST /api/vectors/:id/similar with metadata filter + includeMetadata */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/1/similar",
      NULL,
      "{\"k\":2,\"filters\":{\"team\":\"alpha\"},\"includeMetadata\":true,\"metric\":\"euclidean\"}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":3") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":4") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"team\":\"alpha\"") != NULL);
  free(r.body);

  /* POST /api/vectors/:id/similar with empty body uses defaults */
  ASSERT_TRUE(http_req(port, "POST", "/api/vectors/1/similar", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":3") != NULL);
  free(r.body);

  /* Missing ID -> 404, wrong method -> 405 */
  ASSERT_TRUE(http_req(port, "POST", "/api/vectors/999/similar", NULL, "{\"k\":2}", &r));
  ASSERT_EQ_I64(r.status, 404);
  free(r.body);

  ASSERT_TRUE(http_req(port, "DELETE", "/api/vectors/1/similar", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  /* POST /api/distance across metrics + edge/error paths */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/distance",
      NULL,
      "{\"metric\":\"cosine\",\"query\":[1,0,0],\"vectors\":[[1,0,0],[0,1,0],[-1,0,0]]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"metric\":\"cosine\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"count\":3") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"distances\":[0,1,2]") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/distance",
      NULL,
      "{\"query\":[3,0,0],\"vectors\":[[0,4,0]]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"metric\":\"euclidean\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"distances\":[5]") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/distance",
      NULL,
      "{\"metric\":\"inner_product\",\"query\":[2,1,0],\"vectors\":[]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":0") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"distances\":[]") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/distance",
      NULL,
      "{\"query\":[1,0,0],\"vectors\":[[1,0]]}",
      &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/distance",
      NULL,
      "{\"metric\":\"bad\",\"query\":[1,0,0],\"vectors\":[[1,0,0]]}",
      &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/distance", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_ndjson_import_export(void) {
  nbase_config_t cfg1 = nbase_config_default(3U);
  cfg1.enable_hnsw = false;
  nbase_db_t *db1 = NULL;
  ASSERT_STATUS(nbase_create(&cfg1, &db1), NBASE_OK);

  nbase_http_config_t hcfg1 = nbase_http_config_default();
  hcfg1.port = 0U;
  hcfg1.num_workers = 2U;
  nbase_http_server_t *srv1 = NULL;
  ASSERT_STATUS(nbase_http_server_start(db1, &hcfg1, &srv1), NBASE_OK);
  uint16_t port1 = nbase_http_server_port(srv1);

  http_resp_t r;
  ASSERT_TRUE(http_req(port1, "POST", "/api/vectors/import", NULL, "", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"imported\":0") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port1,
      "POST",
      "/api/vectors/import",
      NULL,
      "{\"id\":10,\"vector\":[1,0,0],\"metadata\":{\"tag\":\"a\"},\"text\":\"doc ten\"}\n"
      "\r\n"
      "{\"id\":20,\"vector\":[0,1,0],\"metadata\":{\"tag\":\"b\"},\"text\":\"doc twenty\"}\n",
      &r));
  ASSERT_EQ_I64(r.status, 201);
  ASSERT_TRUE(strstr(r.body, "\"imported\":2") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port1,
      "POST",
      "/api/vectors/import?upsert=true",
      NULL,
      "{\"id\":20,\"vector\":[0,2,0],\"metadata\":{\"tag\":\"updated\"}}\n"
      "{\"id\":30,\"vector\":[0,0,1]}\n",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"imported\":2") != NULL);
  free(r.body);

  /* Error cases: malformed JSON line -> 400, wrong method -> 405 */
  ASSERT_TRUE(http_req(port1, "POST", "/api/vectors/import", NULL, "{bad json}\n", &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  ASSERT_TRUE(http_req(port1, "GET", "/api/vectors/import", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  ASSERT_TRUE(http_req(port1, "DELETE", "/api/vectors/export", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  /* Export NDJSON from srv1 and import directly into empty srv2 */
  ASSERT_TRUE(http_req(port1, "GET", "/api/vectors/export", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":10") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":20") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"tag\":\"updated\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":30") != NULL);
  char *exported = r.body;

  nbase_config_t cfg2 = nbase_config_default(3U);
  cfg2.enable_hnsw = false;
  nbase_db_t *db2 = NULL;
  ASSERT_STATUS(nbase_create(&cfg2, &db2), NBASE_OK);
  nbase_http_config_t hcfg2 = nbase_http_config_default();
  hcfg2.port = 0U;
  hcfg2.num_workers = 2U;
  nbase_http_server_t *srv2 = NULL;
  ASSERT_STATUS(nbase_http_server_start(db2, &hcfg2, &srv2), NBASE_OK);
  uint16_t port2 = nbase_http_server_port(srv2);

  ASSERT_TRUE(http_req(port2, "POST", "/api/vectors/import", NULL, exported, &r));
  ASSERT_EQ_I64(r.status, 201);
  ASSERT_TRUE(strstr(r.body, "\"imported\":3") != NULL);
  free(r.body);
  free(exported);

  ASSERT_TRUE(http_req(
      port2,
      "GET",
      "/api/vectors/10?includeVector=true&includeMetadata=true&includeText=true",
      NULL,
      NULL,
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"tag\":\"a\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"text\":\"doc ten\"") != NULL);
  free(r.body);

  nbase_http_server_stop(srv2);
  nbase_close(db2);
  nbase_http_server_stop(srv1);
  nbase_close(db1);
}

static void test_http_server_rerank_endpoint(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  uint64_t ids[3] = {1ULL, 2ULL, 3ULL};
  float vecs[9] = {
      1.0f, 0.0f, 0.0f,
      1.0f, 0.01f, 0.0f,
      0.0f, 1.0f, 0.0f
  };
  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, 3U, 3U, 1U), NBASE_OK);

  nbase_meta_field_t m1 = {"score", {.type = NBASE_META_DOUBLE, .as = {.f64 = 0.1}}};
  nbase_meta_field_t m2 = {"score", {.type = NBASE_META_DOUBLE, .as = {.f64 = 0.2}}};
  nbase_meta_field_t m3 = {"score", {.type = NBASE_META_DOUBLE, .as = {.f64 = 5.0}}};
  ASSERT_STATUS(nbase_set_meta(db, 1ULL, &m1, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 2ULL, &m2, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 3ULL, &m3, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 3ULL, "diverse pick"), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  /* 1. MMR diversity rerank pushes orthogonal id=3 ahead of near-duplicate id=2 */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/rerank",
      NULL,
      "{\"method\":\"diversity\",\"k\":2,\"lambda\":0.3,"
      "\"includeMetadata\":true,\"includeVectors\":true,\"includeText\":true,"
      "\"candidates\":[{\"id\":1,\"distance\":0.0},{\"id\":2,\"distance\":0.01},{\"id\":3,\"distance\":0.5}]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"method\":\"diversity\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"count\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"text\":\"diverse pick\"") != NULL);
  const char *p1 = strstr(r.body, "\"id\":1");
  const char *p3 = strstr(r.body, "\"id\":3");
  ASSERT_TRUE(p1 != NULL && p3 != NULL && p1 < p3);
  free(r.body);

  /* 2. Weighted rerank (method omitted -> defaults to weighted when weights provided) */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/rerank",
      NULL,
      "{\"k\":2,\"weights\":{\"score\":0.5},"
      "\"candidates\":[{\"id\":1,\"distance\":0.05},{\"id\":2,\"distance\":0.06},{\"id\":3,\"distance\":0.5}]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"method\":\"weighted\"") != NULL);
  const char *wp3 = strstr(r.body, "\"id\":3");
  const char *wp2 = strstr(r.body, "\"id\":2");
  const char *wp1 = strstr(r.body, "\"id\":1");
  ASSERT_TRUE(wp3 != NULL && wp2 != NULL && wp3 < wp2 && wp1 == NULL);
  free(r.body);

  /* 3. Standard ascending sort by distance */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/rerank",
      NULL,
      "{\"method\":\"standard\",\"k\":2,"
      "\"candidates\":[{\"id\":3,\"distance\":0.8},{\"id\":2,\"distance\":0.1},{\"id\":1,\"distance\":0.2}]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"method\":\"standard\"") != NULL);
  const char *sp2 = strstr(r.body, "\"id\":2");
  const char *sp1 = strstr(r.body, "\"id\":1");
  ASSERT_TRUE(sp2 != NULL && sp1 != NULL && sp2 < sp1);
  free(r.body);

  /* 4. Empty candidates array succeeds */
  ASSERT_TRUE(http_req(port, "POST", "/api/rerank", NULL, "{\"candidates\":[]}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":0") != NULL);
  free(r.body);

  /* 5. Error & method checks */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/rerank",
      NULL,
      "{\"candidates\":[{\"id\":999,\"distance\":0.1}]}",
      &r));
  ASSERT_EQ_I64(r.status, 404);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/rerank",
      NULL,
      "{\"lambda\":1.5,\"candidates\":[{\"id\":1,\"distance\":0.1}]}",
      &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/rerank", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_exists_batch_and_radius_endpoint(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  uint64_t ids[4] = {10ULL, 20ULL, 30ULL, 40ULL};
  float vecs[12] = {
      1.0f, 0.0f, 0.0f,
      1.2f, 0.0f, 0.0f,
      1.4f, 0.0f, 0.0f,
      0.0f, 5.0f, 0.0f
  };
  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, 4U, 3U, 1U), NBASE_OK);
  nbase_meta_field_t m10 = {"group", {.type = NBASE_META_STRING, .as = {.str = "keep"}}};
  nbase_meta_field_t m20 = {"group", {.type = NBASE_META_STRING, .as = {.str = "drop"}}};
  nbase_meta_field_t m30 = {"group", {.type = NBASE_META_STRING, .as = {.str = "keep"}}};
  ASSERT_STATUS(nbase_set_meta(db, 10ULL, &m10, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 20ULL, &m20, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 30ULL, &m30, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 30ULL, "radius match"), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  /* 1. POST /api/vectors/exists batch check */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/exists",
      NULL,
      "{\"ids\":[10,999,30,888]}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":4") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"existingCount\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "{\"id\":10,\"exists\":true}") != NULL);
  ASSERT_TRUE(strstr(r.body, "{\"id\":999,\"exists\":false}") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/vectors/exists", NULL, "{\"ids\":[]}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":0") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/exists", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  /* 2. POST /api/search/radius unfiltered & filtered */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/radius",
      NULL,
      "{\"query\":[1,0,0],\"maxDistance\":0.45,\"maxResults\":10}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":3") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":40") == NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/radius",
      NULL,
      "{\"query\":[1,0,0],\"maxDistance\":0.45,\"filters\":{\"group\":\"keep\"},"
      "\"includeMetadata\":true,\"includeVectors\":true,\"includeText\":true}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":20") == NULL);
  ASSERT_TRUE(strstr(r.body, "\"text\":\"radius match\"") != NULL);
  free(r.body);

  /* Error cases */
  ASSERT_TRUE(http_req(port, "POST", "/api/search/radius", NULL, "{\"query\":[1,0,0]}", &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/search/radius", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_head_and_filtered_export(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  uint64_t ids[3] = {10ULL, 20ULL, 30ULL};
  float vecs[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f
  };
  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, 3U, 3U, 1U), NBASE_OK);
  nbase_meta_field_t m10 = {"env", {.type = NBASE_META_STRING, .as = {.str = "prod"}}};
  nbase_meta_field_t m20 = {"env", {.type = NBASE_META_STRING, .as = {.str = "dev"}}};
  nbase_meta_field_t m30 = {"env", {.type = NBASE_META_STRING, .as = {.str = "prod"}}};
  ASSERT_STATUS(nbase_set_meta(db, 10ULL, &m10, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 20ULL, &m20, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 30ULL, &m30, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 10ULL, "prod ten"), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  hcfg.api_key = "secret123";
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  /* 1. HEAD /health succeeds without auth and returns empty body */
  ASSERT_TRUE(http_req(port, "HEAD", "/health", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_EQ_U64(r.body_len, 0ULL);
  free(r.body);

  /* 2. HEAD /api/vectors/:id enforces auth and returns 200 / 404 */
  ASSERT_TRUE(http_req(port, "HEAD", "/api/vectors/10", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 401);
  free(r.body);

  ASSERT_TRUE(http_req(port, "HEAD", "/api/vectors/10", "secret123", NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_EQ_U64(r.body_len, 0ULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "HEAD", "/api/vectors/999", "secret123", NULL, &r));
  ASSERT_EQ_I64(r.status, 404);
  ASSERT_EQ_U64(r.body_len, 0ULL);
  free(r.body);

  /* 3. POST /api/vectors/export with metadata filter and includeVectors=false */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/export",
      "secret123",
      "{\"filters\":{\"env\":\"prod\"},\"includeVectors\":false,\"includeMetadata\":true,\"includeText\":true}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":10") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":30") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":20") == NULL);
  ASSERT_TRUE(strstr(r.body, "\"vector\":") == NULL);
  ASSERT_TRUE(strstr(r.body, "\"text\":\"prod ten\"") != NULL);
  free(r.body);

  /* 4. Unfiltered POST export + error checks */
  ASSERT_TRUE(http_req(port, "POST", "/api/vectors/export", "secret123", "{}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":20") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/export",
      "secret123",
      "{\"filters\":[{\"bad\":1}]}",
      &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  ASSERT_TRUE(http_req(port, "DELETE", "/api/vectors/export", "secret123", NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_patch_vector_head_subresources_and_export_ids(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  const uint64_t ids[3] = {10ULL, 20ULL, 30ULL};
  const float vecs[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f,
  };
  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, 3U, 3U, 1U), NBASE_OK);
  nbase_meta_field_t m10[2] = {
      {"env", {.type = NBASE_META_STRING, .as = {.str = "prod"}}},
      {"tier", {.type = NBASE_META_INT64, .as = {.i64 = 1}}},
  };
  nbase_meta_field_t m20[1] = {
      {"env", {.type = NBASE_META_STRING, .as = {.str = "staging"}}},
  };
  nbase_meta_field_t m30[1] = {
      {"env", {.type = NBASE_META_STRING, .as = {.str = "prod"}}},
  };
  ASSERT_STATUS(nbase_set_meta(db, 10ULL, m10, 2U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 20ULL, m20, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 30ULL, m30, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 10ULL, "initial text ten"), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  /* 1. HEAD /api/vectors/:id/metadata and /text */
  ASSERT_TRUE(http_req(port, "HEAD", "/api/vectors/10/metadata", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_EQ_U64(r.body_len, 0ULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "HEAD", "/api/vectors/999/metadata", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 404);
  ASSERT_EQ_U64(r.body_len, 0ULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "HEAD", "/api/vectors/10/text", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_EQ_U64(r.body_len, 0ULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "HEAD", "/api/vectors/20/text", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 404);
  ASSERT_EQ_U64(r.body_len, 0ULL);
  free(r.body);

  /* 2. PATCH /api/vectors/:id partial metadata + text update & preserve untouched keys */
  ASSERT_TRUE(http_req(
      port,
      "PATCH",
      "/api/vectors/10",
      NULL,
      "{\"metadata\":{\"tier\":2},\"text\":\"patched text ten\"}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/vectors/10?includeText=true", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"env\":\"prod\"") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"tier\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"text\":\"patched text ten\"") != NULL);
  free(r.body);

  /* 3. PATCH vector + clear text via null */
  ASSERT_TRUE(http_req(
      port,
      "PATCH",
      "/api/vectors/10",
      NULL,
      "{\"vector\":[0.5,0.5,0],\"text\":null}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  free(r.body);

  ASSERT_TRUE(http_req(port, "HEAD", "/api/vectors/10/text", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 404);
  ASSERT_EQ_U64(r.body_len, 0ULL);
  free(r.body);

  /* PATCH errors: missing id -> 404, empty/bad body -> 400 */
  ASSERT_TRUE(http_req(port, "PATCH", "/api/vectors/999", NULL, "{\"metadata\":{\"a\":1}}", &r));
  ASSERT_EQ_I64(r.status, 404);
  free(r.body);

  ASSERT_TRUE(http_req(port, "PATCH", "/api/vectors/10", NULL, "{}", &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  ASSERT_TRUE(http_req(port, "PATCH", "/api/vectors/10", NULL, "{\"text\":123}", &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  /* 4. Selective POST /api/vectors/export with ids and ids + filters */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/export",
      NULL,
      "{\"ids\":[30,10,999],\"includeMetadata\":false}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":30") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":10") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":20") == NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":999") == NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/export",
      NULL,
      "{\"ids\":[20,10],\"filters\":{\"env\":\"prod\"}}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":10") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":20") == NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":30") == NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/vectors/export", NULL, "{\"ids\":[]}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_EQ_U64(r.body_len, 0ULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/vectors/export", NULL, "{\"ids\":[\"bad\"]}", &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

static void test_http_server_text_and_hybrid_batch_search(void) {
  nbase_config_t cfg = nbase_config_default(3U);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  const uint64_t ids[3] = {10ULL, 20ULL, 30ULL};
  const float vecs[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f,
  };
  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, 3U, 3U, 1U), NBASE_OK);
  nbase_meta_field_t m10[1] = {{"group", {.type = NBASE_META_STRING, .as = {.str = "alpha"}}}};
  nbase_meta_field_t m20[1] = {{"group", {.type = NBASE_META_STRING, .as = {.str = "beta"}}}};
  nbase_meta_field_t m30[1] = {{"group", {.type = NBASE_META_STRING, .as = {.str = "alpha"}}}};
  ASSERT_STATUS(nbase_set_meta(db, 10ULL, m10, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 20ULL, m20, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 30ULL, m30, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 10ULL, "simd vector hnsw kernel"), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 20ULL, "bm25 hybrid ranking score"), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 30ULL, "wal durability snapshot compact"), NBASE_OK);

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.port = 0U;
  hcfg.num_workers = 2U;
  nbase_http_server_t *srv = NULL;
  ASSERT_STATUS(nbase_http_server_start(db, &hcfg, &srv), NBASE_OK);
  uint16_t port = nbase_http_server_port(srv);

  http_resp_t r;
  /* 1. POST /api/search/text/batch (with threads parameter) */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/text/batch",
      NULL,
      "{\"queries\":[\"simd hnsw\",\"wal compact\",\"hybrid bm25\"],"
      "\"k\":2,\"threads\":4,\"filters\":{\"group\":\"alpha\"},\"includeMetadata\":true,\"includeText\":true}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"queryCount\":3") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":10") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":30") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":20") == NULL);
  ASSERT_TRUE(strstr(r.body, "\"text\":\"simd vector hnsw kernel\"") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/text/batch",
      NULL,
      "{\"queries\":[\"simd hnsw\"],\"threads\":999}",
      &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  ASSERT_TRUE(http_req(port, "GET", "/api/search/text/batch", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/search/text/batch", NULL, "{\"queries\":[123]}", &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  /* 2. POST /api/search/hybrid/batch (mixed + uniform parallel fast path) */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/hybrid/batch",
      NULL,
      "{\"queries\":["
      "{\"query\":[1,0,0],\"text\":\"simd\"},"
      "{\"query\":[0,1,0]},"
      "{\"text\":\"durability\"}"
      "],\"k\":2,\"alpha\":0.5,\"rrfK\":60,\"includeVectors\":true,\"includeMetadata\":true}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"queryCount\":3") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":10") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":20") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":30") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"vector\":[") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/hybrid/batch",
      NULL,
      "{\"queries\":["
      "{\"query\":[1,0,0],\"text\":\"simd\"},"
      "{\"query\":[0,0,1],\"text\":\"hybrid\"}"
      "],\"k\":2,\"threads\":2,\"alpha\":0.5,\"rrfK\":60}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"queryCount\":2") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":10") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":30") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "DELETE", "/api/search/hybrid/batch", NULL, NULL, &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/search/hybrid/batch", NULL, "{\"queries\":[{}]}", &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  /* 2b. POST /api/search/batch with includeMetadata/includeVectors/includeText flags */
  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/batch",
      NULL,
      "{\"queries\":[[1,0,0],[0,0,1]],\"k\":1,"
      "\"includeMetadata\":true,\"includeVectors\":true,\"includeText\":true}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"success\":true,\"results\":[[") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":10") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"id\":30") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"vector\":[") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"metadata\":{") != NULL);
  ASSERT_TRUE(strstr(r.body, "\"text\":\"simd vector hnsw kernel\"") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/batch",
      NULL,
      "{\"queries\":[[1,0,0]],\"k\":1,\"includeVectors\":\"yes\"}",
      &r));
  ASSERT_EQ_I64(r.status, 400);
  free(r.body);

  /* 3. DELETE /api/vectors/text & POST /api/vectors/text/delete */
  ASSERT_TRUE(http_req(port, "PUT", "/api/vectors/text", NULL, "{\"ids\":[10]}", &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  ASSERT_TRUE(http_req(port, "DELETE", "/api/vectors/text", NULL, "{\"ids\":[20,999]}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"deleted\":1") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/vectors/text/delete", NULL, "{\"ids\":[30]}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"deleted\":1") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/search/text/batch",
      NULL,
      "{\"queries\":[\"hybrid bm25\"],\"k\":2}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"id\":30") == NULL);
  free(r.body);

  /* 4. DELETE /api/vectors/metadata & POST /api/vectors/metadata/delete */
  ASSERT_TRUE(http_req(port, "PUT", "/api/vectors/metadata", NULL, "{\"ids\":[10]}", &r));
  ASSERT_EQ_I64(r.status, 405);
  free(r.body);

  ASSERT_TRUE(http_req(port, "DELETE", "/api/vectors/metadata", NULL, "{\"ids\":[10,999]}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"deleted\":1") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(port, "POST", "/api/vectors/metadata/delete", NULL, "{\"ids\":[30]}", &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"deleted\":1") != NULL);
  free(r.body);

  ASSERT_TRUE(http_req(
      port,
      "POST",
      "/api/vectors/count",
      NULL,
      "{\"filters\":{\"group\":\"alpha\"}}",
      &r));
  ASSERT_EQ_I64(r.status, 200);
  ASSERT_TRUE(strstr(r.body, "\"count\":0") != NULL);
  free(r.body);

  nbase_http_server_stop(srv);
  nbase_close(db);
}

void test_server_run(void) {
  RUN_TEST(test_http_server_crud_search_auth_and_admin);
  RUN_TEST(test_http_server_concurrent_clients_tsan);
  RUN_TEST(test_http_server_cors_and_ratelimit);
  RUN_TEST(test_http_server_keepalive_and_hybrid);
  RUN_TEST(test_http_server_batch_delete_and_telemetry);
  RUN_TEST(test_http_server_weighted_rerank_and_cache);
  RUN_TEST(test_http_server_upsert_and_get_batch);
  RUN_TEST(test_http_server_list_vectors_and_radius_search);
  RUN_TEST(test_http_server_batch_meta_text_and_filtered_graph);
  RUN_TEST(test_http_server_count_and_resource_crud);
  RUN_TEST(test_http_server_post_similar_and_distance);
  RUN_TEST(test_http_server_ndjson_import_export);
  RUN_TEST(test_http_server_rerank_endpoint);
  RUN_TEST(test_http_server_exists_batch_and_radius_endpoint);
  RUN_TEST(test_http_server_head_and_filtered_export);
  RUN_TEST(test_http_server_patch_vector_head_subresources_and_export_ids);
  RUN_TEST(test_http_server_text_and_hybrid_batch_search);
}
