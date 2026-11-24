#include "test_common.h"
#include "nbase_json.h"

#include <string.h>

static void test_json_parse_valid_and_escapes(void) {
  const char *doc =
      "{\n"
      "  \"name\": \"nbase\\n\\t\\\"v0.4\\\" \\u0041\\u03B1\",\n"
      "  \"count\": 42,\n"
      "  \"neg\": -1234567890123,\n"
      "  \"ratio\": 0.5e2,\n"
      "  \"ok\": true,\n"
      "  \"fail\": false,\n"
      "  \"empty\": null,\n"
      "  \"arr\": [1, 2.5, \"x\", []]\n"
      "} ";
  nbase_json_t *root = NULL;
  ASSERT_STATUS(nbase_json_parse(doc, strlen(doc), &root), NBASE_OK);
  ASSERT_TRUE(root != NULL);
  ASSERT_EQ_I64((int64_t)root->type, (int64_t)NBASE_JSON_OBJECT);

  const nbase_json_t *jname = nbase_json_obj_get(root, "name");
  ASSERT_TRUE(jname != NULL && jname->type == NBASE_JSON_STRING);
  ASSERT_TRUE(strstr(jname->str, "nbase\n\t\"v0.4\" A") != NULL);

  const nbase_json_t *jcnt = nbase_json_obj_get(root, "count");
  ASSERT_TRUE(jcnt != NULL && jcnt->is_int && jcnt->i64 == 42);

  const nbase_json_t *jneg = nbase_json_obj_get(root, "neg");
  ASSERT_TRUE(jneg != NULL && jneg->is_int && jneg->i64 == -1234567890123LL);

  const nbase_json_t *jrat = nbase_json_obj_get(root, "ratio");
  ASSERT_TRUE(jrat != NULL && !jrat->is_int && fabs(jrat->num - 50.0) < 1e-9);

  const nbase_json_t *jok = nbase_json_obj_get(root, "ok");
  ASSERT_TRUE(jok != NULL && jok->type == NBASE_JSON_BOOL && jok->b);

  const nbase_json_t *jarr = nbase_json_obj_get(root, "arr");
  ASSERT_TRUE(jarr != NULL && jarr->type == NBASE_JSON_ARRAY && jarr->len == 4U);

  nbase_json_free(root);
}

static void test_json_rejects_malformed_and_deep_nesting(void) {
  nbase_json_t *out = NULL;
  ASSERT_STATUS(nbase_json_parse("", 0U, &out), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_json_parse("{\"a\":1} extra", 13U, &out), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_json_parse("{\"a\":01}", 8U, &out), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_json_parse("{\"a\":1,}", 8U, &out), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_json_parse("\"unterminated", 13U, &out), NBASE_ERR_INVALID_ARG);

  /* Depth > 64 must be rejected without stack overflow. */
  char deep[200];
  for (int i = 0; i < 80; i++) {
    deep[i] = '[';
    deep[159 - i] = ']';
  }
  deep[160] = '\0';
  ASSERT_STATUS(nbase_json_parse(deep, 160U, &out), NBASE_ERR_INVALID_ARG);

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_append_json_str(&sb, "hi\"\\\n");
  ASSERT_TRUE(!sb.oom && strcmp(sb.data, "\"hi\\\"\\\\\\n\"") == 0);
  nbase_sbuf_free(&sb);
}

static uint64_t json_fuzz_rng(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static void test_json_fuzz_deterministic_resilience(void) {
  static const char *const edge_cases[] = {
      "",
      "{",
      "[1,2,",
      "\"\\uD800\"",
      "\"\\uD800\\u0041\"",
      "\"\\u\"",
      "\"\\x41\"",
      "1e99999",
      "-1e99999",
      "{\"a\":1,}",
      "[1,]",
      "{\"a\":true,\"a\":false}",
      "truefalse",
      "01",
      "-",
      "\"",
      "\"hello\x01world\"",
      "null []"};
  for (size_t i = 0U; i < sizeof(edge_cases) / sizeof(edge_cases[0]); i++) {
    nbase_json_t *out = NULL;
    nbase_status_t st = nbase_json_parse(edge_cases[i], strlen(edge_cases[i]), &out);
    (void)st;
    nbase_json_free(out);
  }

  /* Explicit embedded NUL check via length-delimited parser. */
  const char nul_doc[9] = {'"', 'a', '\0', 'b', '"', ' ', ' ', ' ', ' '};
  nbase_json_t *nul_out = NULL;
  (void)nbase_json_parse(nul_doc, 5U, &nul_out);
  nbase_json_free(nul_out);

  /* Mutate a structured base JSON document deterministically across 1000 iterations. */
  static const char seed_json[] =
      "{\"id\":42,\"v\":[1,-2.5e3,true,false,null,\"\\u0041\"],\"m\":{\"k\":\"v\"}}";
  const size_t base_len = strlen(seed_json);
  char buf[128];
  uint64_t rng = 0xCAFEBABE12345678ULL;
  for (uint32_t iter = 0U; iter < 1000U; iter++) {
    memcpy(buf, seed_json, base_len);
    size_t len = base_len;
    uint32_t muts = 1U + (uint32_t)(json_fuzz_rng(&rng) % 5U);
    for (uint32_t m = 0U; m < muts; m++) {
      uint64_t r = json_fuzz_rng(&rng);
      if ((r & 3ULL) == 0ULL && len > 1U) {
        len = 1U + (size_t)(r % len);
      } else {
        size_t pos = (size_t)(r % len);
        buf[pos] = (char)((r >> 8) & 0xFFU);
      }
    }
    buf[len] = '\0';
    nbase_json_t *out = NULL;
    nbase_status_t st = nbase_json_parse(buf, len, &out);
    if (st == NBASE_OK) {
      ASSERT_TRUE(out != NULL);
    }
    nbase_json_free(out);
  }
}

void test_json_run(void) {
  RUN_TEST(test_json_parse_valid_and_escapes);
  RUN_TEST(test_json_rejects_malformed_and_deep_nesting);
  RUN_TEST(test_json_fuzz_deterministic_resilience);
}
