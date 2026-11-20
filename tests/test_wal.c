#include "test_common.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static void test_wal_enable_add_upsert_delete_meta_and_crash_recovery(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  const uint32_t dim = 8U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.seed = 42ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float v1[8] = {1, 0, 0, 0, 0, 0, 0, 0};
  float v2[8] = {0, 1, 0, 0, 0, 0, 0, 0};
  float v3[8] = {0, 0, 1, 0, 0, 0, 0, 0};
  float v3_new[8] = {0.5f, 0.5f, 0, 0, 0, 0, 0, 0};

  ASSERT_STATUS(nbase_add(db, 1ULL, v1, dim), NBASE_OK);
  ASSERT_STATUS(nbase_wal_enable(db, dir), NBASE_OK);
  ASSERT_STATUS(nbase_wal_enable(db, dir), NBASE_ERR_ALREADY_EXISTS);

  /* Mutate without calling nbase_save or checkpoint; close to simulate crash. */
  ASSERT_STATUS(nbase_add(db, 2ULL, v2, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 3ULL, v3, dim), NBASE_OK);
  ASSERT_STATUS(nbase_upsert(db, 3ULL, v3_new, dim), NBASE_OK);

  nbase_meta_field_t mf1[2] = {
      {"tag", {.type = NBASE_META_STRING, .as = {.str = "alpha"}}},
      {"rank", {.type = NBASE_META_INT64, .as = {.i64 = 10}}},
  };
  ASSERT_STATUS(nbase_set_meta(db, 2ULL, mf1, 2U), NBASE_OK);
  nbase_meta_field_t mf_upd = {"rank", {.type = NBASE_META_INT64, .as = {.i64 = 99}}};
  ASSERT_STATUS(nbase_update_meta(db, 2ULL, &mf_upd, 1U), NBASE_OK);

  nbase_meta_field_t mf3 = {"tag", {.type = NBASE_META_STRING, .as = {.str = "temp"}}};
  ASSERT_STATUS(nbase_set_meta(db, 3ULL, &mf3, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_delete_meta(db, 3ULL), NBASE_OK);

  uint64_t batch_ids[3] = {10ULL, 11ULL, 12ULL};
  float batch_vecs[3 * 8];
  test_fill_vectors(batch_vecs, 3U, dim, 515ULL);
  ASSERT_STATUS(nbase_add_batch(db, batch_ids, batch_vecs, 3U, dim, 2U), NBASE_OK);

  ASSERT_STATUS(nbase_delete(db, 1ULL), NBASE_OK);
  ASSERT_STATUS(nbase_compact(db), NBASE_OK);

  nbase_close(db);

  nbase_db_t *reopened = NULL;
  ASSERT_STATUS(nbase_open_wal(dir, &reopened), NBASE_OK);

  nbase_stats_t stats;
  ASSERT_STATUS(nbase_stats(reopened, &stats), NBASE_OK);
  ASSERT_EQ_U64(stats.active_vectors, 5ULL);
  ASSERT_TRUE(!nbase_contains(reopened, 1ULL));
  ASSERT_TRUE(nbase_contains(reopened, 2ULL));
  ASSERT_TRUE(nbase_contains(reopened, 3ULL));
  ASSERT_TRUE(nbase_contains(reopened, 11ULL));

  float out_v[8];
  ASSERT_STATUS(nbase_get(reopened, 3ULL, out_v, dim), NBASE_OK);
  ASSERT_TRUE(fabsf(out_v[0] - 0.5f) < 1e-6f);
  ASSERT_TRUE(fabsf(out_v[1] - 0.5f) < 1e-6f);

  nbase_meta_value_t mv;
  char sbuf[32];
  ASSERT_STATUS(nbase_get_meta(reopened, 2ULL, "tag", &mv, sbuf, sizeof(sbuf)), NBASE_OK);
  ASSERT_EQ_I64((int64_t)mv.type, (int64_t)NBASE_META_STRING);
  ASSERT_TRUE(strcmp(mv.as.str, "alpha") == 0);
  ASSERT_STATUS(nbase_get_meta(reopened, 2ULL, "rank", &mv, NULL, 0U), NBASE_OK);
  ASSERT_EQ_I64((int64_t)mv.type, (int64_t)NBASE_META_INT64);
  ASSERT_EQ_I64(mv.as.i64, 99LL);
  ASSERT_STATUS(nbase_get_meta(reopened, 3ULL, "tag", &mv, sbuf, sizeof(sbuf)), NBASE_ERR_NOT_FOUND);

  nbase_close(reopened);
  test_rm_rf(dir);
}

static void test_wal_torn_tail_truncation_and_checkpoint(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  char wal_path[300];
  snprintf(wal_path, sizeof(wal_path), "%s/wal.nbw", dir);

  const uint32_t dim = 4U;
  nbase_config_t cfg = nbase_config_default(dim);
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  ASSERT_STATUS(nbase_wal_enable(db, dir), NBASE_OK);

  float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_STATUS(nbase_add(db, 100ULL, v, dim), NBASE_OK);
  v[0] = 5.0f;
  ASSERT_STATUS(nbase_add(db, 200ULL, v, dim), NBASE_OK);
  nbase_close(db);

  /* Append incomplete torn tail frame (9 bytes < 16-byte frame header). */
  FILE *fp = fopen(wal_path, "ab");
  ASSERT_TRUE(fp != NULL);
  const uint8_t garbage[9] = {0xAA, 0xBB, 0xCC, 1, 2, 3, 4, 5, 6};
  ASSERT_TRUE(fwrite(garbage, 1, sizeof(garbage), fp) == sizeof(garbage));
  fclose(fp);

  struct stat st_before;
  ASSERT_TRUE(stat(wal_path, &st_before) == 0);

  nbase_db_t *rec = NULL;
  ASSERT_STATUS(nbase_open_wal(dir, &rec), NBASE_OK);
  ASSERT_TRUE(nbase_contains(rec, 100ULL));
  ASSERT_TRUE(nbase_contains(rec, 200ULL));

  struct stat st_after_replay;
  ASSERT_TRUE(stat(wal_path, &st_after_replay) == 0);
  ASSERT_EQ_I64((int64_t)st_after_replay.st_size, (int64_t)st_before.st_size - 9LL);

  /* Append after torn-tail truncation + checkpoint truncation back to 64-byte header. */
  v[0] = 9.0f;
  ASSERT_STATUS(nbase_add(rec, 300ULL, v, dim), NBASE_OK);
  ASSERT_STATUS(nbase_wal_checkpoint(rec), NBASE_OK);

  struct stat st_ckpt;
  ASSERT_TRUE(stat(wal_path, &st_ckpt) == 0);
  ASSERT_EQ_I64((int64_t)st_ckpt.st_size, 64LL);
  nbase_close(rec);

  nbase_db_t *final_db = NULL;
  ASSERT_STATUS(nbase_open(dir, &final_db), NBASE_OK);
  ASSERT_TRUE(nbase_contains(final_db, 100ULL));
  ASSERT_TRUE(nbase_contains(final_db, 200ULL));
  ASSERT_TRUE(nbase_contains(final_db, 300ULL));
  nbase_close(final_db);

  /* Corrupt WAL header must be rejected. */
  ASSERT_TRUE(test_flip_file_byte(wal_path, 4L));
  nbase_db_t *bad_db = NULL;
  ASSERT_STATUS(nbase_open(dir, &bad_db), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
}

void test_wal_run(void) {
  RUN_TEST(test_wal_enable_add_upsert_delete_meta_and_crash_recovery);
  RUN_TEST(test_wal_torn_tail_truncation_and_checkpoint);
}
