#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define NBASE_WAL_MAX_FRAME_PAYLOAD (64U * 1024U * 1024U)

struct nbase_wal {
  int fd;
  char *path;
  uint64_t next_seq;
};

static uint32_t wal_frame_crc(const nbw_frame_hdr_t *fhdr, const void *payload, uint32_t payload_len) {
  size_t hdr_tail_off = offsetof(nbw_frame_hdr_t, op);
  const uint8_t *hdr_bytes = (const uint8_t *)fhdr;
  uint32_t crc = nbase_crc32_update(0U, hdr_bytes + hdr_tail_off, sizeof(*fhdr) - hdr_tail_off);
  if (payload_len > 0U && payload != NULL) {
    crc = nbase_crc32_update(crc, payload, payload_len);
  }
  return crc;
}

static nbase_status_t wal_write_header(nbase_wal_t *wal) {
  nbw_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBW_MAGIC, 8);
  hdr.version = NBASE_NBW_VERSION;
  hdr.flags = 0U;
  hdr.next_seq = wal->next_seq;
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbw_header_t, header_crc32));

  if (lseek(wal->fd, 0, SEEK_SET) < 0) {
    return NBASE_ERR_IO;
  }
  nbase_status_t st = nbase_write_all_fd(wal->fd, &hdr, sizeof(hdr));
  if (st != NBASE_OK) {
    return st;
  }
  if (fsync(wal->fd) != 0) {
    return NBASE_ERR_IO;
  }
  return NBASE_OK;
}

static nbase_status_t wal_append_frame(
    nbase_wal_t *wal,
    nbase_wal_op_t op,
    const void *payload,
    uint32_t payload_len,
    bool do_sync) {
  if (!wal || wal->fd < 0) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (payload_len > 0U && !payload) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (lseek(wal->fd, 0, SEEK_END) < 0) {
    return NBASE_ERR_IO;
  }

  nbw_frame_hdr_t fhdr;
  memset(&fhdr, 0, sizeof(fhdr));
  fhdr.payload_len = payload_len;
  fhdr.op = (uint32_t)op;
  fhdr.reserved = 0U;
  fhdr.frame_crc32 = wal_frame_crc(&fhdr, payload, payload_len);

  nbase_status_t st = nbase_write_all_fd(wal->fd, &fhdr, sizeof(fhdr));
  if (st == NBASE_OK && payload_len > 0U) {
    st = nbase_write_all_fd(wal->fd, payload, payload_len);
  }
  if (st != NBASE_OK) {
    return st;
  }
  if (do_sync && fsync(wal->fd) != 0) {
    return NBASE_ERR_IO;
  }
  wal->next_seq++;
  return NBASE_OK;
}

nbase_status_t nbase_wal_open(const char *wal_path, nbase_wal_t **out_wal) {
  if (!wal_path || !out_wal) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_wal = NULL;

  int fd = open(wal_path, O_RDWR | O_CREAT, 0644);
  if (fd < 0) {
    return NBASE_ERR_IO;
  }

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0) {
    close(fd);
    return NBASE_ERR_IO;
  }

  size_t path_len = strlen(wal_path);
  nbase_wal_t *wal = calloc(1, sizeof(*wal));
  char *path_copy = malloc(path_len + 1U);
  if (!wal || !path_copy) {
    free(wal);
    free(path_copy);
    close(fd);
    return NBASE_ERR_NOMEM;
  }
  memcpy(path_copy, wal_path, path_len + 1U);
  wal->fd = fd;
  wal->path = path_copy;
  wal->next_seq = 1ULL;

  if (st_buf.st_size == 0) {
    nbase_status_t st = wal_write_header(wal);
    if (st != NBASE_OK) {
      nbase_wal_close(wal);
      return st;
    }
  } else {
    if (st_buf.st_size < (off_t)sizeof(nbw_header_t)) {
      nbase_wal_close(wal);
      return NBASE_ERR_CORRUPT;
    }
    nbw_header_t hdr;
    nbase_status_t st = nbase_read_exact(fd, &hdr, sizeof(hdr));
    if (st != NBASE_OK) {
      nbase_wal_close(wal);
      return st;
    }
    if (memcmp(hdr.magic, NBASE_NBW_MAGIC, 8) != 0 || hdr.version != NBASE_NBW_VERSION) {
      nbase_wal_close(wal);
      return NBASE_ERR_CORRUPT;
    }
    if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nbw_header_t, header_crc32))) {
      nbase_wal_close(wal);
      return NBASE_ERR_CORRUPT;
    }
    wal->next_seq = hdr.next_seq;
  }

  *out_wal = wal;
  return NBASE_OK;
}

void nbase_wal_close(nbase_wal_t *wal) {
  if (!wal) {
    return;
  }
  if (wal->fd >= 0) {
    close(wal->fd);
    wal->fd = -1;
  }
  free(wal->path);
  free(wal);
}

nbase_status_t nbase_wal_reset(nbase_wal_t *wal) {
  if (!wal || wal->fd < 0) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (ftruncate(wal->fd, 0) != 0) {
    return NBASE_ERR_IO;
  }
  return wal_write_header(wal);
}

nbase_status_t nbase_wal_append_vec(nbase_wal_t *wal, nbase_wal_op_t op, uint64_t id, const float *vec, uint32_t dim) {
  if (!wal || !vec || dim == 0U || dim > NBASE_MAX_DIM) {
    return NBASE_ERR_INVALID_ARG;
  }
  size_t payload_sz = 16U + (size_t)dim * sizeof(float);
  if (payload_sz > (size_t)NBASE_WAL_MAX_FRAME_PAYLOAD) {
    return NBASE_ERR_INVALID_ARG;
  }
  uint8_t *buf = malloc(payload_sz);
  if (!buf) {
    return NBASE_ERR_NOMEM;
  }
  uint32_t pad = 0U;
  memcpy(buf, &id, sizeof(id));
  memcpy(buf + 8, &dim, sizeof(dim));
  memcpy(buf + 12, &pad, sizeof(pad));
  memcpy(buf + 16, vec, (size_t)dim * sizeof(float));

  nbase_status_t st = wal_append_frame(wal, op, buf, (uint32_t)payload_sz, true);
  free(buf);
  return st;
}

nbase_status_t nbase_wal_append_vec_batch(nbase_wal_t *wal, const uint64_t *ids, const float *vectors, uint32_t count, uint32_t dim) {
  if (!wal || !ids || !vectors || dim == 0U || dim > NBASE_MAX_DIM) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  size_t payload_sz = 16U + (size_t)dim * sizeof(float);
  if (payload_sz > (size_t)NBASE_WAL_MAX_FRAME_PAYLOAD) {
    return NBASE_ERR_INVALID_ARG;
  }
  uint8_t *buf = malloc(payload_sz);
  if (!buf) {
    return NBASE_ERR_NOMEM;
  }

  uint32_t pad = 0U;
  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < count; i++) {
    uint64_t id = ids[i];
    const float *vec = vectors + (size_t)i * (size_t)dim;
    memcpy(buf, &id, sizeof(id));
    memcpy(buf + 8, &dim, sizeof(dim));
    memcpy(buf + 12, &pad, sizeof(pad));
    memcpy(buf + 16, vec, (size_t)dim * sizeof(float));
    st = wal_append_frame(wal, NBASE_WAL_OP_ADD, buf, (uint32_t)payload_sz, false);
    if (st != NBASE_OK) {
      break;
    }
  }
  free(buf);
  if (st == NBASE_OK && fsync(wal->fd) != 0) {
    return NBASE_ERR_IO;
  }
  return st;
}

nbase_status_t nbase_wal_append_id(nbase_wal_t *wal, nbase_wal_op_t op, uint64_t id) {
  if (!wal) {
    return NBASE_ERR_INVALID_ARG;
  }
  return wal_append_frame(wal, op, &id, (uint32_t)sizeof(id), true);
}

nbase_status_t nbase_wal_append_meta(nbase_wal_t *wal, nbase_wal_op_t op, uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
  if (!wal || (!fields && count > 0U)) {
    return NBASE_ERR_INVALID_ARG;
  }

  size_t total = 12U; /* uint64_t id + uint32_t count */
  for (uint32_t i = 0U; i < count; i++) {
    if (!fields[i].key) {
      return NBASE_ERR_INVALID_ARG;
    }
    size_t klen = strlen(fields[i].key);
    if (klen == 0U || klen > 65535U) {
      return NBASE_ERR_INVALID_ARG;
    }
    size_t flen = 12U + klen;
    if (fields[i].value.type == NBASE_META_STRING) {
      const char *s = fields[i].value.as.str ? fields[i].value.as.str : "";
      size_t vlen = strlen(s);
      if (vlen > NBASE_WAL_MAX_FRAME_PAYLOAD) {
        return NBASE_ERR_INVALID_ARG;
      }
      flen += vlen;
    } else if (fields[i].value.type == NBASE_META_INT64 || fields[i].value.type == NBASE_META_DOUBLE) {
      flen += 8U;
    } else if (fields[i].value.type == NBASE_META_BOOL) {
      flen += 4U;
    } else {
      return NBASE_ERR_INVALID_ARG;
    }
    if (total > NBASE_WAL_MAX_FRAME_PAYLOAD - flen) {
      return NBASE_ERR_INVALID_ARG;
    }
    total += flen;
  }

  uint8_t *buf = malloc(total);
  if (!buf) {
    return NBASE_ERR_NOMEM;
  }

  size_t off = 0U;
  memcpy(buf + off, &id, 8U);
  off += 8U;
  memcpy(buf + off, &count, 4U);
  off += 4U;

  for (uint32_t i = 0U; i < count; i++) {
    uint32_t klen = (uint32_t)strlen(fields[i].key);
    uint32_t ftype = (uint32_t)fields[i].value.type;
    uint32_t vlen = 0U;
    const char *s = NULL;
    if (fields[i].value.type == NBASE_META_STRING) {
      s = fields[i].value.as.str ? fields[i].value.as.str : "";
      vlen = (uint32_t)strlen(s);
    }
    memcpy(buf + off, &klen, 4U);
    off += 4U;
    memcpy(buf + off, &ftype, 4U);
    off += 4U;
    memcpy(buf + off, &vlen, 4U);
    off += 4U;
    memcpy(buf + off, fields[i].key, klen);
    off += klen;

    if (fields[i].value.type == NBASE_META_STRING) {
      if (vlen > 0U) {
        memcpy(buf + off, s, vlen);
        off += vlen;
      }
    } else if (fields[i].value.type == NBASE_META_INT64) {
      memcpy(buf + off, &fields[i].value.as.i64, 8U);
      off += 8U;
    } else if (fields[i].value.type == NBASE_META_DOUBLE) {
      memcpy(buf + off, &fields[i].value.as.f64, 8U);
      off += 8U;
    } else {
      uint32_t b = fields[i].value.as.b ? 1U : 0U;
      memcpy(buf + off, &b, 4U);
      off += 4U;
    }
  }

  nbase_status_t st = wal_append_frame(wal, op, buf, (uint32_t)total, true);
  free(buf);
  return st;
}

nbase_status_t nbase_wal_append_text(nbase_wal_t *wal, uint64_t id, const char *text) {
  if (!wal || !text) {
    return NBASE_ERR_INVALID_ARG;
  }
  size_t tlen = strlen(text);
  if (tlen > (size_t)NBASE_WAL_MAX_FRAME_PAYLOAD - 12U) {
    return NBASE_ERR_INVALID_ARG;
  }
  size_t payload_sz = 12U + tlen;
  uint8_t *buf = malloc(payload_sz);
  if (!buf) {
    return NBASE_ERR_NOMEM;
  }
  uint32_t tlen_u32 = (uint32_t)tlen;
  memcpy(buf, &id, 8U);
  memcpy(buf + 8, &tlen_u32, 4U);
  if (tlen > 0U) {
    memcpy(buf + 12, text, tlen);
  }
  nbase_status_t st = wal_append_frame(wal, NBASE_WAL_OP_SET_TEXT, buf, (uint32_t)payload_sz, true);
  free(buf);
  return st;
}

nbase_status_t nbase_wal_append_compact(nbase_wal_t *wal) {
  if (!wal) {
    return NBASE_ERR_INVALID_ARG;
  }
  return wal_append_frame(wal, NBASE_WAL_OP_COMPACT, NULL, 0U, true);
}

static nbase_status_t replay_meta_frame(nbase_db_t *db, nbase_wal_op_t op, const uint8_t *payload, uint32_t payload_len) {
  if (payload_len < 12U) {
    return NBASE_ERR_CORRUPT;
  }
  uint64_t id = 0ULL;
  uint32_t count = 0U;
  memcpy(&id, payload, 8U);
  memcpy(&count, payload + 8, 4U);

  if (count == 0U) {
    if (payload_len != 12U) {
      return NBASE_ERR_CORRUPT;
    }
    return nbase_db_replay_meta(db, op, id, NULL, 0U);
  }
  if ((uint64_t)count > payload_len / 12U) {
    return NBASE_ERR_CORRUPT;
  }

  nbase_meta_field_t *fields = calloc(count, sizeof(*fields));
  char **owned_strs = calloc((size_t)count * 2U, sizeof(char *));
  if (!fields || !owned_strs) {
    free(fields);
    free(owned_strs);
    return NBASE_ERR_NOMEM;
  }

  size_t off = 12U;
  size_t owned_cnt = 0U;
  nbase_status_t st = NBASE_OK;

  for (uint32_t i = 0U; i < count; i++) {
    if (off + 12U > payload_len) {
      st = NBASE_ERR_CORRUPT;
      break;
    }
    uint32_t klen = 0U;
    uint32_t ftype = 0U;
    uint32_t vlen = 0U;
    memcpy(&klen, payload + off, 4U);
    off += 4U;
    memcpy(&ftype, payload + off, 4U);
    off += 4U;
    memcpy(&vlen, payload + off, 4U);
    off += 4U;

    if (klen == 0U || off + klen > payload_len) {
      st = NBASE_ERR_CORRUPT;
      break;
    }
    char *kbuf = malloc((size_t)klen + 1U);
    if (!kbuf) {
      st = NBASE_ERR_NOMEM;
      break;
    }
    memcpy(kbuf, payload + off, klen);
    kbuf[klen] = '\0';
    off += klen;
    owned_strs[owned_cnt++] = kbuf;
    fields[i].key = kbuf;
    fields[i].value.type = (nbase_meta_type_t)ftype;

    if (ftype == (uint32_t)NBASE_META_STRING) {
      if (off + vlen > payload_len) {
        st = NBASE_ERR_CORRUPT;
        break;
      }
      char *vbuf = malloc((size_t)vlen + 1U);
      if (!vbuf) {
        st = NBASE_ERR_NOMEM;
        break;
      }
      if (vlen > 0U) {
        memcpy(vbuf, payload + off, vlen);
      }
      vbuf[vlen] = '\0';
      off += vlen;
      owned_strs[owned_cnt++] = vbuf;
      fields[i].value.as.str = vbuf;
    } else if (ftype == (uint32_t)NBASE_META_INT64) {
      if (vlen != 0U || off + 8U > payload_len) {
        st = NBASE_ERR_CORRUPT;
        break;
      }
      memcpy(&fields[i].value.as.i64, payload + off, 8U);
      off += 8U;
    } else if (ftype == (uint32_t)NBASE_META_DOUBLE) {
      if (vlen != 0U || off + 8U > payload_len) {
        st = NBASE_ERR_CORRUPT;
        break;
      }
      memcpy(&fields[i].value.as.f64, payload + off, 8U);
      off += 8U;
    } else if (ftype == (uint32_t)NBASE_META_BOOL) {
      if (vlen != 0U || off + 4U > payload_len) {
        st = NBASE_ERR_CORRUPT;
        break;
      }
      uint32_t b = 0U;
      memcpy(&b, payload + off, 4U);
      off += 4U;
      fields[i].value.as.b = (b != 0U);
    } else {
      st = NBASE_ERR_CORRUPT;
      break;
    }
  }

  if (st == NBASE_OK && off != payload_len) {
    st = NBASE_ERR_CORRUPT;
  }
  if (st == NBASE_OK) {
    st = nbase_db_replay_meta(db, op, id, fields, count);
  }

  for (size_t i = 0U; i < owned_cnt; i++) {
    free(owned_strs[i]);
  }
  free(owned_strs);
  free(fields);
  return st;
}

nbase_status_t nbase_wal_replay(nbase_wal_t *wal, nbase_db_t *db, uint64_t *out_applied) {
  if (!wal || !db) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (out_applied) {
    *out_applied = 0ULL;
  }

  struct stat st_buf;
  if (fstat(wal->fd, &st_buf) != 0) {
    return NBASE_ERR_IO;
  }
  if (st_buf.st_size < (off_t)sizeof(nbw_header_t)) {
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;
  uint64_t offset = (uint64_t)sizeof(nbw_header_t);
  if (lseek(wal->fd, (off_t)offset, SEEK_SET) < 0) {
    return NBASE_ERR_IO;
  }

  uint64_t applied = 0ULL;
  bool truncate_tail = false;

  while (offset < file_size) {
    uint64_t rem = file_size - offset;
    if (rem < (uint64_t)sizeof(nbw_frame_hdr_t)) {
      truncate_tail = true;
      break;
    }

    nbw_frame_hdr_t fhdr;
    nbase_status_t st = nbase_read_exact(wal->fd, &fhdr, sizeof(fhdr));
    if (st != NBASE_OK) {
      return st;
    }
    uint64_t after_hdr_rem = rem - (uint64_t)sizeof(nbw_frame_hdr_t);
    if (fhdr.payload_len > NBASE_WAL_MAX_FRAME_PAYLOAD || (uint64_t)fhdr.payload_len > after_hdr_rem) {
      truncate_tail = true;
      break;
    }

    uint8_t *payload = NULL;
    if (fhdr.payload_len > 0U) {
      payload = malloc(fhdr.payload_len);
      if (!payload) {
        return NBASE_ERR_NOMEM;
      }
      st = nbase_read_exact(wal->fd, payload, fhdr.payload_len);
      if (st != NBASE_OK) {
        free(payload);
        return st;
      }
    }

    uint32_t calc_crc = wal_frame_crc(&fhdr, payload, fhdr.payload_len);
    if (calc_crc != fhdr.frame_crc32) {
      free(payload);
      truncate_tail = true;
      break;
    }

    nbase_wal_op_t op = (nbase_wal_op_t)fhdr.op;
    if (op == NBASE_WAL_OP_ADD || op == NBASE_WAL_OP_UPSERT) {
      if (fhdr.payload_len < 16U) {
        free(payload);
        truncate_tail = true;
        break;
      }
      uint64_t id = 0ULL;
      uint32_t dim = 0U;
      memcpy(&id, payload, 8U);
      memcpy(&dim, payload + 8, 4U);
      if (dim == 0U || dim > NBASE_MAX_DIM || (uint64_t)fhdr.payload_len != 16ULL + (uint64_t)dim * sizeof(float)) {
        free(payload);
        truncate_tail = true;
        break;
      }
      float *vec = malloc((size_t)dim * sizeof(float));
      if (!vec) {
        free(payload);
        return NBASE_ERR_NOMEM;
      }
      memcpy(vec, payload + 16, (size_t)dim * sizeof(float));
      st = nbase_db_replay_vec(db, op, id, vec, dim);
      free(vec);
    } else if (op == NBASE_WAL_OP_DELETE) {
      if (fhdr.payload_len != 8U) {
        free(payload);
        truncate_tail = true;
        break;
      }
      uint64_t id = 0ULL;
      memcpy(&id, payload, 8U);
      st = nbase_db_replay_delete(db, id);
    } else if (op == NBASE_WAL_OP_SET_META || op == NBASE_WAL_OP_UPDATE_META) {
      st = replay_meta_frame(db, op, payload, fhdr.payload_len);
    } else if (op == NBASE_WAL_OP_DELETE_META) {
      if (fhdr.payload_len != 8U) {
        free(payload);
        truncate_tail = true;
        break;
      }
      uint64_t id = 0ULL;
      memcpy(&id, payload, 8U);
      st = nbase_db_replay_meta(db, op, id, NULL, 0U);
    } else if (op == NBASE_WAL_OP_SET_TEXT) {
      if (fhdr.payload_len < 12U) {
        free(payload);
        truncate_tail = true;
        break;
      }
      uint64_t id = 0ULL;
      uint32_t tlen = 0U;
      memcpy(&id, payload, 8U);
      memcpy(&tlen, payload + 8, 4U);
      if ((uint64_t)fhdr.payload_len != 12ULL + (uint64_t)tlen) {
        free(payload);
        truncate_tail = true;
        break;
      }
      char *tbuf = malloc((size_t)tlen + 1U);
      if (!tbuf) {
        free(payload);
        return NBASE_ERR_NOMEM;
      }
      if (tlen > 0U) {
        memcpy(tbuf, payload + 12, tlen);
      }
      tbuf[tlen] = '\0';
      st = nbase_db_replay_text(db, op, id, tbuf);
      free(tbuf);
    } else if (op == NBASE_WAL_OP_DELETE_TEXT) {
      if (fhdr.payload_len != 8U) {
        free(payload);
        truncate_tail = true;
        break;
      }
      uint64_t id = 0ULL;
      memcpy(&id, payload, 8U);
      st = nbase_db_replay_text(db, op, id, NULL);
    } else if (op == NBASE_WAL_OP_COMPACT) {
      if (fhdr.payload_len != 0U) {
        free(payload);
        truncate_tail = true;
        break;
      }
      st = nbase_db_replay_compact(db);
    } else {
      free(payload);
      truncate_tail = true;
      break;
    }

    free(payload);
    if (st == NBASE_ERR_CORRUPT) {
      truncate_tail = true;
      break;
    }
    if (st != NBASE_OK) {
      return st;
    }

    offset += (uint64_t)sizeof(nbw_frame_hdr_t) + (uint64_t)fhdr.payload_len;
    applied++;
  }

  if (truncate_tail) {
    if (ftruncate(wal->fd, (off_t)offset) != 0 || fsync(wal->fd) != 0) {
      return NBASE_ERR_IO;
    }
    if (lseek(wal->fd, (off_t)offset, SEEK_SET) < 0) {
      return NBASE_ERR_IO;
    }
  }

  if (applied > 0ULL) {
    wal->next_seq += applied;
  }
  if (out_applied) {
    *out_applied = applied;
  }
  return NBASE_OK;
}
