/* Freestanding result records with an explicitly selected output transport. */
#ifndef ATLAS_RESULT_H
#define ATLAS_RESULT_H
#include <stdint.h>
#include <stddef.h>

typedef void (*atlas_result_putchar_fn)(void *, char);
typedef struct {
  atlas_result_putchar_fn putchar;
  void *context;
  const char *name;
  uint32_t expected, completed, failures, finished;
} atlas_result;

static inline void atlas_result_text(atlas_result *s, const char *text) {
  while (*text) s->putchar(s->context, *text++);
}
static inline void atlas_result_number(atlas_result *s, uint64_t value) {
  char digits[20]; unsigned count = 0;
  do { digits[count++] = (char)('0' + value % 10); value /= 10; } while (value);
  while (count) s->putchar(s->context, digits[--count]);
}
static inline void atlas_result_record(atlas_result *s, const char *kind) {
  atlas_result_text(s, "ATLAS_KERNEL_"); atlas_result_text(s, kind);
  atlas_result_text(s, " name="); atlas_result_text(s, s->name);
  atlas_result_text(s, " launches_completed="); atlas_result_number(s, s->completed);
  atlas_result_text(s, " launches_expected="); atlas_result_number(s, s->expected);
}
static inline void atlas_result_begin(atlas_result *s, atlas_result_putchar_fn output,
                                      void *context, const char *name, uint32_t launches) {
  *s = (atlas_result){output, context, name, launches, 0, 0, 0};
  atlas_result_record(s, "BEGIN"); atlas_result_text(s, "\n");
}
static inline void atlas_result_fail(atlas_result *s, const char *check, uint64_t observed, uint64_t expected) {
  ++s->failures;
  atlas_result_record(s, "FAIL"); atlas_result_text(s, " check="); atlas_result_text(s, check);
  atlas_result_text(s, " observed="); atlas_result_number(s, observed);
  atlas_result_text(s, " expected="); atlas_result_number(s, expected); atlas_result_text(s, "\n");
}
static inline int atlas_result_check(atlas_result *s, const char *check, uint64_t observed, uint64_t expected) {
  if (observed == expected) return 1;
  atlas_result_fail(s, check, observed, expected); return 0;
}
static inline void atlas_result_progress(atlas_result *s, uint32_t completed, const char *boundary) {
  if (s->finished || completed <= s->completed || completed > s->expected) {
    atlas_result_fail(s, "launch-order", completed, s->completed + 1); return;
  }
  s->completed = completed;
  atlas_result_record(s, "PROGRESS"); atlas_result_text(s, " boundary=");
  atlas_result_text(s, boundary); atlas_result_text(s, "\n");
}
static inline int atlas_result_pass(atlas_result *s) {
  if (s->finished) { atlas_result_fail(s, "duplicate-finish", 1, 0); return 0; }
  s->finished = 1;
  if (!s->expected || s->completed != s->expected)
    atlas_result_fail(s, "launch-count", s->completed, s->expected);
  if (s->failures) return 0;
  atlas_result_record(s, "PASS"); atlas_result_text(s, "\n"); return 1;
}

#endif
