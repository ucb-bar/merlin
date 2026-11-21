/* OUT_BIN v1 frames with explicit input/result shape companions. */
#ifndef MERLIN_OWNED_RESULT_READBACK_H
#define MERLIN_OWNED_RESULT_READBACK_H
#include "owned_result_driver.h"
typedef struct {
  int (*write)(void *, const void *, size_t);
  void *context;
} MerlinOwnedReadback;
/* Use as the driver's observer. The executor separately writes DONE only
 * after the complete source call, result copies and all callbacks succeed. */
int merlin_owned_readback_observe(void *, int, size_t, const char *, size_t,
                                 const intptr_t *, const void *, size_t);
#endif
