/* Checked logical views of invocation-owned MLIR results. */
#ifndef MERLIN_OWNED_RESULT_VIEWS_H
#define MERLIN_OWNED_RESULT_VIEWS_H

#include <stddef.h>
#include <stdint.h>

#define MERLIN_OWNED_RESULT_MAX_RANK 16
#define MERLIN_OWNED_RESULT_MAX_COUNT 64

typedef struct {
  void *allocated, *aligned;
  intptr_t offset;
  size_t rank, element_bytes, element_alignment;
  const intptr_t *sizes, *strides, *static_sizes;
} MerlinOwnedResultDescriptor;

typedef struct {
  void *allocated, *aligned;
  intptr_t offset;
  size_t rank, element_bytes, element_alignment, count, bytes;
  intptr_t sizes[MERLIN_OWNED_RESULT_MAX_RANK];
  intptr_t strides[MERLIN_OWNED_RESULT_MAX_RANK];
  const unsigned char *data;
} MerlinOwnedResultView;

enum {
  MERLIN_OWNED_RESULT_OK,
  MERLIN_OWNED_RESULT_TYPE,
  MERLIN_OWNED_RESULT_EXTENT,
  MERLIN_OWNED_RESULT_OWNER,
  MERLIN_OWNED_RESULT_FOOTPRINT,
  MERLIN_OWNED_RESULT_BUDGET,
  MERLIN_OWNED_RESULT_UNRETURNED,
  MERLIN_OWNED_RESULT_COPY
};

/* Descriptor arrays/types are compiler-owned metadata. Their pointer values,
 * offsets, extents and strides may be untrusted execution results. No payload
 * is read here. Failure clears every output view. */
int merlin_owned_result_validate_all(const MerlinOwnedResultDescriptor *, size_t,
                                     MerlinOwnedResultView *, size_t);
/* The caller provides valid independent destination storage. This rechecks
 * the live source view before reading, and refuses overlapping storage. */
int merlin_owned_result_copy(const MerlinOwnedResultView *, void *, size_t);
const char *merlin_owned_result_error(int);

#endif
