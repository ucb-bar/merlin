/* Shared rank-exact descriptor packing; include only from runtime owners. */
#ifndef MERLIN_DESCRIPTOR_INTERNAL_H
#define MERLIN_DESCRIPTOR_INTERNAL_H
#include "merlin_model.h"
/* MLIR's C interface reads a RANK-EXACT descriptor: {ptr, ptr, i64, [rank x i64], [rank x i64]}.
 * merlin_descriptor_t reserves MERLIN_MAX_RANK slots for each array so one struct fits any rank,
 * which makes it big enough but NOT the same layout -- writing d->strides[i] puts the strides at
 * word 3 + MERLIN_MAX_RANK while the model reads them at word 3 + rank. For every argument of rank
 * below MERLIN_MAX_RANK the model therefore read uninitialized stack as its strides. Whether that
 * showed depended on the ops: a static-shape kernel recomputes its own strides and never noticed,
 * while anything that materializes the descriptor (an unranked `memrefCopy`, which is how an input
 * gets copied into a buffer) indexed with garbage and faulted. So: write the fields PACKED to the
 * argument's own rank, into the same over-sized storage. */
_Static_assert(sizeof(void *) == sizeof(int64_t),
               "merlin_descriptor_t is addressed as int64_t words; a non-LP64 ABI needs its own "
               "packing");
_Static_assert(sizeof(long) == sizeof(int64_t), "ranked descriptor helper requires LP64 long");
_Static_assert(sizeof(merlin_descriptor_t) == (size_t)(3 + 2 * MERLIN_MAX_RANK) * sizeof(int64_t),
               "merlin_descriptor_t has padding; the packed write below would land in it");

static void fill_descriptor(merlin_descriptor_t *d, void *data, const merlin_arg_t *a) {
  int64_t *w = (int64_t *)d;               /* {allocated, aligned, offset, sizes[], strides[]} */
  d->allocated = data;
  d->aligned = data;
  w[2] = 0;                                /* offset */
  int64_t *sizes = w + 3;
  int64_t *strides = sizes + a->rank;
  /* row-major contiguous strides */
  long stride = 1;
  for (int i = a->rank - 1; i >= 0; i--) {
    sizes[i] = a->dims[i];
    strides[i] = stride;
    stride *= a->dims[i];
  }
}

#endif
