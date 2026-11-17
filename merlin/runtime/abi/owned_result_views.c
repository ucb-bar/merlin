/* No target layout or scheduling facts: use the selected C ABI and actual
 * invocation allocation records. Result ownership remains with the caller
 * until merlin_owned_results_end releases the tracked owners exactly once. */
#include "owned_result_views.h"
#include <string.h>

_Static_assert(SIZE_MAX == UINTPTR_MAX, "result views require matching size/address widths");

extern size_t merlin_owned_results_capacity(void *);
extern size_t merlin_owned_results_live_count(void);

static uintptr_t magnitude(intptr_t value) {
  /* Avoid negating INTPTR_MIN in signed arithmetic. */
  return value < 0 ? (uintptr_t)(-(value + 1)) + 1 : (uintptr_t)value;
}

static int product(size_t left, size_t right, size_t *out) {
  if (right && left > SIZE_MAX / right)
    return 0;
  *out = left * right;
  return 1;
}

static int checked_view(const MerlinOwnedResultDescriptor *d,
                        MerlinOwnedResultView *view, size_t budget) {
  if (!d || d->rank > MERLIN_OWNED_RESULT_MAX_RANK || !d->element_bytes ||
      !d->element_alignment ||
      (d->element_alignment & (d->element_alignment - 1)) ||
      d->element_bytes % d->element_alignment ||
      (d->rank && (!d->sizes || !d->strides)))
    return MERLIN_OWNED_RESULT_TYPE;
  int empty = 0;
  for (size_t axis = 0; axis < d->rank; ++axis) {
    if (d->sizes[axis] < 0 ||
        (d->static_sizes && d->static_sizes[axis] >= 0 &&
         d->static_sizes[axis] != d->sizes[axis]))
      return MERLIN_OWNED_RESULT_EXTENT;
    empty |= d->sizes[axis] == 0;
  }
  size_t capacity = merlin_owned_results_capacity(d->allocated);
  if (!d->allocated || !capacity)
    return MERLIN_OWNED_RESULT_OWNER;
  memset(view, 0, sizeof(*view));
  view->allocated = d->allocated;
  view->aligned = d->aligned;
  view->offset = d->offset;
  view->rank = d->rank;
  view->element_bytes = d->element_bytes;
  view->element_alignment = d->element_alignment;
  for (size_t axis = 0; axis < d->rank; ++axis) {
    view->sizes[axis] = d->sizes[axis];
    view->strides[axis] = d->strides[axis];
  }
  /* All extents/static types and ownership were checked. Empty tensors do not
   * evaluate inactive pointer arithmetic, including extreme offsets/strides. */
  if (empty)
    return MERLIN_OWNED_RESULT_OK;
  size_t count = 1, bytes;
  for (size_t axis = 0; axis < d->rank; ++axis)
    if (!product(count, (size_t)d->sizes[axis], &count))
      return MERLIN_OWNED_RESULT_BUDGET;
  if (!product(count, d->element_bytes, &bytes) || bytes > budget)
    return MERLIN_OWNED_RESULT_BUDGET;
  uintptr_t base = (uintptr_t)d->allocated, aligned = (uintptr_t)d->aligned;
  if (!aligned || aligned % d->element_alignment ||
      capacity - 1 > UINTPTR_MAX - base)
    return MERLIN_OWNED_RESULT_FOOTPRINT;
  size_t offset;
  if (!product(magnitude(d->offset), d->element_bytes, &offset) ||
      (d->offset < 0 ? offset > aligned : offset > UINTPTR_MAX - aligned))
    return MERLIN_OWNED_RESULT_FOOTPRINT;
  uintptr_t start = d->offset < 0 ? aligned - offset : aligned + offset;
  uintptr_t last = base + capacity - 1;
  if (start < base || start > last)
    return MERLIN_OWNED_RESULT_FOOTPRINT;
  size_t before = 0, after = 0;
  for (size_t axis = 0; axis < d->rank; ++axis) {
    size_t span;
    if (!product((size_t)d->sizes[axis] - 1, magnitude(d->strides[axis]), &span) ||
        !product(span, d->element_bytes, &span))
      return MERLIN_OWNED_RESULT_FOOTPRINT;
    size_t *sum = d->strides[axis] < 0 ? &before : &after;
    if (span > SIZE_MAX - *sum)
      return MERLIN_OWNED_RESULT_FOOTPRINT;
    *sum += span;
  }
  if (before > start - base || after > last - start ||
      d->element_bytes - 1 > last - start - after)
    return MERLIN_OWNED_RESULT_FOOTPRINT;
  view->data = (const unsigned char *)start;
  view->count = count;
  view->bytes = bytes;
  return MERLIN_OWNED_RESULT_OK;
}

int merlin_owned_result_validate_all(const MerlinOwnedResultDescriptor *descriptors,
                                     size_t count, MerlinOwnedResultView *views,
                                     size_t budget) {
  if (!descriptors || !views || !count || count > MERLIN_OWNED_RESULT_MAX_COUNT)
    return MERLIN_OWNED_RESULT_TYPE;
  memset(views, 0, count * sizeof(*views));
  size_t bytes = 0, owners = 0;
  int status = MERLIN_OWNED_RESULT_OK;
  for (size_t i = 0; i < count; ++i) {
    status = checked_view(&descriptors[i], &views[i], budget - bytes);
    if (status)
      break;
    bytes += views[i].bytes;
    int unique = 1;
    for (size_t j = 0; j < i; ++j)
      if (views[j].allocated == views[i].allocated)
        unique = 0;
    owners += unique;
  }
  if (!status && owners != merlin_owned_results_live_count())
    status = MERLIN_OWNED_RESULT_UNRETURNED;
  if (status)
    memset(views, 0, count * sizeof(*views));
  return status;
}

int merlin_owned_result_copy(const MerlinOwnedResultView *view, void *destination,
                             size_t capacity) {
  if (!view)
    return MERLIN_OWNED_RESULT_COPY;
  MerlinOwnedResultDescriptor d = {
      view->allocated, view->aligned, view->offset, view->rank,
      view->element_bytes, view->element_alignment,
      view->sizes, view->strides, view->sizes};
  MerlinOwnedResultView checked;
  int status = checked_view(&d, &checked, capacity);
  if (status)
    return status;
  if (checked.data != view->data || checked.count != view->count || checked.bytes != view->bytes)
    return MERLIN_OWNED_RESULT_COPY;
  if (!checked.bytes)
    return MERLIN_OWNED_RESULT_OK;
  uintptr_t target = (uintptr_t)destination, owner = (uintptr_t)view->allocated;
  size_t owned_bytes = merlin_owned_results_capacity(view->allocated);
  if (!target || checked.bytes - 1 > UINTPTR_MAX - target ||
      !(target > owner + owned_bytes - 1 || target + checked.bytes - 1 < owner))
    return MERLIN_OWNED_RESULT_COPY;
  for (size_t index = 0; index < checked.count; ++index) {
    uintptr_t address = (uintptr_t)checked.data;
    size_t remaining = index;
    for (size_t axis = checked.rank; axis-- > 0;) {
      size_t coordinate = remaining % (size_t)checked.sizes[axis];
      remaining /= (size_t)checked.sizes[axis];
      size_t delta = coordinate * magnitude(checked.strides[axis]) * checked.element_bytes;
      address = checked.strides[axis] < 0 ? address - delta : address + delta;
    }
    memcpy((unsigned char *)destination + index * checked.element_bytes,
           (const void *)address, checked.element_bytes);
  }
  return MERLIN_OWNED_RESULT_OK;
}

const char *merlin_owned_result_error(int status) {
  switch (status) {
  case MERLIN_OWNED_RESULT_OK: return "success";
  case MERLIN_OWNED_RESULT_TYPE: return "invalid result type or descriptor metadata";
  case MERLIN_OWNED_RESULT_EXTENT: return "negative or changed static result extent";
  case MERLIN_OWNED_RESULT_OWNER: return "result has no live invocation-owned allocation";
  case MERLIN_OWNED_RESULT_FOOTPRINT: return "result footprint escapes its allocation";
  case MERLIN_OWNED_RESULT_BUDGET: return "result exceeds the output byte budget";
  case MERLIN_OWNED_RESULT_UNRETURNED: return "unreturned invocation allocations remain";
  case MERLIN_OWNED_RESULT_COPY: return "invalid or overlapping result copy destination";
  default: return "unknown result validation status";
  }
}
