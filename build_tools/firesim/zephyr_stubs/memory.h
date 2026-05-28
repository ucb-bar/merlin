/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stub <memory.h> for libc implementations (e.g. picolibc) that omit the
 * deprecated newlib/glibc alias header. <memory.h> is just a backwards-
 * compatibility wrapper for <string.h> -- forwarding is sufficient.
 *
 * IREE's runtime/src/iree/base/allocator.h includes <memory.h>
 * unconditionally; this overlay keeps that working under picolibc on
 * Zephyr without modifying iree_bar.
 */

#ifndef MERLIN_ZEPHYR_STUB_MEMORY_H_
#define MERLIN_ZEPHYR_STUB_MEMORY_H_

#include <string.h>

#endif /* MERLIN_ZEPHYR_STUB_MEMORY_H_ */
