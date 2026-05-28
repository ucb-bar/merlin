/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stub <alloca.h> for libc implementations (e.g. picolibc) that omit it.
 * GCC provides alloca() as a builtin (`__builtin_alloca`); this header
 * just exposes the conventional name.
 */

#ifndef MERLIN_ZEPHYR_STUB_ALLOCA_H_
#define MERLIN_ZEPHYR_STUB_ALLOCA_H_

#include <stddef.h>

#define alloca(size) __builtin_alloca(size)

#endif /* MERLIN_ZEPHYR_STUB_ALLOCA_H_ */
