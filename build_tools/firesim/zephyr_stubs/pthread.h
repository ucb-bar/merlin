/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stub <pthread.h> for picolibc-on-Zephyr where the pthread API is not
 * available. IREE's iree/base/threading/{mutex,notification}.h include
 * <pthread.h> in the no-IREE_SYNCHRONIZATION_DISABLE_UNSAFE branch and
 * embed pthread_mutex_t / pthread_cond_t fields in their structs.
 *
 * Layout of pthread_mutex_t / pthread_cond_t below is opaque from IREE's
 * point of view (only used as struct members). The stub provides
 * forward-compatible-size opaque types (sized to glibc/musl values so
 * sizeof() comparisons in static_assert don't fire) plus declarations
 * for the pthread_* functions IREE references inline. The functions
 * themselves are never resolved at link time because the consuming
 * Zephyr application uses only the IREE local-sync HAL (never
 * instantiates iree_thread_*, iree_notification_t, or any
 * pthread-backed primitive); --gc-sections drops the unresolved
 * references.
 */

#ifndef MERLIN_ZEPHYR_STUB_PTHREAD_H_
#define MERLIN_ZEPHYR_STUB_PTHREAD_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Glibc-sized opaque types so any sizeof()-based assertions in IREE
 * pass identically across the two views (this consumer side and the
 * IREE archives, which were built with IREE_SYNCHRONIZATION_DISABLE_UNSAFE
 * and therefore do *not* have a pthread.h dependency at all). The struct
 * members are read but never written by inline code we link. */
typedef struct {
	long __padding[6];
} pthread_mutex_t;

typedef struct {
	long __padding[6];
} pthread_cond_t;

typedef struct {
	long __padding[7];
} pthread_t;

typedef struct {
	int __initialized;
} pthread_attr_t;

typedef struct {
	int __initialized;
} pthread_mutexattr_t;

typedef struct {
	int __initialized;
} pthread_condattr_t;

typedef struct {
	int __initialized;
} pthread_key_t;

typedef int pthread_once_t;

#define PTHREAD_MUTEX_INITIALIZER                                              \
	{ 0 }
#define PTHREAD_COND_INITIALIZER                                               \
	{ 0 }
#define PTHREAD_ONCE_INIT 0

/* Function declarations -- linker GCs unreferenced ones. */
int pthread_mutex_init(pthread_mutex_t *, const pthread_mutexattr_t *);
int pthread_mutex_destroy(pthread_mutex_t *);
int pthread_mutex_lock(pthread_mutex_t *);
int pthread_mutex_unlock(pthread_mutex_t *);
int pthread_mutex_trylock(pthread_mutex_t *);

int pthread_cond_init(pthread_cond_t *, const pthread_condattr_t *);
int pthread_cond_destroy(pthread_cond_t *);
int pthread_cond_signal(pthread_cond_t *);
int pthread_cond_broadcast(pthread_cond_t *);
int pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *);

int pthread_create(
	pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int pthread_join(pthread_t, void **);
int pthread_detach(pthread_t);

int pthread_key_create(pthread_key_t *, void (*)(void *));
int pthread_key_delete(pthread_key_t);
void *pthread_getspecific(pthread_key_t);
int pthread_setspecific(pthread_key_t, const void *);

int pthread_once(pthread_once_t *, void (*)(void));

#ifdef __cplusplus
}
#endif

#endif /* MERLIN_ZEPHYR_STUB_PTHREAD_H_ */
