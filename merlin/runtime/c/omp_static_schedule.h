/* The OpenMP static worksharing split — pure integer math, no Zephyr, no OS.
 *
 * Split out of libomp_zephyr.c on purpose: this is the one piece of the shim that silently
 * produces WRONG NUMBERS (rather than a crash) if it is off by one — two harts claiming the
 * same rows, or a row claimed by nobody, both look like a successful run. Keeping it free of
 * kernel headers lets merlin/tests/runtime/test_omp_static_schedule.py compile it with the
 * host compiler and exhaustively check the partition invariants (disjoint, complete, exactly
 * one lastiter) across thread counts and trip counts, with no board or simulator involved.
 */
#ifndef MERLIN_OMP_STATIC_SCHEDULE_H
#define MERLIN_OMP_STATIC_SCHEDULE_H

#include <stdint.h>

/* kmp schedule types merlin's lowering emits. 34 (static, even) is what convert-scf-to-openmp
 * produces; 33 (static, chunked) is accepted and handled as an even split because merlin
 * never emits a chunk size. */
#define MERLIN_KMP_SCH_STATIC_CHUNKED 33
#define MERLIN_KMP_SCH_STATIC 34

/* Compute thread `tid`'s contiguous sub-range of the loop [*plower, *pupper] stepping `incr`.
 *
 * Mirrors libomp's STATIC_EVEN: the trip count is split into `nth` blocks whose sizes differ
 * by at most one, with the larger blocks going to the lowest thread ids. Writes the thread's
 * bounds back through plower/pupper, sets *plastiter iff this thread owns the final
 * iteration, and *pstride to the whole-loop span (what libomp reports for a static schedule).
 *
 * An EMPTY assignment is encoded the way OpenMP expects — *pupper = *plower - incr — so the
 * generated `for (i = lower; i <= upper; i += incr)` executes zero times. Returns the number
 * of iterations assigned to this thread (0 when empty), which the tests assert on directly.
 */
static inline int64_t merlin_omp_static_split(int32_t tid, int32_t nth,
					      int64_t *plower, int64_t *pupper,
					      int64_t *pstride, int64_t incr,
					      int32_t *plastiter)
{
	int64_t lo = *plower, up = *pupper;
	int64_t trip, q, r, start, count;

	if (incr == 0) {
		incr = 1;
	}
	if (nth <= 1 || tid < 0 || tid >= nth) {
		/* Serial (or an out-of-range id): keep the whole range. */
		*pstride = up - lo + incr;
		*plastiter = 1;
		return (up - lo) / incr + 1 > 0 ? (up - lo) / incr + 1 : 0;
	}

	trip = (up - lo) / incr + 1;
	if (trip <= 0) {
		*pupper = lo - incr;
		*plastiter = 0;
		*pstride = 0;
		return 0;
	}

	q = trip / nth;
	r = trip % nth;
	if ((int64_t)tid < r) {
		start = (int64_t)tid * (q + 1);
		count = q + 1;
	} else {
		start = r * (q + 1) + ((int64_t)tid - r) * q;
		count = q;
	}

	*pstride = trip * incr;
	*plower = lo + start * incr;
	if (count <= 0) {
		*pupper = *plower - incr;   /* zero iterations for this thread */
		*plastiter = 0;
		return 0;
	}
	*pupper = *plower + (count - 1) * incr;
	*plastiter = (start + count == trip);
	return count;
}

#endif /* MERLIN_OMP_STATIC_SCHEDULE_H */
