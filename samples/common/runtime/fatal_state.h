/** @file fatal_state.h
 *  @brief Atomic fatal-error tracking for multi-threaded IREE workloads.
 */

#ifndef MERLIN_RUNTIME_FATAL_STATE_H_
#define MERLIN_RUNTIME_FATAL_STATE_H_

#include <atomic>
#include <cstdio>

#include "iree/base/api.h"

namespace merlin_bench {

/** @brief Shared state holding an atomic fatal error code.
 *
 *  Multiple threads can race to report the first fatal error via SetFatalOnce;
 *  only the first non-OK status is recorded.
 */
struct SharedState {
	std::atomic<int> fatal_code{IREE_STATUS_OK};
};

/** @brief Check whether a fatal error has been recorded.
 *  @param s Shared state to inspect.
 *  @return true if a non-OK status was stored.
 */
inline bool HasFatal(const SharedState *s) {
	return s->fatal_code.load(std::memory_order_relaxed) != IREE_STATUS_OK;
}

/** @brief Atomically record the first fatal error.
 *
 *  If @p st is OK the call is a no-op.  Otherwise the status code is stored
 *  (compare-exchange) and the error is printed to stderr exactly once.
 *  The status is always consumed (ignored) regardless of who wins the race.
 *
 *  @param s   Shared state to update.
 *  @param st  IREE status to check; consumed on return.
 *  @param tag Optional label printed before the error message (may be NULL).
 */
inline void SetFatalOnce(SharedState *s, iree_status_t st, const char *tag) {
	if (iree_status_is_ok(st))
		return;
	const int code = static_cast<int>(iree_status_code(st));
	int expected = IREE_STATUS_OK;
	if (s->fatal_code.compare_exchange_strong(
			expected, code, std::memory_order_relaxed)) {
		if (tag && tag[0])
			fprintf(stderr, "%s\n", tag);
		iree_status_fprint(stderr, st);
	}
	iree_status_ignore(st);
}

} // namespace merlin_bench

#endif // MERLIN_RUNTIME_FATAL_STATE_H_
