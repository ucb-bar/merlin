#ifndef MERLIN_IREE_BENCH_FATAL_STATE_H_
#define MERLIN_IREE_BENCH_FATAL_STATE_H_

#include <atomic>
#include <cstdio>

#include "iree/base/api.h"

namespace merlin_bench {

struct SharedState {
	std::atomic<int> fatal_code{IREE_STATUS_OK};
};

inline bool HasFatal(const SharedState *s) {
	return s->fatal_code.load(std::memory_order_relaxed) != IREE_STATUS_OK;
}

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

#endif // MERLIN_IREE_BENCH_FATAL_STATE_H_
