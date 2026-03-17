#ifndef MERLIN_IREE_BENCH_STATS_H_
#define MERLIN_IREE_BENCH_STATS_H_

#include <climits>
#include <cstdint>

namespace merlin_bench {

struct Log2Histogram {
	static constexpr int kBuckets = 64;
	uint64_t buckets[kBuckets];

	Log2Histogram() {
		Reset();
	}

	void Reset() {
		for (int i = 0; i < kBuckets; ++i)
			buckets[i] = 0;
	}

	static int BucketForUs(uint64_t us) {
		if (us == 0)
			return 0;
		int b = 0;
		while (us) {
			++b;
			us >>= 1;
			if (b >= kBuckets - 1)
				return kBuckets - 1;
		}
		return b;
	}

	void Add(uint64_t us) {
		buckets[BucketForUs(us)]++;
	}

	uint64_t Count() const {
		uint64_t c = 0;
		for (int i = 0; i < kBuckets; ++i)
			c += buckets[i];
		return c;
	}

	uint64_t ApproxPercentile(double pct) const {
		if (pct <= 0.0)
			return 0;
		if (pct >= 1.0)
			pct = 1.0;
		const uint64_t total = Count();
		if (total == 0)
			return 0;
		const uint64_t target =
			static_cast<uint64_t>(static_cast<double>(total) * pct);
		uint64_t run = 0;
		for (int b = 0; b < kBuckets; ++b) {
			run += buckets[b];
			if (run >= target) {
				if (b == 0)
					return 0;
				if (b >= kBuckets - 1)
					return static_cast<uint64_t>(-1);
				return (1ull << b);
			}
		}
		return (1ull << 63);
	}
};

struct RunningStats {
	uint64_t count = 0;
	uint64_t sum_us = 0;
	uint64_t min_us = UINT64_MAX;
	uint64_t max_us = 0;
	Log2Histogram hist;

	void Reset() {
		count = 0;
		sum_us = 0;
		min_us = UINT64_MAX;
		max_us = 0;
		hist.Reset();
	}

	void Add(uint64_t us) {
		++count;
		sum_us += us;
		if (us < min_us)
			min_us = us;
		if (us > max_us)
			max_us = us;
		hist.Add(us);
	}

	double AvgMs() const {
		if (count == 0)
			return 0.0;
		return (static_cast<double>(sum_us) / 1000.0) /
			static_cast<double>(count);
	}

	double MinMs() const {
		return count == 0 ? 0.0 : static_cast<double>(min_us) / 1000.0;
	}
	double MaxMs() const {
		return count == 0 ? 0.0 : static_cast<double>(max_us) / 1000.0;
	}
	double P50Ms() const {
		return static_cast<double>(hist.ApproxPercentile(0.50)) / 1000.0;
	}
	double P90Ms() const {
		return static_cast<double>(hist.ApproxPercentile(0.90)) / 1000.0;
	}
	double P99Ms() const {
		return static_cast<double>(hist.ApproxPercentile(0.99)) / 1000.0;
	}
};

} // namespace merlin_bench

#endif // MERLIN_IREE_BENCH_STATS_H_
