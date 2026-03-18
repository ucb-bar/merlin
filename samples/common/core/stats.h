/** @file stats.h
 *  @brief Log2-bucketed histogram and running statistics for latency tracking.
 */
#ifndef MERLIN_CORE_STATS_H_
#define MERLIN_CORE_STATS_H_

#include <climits>
#include <cstdint>

namespace merlin_bench {

/** @brief Fixed-size histogram with log2-spaced buckets.
 *
 *  Bucket @c b covers the half-open range [2^(b-1), 2^b) microseconds,
 *  with bucket 0 holding the count of zero-valued samples and bucket 63
 *  acting as an overflow bin.
 */
struct Log2Histogram {
	static constexpr int kBuckets = 64; /**< Number of histogram buckets. */
	uint64_t buckets[kBuckets]; /**< Per-bucket sample counts. */

	Log2Histogram() {
		Reset();
	}

	/** @brief Zero all bucket counts. */
	void Reset() {
		for (int i = 0; i < kBuckets; ++i)
			buckets[i] = 0;
	}

	/** @brief Compute the bucket index for a given microsecond value.
	 *  @param us  Latency in microseconds.
	 *  @return Bucket index in [0, kBuckets).
	 */
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

	/** @brief Record a single sample.
	 *  @param us  Latency in microseconds.
	 */
	void Add(uint64_t us) {
		buckets[BucketForUs(us)]++;
	}

	/** @brief Total number of samples recorded across all buckets. */
	uint64_t Count() const {
		uint64_t c = 0;
		for (int i = 0; i < kBuckets; ++i)
			c += buckets[i];
		return c;
	}

	/** @brief Approximate a percentile from the histogram.
	 *  @param pct  Percentile as a fraction in [0.0, 1.0].
	 *  @return Upper bound of the bucket that contains the target percentile.
	 */
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

/** @brief Accumulates min/max/avg/percentile statistics for a stream of
 *         microsecond latency samples.
 */
struct RunningStats {
	uint64_t count = 0; /**< Number of samples recorded. */
	uint64_t sum_us = 0; /**< Sum of all samples (microseconds). */
	uint64_t min_us = UINT64_MAX; /**< Minimum observed sample. */
	uint64_t max_us = 0; /**< Maximum observed sample. */
	Log2Histogram hist; /**< Backing histogram for percentiles. */

	/** @brief Reset all counters and the histogram. */
	void Reset() {
		count = 0;
		sum_us = 0;
		min_us = UINT64_MAX;
		max_us = 0;
		hist.Reset();
	}

	/** @brief Record a single latency sample.
	 *  @param us  Latency in microseconds.
	 */
	void Add(uint64_t us) {
		++count;
		sum_us += us;
		if (us < min_us)
			min_us = us;
		if (us > max_us)
			max_us = us;
		hist.Add(us);
	}

	/** @brief Average latency in milliseconds (0.0 if no samples). */
	double AvgMs() const {
		if (count == 0)
			return 0.0;
		return (static_cast<double>(sum_us) / 1000.0) /
			static_cast<double>(count);
	}

	/** @brief Minimum latency in milliseconds. */
	double MinMs() const {
		return count == 0 ? 0.0 : static_cast<double>(min_us) / 1000.0;
	}
	/** @brief Maximum latency in milliseconds. */
	double MaxMs() const {
		return count == 0 ? 0.0 : static_cast<double>(max_us) / 1000.0;
	}
	/** @brief Approximate 50th-percentile latency in milliseconds. */
	double P50Ms() const {
		return static_cast<double>(hist.ApproxPercentile(0.50)) / 1000.0;
	}
	/** @brief Approximate 90th-percentile latency in milliseconds. */
	double P90Ms() const {
		return static_cast<double>(hist.ApproxPercentile(0.90)) / 1000.0;
	}
	/** @brief Approximate 99th-percentile latency in milliseconds. */
	double P99Ms() const {
		return static_cast<double>(hist.ApproxPercentile(0.99)) / 1000.0;
	}
};

} // namespace merlin_bench

#endif // MERLIN_CORE_STATS_H_
