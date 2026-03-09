#ifndef MERLIN_SAMPLES_BASELINE_DUAL_MODEL_ASYNC_SENSOR_GENERATOR_H_
#define MERLIN_SAMPLES_BASELINE_DUAL_MODEL_ASYNC_SENSOR_GENERATOR_H_

#include <stdint.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "iree/runtime/api.h"

// A simple periodic synthetic tensor source used to model sensor input streams.
class PeriodicTensorSensor {
  public:
	PeriodicTensorSensor(const std::string &name,
		const std::vector<iree_hal_dim_t> &shape, double frequency_hz,
		float base_value, float amplitude);
	~PeriodicTensorSensor();

	PeriodicTensorSensor(const PeriodicTensorSensor &) = delete;
	PeriodicTensorSensor &operator=(const PeriodicTensorSensor &) = delete;

	void Start();
	void Stop();

	const std::string &name() const {
		return name_;
	}
	const std::vector<iree_hal_dim_t> &shape() const {
		return shape_;
	}
	iree_host_size_t element_count() const {
		return element_count_;
	}
	double frequency_hz() const {
		return frequency_hz_;
	}

	// Copies the latest generated tensor into |out_data| and returns the latest
	// sequence number.
	uint64_t Snapshot(
		std::vector<float> *out_data, int64_t *out_generated_time_ns) const;

	uint64_t generated_count() const {
		return generated_count_.load(std::memory_order_relaxed);
	}

  private:
	void ThreadMain();
	void GenerateSample(uint64_t sample_index, std::vector<float> *out) const;
	static int64_t NowSteadyClockNs();

	std::string name_;
	std::vector<iree_hal_dim_t> shape_;
	iree_host_size_t element_count_ = 0;
	double frequency_hz_ = 0.0;
	float base_value_ = 0.0f;
	float amplitude_ = 1.0f;

	mutable std::mutex data_mutex_;
	std::vector<float> latest_data_;
	uint64_t latest_sequence_ = 0;
	int64_t latest_generated_time_ns_ = 0;

	std::atomic<bool> stop_requested_{false};
	std::atomic<bool> started_{false};
	std::atomic<uint64_t> generated_count_{0};
	std::thread worker_thread_;
};

#endif // MERLIN_SAMPLES_BASELINE_DUAL_MODEL_ASYNC_SENSOR_GENERATOR_H_
