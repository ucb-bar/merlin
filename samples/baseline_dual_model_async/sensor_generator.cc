#include "sensor_generator.h"

#include <cmath>
#include <cstdint>

namespace {
constexpr float kPi = 3.14159265358979323846f;
}  // namespace

PeriodicTensorSensor::PeriodicTensorSensor(
    const std::string& name, const std::vector<iree_hal_dim_t>& shape,
    double frequency_hz, float base_value, float amplitude)
    : name_(name),
      shape_(shape),
      frequency_hz_(frequency_hz),
      base_value_(base_value),
      amplitude_(amplitude) {
  element_count_ = 1;
  for (iree_hal_dim_t dim : shape_) {
    if (dim < 1) dim = 1;
    element_count_ *= (iree_host_size_t)dim;
  }
  latest_data_.assign(element_count_, base_value_);
}

PeriodicTensorSensor::~PeriodicTensorSensor() { Stop(); }

void PeriodicTensorSensor::Start() {
  bool expected = false;
  if (!started_.compare_exchange_strong(expected, true)) {
    return;
  }
  stop_requested_.store(false, std::memory_order_relaxed);
  worker_thread_ = std::thread(&PeriodicTensorSensor::ThreadMain, this);
}

void PeriodicTensorSensor::Stop() {
  stop_requested_.store(true, std::memory_order_relaxed);
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

uint64_t PeriodicTensorSensor::Snapshot(std::vector<float>* out_data,
                                        int64_t* out_generated_time_ns) const {
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (out_data) {
    *out_data = latest_data_;
  }
  if (out_generated_time_ns) {
    *out_generated_time_ns = latest_generated_time_ns_;
  }
  return latest_sequence_;
}

void PeriodicTensorSensor::ThreadMain() {
  int64_t period_ns = (int64_t)(1000000000.0 / frequency_hz_);
  if (period_ns < 1) period_ns = 1;
  auto next_release = std::chrono::steady_clock::now();

  uint64_t sample_index = 0;
  std::vector<float> sample(element_count_);
  while (!stop_requested_.load(std::memory_order_relaxed)) {
    GenerateSample(sample_index, &sample);
    const int64_t generated_at = NowSteadyClockNs();

    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      latest_data_.swap(sample);
      latest_sequence_ = sample_index;
      latest_generated_time_ns_ = generated_at;
    }

    generated_count_.fetch_add(1, std::memory_order_relaxed);
    ++sample_index;

    next_release += std::chrono::nanoseconds(period_ns);
    const auto now = std::chrono::steady_clock::now();
    if (next_release > now) {
      std::this_thread::sleep_until(next_release);
    } else {
      // If generation is late, continue immediately and re-anchor the next
      // release to avoid unbounded drift.
      next_release = now;
    }
  }
}

void PeriodicTensorSensor::GenerateSample(uint64_t sample_index,
                                          std::vector<float>* out) const {
  // Deterministic, low-overhead synthetic signal:
  // - slow sinusoid over sample index (time)
  // - per-element phase offset
  const float t = (float)sample_index;
  for (iree_host_size_t i = 0; i < element_count_; ++i) {
    const float phase = (float)(i % 257) * 0.03125f;
    const float value = base_value_ +
                        amplitude_ * std::sin((2.0f * kPi * 0.01f * t) + phase);
    (*out)[i] = value;
  }
}

int64_t PeriodicTensorSensor::NowSteadyClockNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
