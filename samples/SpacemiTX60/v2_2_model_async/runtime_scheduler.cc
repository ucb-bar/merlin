#include "runtime_scheduler.h"

#include <inttypes.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kDronetMaxInFlight = 8;
constexpr uint64_t kMlpMaxInFlight = 2;
constexpr int64_t kSubmitIdleSleepNs = 200000;  // 0.2ms

struct Invocation {
  uint64_t id = 0;

  // Keep alive until reaped:
  iree_hal_buffer_view_t* input_view = nullptr;
  iree_vm_list_t* outputs = nullptr;

  // Completion primitive:
  iree_hal_fence_t* signal_fence = nullptr;

  iree_host_size_t expected_outputs = 0;
  Clock::time_point submit_time;
};

struct QueueState {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Invocation> q;

  std::atomic<uint64_t> submitted{0};
  std::atomic<uint64_t> completed{0};

  std::atomic<uint64_t> done_count{0};
  std::atomic<uint64_t> total_latency_us{0};
};

struct SharedState {
  std::atomic<bool> stop{false};
  std::atomic<int> fatal_code{IREE_STATUS_OK};
};

static bool HasFatal(const SharedState* s) {
  return s->fatal_code.load(std::memory_order_relaxed) != IREE_STATUS_OK;
}

static void SetFatalOnce(SharedState* s, iree_status_t st, const char* tag,
                         QueueState* q0, QueueState* q1) {
  if (iree_status_is_ok(st)) return;
  const int code = (int)iree_status_code(st);
  int expected = IREE_STATUS_OK;
  if (s->fatal_code.compare_exchange_strong(expected, code,
                                            std::memory_order_relaxed)) {
    if (tag && tag[0]) fprintf(stderr, "%s\n", tag);
    iree_status_fprint(stderr, st);
  }
  iree_status_ignore(st);
  if (q0) q0->cv.notify_all();
  if (q1) q1->cv.notify_all();
}

static std::chrono::nanoseconds PeriodFromHz(double hz) {
  int64_t ns = (int64_t)(1000000000.0 / hz);
  if (ns < 1) ns = 1;
  return std::chrono::nanoseconds(ns);
}

static iree_status_t MakeF32InputViewFromHostData(
    iree_runtime_session_t* session,
    const std::vector<iree_hal_dim_t>& shape,
    const float* host_data,
    iree_host_size_t element_count,
    iree_hal_buffer_view_t** out_view) {
  *out_view = nullptr;
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* allocator = iree_runtime_session_device_allocator(session);

  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

  return iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator,
      (iree_host_size_t)shape.size(), shape.data(),
      IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      params,
      iree_make_const_byte_span(host_data, element_count * sizeof(float)),
      out_view);
}

static iree_status_t ValidateOutputs(iree_vm_list_t* outputs,
                                     iree_host_size_t expected) {
  const iree_host_size_t n = iree_vm_list_size(outputs);
  if (n != expected) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected %" PRIu64 " outputs, got %" PRIu64,
                            (uint64_t)expected, (uint64_t)n);
  }
  for (iree_host_size_t i = 0; i < n; ++i) {
    iree_hal_buffer_view_t* v = iree_vm_list_get_buffer_view_assign(outputs, i);
    if (!v) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "output %" PRIu64 " not a buffer_view",
                              (uint64_t)i);
    }
  }
  return iree_ok_status();
}

// Produces device-local input tensors periodically and signals an input timeline.
// Consumers can create wait_fences at (timeline, latest_epoch).
class InputProducer {
 public:
  InputProducer() = default;

  iree_status_t Initialize(iree_runtime_session_t* session,
                           const char* name,
                           std::vector<iree_hal_dim_t> shape,
                           double frequency_hz,
                           float fill_value) {
    session_ = session;
    name_ = name ? name : "input";
    shape_ = std::move(shape);
    frequency_hz_ = frequency_hz;
    fill_value_ = fill_value;

    if (!session_ || frequency_hz_ <= 0.0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid producer init args");
    }

    // Compute element count.
    element_count_ = 1;
    for (iree_hal_dim_t d : shape_) element_count_ *= (iree_host_size_t)d;

    host_.assign(element_count_, fill_value_);

    iree_hal_device_t* device = iree_runtime_session_device(session_);
    iree_status_t st = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0ull,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &timeline_);
    if (!iree_status_is_ok(st)) return st;

    // epoch 0 is "ready" by definition (timeline starts at 0).
    latest_epoch_.store(0, std::memory_order_relaxed);

    return iree_ok_status();
  }

  void Start(SharedState* shared, QueueState* q0, QueueState* q1) {
    stop_.store(false, std::memory_order_relaxed);
    thread_ = std::thread([this, shared, q0, q1]() { ThreadMain(shared, q0, q1); });
  }

  void Stop() {
    stop_.store(true, std::memory_order_relaxed);
    if (thread_.joinable()) thread_.join();

    {
      std::lock_guard<std::mutex> lock(mu_);
      iree_hal_buffer_view_release(latest_view_);
      latest_view_ = nullptr;
    }

    iree_hal_semaphore_release(timeline_);
    timeline_ = nullptr;
  }

  iree_hal_semaphore_t* timeline() const { return timeline_; }

  uint64_t latest_epoch() const {
    return latest_epoch_.load(std::memory_order_relaxed);
  }

  // Returns a retained reference to the latest view, or nullptr if none yet.
  iree_hal_buffer_view_t* AcquireLatestViewRetained() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!latest_view_) return nullptr;
    iree_hal_buffer_view_retain(latest_view_);  // returns void in your build
    return latest_view_;
  }
  uint64_t generated_count() const {
    return generated_count_.load(std::memory_order_relaxed);
  }

 private:
  void ThreadMain(SharedState* shared, QueueState* q0, QueueState* q1) {
    const auto period = PeriodFromHz(frequency_hz_);
    auto next_release = Clock::now();

    // Produce an initial sample as soon as possible.
    while (!stop_.load(std::memory_order_relaxed) && !HasFatal(shared)) {
      const auto now = Clock::now();
      if (now < next_release) {
        std::this_thread::sleep_until(next_release);
      } else {
        // If we're late, re-anchor to avoid drift.
        next_release = now;
      }

      // Generate host data (simple constant fill; you can replace with real data).
      // Small deterministic variation so you can confirm it changes if you want.
      const uint64_t idx = generated_count_.load(std::memory_order_relaxed);
      const float v = fill_value_ + (float)((idx % 7) * 0.0001f);
      for (iree_host_size_t i = 0; i < element_count_; ++i) host_[i] = v;

      iree_hal_buffer_view_t* new_view = nullptr;
      iree_status_t st = MakeF32InputViewFromHostData(
          session_, shape_, host_.data(), element_count_, &new_view);
      if (!iree_status_is_ok(st)) {
        SetFatalOnce(shared, st, "[producer] input allocate/copy failed", q0, q1);
        break;
      }

      const uint64_t new_epoch = latest_epoch_.load(std::memory_order_relaxed) + 1;

      // Publish latest view (producer holds one ref).
      {
        std::lock_guard<std::mutex> lock(mu_);
        iree_hal_buffer_view_release(latest_view_);
        latest_view_ = new_view;  // already owned
      }

      // Signal input timeline to new_epoch (via fence_signal).
      iree_hal_device_t* device = iree_runtime_session_device(session_);
      iree_allocator_t dev_host_alloc = iree_hal_device_host_allocator(device);

      iree_hal_fence_t* fence = nullptr;
      st = iree_hal_fence_create_at(timeline_, new_epoch, dev_host_alloc, &fence);
      if (!iree_status_is_ok(st)) {
        SetFatalOnce(shared, st, "[producer] fence_create_at failed", q0, q1);
        break;
      }
      st = iree_hal_fence_signal(fence);
      iree_hal_fence_release(fence);
      if (!iree_status_is_ok(st)) {
        SetFatalOnce(shared, st, "[producer] fence_signal failed", q0, q1);
        break;
      }

      latest_epoch_.store(new_epoch, std::memory_order_relaxed);
      generated_count_.fetch_add(1, std::memory_order_relaxed);

      next_release += period;
    }
  }

  iree_runtime_session_t* session_ = nullptr;
  const char* name_ = nullptr;

  std::vector<iree_hal_dim_t> shape_;
  double frequency_hz_ = 0.0;
  float fill_value_ = 0.0f;

  iree_host_size_t element_count_ = 0;
  std::vector<float> host_;

  std::atomic<bool> stop_{false};
  std::thread thread_;

  // Input-ready timeline:
  iree_hal_semaphore_t* timeline_ = nullptr;
  std::atomic<uint64_t> latest_epoch_{0};

  // Latest produced input:
  std::mutex mu_;
  iree_hal_buffer_view_t* latest_view_ = nullptr;  // retained by producer

  std::atomic<uint64_t> generated_count_{0};
};

// Submits one async-external coarse-fences call.
// inputs = (input_view, wait_fence(sensor_timeline@epoch), signal_fence)
// signal_fence is backed by a fresh semaphore per invocation (robust).
static iree_status_t SubmitOne(
    iree_runtime_session_t* session,
    const char* function_name,
    iree_hal_semaphore_t* wait_timeline,
    uint64_t wait_epoch,
    iree_hal_buffer_view_t* input_view,
    iree_host_size_t expected_outputs,
    iree_allocator_t host_alloc,
    iree_hal_fence_t** out_signal_fence,
    iree_vm_list_t** out_outputs,
    double* out_call_ms) {
  *out_signal_fence = nullptr;
  *out_outputs = nullptr;
  if (out_call_ms) *out_call_ms = 0.0;

  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_allocator_t dev_host_alloc = iree_hal_device_host_allocator(device);

  iree_status_t st = iree_ok_status();
  iree_vm_list_t* inputs = nullptr;
  iree_vm_list_t* outputs = nullptr;

  iree_hal_fence_t* wait_fence = nullptr;

  iree_hal_semaphore_t* semaphore = nullptr;
  iree_hal_fence_t* signal_fence = nullptr;

  st = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                           /*capacity=*/3, host_alloc, &inputs);
  if (!iree_status_is_ok(st)) goto cleanup;

  st = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                           /*capacity=*/expected_outputs, host_alloc, &outputs);
  if (!iree_status_is_ok(st)) goto cleanup;

  // arg0: input_view
  {
    iree_vm_ref_t r = iree_hal_buffer_view_retain_ref(input_view);
    st = iree_vm_list_push_ref_move(inputs, &r);
    if (!iree_status_is_ok(st)) goto cleanup;
  }

  // arg1: wait fence at (wait_timeline, wait_epoch)
  st = iree_hal_fence_create_at(wait_timeline, wait_epoch, dev_host_alloc, &wait_fence);
  if (!iree_status_is_ok(st)) goto cleanup;
  {
    iree_vm_ref_t r = iree_hal_fence_retain_ref(wait_fence);
    st = iree_vm_list_push_ref_move(inputs, &r);
    if (!iree_status_is_ok(st)) goto cleanup;
  }

  // Fresh semaphore at 0, signal fence at 1.
  st = iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                 /*initial_value=*/0ull,
                                 IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                 &semaphore);
  if (!iree_status_is_ok(st)) goto cleanup;

  st = iree_hal_fence_create_at(semaphore, /*value=*/1ull, dev_host_alloc,
                                &signal_fence);
  if (!iree_status_is_ok(st)) goto cleanup;

  // Drop semaphore ref; fence retains what it needs.
  iree_hal_semaphore_release(semaphore);
  semaphore = nullptr;

  // arg2: signal fence
  {
    iree_vm_ref_t r = iree_hal_fence_retain_ref(signal_fence);
    st = iree_vm_list_push_ref_move(inputs, &r);
    if (!iree_status_is_ok(st)) goto cleanup;
  }

  // Call (should return after scheduling).
  {
    auto t0 = Clock::now();
    st = iree_runtime_session_call_by_name(
        session, iree_make_cstring_view(function_name), inputs, outputs);
    auto t1 = Clock::now();
    if (out_call_ms) {
      *out_call_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
  }
  if (!iree_status_is_ok(st)) goto cleanup;

  *out_signal_fence = signal_fence;
  signal_fence = nullptr;
  *out_outputs = outputs;
  outputs = nullptr;

cleanup:
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_fence_release(wait_fence);
  iree_hal_fence_release(signal_fence);
  iree_hal_semaphore_release(semaphore);
  return st;
}

static void DrainQueue(QueueState* q) {
  std::deque<Invocation> tmp;
  {
    std::lock_guard<std::mutex> lock(q->mu);
    tmp.swap(q->q);
  }
  while (!tmp.empty()) {
    Invocation inv = std::move(tmp.front());
    tmp.pop_front();
    iree_vm_list_release(inv.outputs);
    iree_hal_fence_release(inv.signal_fence);
    iree_hal_buffer_view_release(inv.input_view);
  }
}

static void ReaperThread(SharedState* s, QueueState* q, const char* tag) {
  while (true) {
    Invocation inv;

    {
      std::unique_lock<std::mutex> lock(q->mu);
      q->cv.wait(lock, [&]() {
        return s->stop.load(std::memory_order_relaxed) || HasFatal(s) ||
               !q->q.empty();
      });

      if ((s->stop.load(std::memory_order_relaxed) || HasFatal(s)) &&
          q->q.empty()) {
        break;
      }

      inv = std::move(q->q.front());
      q->q.pop_front();
    }

    // Wait for completion (robust path).
    iree_status_t st = iree_hal_fence_wait(
        inv.signal_fence, iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT);
    if (!iree_status_is_ok(st)) {
      SetFatalOnce(s, st, tag, q, q);
      iree_vm_list_release(inv.outputs);
      iree_hal_fence_release(inv.signal_fence);
      iree_hal_buffer_view_release(inv.input_view);
      break;
    }

    st = ValidateOutputs(inv.outputs, inv.expected_outputs);

    iree_vm_list_release(inv.outputs);
    iree_hal_fence_release(inv.signal_fence);
    iree_hal_buffer_view_release(inv.input_view);

    if (!iree_status_is_ok(st)) {
      SetFatalOnce(s, st, "[reaper] invalid outputs", q, q);
      break;
    }

    const uint64_t lat_us =
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - inv.submit_time)
            .count();

    q->completed.fetch_add(1, std::memory_order_relaxed);
    q->done_count.fetch_add(1, std::memory_order_relaxed);
    q->total_latency_us.fetch_add(lat_us, std::memory_order_relaxed);
  }
}

}  // namespace

extern "C" int dual_model_async_scheduler_run(const dual_model_async_config_t* c) {
  if (!c || !c->dronet_vmfb_path || !c->mlp_vmfb_path ||
      !c->dronet_function || !c->mlp_function || !c->driver_name ||
      c->mlp_frequency_hz <= 0.0 || c->report_frequency_hz <= 0.0 ||
      c->dronet_sensor_frequency_hz <= 0.0 || c->mlp_sensor_frequency_hz <= 0.0) {
    fprintf(stderr, "Invalid config\n");
    return 1;
  }

  iree_allocator_t host_alloc = iree_allocator_system();
  iree_status_t st = iree_ok_status();

  iree_runtime_instance_t* instance = nullptr;
  iree_hal_device_t* device = nullptr;
  iree_runtime_session_t* session = nullptr;

  SharedState shared;
  QueueState dronet_q;
  QueueState mlp_q;

  std::thread dronet_reaper;
  std::thread mlp_reaper;

  // Producer-driven (input-ready) timelines.
  InputProducer dronet_input;
  InputProducer mlp_input;

  // Shapes (your current known-good shapes).
  const std::vector<iree_hal_dim_t> dronet_shape = {1, 3, 112, 112};
  const std::vector<iree_hal_dim_t> mlp_shape = {1, 10};

  // MLP scheduling misses.
  std::atomic<uint64_t> mlp_misses{0};

  do {
    // Instance/device/session.
    iree_runtime_instance_options_t instance_opts;
    iree_runtime_instance_options_initialize(&instance_opts);
    iree_runtime_instance_options_use_all_available_drivers(&instance_opts);
    st = iree_runtime_instance_create(&instance_opts, host_alloc, &instance);
    if (!iree_status_is_ok(st)) break;

    st = iree_runtime_instance_try_create_default_device(
        instance, iree_make_cstring_view(c->driver_name), &device);
    if (!iree_status_is_ok(st)) break;

    iree_runtime_session_options_t session_opts;
    iree_runtime_session_options_initialize(&session_opts);
    st = iree_runtime_session_create_with_device(
        instance, &session_opts, device,
        iree_runtime_instance_host_allocator(instance), &session);
    if (!iree_status_is_ok(st)) break;

    st = iree_runtime_session_append_bytecode_module_from_file(
        session, c->dronet_vmfb_path);
    if (!iree_status_is_ok(st)) break;

    st = iree_runtime_session_append_bytecode_module_from_file(
        session, c->mlp_vmfb_path);
    if (!iree_status_is_ok(st)) break;

    // Init input producers (these create their own timelines).
    st = dronet_input.Initialize(session, "dronet_input", dronet_shape,
                                 c->dronet_sensor_frequency_hz,
                                 /*fill_value=*/0.01f);
    if (!iree_status_is_ok(st)) break;

    st = mlp_input.Initialize(session, "mlp_input", mlp_shape,
                              c->mlp_sensor_frequency_hz,
                              /*fill_value=*/0.25f);
    if (!iree_status_is_ok(st)) break;

    dronet_input.Start(&shared, &dronet_q, &mlp_q);
    mlp_input.Start(&shared, &dronet_q, &mlp_q);

    // Start reapers.
    dronet_reaper = std::thread(ReaperThread, &shared, &dronet_q,
                                "[dronet/reaper] fence wait failed");
    mlp_reaper    = std::thread(ReaperThread, &shared, &mlp_q,
                                "[mlp/reaper] fence wait failed");

    const auto start = Clock::now();
    const auto mlp_period = PeriodFromHz(c->mlp_frequency_hz);
    const auto report_period = PeriodFromHz(c->report_frequency_hz);

    auto next_mlp = start;
    auto next_report = start + report_period;

    // For input-driven dronet: only submit when input epoch advances.
    uint64_t last_dronet_input_epoch_submitted = 0;

    while (!HasFatal(&shared)) {
      const auto now = Clock::now();

      if (c->run_duration_ms > 0) {
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed_ms >= c->run_duration_ms) break;
      }

      bool did_submit = false;

      // Track missed MLP releases if we fell behind.
      while (now >= next_mlp + mlp_period) {
        mlp_misses.fetch_add(1, std::memory_order_relaxed);
        next_mlp += mlp_period;
      }

      // MLP periodic, bounded inflight, uses latest available input epoch.
      if (now >= next_mlp) {
        const uint64_t inflight =
            mlp_q.submitted.load(std::memory_order_relaxed) -
            mlp_q.completed.load(std::memory_order_relaxed);

        if (inflight < kMlpMaxInFlight) {
          const uint64_t input_epoch = mlp_input.latest_epoch();
          if (input_epoch == 0) {
            // No produced sample yet (very early startup).
            mlp_misses.fetch_add(1, std::memory_order_relaxed);
          } else {
            iree_hal_buffer_view_t* in_view = mlp_input.AcquireLatestViewRetained();
            if (!in_view) {
              // Shouldn't happen if epoch>0, but handle gracefully.
              mlp_misses.fetch_add(1, std::memory_order_relaxed);
            } else {
              iree_vm_list_t* outputs = nullptr;
              iree_hal_fence_t* signal_fence = nullptr;
              double call_ms = 0.0;

              st = SubmitOne(session, c->mlp_function,
                             /*wait_timeline=*/mlp_input.timeline(),
                             /*wait_epoch=*/input_epoch,
                             in_view, /*expected_outputs=*/1,
                             host_alloc, &signal_fence, &outputs, &call_ms);
              if (!iree_status_is_ok(st)) {
                iree_hal_buffer_view_release(in_view);
                SetFatalOnce(&shared, st, "[mlp] submit failed", &dronet_q, &mlp_q);
                break;
              }

              const uint64_t id = mlp_q.submitted.fetch_add(1, std::memory_order_relaxed) + 1;
              {
                std::lock_guard<std::mutex> lock(mlp_q.mu);
                mlp_q.q.push_back(
                    {id, in_view, outputs, signal_fence, 1, Clock::now()});
              }
              mlp_q.cv.notify_one();
              did_submit = true;
            }
          }
        } else {
          // Scheduler tick happened but we were saturated.
          mlp_misses.fetch_add(1, std::memory_order_relaxed);
        }

        do { next_mlp += mlp_period; } while (next_mlp <= now);
      }

      // DRONET: "as fast as possible" but **input-driven**:
      // only submit when a new input epoch is available and inflight allows.
      while (!HasFatal(&shared)) {
        const uint64_t inflight =
            dronet_q.submitted.load(std::memory_order_relaxed) -
            dronet_q.completed.load(std::memory_order_relaxed);
        if (inflight >= kDronetMaxInFlight) break;

        const uint64_t input_epoch = dronet_input.latest_epoch();
        if (input_epoch == 0 || input_epoch == last_dronet_input_epoch_submitted) {
          // No new input yet; don't spam identical work.
          break;
        }

        iree_hal_buffer_view_t* in_view = dronet_input.AcquireLatestViewRetained();
        if (!in_view) break;

        iree_vm_list_t* outputs = nullptr;
        iree_hal_fence_t* signal_fence = nullptr;
        double call_ms = 0.0;

        st = SubmitOne(session, c->dronet_function,
                       /*wait_timeline=*/dronet_input.timeline(),
                       /*wait_epoch=*/input_epoch,
                       in_view, /*expected_outputs=*/2,
                       host_alloc, &signal_fence, &outputs, &call_ms);
        if (!iree_status_is_ok(st)) {
          iree_hal_buffer_view_release(in_view);
          SetFatalOnce(&shared, st, "[dronet] submit failed", &dronet_q, &mlp_q);
          break;
        }

        const uint64_t id = dronet_q.submitted.fetch_add(1, std::memory_order_relaxed) + 1;
        {
          std::lock_guard<std::mutex> lock(dronet_q.mu);
          dronet_q.q.push_back(
              {id, in_view, outputs, signal_fence, 2, Clock::now()});
        }
        dronet_q.cv.notify_one();

        last_dronet_input_epoch_submitted = input_epoch;
        did_submit = true;
      }

      // Report.
      if (now >= next_report) {
        const uint64_t d_done = dronet_q.done_count.load(std::memory_order_relaxed);
        const uint64_t m_done = mlp_q.done_count.load(std::memory_order_relaxed);
        const uint64_t d_inflight =
            dronet_q.submitted.load(std::memory_order_relaxed) -
            dronet_q.completed.load(std::memory_order_relaxed);
        const uint64_t m_inflight =
            mlp_q.submitted.load(std::memory_order_relaxed) -
            mlp_q.completed.load(std::memory_order_relaxed);
        const uint64_t misses = mlp_misses.load(std::memory_order_relaxed);

        const uint64_t d_gen = dronet_input.generated_count();
        const uint64_t m_gen = mlp_input.generated_count();

        fprintf(stdout,
                "[stats] dronet_total=%" PRIu64 " mlp_total=%" PRIu64
                " mlp_misses=%" PRIu64
                " dronet_inflight=%" PRIu64 " mlp_inflight=%" PRIu64
                " dronet_input_gen=%" PRIu64 " mlp_input_gen=%" PRIu64 "\n",
                d_done, m_done, misses, d_inflight, m_inflight, d_gen, m_gen);
        fflush(stdout);

        do { next_report += report_period; } while (next_report <= now);
      }

      if (!did_submit) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(kSubmitIdleSleepNs));
      }
    }

  } while (false);

  // Stop everything.
  shared.stop.store(true, std::memory_order_relaxed);
  dronet_q.cv.notify_all();
  mlp_q.cv.notify_all();

  if (dronet_reaper.joinable()) dronet_reaper.join();
  if (mlp_reaper.joinable()) mlp_reaper.join();

  // Drain any leftover queued invocations.
  DrainQueue(&dronet_q);
  DrainQueue(&mlp_q);

  // Stop producers (releases their resources).
  dronet_input.Stop();
  mlp_input.Stop();

  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);

  if (!iree_status_is_ok(st)) {
    iree_status_fprint(stderr, st);
    iree_status_ignore(st);
    return 1;
  }
  return HasFatal(&shared) ? 1 : 0;
}