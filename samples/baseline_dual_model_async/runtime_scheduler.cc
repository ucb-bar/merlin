#include "runtime_scheduler.h"

#include <inttypes.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/runtime/api.h"
#include "sensor_generator.h"

namespace {

using Clock = std::chrono::steady_clock;

struct SharedRuntimeState {
  iree_runtime_session_t* session = nullptr;

  std::mutex session_mutex;
  std::atomic<bool> stop_requested{false};

  std::mutex status_mutex;
  iree_status_t fatal_status = iree_ok_status();

  std::atomic<uint64_t> dronet_invocations{0};
  std::atomic<uint64_t> mlp_invocations{0};
  std::atomic<uint64_t> mlp_deadline_misses{0};
  std::atomic<uint64_t> dronet_fresh_inputs{0};
  std::atomic<uint64_t> mlp_fresh_inputs{0};
  std::atomic<uint64_t> dronet_total_latency_us{0};
  std::atomic<uint64_t> mlp_total_latency_us{0};
};

struct WorkerParams {
  const char* worker_name = nullptr;
  const char* function_name = nullptr;
  iree_host_size_t expected_outputs = 0;
  iree_hal_device_t* device = nullptr;
  iree_hal_allocator_t* device_allocator = nullptr;
  const PeriodicTensorSensor* sensor = nullptr;

  // 0.0 means run as fast as possible.
  double target_frequency_hz = 0.0;

  std::atomic<uint64_t>* invocation_counter = nullptr;
  std::atomic<uint64_t>* total_latency_us_counter = nullptr;
  std::atomic<uint64_t>* deadline_miss_counter = nullptr;
  std::atomic<uint64_t>* fresh_input_counter = nullptr;
};

static void StoreFatalStatusIfFirst(SharedRuntimeState* state,
                                    iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  {
    std::lock_guard<std::mutex> lock(state->status_mutex);
    if (iree_status_is_ok(state->fatal_status)) {
      state->fatal_status = status;
    } else {
      iree_status_ignore(status);
    }
  }
  state->stop_requested.store(true, std::memory_order_relaxed);
}

static bool HasFatalStatus(SharedRuntimeState* state) {
  std::lock_guard<std::mutex> lock(state->status_mutex);
  return !iree_status_is_ok(state->fatal_status);
}

static iree_status_t ConsumeFatalStatus(SharedRuntimeState* state) {
  std::lock_guard<std::mutex> lock(state->status_mutex);
  iree_status_t status = state->fatal_status;
  state->fatal_status = iree_ok_status();
  return status;
}

static iree_status_t CreateF32InputViewFromData(
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    const float* input_data, iree_host_size_t element_count,
    iree_hal_buffer_view_t** out_view) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(shape);
  IREE_ASSERT_ARGUMENT(input_data);
  IREE_ASSERT_ARGUMENT(out_view);
  *out_view = nullptr;

  iree_hal_buffer_params_t buffer_params;
  memset(&buffer_params, 0, sizeof(buffer_params));
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

  return iree_hal_buffer_view_allocate_buffer_copy(
      device, device_allocator, shape_rank, shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      buffer_params,
      iree_make_const_byte_span(input_data, element_count * sizeof(float)),
      out_view);
}

static iree_status_t DrainOutputs(iree_runtime_call_t* call,
                                  iree_host_size_t expected_outputs) {
  iree_vm_list_t* outputs = iree_runtime_call_outputs(call);
  iree_host_size_t output_count = iree_vm_list_size(outputs);
  if (output_count != expected_outputs) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected %" PRIu64
                            " outputs from invocation but got %" PRIu64,
                            (uint64_t)expected_outputs, (uint64_t)output_count);
  }

  for (iree_host_size_t i = 0; i < output_count; ++i) {
    iree_hal_buffer_view_t* output_view = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_runtime_call_outputs_pop_front_buffer_view(call, &output_view));
    iree_hal_buffer_view_release(output_view);
  }
  return iree_ok_status();
}

static iree_status_t InvokeModelOnce(SharedRuntimeState* state,
                                     iree_runtime_call_t* call,
                                     iree_hal_buffer_view_t* input_view,
                                     iree_host_size_t expected_outputs,
                                     uint64_t* out_latency_us) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(call);
  IREE_ASSERT_ARGUMENT(input_view);

  iree_runtime_call_reset(call);
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(call, input_view));

  const auto t0 = Clock::now();
  {
    // Runtime sessions are thread-compatible, so invoke must be externally
    // synchronized when used from multiple threads.
    std::lock_guard<std::mutex> lock(state->session_mutex);
    IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(call, /*flags=*/0));
  }
  const auto t1 = Clock::now();

  IREE_RETURN_IF_ERROR(DrainOutputs(call, expected_outputs));

  if (out_latency_us) {
    *out_latency_us =
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
            .count();
  }
  return iree_ok_status();
}

static void WorkerMain(SharedRuntimeState* state, WorkerParams params) {
  if (!params.device || !params.device_allocator || !params.sensor) {
    StoreFatalStatusIfFirst(
        state, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "worker '%s' missing device/allocator/sensor",
                                params.worker_name ? params.worker_name
                                                   : "<unknown>"));
    return;
  }

  iree_runtime_call_t call;
  iree_status_t status = iree_ok_status();
  {
    std::lock_guard<std::mutex> lock(state->session_mutex);
    status = iree_runtime_call_initialize_by_name(
        state->session, iree_make_cstring_view(params.function_name), &call);
  }
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "[%s] failed to initialize call: %s\n", params.worker_name,
            params.function_name);
    StoreFatalStatusIfFirst(state, status);
    return;
  }

  const bool is_periodic = params.target_frequency_hz > 0.0;
  std::chrono::nanoseconds period_ns(0);
  if (is_periodic) {
    int64_t nanos = (int64_t)(1000000000.0 / params.target_frequency_hz);
    if (nanos < 1) nanos = 1;
    period_ns = std::chrono::nanoseconds(nanos);
  }

  std::vector<float> host_input;
  host_input.resize(params.sensor->element_count());
  uint64_t last_sequence = UINT64_MAX;

  auto next_release = Clock::now();
  while (!state->stop_requested.load(std::memory_order_relaxed)) {
    if (is_periodic) {
      const auto now = Clock::now();
      if (now < next_release) {
        std::this_thread::sleep_until(next_release);
      } else if (params.deadline_miss_counter && now > next_release) {
        const uint64_t missed =
            (uint64_t)((now - next_release) / period_ns);
        if (missed > 0) {
          params.deadline_miss_counter->fetch_add(missed,
                                                  std::memory_order_relaxed);
        }
      }
    }

    if (state->stop_requested.load(std::memory_order_relaxed)) break;

    uint64_t sample_sequence = params.sensor->Snapshot(&host_input, nullptr);
    if (sample_sequence != last_sequence) {
      last_sequence = sample_sequence;
      if (params.fresh_input_counter) {
        params.fresh_input_counter->fetch_add(1, std::memory_order_relaxed);
      }
    }

    iree_hal_buffer_view_t* input_view = nullptr;
    status = CreateF32InputViewFromData(
        params.device, params.device_allocator, params.sensor->shape().data(),
        params.sensor->shape().size(), host_input.data(), host_input.size(),
        &input_view);
    if (!iree_status_is_ok(status)) {
      fprintf(stderr, "[%s] failed to create input view from sensor '%s'\n",
              params.worker_name, params.sensor->name().c_str());
      StoreFatalStatusIfFirst(state, status);
      break;
    }

    uint64_t latency_us = 0;
    status = InvokeModelOnce(state, &call, input_view,
                             params.expected_outputs, &latency_us);
    iree_hal_buffer_view_release(input_view);
    if (!iree_status_is_ok(status)) {
      fprintf(stderr, "[%s] invocation failed: %s\n", params.worker_name,
              params.function_name);
      StoreFatalStatusIfFirst(state, status);
      break;
    }

    params.invocation_counter->fetch_add(1, std::memory_order_relaxed);
    params.total_latency_us_counter->fetch_add(latency_us,
                                               std::memory_order_relaxed);

    if (is_periodic) {
      next_release += period_ns;
      if (params.deadline_miss_counter) {
        const auto done = Clock::now();
        while (next_release < done) {
          params.deadline_miss_counter->fetch_add(1, std::memory_order_relaxed);
          next_release += period_ns;
        }
      }
    }
  }

  iree_runtime_call_deinitialize(&call);
}

}  // namespace

extern "C" int merlin_dual_model_runtime_run(
    const merlin_dual_model_runtime_config_t* config) {
  if (!config || !config->dronet_vmfb_path || !config->mlp_vmfb_path ||
      !config->dronet_function || !config->mlp_function ||
      !config->driver_name) {
    fprintf(stderr, "Invalid config passed to merlin_dual_model_runtime_run.\n");
    return 1;
  }
  if (config->mlp_frequency_hz <= 0.0) {
    fprintf(stderr, "mlp_frequency_hz must be > 0.\n");
    return 1;
  }
  if (config->report_frequency_hz <= 0.0) {
    fprintf(stderr, "report_frequency_hz must be > 0.\n");
    return 1;
  }
  if (config->dronet_sensor_frequency_hz <= 0.0) {
    fprintf(stderr, "dronet_sensor_frequency_hz must be > 0.\n");
    return 1;
  }
  if (config->mlp_sensor_frequency_hz <= 0.0) {
    fprintf(stderr, "mlp_sensor_frequency_hz must be > 0.\n");
    return 1;
  }

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();

  iree_runtime_instance_t* instance = nullptr;
  iree_hal_device_t* device = nullptr;
  iree_runtime_session_t* session = nullptr;
  std::unique_ptr<PeriodicTensorSensor> dronet_sensor;
  std::unique_ptr<PeriodicTensorSensor> mlp_sensor;

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  status = iree_runtime_instance_create(&instance_options, host_allocator,
                                        &instance);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view(config->driver_name), &device);
  if (!iree_status_is_ok(status)) goto cleanup;

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  status = iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = iree_runtime_session_append_bytecode_module_from_file(
      session, config->dronet_vmfb_path);
  if (!iree_status_is_ok(status)) goto cleanup;
  status = iree_runtime_session_append_bytecode_module_from_file(
      session, config->mlp_vmfb_path);
  if (!iree_status_is_ok(status)) goto cleanup;

  // Validate function names early (must be fully-qualified module.func).
  {
    iree_vm_function_t dronet_function;
    status = iree_runtime_session_lookup_function(
        session, iree_make_cstring_view(config->dronet_function),
        &dronet_function);
    if (!iree_status_is_ok(status)) goto cleanup;
  }
  {
    iree_vm_function_t mlp_function;
    status = iree_runtime_session_lookup_function(
        session, iree_make_cstring_view(config->mlp_function), &mlp_function);
    if (!iree_status_is_ok(status)) goto cleanup;
  }

  const std::vector<iree_hal_dim_t> dronet_shape = {1, 1, 224, 224};
  const std::vector<iree_hal_dim_t> mlp_shape = {1, 10};

  dronet_sensor = std::make_unique<PeriodicTensorSensor>(
      "dronet_sensor", dronet_shape, config->dronet_sensor_frequency_hz,
      /*base_value=*/0.01f, /*amplitude=*/1.0f);
  mlp_sensor = std::make_unique<PeriodicTensorSensor>(
      "mlp_sensor", mlp_shape, config->mlp_sensor_frequency_hz,
      /*base_value=*/0.25f, /*amplitude=*/0.5f);

  dronet_sensor->Start();
  mlp_sensor->Start();

  {
    SharedRuntimeState state;
    state.session = session;

    WorkerParams dronet_params;
    dronet_params.worker_name = "dronet";
    dronet_params.function_name = config->dronet_function;
    dronet_params.expected_outputs = 2;
    dronet_params.device = iree_runtime_session_device(session);
    dronet_params.device_allocator = iree_runtime_session_device_allocator(session);
    dronet_params.sensor = dronet_sensor.get();
    dronet_params.target_frequency_hz = 0.0;
    dronet_params.invocation_counter = &state.dronet_invocations;
    dronet_params.total_latency_us_counter = &state.dronet_total_latency_us;
    dronet_params.deadline_miss_counter = nullptr;
    dronet_params.fresh_input_counter = &state.dronet_fresh_inputs;

    WorkerParams mlp_params;
    mlp_params.worker_name = "mlp";
    mlp_params.function_name = config->mlp_function;
    mlp_params.expected_outputs = 1;
    mlp_params.device = iree_runtime_session_device(session);
    mlp_params.device_allocator = iree_runtime_session_device_allocator(session);
    mlp_params.sensor = mlp_sensor.get();
    mlp_params.target_frequency_hz = config->mlp_frequency_hz;
    mlp_params.invocation_counter = &state.mlp_invocations;
    mlp_params.total_latency_us_counter = &state.mlp_total_latency_us;
    mlp_params.deadline_miss_counter = &state.mlp_deadline_misses;
    mlp_params.fresh_input_counter = &state.mlp_fresh_inputs;

    std::thread dronet_thread(WorkerMain, &state, dronet_params);
    std::thread mlp_thread(WorkerMain, &state, mlp_params);

    const auto app_start = Clock::now();
    int64_t report_period_raw_ns =
        (int64_t)(1000000000.0 / config->report_frequency_hz);
    if (report_period_raw_ns < 1) report_period_raw_ns = 1;
    const auto report_period_ns =
        std::chrono::nanoseconds(report_period_raw_ns);
    auto next_report = app_start + report_period_ns;

    uint64_t last_dronet = 0;
    uint64_t last_mlp = 0;
    auto last_report = app_start;

    while (!state.stop_requested.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_until(next_report);
      const auto now = Clock::now();

      const uint64_t dronet_now =
          state.dronet_invocations.load(std::memory_order_relaxed);
      const uint64_t mlp_now = state.mlp_invocations.load(std::memory_order_relaxed);
      const uint64_t dronet_delta = dronet_now - last_dronet;
      const uint64_t mlp_delta = mlp_now - last_mlp;
      const double dt_s = std::chrono::duration<double>(now - last_report).count();
      const double dronet_hz = (dt_s > 0.0) ? ((double)dronet_delta / dt_s) : 0.0;
      const double mlp_hz = (dt_s > 0.0) ? ((double)mlp_delta / dt_s) : 0.0;
      const uint64_t misses =
          state.mlp_deadline_misses.load(std::memory_order_relaxed);
      const uint64_t dronet_fresh =
          state.dronet_fresh_inputs.load(std::memory_order_relaxed);
      const uint64_t mlp_fresh =
          state.mlp_fresh_inputs.load(std::memory_order_relaxed);
      const uint64_t dronet_generated = dronet_sensor->generated_count();
      const uint64_t mlp_generated = mlp_sensor->generated_count();

      fprintf(stdout,
              "[stats] dronet_hz=%.2f mlp_hz=%.2f mlp_misses=%" PRIu64
              " dronet_total=%" PRIu64 " mlp_total=%" PRIu64
              " dronet_fresh=%" PRIu64 " mlp_fresh=%" PRIu64
              " dronet_sensor_generated=%" PRIu64
              " mlp_sensor_generated=%" PRIu64 "\n",
              dronet_hz, mlp_hz, misses, dronet_now, mlp_now, dronet_fresh,
              mlp_fresh, dronet_generated, mlp_generated);
      fflush(stdout);

      last_dronet = dronet_now;
      last_mlp = mlp_now;
      last_report = now;
      next_report += report_period_ns;

      if (config->run_duration_ms > 0) {
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now - app_start)
                                    .count();
        if (elapsed_ms >= config->run_duration_ms) {
          state.stop_requested.store(true, std::memory_order_relaxed);
          break;
        }
      }

      if (HasFatalStatus(&state)) {
        state.stop_requested.store(true, std::memory_order_relaxed);
        break;
      }
    }

    state.stop_requested.store(true, std::memory_order_relaxed);
    if (dronet_thread.joinable()) dronet_thread.join();
    if (mlp_thread.joinable()) mlp_thread.join();

    iree_status_t worker_status = ConsumeFatalStatus(&state);
    if (!iree_status_is_ok(worker_status)) {
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
      }
      status = worker_status;
      goto cleanup;
    }

    const uint64_t dronet_total =
        state.dronet_invocations.load(std::memory_order_relaxed);
    const uint64_t mlp_total =
        state.mlp_invocations.load(std::memory_order_relaxed);
    const uint64_t misses =
        state.mlp_deadline_misses.load(std::memory_order_relaxed);
    const uint64_t dronet_fresh =
        state.dronet_fresh_inputs.load(std::memory_order_relaxed);
    const uint64_t mlp_fresh =
        state.mlp_fresh_inputs.load(std::memory_order_relaxed);
    const uint64_t dronet_latency_us =
        state.dronet_total_latency_us.load(std::memory_order_relaxed);
    const uint64_t mlp_latency_us =
        state.mlp_total_latency_us.load(std::memory_order_relaxed);

    const double dronet_avg_ms =
        (dronet_total > 0)
            ? ((double)dronet_latency_us / 1000.0) / (double)dronet_total
            : 0.0;
    const double mlp_avg_ms =
        (mlp_total > 0) ? ((double)mlp_latency_us / 1000.0) / (double)mlp_total
                        : 0.0;

    fprintf(stdout,
            "Run complete:\n"
            "  dronet_total_invocations = %" PRIu64 "\n"
            "  mlp_total_invocations    = %" PRIu64 "\n"
            "  mlp_deadline_misses      = %" PRIu64 "\n"
            "  dronet_fresh_inputs      = %" PRIu64 "\n"
            "  mlp_fresh_inputs         = %" PRIu64 "\n"
            "  dronet_sensor_generated  = %" PRIu64 "\n"
            "  mlp_sensor_generated     = %" PRIu64 "\n"
            "  dronet_avg_latency_ms    = %.3f\n"
            "  mlp_avg_latency_ms       = %.3f\n",
            dronet_total, mlp_total, misses, dronet_fresh, mlp_fresh,
            dronet_sensor->generated_count(), mlp_sensor->generated_count(),
            dronet_avg_ms, mlp_avg_ms);
    fflush(stdout);
  }

cleanup:
  if (dronet_sensor) dronet_sensor->Stop();
  if (mlp_sensor) mlp_sensor->Stop();
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}
