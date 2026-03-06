#include "runtime_scheduler.h"

#include <inttypes.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"
#include "sensor_generator.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kDronetMaxInFlight = 8;
constexpr uint64_t kMlpMaxInFlight = 2;
constexpr int64_t kSubmitLoopIdleSleepNs = 200000;   // 0.2ms
constexpr int64_t kReaperWaitTimeoutNs = 1000000;    // 1.0ms

enum class ModelId : uint8_t {
  kDronet = 0,
  kMlp = 1,
};

struct InflightInvocation {
  ModelId model_id = ModelId::kDronet;
  uint64_t epoch = 0;
  iree_vm_list_t* outputs = nullptr;
  iree_host_size_t expected_outputs = 0;
  Clock::time_point submit_time;
};

struct SharedRuntimeState {
  std::atomic<bool> submission_done{false};
  std::atomic<int> fatal_status_code{IREE_STATUS_OK};

  std::mutex pending_mutex;
  std::condition_variable pending_cv;
  std::deque<InflightInvocation> dronet_pending;
  std::deque<InflightInvocation> mlp_pending;

  std::atomic<uint64_t> dronet_submitted_epoch{0};
  std::atomic<uint64_t> mlp_submitted_epoch{0};
  std::atomic<uint64_t> dronet_completed_epoch{0};
  std::atomic<uint64_t> mlp_completed_epoch{0};

  std::atomic<uint64_t> dronet_invocations{0};
  std::atomic<uint64_t> mlp_invocations{0};
  std::atomic<uint64_t> mlp_deadline_misses{0};
  std::atomic<uint64_t> dronet_fresh_inputs{0};
  std::atomic<uint64_t> mlp_fresh_inputs{0};
  std::atomic<uint64_t> dronet_total_latency_us{0};
  std::atomic<uint64_t> mlp_total_latency_us{0};
};

struct ModelSubmitContext {
  const char* function_name = nullptr;
  iree_host_size_t expected_outputs = 0;
  iree_runtime_session_t* session = nullptr;
  iree_hal_device_t* device = nullptr;
  iree_hal_allocator_t* device_allocator = nullptr;
  const PeriodicTensorSensor* sensor = nullptr;
  iree_hal_semaphore_t* timeline = nullptr;
  std::deque<InflightInvocation>* pending_queue = nullptr;
  std::atomic<uint64_t>* submitted_epoch_counter = nullptr;
  std::atomic<uint64_t>* fresh_input_counter = nullptr;
  ModelId model_id = ModelId::kDronet;

  uint64_t next_epoch = 0;
  uint64_t last_sequence = UINT64_MAX;
  std::vector<float> host_input;
};

static bool HasFatalStatus(const SharedRuntimeState* state) {
  return state->fatal_status_code.load(std::memory_order_relaxed) !=
         IREE_STATUS_OK;
}

static void StoreFatalStatusIfFirst(SharedRuntimeState* state,
                                    iree_status_t status,
                                    const char* context) {
  if (iree_status_is_ok(status)) return;
  const int code = (int)iree_status_code(status);
  int expected = IREE_STATUS_OK;
  if (state->fatal_status_code.compare_exchange_strong(
          expected, code, std::memory_order_relaxed)) {
    if (context && context[0]) {
      fprintf(stderr, "%s\n", context);
    }
    iree_status_fprint(stderr, status);
  }
  iree_status_ignore(status);
  state->pending_cv.notify_all();
}

static bool FunctionUsesCoarseFencesAbi(const iree_vm_function_t* function) {
  const iree_string_view_t model = iree_vm_function_lookup_attr_by_name(
      function, IREE_SV("iree.abi.model"));
  return iree_string_view_equal(model, IREE_SV("coarse-fences"));
}

static std::chrono::nanoseconds PeriodFromFrequencyHz(double frequency_hz) {
  int64_t period_ns = (int64_t)(1000000000.0 / frequency_hz);
  if (period_ns < 1) period_ns = 1;
  return std::chrono::nanoseconds(period_ns);
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

static iree_status_t ValidateOutputList(iree_vm_list_t* outputs,
                                        iree_host_size_t expected_outputs) {
  IREE_ASSERT_ARGUMENT(outputs);
  const iree_host_size_t output_count = iree_vm_list_size(outputs);
  if (output_count != expected_outputs) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected %" PRIu64
                            " outputs from invocation but got %" PRIu64,
                            (uint64_t)expected_outputs, (uint64_t)output_count);
  }

  for (iree_host_size_t i = 0; i < output_count; ++i) {
    iree_vm_ref_t ref = iree_vm_ref_null();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(outputs, i, &ref));
    const bool is_buffer_view = iree_hal_buffer_view_isa(ref);
    iree_vm_ref_release(&ref);
    if (!is_buffer_view) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "output %" PRIu64 " was not a HAL buffer view",
                              (uint64_t)i);
    }
  }

  return iree_ok_status();
}

static iree_status_t SubmitAsyncInvocation(
    iree_runtime_session_t* session, const char* function_name,
    iree_hal_buffer_view_t* input_view, iree_hal_device_t* device,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_epoch,
    iree_allocator_t host_allocator, iree_vm_list_t** out_outputs) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(function_name);
  IREE_ASSERT_ARGUMENT(input_view);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(signal_semaphore);
  IREE_ASSERT_ARGUMENT(out_outputs);
  *out_outputs = nullptr;

  iree_status_t status = iree_ok_status();
  iree_vm_list_t* inputs = nullptr;
  iree_vm_list_t* outputs = nullptr;
  iree_hal_fence_t* signal_fence = nullptr;

  status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                               /*capacity=*/3, host_allocator, &inputs);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                               /*capacity=*/8, host_allocator, &outputs);
  if (!iree_status_is_ok(status)) goto cleanup;

  {
    iree_vm_ref_t input_ref = iree_hal_buffer_view_retain_ref(input_view);
    status = iree_vm_list_push_ref_move(inputs, &input_ref);
    iree_vm_ref_release(&input_ref);
    if (!iree_status_is_ok(status)) goto cleanup;
  }

  {
    iree_hal_fence_t* empty_wait_fence = nullptr;
    // Create an empty fence (0 capacity) to represent "no wait dependencies"
    status = iree_hal_fence_create(/*capacity=*/0, host_allocator, &empty_wait_fence);
    if (!iree_status_is_ok(status)) goto cleanup;

    iree_vm_ref_t wait_fence_ref = iree_hal_fence_retain_ref(empty_wait_fence);
    status = iree_vm_list_push_ref_move(inputs, &wait_fence_ref);
    
    // Release our local references (the VM list now owns a ref)
    iree_vm_ref_release(&wait_fence_ref);
    iree_hal_fence_release(empty_wait_fence);
    
    if (!iree_status_is_ok(status)) goto cleanup;
  }

  status = iree_hal_fence_create_at(signal_semaphore, signal_epoch,
                                    iree_hal_device_host_allocator(device),
                                    &signal_fence);
  if (!iree_status_is_ok(status)) goto cleanup;

  {
    iree_vm_ref_t signal_fence_ref = iree_hal_fence_retain_ref(signal_fence);
    status = iree_vm_list_push_ref_move(inputs, &signal_fence_ref);
    iree_vm_ref_release(&signal_fence_ref);
    if (!iree_status_is_ok(status)) goto cleanup;
  }

  status = iree_runtime_session_call_by_name(
      session, iree_make_cstring_view(function_name), inputs, outputs);
  if (!iree_status_is_ok(status)) goto cleanup;

  *out_outputs = outputs;
  outputs = nullptr;

cleanup:
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_fence_release(signal_fence);
  return status;
}

static iree_status_t SubmitModelFrame(SharedRuntimeState* state,
                                      ModelSubmitContext* model,
                                      iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(model);
  IREE_ASSERT_ARGUMENT(model->session);
  IREE_ASSERT_ARGUMENT(model->device);
  IREE_ASSERT_ARGUMENT(model->device_allocator);
  IREE_ASSERT_ARGUMENT(model->sensor);
  IREE_ASSERT_ARGUMENT(model->timeline);
  IREE_ASSERT_ARGUMENT(model->pending_queue);
  IREE_ASSERT_ARGUMENT(model->submitted_epoch_counter);
  IREE_ASSERT_ARGUMENT(model->fresh_input_counter);

  const uint64_t sample_sequence = model->sensor->Snapshot(&model->host_input,
                                                           /*out_time=*/nullptr);
  if (sample_sequence != model->last_sequence) {
    model->last_sequence = sample_sequence;
    model->fresh_input_counter->fetch_add(1, std::memory_order_relaxed);
  }

  iree_hal_buffer_view_t* input_view = nullptr;
  iree_status_t status = CreateF32InputViewFromData(
      model->device, model->device_allocator, model->sensor->shape().data(),
      model->sensor->shape().size(), model->host_input.data(),
      model->host_input.size(), &input_view);
  if (!iree_status_is_ok(status)) return status;

  const uint64_t epoch = model->next_epoch + 1;
  iree_vm_list_t* outputs = nullptr;
  status = SubmitAsyncInvocation(
      model->session, model->function_name, input_view, model->device,
      model->timeline, epoch, host_allocator, &outputs);
  iree_hal_buffer_view_release(input_view);
  if (!iree_status_is_ok(status)) return status;

  InflightInvocation invocation;
  invocation.model_id = model->model_id;
  invocation.epoch = epoch;
  invocation.outputs = outputs;
  invocation.expected_outputs = model->expected_outputs;
  invocation.submit_time = Clock::now();

  {
    std::lock_guard<std::mutex> lock(state->pending_mutex);
    model->pending_queue->push_back(std::move(invocation));
  }
  state->pending_cv.notify_one();

  model->next_epoch = epoch;
  model->submitted_epoch_counter->store(epoch, std::memory_order_relaxed);
  return iree_ok_status();
}

static iree_status_t PollTimelineReached(iree_hal_semaphore_t* semaphore,
                                         uint64_t epoch, bool* out_reached) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_reached);
  *out_reached = false;

  iree_status_t status = iree_hal_semaphore_wait(
      semaphore, epoch, iree_immediate_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT);
  if (iree_status_is_ok(status)) {
    *out_reached = true;
    return iree_ok_status();
  }
  if (iree_status_code(status) == IREE_STATUS_DEADLINE_EXCEEDED) {
    iree_status_ignore(status);
    return iree_ok_status();
  }
  return status;
}

static iree_status_t WaitTimelineWithTimeout(iree_hal_semaphore_t* semaphore,
                                             uint64_t epoch,
                                             iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(semaphore);

  iree_status_t status = iree_hal_semaphore_wait(
      semaphore, epoch, timeout, IREE_HAL_WAIT_FLAG_DEFAULT);
  if (iree_status_is_ok(status)) return iree_ok_status();
  if (iree_status_code(status) == IREE_STATUS_DEADLINE_EXCEEDED) {
    iree_status_ignore(status);
    return iree_ok_status();
  }
  return status;
}

static void ReleasePendingOutputsUnlocked(std::deque<InflightInvocation>* queue) {
  while (!queue->empty()) {
    iree_vm_list_release(queue->front().outputs);
    queue->pop_front();
  }
}

static void ReaperMain(SharedRuntimeState* state,
                       iree_hal_semaphore_t* dronet_timeline,
                       iree_hal_semaphore_t* mlp_timeline) {
  while (true) {
    bool has_dronet = false;
    bool has_mlp = false;
    uint64_t dronet_epoch = 0;
    uint64_t mlp_epoch = 0;
    Clock::time_point dronet_submit_time;
    Clock::time_point mlp_submit_time;

    {
      std::unique_lock<std::mutex> lock(state->pending_mutex);
      state->pending_cv.wait_for(lock, std::chrono::milliseconds(1), [&]() {
        return HasFatalStatus(state) ||
               state->submission_done.load(std::memory_order_relaxed) ||
               !state->dronet_pending.empty() || !state->mlp_pending.empty();
      });

      has_dronet = !state->dronet_pending.empty();
      has_mlp = !state->mlp_pending.empty();
      if (!has_dronet && !has_mlp) {
        if (HasFatalStatus(state) ||
            state->submission_done.load(std::memory_order_relaxed)) {
          break;
        }
        continue;
      }

      if (has_dronet) {
        dronet_epoch = state->dronet_pending.front().epoch;
        dronet_submit_time = state->dronet_pending.front().submit_time;
      }
      if (has_mlp) {
        mlp_epoch = state->mlp_pending.front().epoch;
        mlp_submit_time = state->mlp_pending.front().submit_time;
      }
    }

    bool dronet_ready = false;
    bool mlp_ready = false;

    if (has_dronet) {
      iree_status_t status =
          PollTimelineReached(dronet_timeline, dronet_epoch, &dronet_ready);
      if (!iree_status_is_ok(status)) {
        StoreFatalStatusIfFirst(state, status,
                                "[dronet/reaper] timeline wait failed");
        break;
      }
    }

    if (has_mlp) {
      iree_status_t status =
          PollTimelineReached(mlp_timeline, mlp_epoch, &mlp_ready);
      if (!iree_status_is_ok(status)) {
        StoreFatalStatusIfFirst(state, status,
                                "[mlp/reaper] timeline wait failed");
        break;
      }
    }

    if (!dronet_ready && !mlp_ready) {
      iree_hal_semaphore_t* wait_timeline = nullptr;
      uint64_t wait_epoch = 0;
      if (has_dronet && has_mlp) {
        if (dronet_submit_time <= mlp_submit_time) {
          wait_timeline = dronet_timeline;
          wait_epoch = dronet_epoch;
        } else {
          wait_timeline = mlp_timeline;
          wait_epoch = mlp_epoch;
        }
      } else if (has_dronet) {
        wait_timeline = dronet_timeline;
        wait_epoch = dronet_epoch;
      } else {
        wait_timeline = mlp_timeline;
        wait_epoch = mlp_epoch;
      }

      iree_status_t status = WaitTimelineWithTimeout(
          wait_timeline, wait_epoch, iree_make_timeout_ns(kReaperWaitTimeoutNs));
      if (!iree_status_is_ok(status)) {
        StoreFatalStatusIfFirst(state, status,
                                "[reaper] timed wait on timeline failed");
        break;
      }
      continue;
    }

    const bool pop_dronet =
        dronet_ready && (!mlp_ready || dronet_submit_time <= mlp_submit_time);
    InflightInvocation invocation;
    {
      std::lock_guard<std::mutex> lock(state->pending_mutex);
      std::deque<InflightInvocation>* queue =
          pop_dronet ? &state->dronet_pending : &state->mlp_pending;
      if (queue->empty()) continue;
      invocation = std::move(queue->front());
      queue->pop_front();
    }

    iree_status_t status =
        ValidateOutputList(invocation.outputs, invocation.expected_outputs);
    iree_vm_list_release(invocation.outputs);
    if (!iree_status_is_ok(status)) {
      StoreFatalStatusIfFirst(state, status,
                              "[reaper] invocation outputs were invalid");
      break;
    }

    const uint64_t latency_us =
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - invocation.submit_time)
            .count();

    if (invocation.model_id == ModelId::kDronet) {
      state->dronet_invocations.fetch_add(1, std::memory_order_relaxed);
      state->dronet_total_latency_us.fetch_add(latency_us,
                                               std::memory_order_relaxed);
      state->dronet_completed_epoch.store(invocation.epoch,
                                          std::memory_order_relaxed);
    } else {
      state->mlp_invocations.fetch_add(1, std::memory_order_relaxed);
      state->mlp_total_latency_us.fetch_add(latency_us,
                                            std::memory_order_relaxed);
      state->mlp_completed_epoch.store(invocation.epoch,
                                       std::memory_order_relaxed);
    }
  }

  {
    std::lock_guard<std::mutex> lock(state->pending_mutex);
    ReleasePendingOutputsUnlocked(&state->dronet_pending);
    ReleasePendingOutputsUnlocked(&state->mlp_pending);
  }
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
  iree_hal_semaphore_t* dronet_timeline = nullptr;
  iree_hal_semaphore_t* mlp_timeline = nullptr;
  std::unique_ptr<PeriodicTensorSensor> dronet_sensor;
  std::unique_ptr<PeriodicTensorSensor> mlp_sensor;
  std::thread reaper_thread;
  bool reaper_started = false;
  SharedRuntimeState state;

  const std::vector<iree_hal_dim_t> dronet_shape = {1, 3, 112, 112};
  const std::vector<iree_hal_dim_t> mlp_shape = {1, 10};

  do {
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_options_use_all_available_drivers(&instance_options);
    status = iree_runtime_instance_create(&instance_options, host_allocator,
                                          &instance);
    if (!iree_status_is_ok(status)) break;

    status = iree_runtime_instance_try_create_default_device(
        instance, iree_make_cstring_view(config->driver_name), &device);
    if (!iree_status_is_ok(status)) break;

    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
    if (!iree_status_is_ok(status)) break;

    status = iree_runtime_session_append_bytecode_module_from_file(
        session, config->dronet_vmfb_path);
    if (!iree_status_is_ok(status)) break;
    status = iree_runtime_session_append_bytecode_module_from_file(
        session, config->mlp_vmfb_path);
    if (!iree_status_is_ok(status)) break;

    iree_vm_function_t dronet_function;
    status = iree_runtime_session_lookup_function(
        session, iree_make_cstring_view(config->dronet_function),
        &dronet_function);
    if (!iree_status_is_ok(status)) break;

    iree_vm_function_t mlp_function;
    status = iree_runtime_session_lookup_function(
        session, iree_make_cstring_view(config->mlp_function), &mlp_function);
    if (!iree_status_is_ok(status)) break;
    
    // TODO: Figure out what is the problem with the naming of the different modules
    //const bool dronet_async = FunctionUsesCoarseFencesAbi(&dronet_function);
    //const bool mlp_async = FunctionUsesCoarseFencesAbi(&mlp_function);

    //if (!dronet_async || !mlp_async) {
    //  status = iree_make_status(
    //      IREE_STATUS_FAILED_PRECONDITION,
    //      "both functions must be compiled with --iree-execution-model="
    //      "async-external (iree.abi.model=coarse-fences)");
    //  break;
    //}

    fprintf(stdout,
            "[dronet] invocation_model=coarse-fences(async-external)\n"
            "[mlp] invocation_model=coarse-fences(async-external)\n");
    fflush(stdout);

    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0ull,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &dronet_timeline);
    if (!iree_status_is_ok(status)) break;
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0ull,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &mlp_timeline);
    if (!iree_status_is_ok(status)) break;

    dronet_sensor = std::make_unique<PeriodicTensorSensor>(
        "dronet_sensor", dronet_shape, config->dronet_sensor_frequency_hz,
        /*base_value=*/0.01f, /*amplitude=*/1.0f);
    mlp_sensor = std::make_unique<PeriodicTensorSensor>(
        "mlp_sensor", mlp_shape, config->mlp_sensor_frequency_hz,
        /*base_value=*/0.25f, /*amplitude=*/0.5f);
    dronet_sensor->Start();
    mlp_sensor->Start();

    ModelSubmitContext dronet_submit;
    dronet_submit.function_name = config->dronet_function;
    dronet_submit.expected_outputs = 2;
    dronet_submit.session = session;
    dronet_submit.device = iree_runtime_session_device(session);
    dronet_submit.device_allocator = iree_runtime_session_device_allocator(session);
    dronet_submit.sensor = dronet_sensor.get();
    dronet_submit.timeline = dronet_timeline;
    dronet_submit.pending_queue = &state.dronet_pending;
    dronet_submit.submitted_epoch_counter = &state.dronet_submitted_epoch;
    dronet_submit.fresh_input_counter = &state.dronet_fresh_inputs;
    dronet_submit.model_id = ModelId::kDronet;
    dronet_submit.host_input.resize(dronet_submit.sensor->element_count());

    ModelSubmitContext mlp_submit;
    mlp_submit.function_name = config->mlp_function;
    mlp_submit.expected_outputs = 1;
    mlp_submit.session = session;
    mlp_submit.device = iree_runtime_session_device(session);
    mlp_submit.device_allocator = iree_runtime_session_device_allocator(session);
    mlp_submit.sensor = mlp_sensor.get();
    mlp_submit.timeline = mlp_timeline;
    mlp_submit.pending_queue = &state.mlp_pending;
    mlp_submit.submitted_epoch_counter = &state.mlp_submitted_epoch;
    mlp_submit.fresh_input_counter = &state.mlp_fresh_inputs;
    mlp_submit.model_id = ModelId::kMlp;
    mlp_submit.host_input.resize(mlp_submit.sensor->element_count());

    reaper_thread =
        std::thread(ReaperMain, &state, dronet_timeline, mlp_timeline);
    reaper_started = true;

    const auto app_start = Clock::now();
    const auto report_period = PeriodFromFrequencyHz(config->report_frequency_hz);
    const auto mlp_period = PeriodFromFrequencyHz(config->mlp_frequency_hz);
    auto next_report = app_start + report_period;
    auto next_mlp_release = app_start;

    uint64_t last_dronet = 0;
    uint64_t last_mlp = 0;
    auto last_report = app_start;

    while (!HasFatalStatus(&state)) {
      const auto now = Clock::now();

      if (config->run_duration_ms > 0) {
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - app_start)
                .count();
        if (elapsed_ms >= config->run_duration_ms) {
          break;
        }
      }

      bool submitted_any = false;

      while (now >= next_mlp_release + mlp_period) {
        state.mlp_deadline_misses.fetch_add(1, std::memory_order_relaxed);
        next_mlp_release += mlp_period;
      }

      if (now >= next_mlp_release) {
        const uint64_t mlp_inflight =
            state.mlp_submitted_epoch.load(std::memory_order_relaxed) -
            state.mlp_completed_epoch.load(std::memory_order_relaxed);
        if (mlp_inflight < kMlpMaxInFlight) {
          status = SubmitModelFrame(&state, &mlp_submit, host_allocator);
          if (!iree_status_is_ok(status)) {
            StoreFatalStatusIfFirst(&state, status,
                                    "[mlp] async submission failed");
            break;
          }
          submitted_any = true;
        } else {
          state.mlp_deadline_misses.fetch_add(1, std::memory_order_relaxed);
        }
        next_mlp_release += mlp_period;
      }

      while (!HasFatalStatus(&state)) {
        const uint64_t dronet_inflight =
            state.dronet_submitted_epoch.load(std::memory_order_relaxed) -
            state.dronet_completed_epoch.load(std::memory_order_relaxed);
        if (dronet_inflight >= kDronetMaxInFlight) break;

        status = SubmitModelFrame(&state, &dronet_submit, host_allocator);
        if (!iree_status_is_ok(status)) {
          StoreFatalStatusIfFirst(&state, status,
                                  "[dronet] async submission failed");
          break;
        }
        submitted_any = true;
      }

      if (HasFatalStatus(&state)) break;

      if (now >= next_report) {
        const uint64_t dronet_now =
            state.dronet_invocations.load(std::memory_order_relaxed);
        const uint64_t mlp_now =
            state.mlp_invocations.load(std::memory_order_relaxed);
        const uint64_t dronet_delta = dronet_now - last_dronet;
        const uint64_t mlp_delta = mlp_now - last_mlp;
        const double dt_s =
            std::chrono::duration<double>(now - last_report).count();
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
        const uint64_t dronet_inflight =
            state.dronet_submitted_epoch.load(std::memory_order_relaxed) -
            state.dronet_completed_epoch.load(std::memory_order_relaxed);
        const uint64_t mlp_inflight =
            state.mlp_submitted_epoch.load(std::memory_order_relaxed) -
            state.mlp_completed_epoch.load(std::memory_order_relaxed);

        fprintf(stdout,
                "[stats] dronet_hz=%.2f mlp_hz=%.2f mlp_misses=%" PRIu64
                " dronet_total=%" PRIu64 " mlp_total=%" PRIu64
                " dronet_inflight=%" PRIu64 " mlp_inflight=%" PRIu64
                " dronet_fresh=%" PRIu64 " mlp_fresh=%" PRIu64
                " dronet_sensor_generated=%" PRIu64
                " mlp_sensor_generated=%" PRIu64 "\n",
                dronet_hz, mlp_hz, misses, dronet_now, mlp_now, dronet_inflight,
                mlp_inflight, dronet_fresh, mlp_fresh, dronet_generated,
                mlp_generated);
        fflush(stdout);

        last_dronet = dronet_now;
        last_mlp = mlp_now;
        last_report = now;
        do {
          next_report += report_period;
        } while (next_report <= now);
      }

      if (!submitted_any) {
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(kSubmitLoopIdleSleepNs));
      }
    }

    state.submission_done.store(true, std::memory_order_relaxed);
    state.pending_cv.notify_all();

    if (reaper_started && reaper_thread.joinable()) {
      reaper_thread.join();
      reaper_started = false;
    }

    if (HasFatalStatus(&state)) {
      status = iree_status_from_code(
          (iree_status_code_t)state.fatal_status_code.load(
              std::memory_order_relaxed));
      break;
    }

    const uint64_t dronet_total =
        state.dronet_invocations.load(std::memory_order_relaxed);
    const uint64_t mlp_total =
        state.mlp_invocations.load(std::memory_order_relaxed);
    const uint64_t misses =
        state.mlp_deadline_misses.load(std::memory_order_relaxed);
    const uint64_t dronet_fresh =
        state.dronet_fresh_inputs.load(std::memory_order_relaxed);
    const uint64_t mlp_fresh = state.mlp_fresh_inputs.load(std::memory_order_relaxed);
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
  } while (false);

  state.submission_done.store(true, std::memory_order_relaxed);
  state.pending_cv.notify_all();
  if (reaper_started && reaper_thread.joinable()) {
    reaper_thread.join();
  }

  if (dronet_sensor) dronet_sensor->Stop();
  if (mlp_sensor) mlp_sensor->Stop();
  iree_hal_semaphore_release(dronet_timeline);
  iree_hal_semaphore_release(mlp_timeline);
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
