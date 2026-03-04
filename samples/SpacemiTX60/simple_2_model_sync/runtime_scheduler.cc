#include "runtime_scheduler.h"

#include <inttypes.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kPollSleepNs = 50000;  // 0.05ms

static iree_status_t WaitUntilReached(iree_hal_semaphore_t* sem,
                                      uint64_t value) {
  while (true) {
    iree_status_t status = iree_hal_semaphore_wait(
        sem, value, iree_immediate_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT);
    if (iree_status_is_ok(status)) return iree_ok_status();

    if (iree_status_code(status) == IREE_STATUS_DEADLINE_EXCEEDED) {
      iree_status_ignore(status);
      std::this_thread::sleep_for(std::chrono::nanoseconds(kPollSleepNs));
      continue;
    }

    // ABORTED or any other real error.
    return status;
  }
}

static iree_status_t ValidateAllOutputsAreBufferViews(iree_vm_list_t* outputs) {
  iree_host_size_t n = iree_vm_list_size(outputs);
  for (iree_host_size_t i = 0; i < n; ++i) {
    iree_vm_ref_t ref = iree_vm_ref_null();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(outputs, i, &ref));
    if (!iree_hal_buffer_view_isa(ref)) {
      // Don't release ref here; ref is just a view into the list item.
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "output %" PRIu64 " is not a HAL buffer_view",
                              (uint64_t)i);
    }
  }
  return iree_ok_status();
}

static iree_status_t CreateF32Input(iree_hal_device_t* device,
                                    iree_hal_allocator_t* device_allocator,
                                    const std::vector<iree_hal_dim_t>& shape,
                                    std::vector<float>* host_data_out,
                                    iree_hal_buffer_view_t** out_view) {
  *out_view = nullptr;

  iree_host_size_t element_count = 1;
  for (iree_hal_dim_t d : shape) element_count *= (iree_host_size_t)d;

  host_data_out->resize(element_count);
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    (*host_data_out)[i] = (float)(i % 17) * 0.1f;
  }

  iree_hal_buffer_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

  return iree_hal_buffer_view_allocate_buffer_copy(
      device, device_allocator,
      (iree_host_size_t)shape.size(), shape.data(),
      IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      params,
      iree_make_const_byte_span(host_data_out->data(),
                                host_data_out->size() * sizeof(float)),
      out_view);
}

static iree_status_t CallAsyncOnce(iree_runtime_session_t* session,
                                   const char* function_name,
                                   iree_hal_device_t* device,
                                   iree_hal_semaphore_t* timeline,
                                   uint64_t epoch,
                                   iree_hal_buffer_view_t* input_view,
                                   iree_vm_list_t** out_outputs,
                                   iree_hal_fence_t** out_signal_fence) {
  *out_outputs = nullptr;
  *out_signal_fence = nullptr;

  const iree_allocator_t host_alloc = iree_allocator_system();
  const iree_allocator_t device_host_alloc = iree_hal_device_host_allocator(device);

  iree_vm_list_t* inputs = nullptr;
  iree_vm_list_t* outputs = nullptr;
  iree_hal_fence_t* wait_fence = nullptr;
  iree_hal_fence_t* signal_fence = nullptr;

  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                          /*capacity=*/3, host_alloc, &inputs));
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                          /*capacity=*/8, host_alloc, &outputs));

  // 1) input buffer_view
  {
    iree_vm_ref_t input_ref = iree_hal_buffer_view_retain_ref(input_view);
    IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &input_ref));
    // DO NOT release input_ref after push_ref_move.
  }

  // 2) wait fence: explicit empty fence (0 capacity)
  IREE_RETURN_IF_ERROR(iree_hal_fence_create(/*capacity=*/0, device_host_alloc,
                                            &wait_fence));
  {
    iree_vm_ref_t wait_ref = iree_hal_fence_retain_ref(wait_fence);
    IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &wait_ref));
  }
  iree_hal_fence_release(wait_fence);
  wait_fence = nullptr;

  // 3) signal fence: timeline@epoch
  IREE_RETURN_IF_ERROR(iree_hal_fence_create_at(timeline, epoch, device_host_alloc,
                                               &signal_fence));
  {
    iree_vm_ref_t signal_ref = iree_hal_fence_retain_ref(signal_fence);
    IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &signal_ref));
  }

  // Call (should return quickly if async-external is functioning)
  IREE_RETURN_IF_ERROR(iree_runtime_session_call_by_name(
      session, iree_make_cstring_view(function_name), inputs, outputs));

  iree_vm_list_release(inputs);
  inputs = nullptr;

  *out_outputs = outputs;
  outputs = nullptr;

  *out_signal_fence = signal_fence;  // keep alive in caller
  signal_fence = nullptr;

  return iree_ok_status();
}

}  // namespace

extern "C" int merlin_async_smoke_run(const merlin_async_smoke_config_t* config) {
  if (!config || !config->dronet_vmfb_path || !config->mlp_vmfb_path ||
      !config->dronet_function || !config->mlp_function || !config->driver_name) {
    fprintf(stderr, "Invalid config.\n");
    return 1;
  }

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();

  iree_runtime_instance_t* instance = nullptr;
  iree_hal_device_t* device = nullptr;
  iree_runtime_session_t* session = nullptr;

  iree_hal_semaphore_t* dronet_timeline = nullptr;
  iree_hal_semaphore_t* mlp_timeline = nullptr;

  do {
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_options_use_all_available_drivers(&instance_options);
    status = iree_runtime_instance_create(&instance_options, host_allocator, &instance);
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

    // Create timelines.
    status = iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                       /*initial_value=*/0ull,
                                       IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                       &dronet_timeline);
    if (!iree_status_is_ok(status)) break;
    status = iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                       /*initial_value=*/0ull,
                                       IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                       &mlp_timeline);
    if (!iree_status_is_ok(status)) break;

    // Device allocator.
    iree_hal_allocator_t* device_allocator = iree_runtime_session_device_allocator(session);
    iree_hal_device_t* session_device = iree_runtime_session_device(session);

    // Hardcoded shapes (same as your baseline).
    const std::vector<iree_hal_dim_t> dronet_shape = {1, 3, 112, 112};
    const std::vector<iree_hal_dim_t> mlp_shape = {1, 10};

    // --- 1) MLP first (isolate failures) ---
    fprintf(stdout, "[smoke] calling MLP once...\n");
    fflush(stdout);

    std::vector<float> mlp_host;
    iree_hal_buffer_view_t* mlp_input = nullptr;
    status = CreateF32Input(session_device, device_allocator, mlp_shape, &mlp_host,
                            &mlp_input);
    if (!iree_status_is_ok(status)) break;

    iree_vm_list_t* mlp_outputs = nullptr;
    iree_hal_fence_t* mlp_signal_fence = nullptr;

    auto t0 = Clock::now();
    status = CallAsyncOnce(session, config->mlp_function, session_device,
                           mlp_timeline, /*epoch=*/1, mlp_input,
                           &mlp_outputs, &mlp_signal_fence);
    auto t1 = Clock::now();
    if (!iree_status_is_ok(status)) break;

    fprintf(stdout, "[smoke] MLP call returned in %.3f ms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    fflush(stdout);

    status = WaitUntilReached(mlp_timeline, 1);
    if (!iree_status_is_ok(status)) {
      fprintf(stderr, "[smoke] MLP timeline wait failed\n");
      iree_status_fprint(stderr, status);
      break;
    }

    fprintf(stdout, "[smoke] MLP timeline reached. outputs=%" PRIu64 "\n",
            (uint64_t)iree_vm_list_size(mlp_outputs));
    fflush(stdout);

    status = ValidateAllOutputsAreBufferViews(mlp_outputs);
    if (!iree_status_is_ok(status)) break;

    iree_vm_list_release(mlp_outputs);
    iree_hal_fence_release(mlp_signal_fence);
    iree_hal_buffer_view_release(mlp_input);

    // --- 2) DRONET next ---
    fprintf(stdout, "[smoke] calling DRONET once...\n");
    fflush(stdout);

    std::vector<float> dronet_host;
    iree_hal_buffer_view_t* dronet_input = nullptr;
    status = CreateF32Input(session_device, device_allocator, dronet_shape,
                            &dronet_host, &dronet_input);
    if (!iree_status_is_ok(status)) break;

    iree_vm_list_t* dronet_outputs = nullptr;
    iree_hal_fence_t* dronet_signal_fence = nullptr;

    t0 = Clock::now();
    status = CallAsyncOnce(session, config->dronet_function, session_device,
                           dronet_timeline, /*epoch=*/1, dronet_input,
                           &dronet_outputs, &dronet_signal_fence);
    t1 = Clock::now();
    if (!iree_status_is_ok(status)) break;

    fprintf(stdout, "[smoke] DRONET call returned in %.3f ms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    fflush(stdout);

    status = WaitUntilReached(dronet_timeline, 1);
    if (!iree_status_is_ok(status)) {
      fprintf(stderr, "[smoke] DRONET timeline wait failed\n");
      iree_status_fprint(stderr, status);
      break;
    }

    fprintf(stdout, "[smoke] DRONET timeline reached. outputs=%" PRIu64 "\n",
            (uint64_t)iree_vm_list_size(dronet_outputs));
    fflush(stdout);

    status = ValidateAllOutputsAreBufferViews(dronet_outputs);
    if (!iree_status_is_ok(status)) break;

    iree_vm_list_release(dronet_outputs);
    iree_hal_fence_release(dronet_signal_fence);
    iree_hal_buffer_view_release(dronet_input);

    fprintf(stdout, "[smoke] SUCCESS: both async calls completed.\n");
    fflush(stdout);

  } while (false);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  }

  iree_hal_semaphore_release(dronet_timeline);
  iree_hal_semaphore_release(mlp_timeline);
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);

  return iree_status_is_ok(status) ? 0 : 1;
}