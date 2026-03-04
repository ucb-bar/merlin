#include "runtime_scheduler.h"

#include <inttypes.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

namespace {

#define CHECK_OK(expr)                                                         \
  do {                                                                         \
    iree_status_t _status = (expr);                                            \
    if (!iree_status_is_ok(_status)) {                                         \
      iree_status_fprint(stderr, _status);                                     \
      iree_status_ignore(_status);                                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static void DumpBufferPrefix(iree_hal_buffer_view_t* view, int bytes) {
  iree_hal_buffer_t* buf = iree_hal_buffer_view_buffer(view);
  if (!buf) return;

  std::vector<uint8_t> tmp(bytes, 0);
  iree_status_t st =
      iree_hal_buffer_map_read(buf, /*source_offset=*/0, tmp.data(), tmp.size());
  if (!iree_status_is_ok(st)) {
    iree_status_ignore(st);
    fprintf(stdout, "    [dump] map_read failed\n");
    return;
  }
  fprintf(stdout, "    [dump] first %d bytes:", bytes);
  for (int i = 0; i < bytes; ++i) fprintf(stdout, " %02X", tmp[i]);
  fprintf(stdout, "\n");
}

static void PrintBufferViewInfo(const char* tag, iree_hal_buffer_view_t* view,
                               int dump_bytes) {
  if (!view) {
    fprintf(stdout, "  [%s] output view = (null)\n", tag);
    return;
  }

  iree_hal_element_type_t et = iree_hal_buffer_view_element_type(view);
  iree_host_size_t rank = iree_hal_buffer_view_shape_rank(view);
  const iree_hal_dim_t* dims = iree_hal_buffer_view_shape_dims(view);

  iree_device_size_t byte_len = iree_hal_buffer_view_byte_length(view);

  fprintf(stdout, "  [%s] element_type=0x%08X rank=%" PRIu64 " bytes=%" PRIu64 " shape=[",
          tag, (uint32_t)et, (uint64_t)rank, (uint64_t)byte_len);
  for (iree_host_size_t i = 0; i < rank; ++i) {
    fprintf(stdout, "%s%" PRId64, (i ? "," : ""), (int64_t)dims[i]);
  }
  fprintf(stdout, "]\n");

  if (dump_bytes) DumpBufferPrefix(view, /*bytes=*/16);
}

// Creates a simple device-local f32 tensor filled with deterministic values.
static iree_hal_buffer_view_t* MakeF32Input(iree_runtime_session_t* session,
                                           const std::vector<iree_hal_dim_t>& shape) {
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* allocator = iree_runtime_session_device_allocator(session);

  iree_host_size_t element_count = 1;
  for (auto d : shape) element_count *= (iree_host_size_t)d;

  std::vector<float> host(element_count);
  for (iree_host_size_t i = 0; i < element_count; ++i) host[i] = (float)(i % 23) * 0.01f;

  iree_hal_buffer_view_t* input_view = nullptr;
  iree_status_t st = iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator,
      (iree_host_size_t)shape.size(), shape.data(),
      IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
      iree_make_const_byte_span(host.data(), host.size() * sizeof(float)),
      &input_view);

  if (!iree_status_is_ok(st)) {
    iree_status_fprint(stderr, st);
    iree_status_ignore(st);
    return nullptr;
  }
  return input_view;
}

// Does the same “T=0 -> fences at T=1,T=2 -> signal T=1 -> call -> wait T=2”
// as the IREE sample.
static int CallOneModelGated(iree_runtime_session_t* session,
                            const char* tag,
                            const char* function_name,
                            const std::vector<iree_hal_dim_t>& input_shape,
                            int dump_output_bytes) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_allocator_t device_host_allocator = iree_hal_device_host_allocator(device);

  iree_vm_list_t* inputs = nullptr;
  iree_vm_list_t* outputs = nullptr;

  iree_hal_semaphore_t* semaphore = nullptr;
  iree_hal_fence_t* fence_t1 = nullptr;
  iree_hal_fence_t* fence_t2 = nullptr;

  iree_hal_buffer_view_t* input_view = nullptr;

  // Lists: (input_view, wait_fence, signal_fence)
  CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                              /*capacity=*/3, host_allocator, &inputs));
  CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                              /*capacity=*/8, host_allocator, &outputs));

  input_view = MakeF32Input(session, input_shape);
  if (!input_view) return 1;

  // Push input (retain so we can hold it alive until after fence_t2).
  {
    iree_vm_ref_t ref = iree_hal_buffer_view_retain_ref(input_view);
    CHECK_OK(iree_vm_list_push_ref_move(inputs, &ref));
  }

  // Create a timeline at T=0 and fences at T=1 and T=2.
  CHECK_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                     /*initial_value=*/0ull,
                                     IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                     &semaphore));
  CHECK_OK(iree_hal_fence_create_at(semaphore, 1ull, device_host_allocator, &fence_t1));
  CHECK_OK(iree_hal_fence_create_at(semaphore, 2ull, device_host_allocator, &fence_t2));
  iree_hal_semaphore_release(semaphore);
  semaphore = nullptr;

  fprintf(stdout, "[%s] INITIALIZE T=0\n", tag);
  fflush(stdout);

  // Push wait fence (T=1) and signal fence (T=2)
  {
    iree_vm_ref_t ref = iree_hal_fence_retain_ref(fence_t1);
    CHECK_OK(iree_vm_list_push_ref_move(inputs, &ref));
  }
  {
    iree_vm_ref_t ref = iree_hal_fence_retain_ref(fence_t2);
    CHECK_OK(iree_vm_list_push_ref_move(inputs, &ref));
  }

  // Workaround like sample: signal T=1 before invoking.
  CHECK_OK(iree_hal_fence_signal(fence_t1));
  fprintf(stdout, "[%s] SIGNALED T=1\n", tag);
  fflush(stdout);

  // Invoke function (should return after scheduling).
  iree_string_view_t ep = iree_make_cstring_view(function_name);
  fprintf(stdout, "[%s] VM INVOKE BEGIN %.*s\n", tag, (int)ep.size, ep.data);
  fflush(stdout);

  auto t0 = std::chrono::steady_clock::now();
  CHECK_OK(iree_runtime_session_call_by_name(session, ep, inputs, outputs));
  auto t1 = std::chrono::steady_clock::now();

  fprintf(stdout, "[%s] VM INVOKE END (returned in %.3f ms)\n",
          tag,
          std::chrono::duration<double, std::milli>(t1 - t0).count());
  fflush(stdout);

  // Wait for completion fence (T=2).
  CHECK_OK(iree_hal_fence_wait(fence_t2, iree_infinite_timeout(),
                              IREE_HAL_WAIT_FLAG_DEFAULT));
  fprintf(stdout, "[%s] REACHED T=2\n", tag);
  fflush(stdout);

  // Validate outputs are buffer_views and print basic info.
  iree_host_size_t out_count = iree_vm_list_size(outputs);
  fprintf(stdout, "[%s] outputs=%" PRIu64 "\n", tag, (uint64_t)out_count);

  for (iree_host_size_t i = 0; i < out_count; ++i) {
    iree_vm_ref_t out_ref = iree_vm_ref_null();
    CHECK_OK(iree_vm_list_get_ref_assign(outputs, i, &out_ref));
    if (!iree_hal_buffer_view_isa(out_ref)) {
      fprintf(stderr, "[%s] output %" PRIu64 " is not a buffer_view\n",
              tag, (uint64_t)i);
      return 1;
    }
    iree_hal_buffer_view_t* out_view = iree_hal_buffer_view_deref(out_ref);
    char out_tag[64];
    std::snprintf(out_tag, sizeof(out_tag), "%s:out%" PRIu64, tag, (uint64_t)i);
    PrintBufferViewInfo(out_tag, out_view, dump_output_bytes);
  }

  // Cleanup.
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_fence_release(fence_t1);
  iree_hal_fence_release(fence_t2);
  iree_hal_buffer_view_release(input_view);
  return 0;
}

}  // namespace

extern "C" int merlin_dual_model_async_gate_run(
    const merlin_dual_model_async_gate_config_t* config) {
  if (!config || !config->dronet_vmfb_path || !config->mlp_vmfb_path ||
      !config->dronet_function || !config->mlp_function || !config->driver_name) {
    fprintf(stderr, "Invalid config.\n");
    return 1;
  }

  iree_allocator_t host_allocator = iree_allocator_system();

  // Instance.
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = nullptr;
  CHECK_OK(iree_runtime_instance_create(&instance_options, host_allocator, &instance));

  // Device.
  iree_hal_device_t* device = nullptr;
  CHECK_OK(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view(config->driver_name), &device));

  // Session.
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = nullptr;
  CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));

  // Load both modules into same session.
  CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(
      session, config->dronet_vmfb_path));
  CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(
      session, config->mlp_vmfb_path));

  // Use the shapes you’ve been using everywhere.
  const std::vector<iree_hal_dim_t> dronet_shape = {1, 3, 112, 112};
  const std::vector<iree_hal_dim_t> mlp_shape = {1, 10};

  // Call both (you can swap order if you want).
  int rc = 0;
  rc = CallOneModelGated(session, "mlp", config->mlp_function, mlp_shape,
                         config->dump_output_bytes);
  if (rc == 0) {
    rc = CallOneModelGated(session, "dronet", config->dronet_function, dronet_shape,
                           config->dump_output_bytes);
  }

  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  return rc;
}