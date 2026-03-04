#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "iree/modules/hal/types.h"
#include "iree/hal/api.h"
#include "iree/runtime/api.h"

typedef struct invocation_stats_t {
  uint64_t count;
  uint64_t total_ns;
  uint64_t min_ns;
  uint64_t max_ns;
} invocation_stats_t;

static void print_usage(const char* argv0) {
  fprintf(stderr,
          "Usage:\n"
          "  %s <dronet.vmfb> <mlp.vmfb> [dronet_fn] [mlp_fn] [driver] [iters] "
          "[report_every]\n"
          "\n"
          "Defaults:\n"
          "  dronet_fn = dronet.main_graph$async\n"
          "  mlp_fn    = mlp.main_graph$async\n"
          "  driver    = local-task\n"
          "  iters     = 1\n"
          "  report_every = 0 (only print final summary)\n"
          "\n"
          "Notes:\n"
          "  Loads both VMFBs into one IREE session and runs dronet then mlp\n"
          "  sequentially.\n",
          argv0);
}

static int parse_int_or_default(const char* text, int default_value) {
  if (!text || !text[0]) return default_value;
  char* end = NULL;
  long v = strtol(text, &end, 10);
  if (end == text) return default_value;
  if (v < 1) v = 1;
  if (v > 1000000) v = 1000000;
  return (int)v;
}

static uint64_t now_monotonic_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void stats_update(invocation_stats_t* stats, uint64_t elapsed_ns) {
  if (!stats) return;
  if (stats->count == 0) {
    stats->min_ns = elapsed_ns;
    stats->max_ns = elapsed_ns;
  } else {
    if (elapsed_ns < stats->min_ns) stats->min_ns = elapsed_ns;
    if (elapsed_ns > stats->max_ns) stats->max_ns = elapsed_ns;
  }
  stats->count++;
  stats->total_ns += elapsed_ns;
}

static double ns_to_ms(uint64_t ns) { return (double)ns / 1000000.0; }

static void stats_print(const char* name, const invocation_stats_t* stats) {
  if (!name || !stats) return;
  const double avg_ms =
      (stats->count > 0) ? ns_to_ms(stats->total_ns) / (double)stats->count : 0.0;
  fprintf(stdout,
          "  %s: count=%" PRIu64 " avg=%.3fms min=%.3fms max=%.3fms\n", name,
          stats->count, avg_ms, ns_to_ms(stats->min_ns), ns_to_ms(stats->max_ns));
}

static iree_status_t create_f32_input_view_from_data(
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    const float* input_data, iree_host_size_t element_count,
    iree_hal_buffer_view_t** out_view) {
  *out_view = NULL;

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

static iree_status_t run_async_external_once(iree_runtime_session_t* session,
                                             const char* function_name,
                                             iree_hal_semaphore_t* timeline,
                                             uint64_t epoch,
                                             iree_hal_buffer_view_t* input_view,
                                             iree_host_size_t expected_outputs,
                                             iree_allocator_t host_allocator) {
  iree_status_t status = iree_ok_status();
  iree_vm_list_t* inputs = NULL;
  iree_vm_list_t* outputs = NULL;
  iree_hal_fence_t* empty_wait_fence = NULL;
  iree_hal_fence_t* signal_fence = NULL;

  status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                               /*capacity=*/3, host_allocator, &inputs);
  if (!iree_status_is_ok(status)) goto cleanup;
  status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                               /*capacity=*/8, host_allocator, &outputs);
  if (!iree_status_is_ok(status)) goto cleanup;

  // input buffer_view
  {
    iree_vm_ref_t input_ref = iree_hal_buffer_view_retain_ref(input_view);
    status = iree_vm_list_push_ref_move(inputs, &input_ref);
    iree_vm_ref_release(&input_ref);
    if (!iree_status_is_ok(status)) goto cleanup;
  }

  // empty wait fence (no deps)
  status = iree_hal_fence_create(/*capacity=*/0, host_allocator,
                                 &empty_wait_fence);
  if (!iree_status_is_ok(status)) goto cleanup;
  {
    iree_vm_ref_t wait_fence_ref =
        iree_hal_fence_retain_ref(empty_wait_fence);
    status = iree_vm_list_push_ref_move(inputs, &wait_fence_ref);
    iree_vm_ref_release(&wait_fence_ref);
    if (!iree_status_is_ok(status)) goto cleanup;
  }

  // signal fence at (timeline, epoch)
  status = iree_hal_fence_create_at(timeline, epoch, host_allocator,
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

  // Wait for completion (coarse-fences).
  status = iree_hal_semaphore_wait(timeline, epoch, iree_infinite_timeout(),
                                  IREE_HAL_WAIT_FLAG_DEFAULT);
  if (!iree_status_is_ok(status)) goto cleanup;

  if (expected_outputs > 0) {
    const iree_host_size_t got = iree_vm_list_size(outputs);
    if (got != expected_outputs) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "expected %" PRIu64 " outputs but got %" PRIu64,
                                (uint64_t)expected_outputs, (uint64_t)got);
      goto cleanup;
    }
  }

cleanup:
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_fence_release(signal_fence);
  iree_hal_fence_release(empty_wait_fence);
  return status;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  const char* dronet_vmfb = argv[1];
  const char* mlp_vmfb = argv[2];
  const char* dronet_fn = (argc >= 4) ? argv[3] : "dronet.main_graph$async";
  const char* mlp_fn = (argc >= 5) ? argv[4] : "mlp.main_graph$async";
  const char* driver_name = (argc >= 6) ? argv[5] : "local-task";
  const int iters = (argc >= 7) ? parse_int_or_default(argv[6], 1) : 1;
  const int report_every = (argc >= 8) ? parse_int_or_default(argv[7], 0) : 0;

  // Fixed shapes (matching the older scheduler sample).
  static const iree_hal_dim_t dronet_shape[4] = {1, 3, 112, 112};
  static const iree_host_size_t dronet_rank = 4;
  static const iree_host_size_t dronet_elems = 1 * 3 * 112 * 112;
  static const iree_hal_dim_t mlp_shape[2] = {1, 10};
  static const iree_host_size_t mlp_rank = 2;
  static const iree_host_size_t mlp_elems = 1 * 10;

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();

  iree_runtime_instance_t* instance = NULL;
  iree_hal_device_t* device = NULL;
  iree_runtime_session_t* session = NULL;
  iree_hal_semaphore_t* timeline = NULL;

  float* dronet_input = NULL;
  float* mlp_input = NULL;
  iree_hal_buffer_view_t* dronet_input_view = NULL;
  iree_hal_buffer_view_t* mlp_input_view = NULL;

  uint64_t epoch = 0;
  invocation_stats_t dronet_stats = {0};
  invocation_stats_t mlp_stats = {0};

  fprintf(stdout,
          "Config:\n"
          "  dronet_vmfb = %s\n"
          "  mlp_vmfb    = %s\n"
          "  dronet_fn   = %s\n"
          "  mlp_fn      = %s\n"
          "  driver      = %s\n"
          "  iters       = %d\n"
          "  report_every= %d\n",
          dronet_vmfb, mlp_vmfb, dronet_fn, mlp_fn, driver_name, iters,
          report_every);
  fflush(stdout);

  dronet_input = (float*)malloc(sizeof(float) * dronet_elems);
  mlp_input = (float*)malloc(sizeof(float) * mlp_elems);
  if (!dronet_input || !mlp_input) {
    fprintf(stderr, "malloc failed\n");
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "malloc failed");
    goto cleanup;
  }
  for (iree_host_size_t i = 0; i < dronet_elems; ++i) dronet_input[i] = 0.01f;
  for (iree_host_size_t i = 0; i < mlp_elems; ++i) mlp_input[i] = 0.25f;

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  status =
      iree_runtime_instance_create(&instance_options, host_allocator, &instance);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view(driver_name), &device);
  if (!iree_status_is_ok(status)) goto cleanup;

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  status = iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session);
  if (!iree_status_is_ok(status)) goto cleanup;

  status =
      iree_runtime_session_append_bytecode_module_from_file(session, dronet_vmfb);
  if (!iree_status_is_ok(status)) goto cleanup;
  status =
      iree_runtime_session_append_bytecode_module_from_file(session, mlp_vmfb);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                     /*initial_value=*/0ull,
                                     IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                     &timeline);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = create_f32_input_view_from_data(
      device, iree_runtime_session_device_allocator(session), dronet_shape,
      dronet_rank, dronet_input, dronet_elems, &dronet_input_view);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = create_f32_input_view_from_data(
      device, iree_runtime_session_device_allocator(session), mlp_shape, mlp_rank,
      mlp_input, mlp_elems, &mlp_input_view);
  if (!iree_status_is_ok(status)) goto cleanup;

  const uint64_t run_start_ns = now_monotonic_ns();
  for (int i = 0; i < iters; ++i) {
    uint64_t t0 = 0, t1 = 0;

    epoch++;
    t0 = now_monotonic_ns();
    status = run_async_external_once(session, dronet_fn, timeline, epoch,
                                     dronet_input_view,
                                     /*expected_outputs=*/2, host_allocator);
    t1 = now_monotonic_ns();
    stats_update(&dronet_stats, t1 - t0);
    if (!iree_status_is_ok(status)) goto cleanup;

    epoch++;
    t0 = now_monotonic_ns();
    status = run_async_external_once(session, mlp_fn, timeline, epoch,
                                     mlp_input_view,
                                     /*expected_outputs=*/1, host_allocator);
    t1 = now_monotonic_ns();
    stats_update(&mlp_stats, t1 - t0);
    if (!iree_status_is_ok(status)) goto cleanup;

    if (report_every > 0 && ((i + 1) % report_every) == 0) {
      fprintf(stdout, "[iter %d/%d]\n", i + 1, iters);
      stats_print("dronet", &dronet_stats);
      stats_print("mlp", &mlp_stats);
      fflush(stdout);
    }
  }
  const uint64_t run_end_ns = now_monotonic_ns();
  const uint64_t total_ns = run_end_ns - run_start_ns;
  const double total_s = (double)total_ns / 1000000000.0;
  const double pairs_per_s = (total_s > 0.0) ? ((double)iters / total_s) : 0.0;

  fprintf(stdout,
          "Run complete:\n"
          "  total_wall_ms=%.3f\n"
          "  iter_pairs_per_s=%.3f\n",
          ns_to_ms(total_ns), pairs_per_s);
  stats_print("dronet", &dronet_stats);
  stats_print("mlp", &mlp_stats);
  fprintf(stdout, "Done.\n");
  fflush(stdout);

cleanup:
  iree_hal_buffer_view_release(mlp_input_view);
  iree_hal_buffer_view_release(dronet_input_view);
  iree_hal_semaphore_release(timeline);
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  free(dronet_input);
  free(mlp_input);
  

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}
