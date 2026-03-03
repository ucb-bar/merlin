// Copyright 2024 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// IREE Runtime & HAL Headers
#include "iree/runtime/api.h"
#include "iree/base/api.h"
#include "iree/modules/hal/module.h"  // Required for manual multi-device setup
#include "iree/modules/hal/types.h"

// IREE Task System Headers (Internal)
#include "iree/task/api.h"
#include "iree/task/topology.h"
#include "iree/hal/drivers/local_task/driver.h"
#include "iree/base/internal/threading.h"

// ============================================================================
// 1. Topology Helpers (Manual Masking)
// ============================================================================

// Helper to convert a bitmask (e.g., 0b101) into an IREE topology
void iree_task_topology_initialize_from_mask(iree_task_topology_t* out_topology,
                                             uint64_t mask) {
  iree_task_topology_initialize(out_topology);

  for (int core_id = 0; core_id < 64; ++core_id) {
    if ((mask >> core_id) & 1) {
      iree_task_topology_group_t group;
      iree_task_topology_group_initialize(out_topology->group_count, &group);
      
      group.processor_index = core_id;
      memset(&group.ideal_thread_affinity, 0, sizeof(group.ideal_thread_affinity));
      iree_thread_affinity_set_bit(&group.ideal_thread_affinity, core_id);

      iree_status_t status = iree_task_topology_push_group(out_topology, &group);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        break;
      }
    }
  }
}

// Helper to create a fully configured HAL device pinned to specific cores
iree_status_t create_device_with_mask(iree_allocator_t host_allocator,
                                      uint64_t core_mask,
                                      const char* label,
                                      iree_hal_device_t** out_device) {
  iree_task_topology_t topology;
  iree_task_topology_initialize_from_mask(&topology, core_mask);

  iree_task_executor_options_t options = iree_task_executor_options_default();
  options.worker_local_memory_size = 64 * 1024; // 64KB local memory per worker

  iree_task_executor_t* executor = NULL;
  IREE_RETURN_IF_ERROR(iree_task_executor_create(options, &topology, 
                                                 host_allocator, &executor));

  iree_hal_task_device_params_t params = iree_hal_task_device_params_default();
  
  // Create the device using the pinned executor
  // Note: queue_count=1 is sufficient; the executor handles the thread pool width internally.
  iree_status_t status = iree_hal_task_device_create(
      iree_make_cstring_view(label), &params, executor, 1, NULL,
      host_allocator, out_device);

  iree_task_executor_release(executor);
  iree_task_topology_deinitialize(&topology);
  return status;
}

// ============================================================================
// 2. Main Execution
// ============================================================================

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <path_to_vmfb> <entry_point_function>\n", argv[0]);
    return -1;
  }

  const char* module_path = argv[1];
  const char* entry_point_name = argv[2];

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_runtime_instance_options_t instance_opts;
  iree_runtime_instance_options_initialize(&instance_opts);
  
  iree_runtime_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_runtime_instance_create(&instance_opts, host_allocator, &instance));

  fprintf(stdout, "--- Creating Hardware Devices ---\n");

  // 1. Create Core 0 Device (device_a)
  iree_hal_device_t* device_a = NULL;
  IREE_CHECK_OK(create_device_with_mask(host_allocator, 1, "device_a", &device_a));

  // 2. Create Core 1 Device (device_b)
  iree_hal_device_t* device_b = NULL;
  IREE_CHECK_OK(create_device_with_mask(host_allocator, 2, "device_b", &device_b));

  // 3. Create Cluster Device (device_ab)
  iree_hal_device_t* device_ab = NULL;
  IREE_CHECK_OK(create_device_with_mask(host_allocator, 3, "device_ab", &device_ab));

  // --- Session Setup ---
  
  iree_runtime_session_options_t session_opts;
  iree_runtime_session_options_initialize(&session_opts);
  iree_runtime_session_t* session = NULL;
  
  // A. Create a bare session (no default device)
  IREE_CHECK_OK(iree_runtime_session_create(instance, &session_opts, host_allocator, &session));

  // B. Register the HAL Module with ALL devices
  // This allows the VMFB to resolve @device_a, @device_b, etc.
  // The order here maps to the compiler's device index (0, 1, 2...).
  iree_hal_device_t* devices[] = {device_a, device_b, device_ab};
  iree_vm_module_t* hal_module = NULL;
  
  IREE_CHECK_OK(iree_hal_module_create(
      iree_runtime_instance_vm_instance(instance),
      IREE_ARRAYSIZE(devices), devices,
      IREE_HAL_MODULE_FLAG_NONE,
      iree_runtime_instance_host_allocator(instance),
      &hal_module));

  IREE_CHECK_OK(iree_runtime_session_append_module(session, hal_module));
  iree_vm_module_release(hal_module);

  // --- Load User Module (VMFB) ---

  fprintf(stdout, "Loading module: %s...\n", module_path);
  IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(session, module_path));

  // --- Prepare Call Inputs ---

  iree_vm_list_t* inputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1, host_allocator, &inputs));
  
  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1, host_allocator, &outputs));

  // Data: Input tensor<4xf32> = {1.0, 2.0, 3.0, 4.0}
  const float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const iree_hal_dim_t shape[] = {4};

  // Allocating on Device A (Core 0) as required by the example MLIR
  iree_hal_buffer_view_t* input_buffer = NULL;
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
      device_a, iree_hal_device_allocator(device_a), IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(input_data, sizeof(input_data)), &input_buffer));

  // Push input to list
  iree_vm_ref_t input_ref = iree_hal_buffer_view_move_ref(input_buffer);
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &input_ref));

  // --- Invoke ---

  fprintf(stdout, "Invoking '%s'...\n", entry_point_name);
  IREE_CHECK_OK(iree_runtime_session_call_by_name(
      session, iree_make_cstring_view(entry_point_name), inputs, outputs));

  // --- Process Outputs ---

  fprintf(stdout, "Execution Complete. Reading output...\n");
  
  // The result might be on a different device (Core 3 per your MLIR), 
  // but iree_hal_buffer_map_read handles mapping regardless of device location
  // for local-task devices (shared memory).
  iree_hal_buffer_view_t* output_view = iree_vm_list_get_buffer_view_assign(outputs, 0);
  
  float result_data[4] = {0};
  IREE_CHECK_OK(iree_hal_buffer_map_read(
      iree_hal_buffer_view_buffer(output_view), 0,
      result_data, sizeof(result_data)));

  fprintf(stdout, "Result: [%.1f, %.1f, %.1f, %.1f]\n", 
          result_data[0], result_data[1], result_data[2], result_data[3]);

  // --- Cleanup ---
  
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_runtime_session_release(session);
  iree_hal_device_release(device_a);
  iree_hal_device_release(device_b);
  iree_hal_device_release(device_ab);
  iree_runtime_instance_release(instance);

  return 0;
}