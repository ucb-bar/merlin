// main.c
#include <stdio.h>
#include <stdlib.h>
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/runtime/api.h"

// IREE C API is C, so no extern "C" needed
iree_status_t iree_hal_local_sync_driver_module_register(
    iree_hal_driver_registry_t* registry);
iree_status_t iree_allocator_libc_ctl(
    void* self, iree_allocator_command_t command,
    const void* params, void** inout_ptr);

// Helper to create an input tensor
iree_status_t create_input_tensor(iree_hal_device_t* device, 
                                  iree_hal_buffer_view_t** out_buffer_view) {
  const iree_hal_dim_t shape[] = {1, 1024}; // model's input shape
  iree_host_size_t shape_rank = 2;
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;

  // Create some dummy float data
  static float input_data[1024];
  for(int i = 0; i < 1024; ++i) {
    input_data[i] = (float)i / 1024.0f;
  }

  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

  iree_host_size_t byte_length = 1024 * sizeof(float);
  return iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), shape_rank, shape, element_type, 
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params, 
      iree_make_const_byte_span(input_data, byte_length), out_buffer_view);
}

int main(int argc, char** argv) {
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  iree_allocator_t host_allocator = { .self = NULL, .ctl = iree_allocator_libc_ctl };
  IREE_CHECK_OK(iree_runtime_instance_create(&instance_options, host_allocator, &instance));

  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(instance, 
      iree_make_cstring_view("local-sync"), &device));

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  IREE_CHECK_OK(iree_runtime_session_create_with_device(instance, &session_options, 
      device, iree_runtime_instance_host_allocator(instance), &session));

  // MODEL_VMFB_FILENAME is set by CMake
  printf("Loading model: %s\n", MODEL_VMFB_FILENAME);
  IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(
      session, MODEL_VMFB_FILENAME));
  
  iree_runtime_call_t call;
  IREE_CHECK_OK(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.main_graph$async"), &call));

  // Create and push inputs
  iree_hal_buffer_view_t* input_view = NULL;
  IREE_CHECK_OK(create_input_tensor(device, &input_view));
  IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, input_view));
  iree_hal_buffer_view_release(input_view);
  
  // (Note: Your MLIR has 3 inputs: %arg0, %arg1, %arg2. %arg1 and %arg2 are 
  // !hal.fence. We are in local-sync, so we can pass null fences)
  IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, NULL));
  IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, NULL));

  printf("Invoking model...\n");
  IREE_CHECK_OK(iree_runtime_call_invoke(&call, /*flags=*/0));

  // Get and print output
  iree_hal_buffer_view_t* output_view = NULL;
  IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &output_view));

  iree_hal_buffer_mapping_t mapped_memory;
  IREE_CHECK_OK(iree_hal_buffer_map_range(iree_hal_buffer_view_buffer(output_view), 
      IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 
      0, IREE_HAL_WHOLE_BUFFER, &mapped_memory));

  printf("Output (tensor<1x10xf32>):\n[ ");
  const float* output_data = (const float*)mapped_memory.contents.data;
  for(int i = 0; i < 10; ++i) {
    printf("%f ", output_data[i]);
  }
  printf("]\n");

  iree_hal_buffer_unmap_range(&mapped_memory);
  iree_hal_buffer_view_release(output_view);
  iree_runtime_call_deinitialize(&call);
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  printf("Run complete.\n");
  return 0;
}