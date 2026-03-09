#include "xpurt_scheduler_core.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "iree/hal/api.h"
#include "iree/runtime/api.h"

typedef struct {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_allocator_t host_allocator;
} xpurt_iree_context_t;

typedef struct {
  iree_runtime_session_t* session;
  iree_vm_function_t entry_fn;
  iree_vm_list_t* inputs;
} xpurt_iree_node_t;

static iree_status_t pick_dispatch_entry_function(iree_vm_module_t* module,
                                                  iree_vm_function_t* out_fn) {
  *out_fn = (iree_vm_function_t){0};
  const iree_vm_module_signature_t sig = iree_vm_module_signature(module);
  if (sig.export_function_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "module has no exported functions");
  }
  if (sig.export_function_count == 1) {
    return iree_vm_module_lookup_function_by_ordinal(
        module, IREE_VM_FUNCTION_LINKAGE_EXPORT, 0, out_fn);
  }
  for (iree_host_size_t i = 0; i < sig.export_function_count; ++i) {
    iree_vm_function_t fn = {0};
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &fn));
    const iree_vm_function_signature_t fsig = iree_vm_function_signature(&fn);
    const iree_string_view_t cc = fsig.calling_convention;
    if (cc.size >= 2 && cc.data[0] == '0' && cc.data[1] == 'i') {
      const char* u = memchr(cc.data, '_', cc.size);
      if (!u || (size_t)(u - cc.data) == 2) {
        *out_fn = fn;
        return iree_ok_status();
      }
    }
  }
  return iree_vm_module_lookup_function_by_ordinal(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT, 0, out_fn);
}

static iree_status_t prepare_node(iree_runtime_instance_t* instance,
                                  iree_hal_device_t* device,
                                  const char* vmfb_path,
                                  int32_t dispatch_iters,
                                  iree_allocator_t host_allocator,
                                  xpurt_iree_node_t* node) {
  node->session = NULL;
  node->entry_fn = (iree_vm_function_t){0};
  node->inputs = NULL;

  iree_runtime_session_options_t opts;
  iree_runtime_session_options_initialize(&opts);
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      instance, &opts, device, host_allocator, &node->session));
  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      node->session, vmfb_path));

  iree_vm_context_t* ctx = iree_runtime_session_context(node->session);
  iree_host_size_t nmod = iree_vm_context_module_count(ctx);
  if (nmod == 0) {
    iree_runtime_session_release(node->session);
    node->session = NULL;
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "no modules in session");
  }
  iree_vm_module_t* mod = iree_vm_context_module_at(ctx, nmod - 1);
  IREE_RETURN_IF_ERROR(pick_dispatch_entry_function(mod, &node->entry_fn));

  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           1, host_allocator, &node->inputs));
  iree_vm_value_t v = iree_vm_value_make_i32(dispatch_iters);
  IREE_RETURN_IF_ERROR(iree_vm_list_push_value(node->inputs, &v));
  return iree_ok_status();
}

static iree_status_t execute_node(xpurt_iree_node_t* node) {
  return iree_runtime_session_call(node->session, &node->entry_fn, node->inputs, NULL);
}

static void release_node(xpurt_iree_node_t* node) {
  if (!node) return;
  iree_vm_list_release(node->inputs);
  iree_runtime_session_release(node->session);
  node->inputs = NULL;
  node->session = NULL;
}

xpurt_status_t xpurt_run_dispatch_graph(const char* json_path, const char* driver_name) {
  if (!json_path || !driver_name) return XPURT_STATUS_ERROR;

  xpurt_graph_t* graph = NULL;
  if (xpurt_graph_load(json_path, &graph) != XPURT_STATUS_OK) return XPURT_STATUS_ERROR;
  size_t node_count = xpurt_graph_node_count(graph);
  const int* order = xpurt_graph_execution_order(graph);
  if (!order || node_count == 0) {
    xpurt_graph_destroy(graph);
    return XPURT_STATUS_ERROR;
  }

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_runtime_instance_t* instance = NULL;
  iree_hal_device_t* device = NULL;
  xpurt_iree_node_t* prepared = (xpurt_iree_node_t*)calloc(node_count, sizeof(xpurt_iree_node_t));
  xpurt_status_t result = XPURT_STATUS_OK;
  iree_status_t status = iree_ok_status();

  if (!prepared) {
    result = XPURT_STATUS_ERROR;
    goto cleanup_graph;
  }

  iree_runtime_instance_options_t inst_opts;
  iree_runtime_instance_options_initialize(&inst_opts);
  iree_runtime_instance_options_use_all_available_drivers(&inst_opts);
  status = iree_runtime_instance_create(&inst_opts, host_allocator, &instance);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    result = XPURT_STATUS_ERROR;
    goto cleanup_graph;
  }
  status = iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view(driver_name), &device);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    result = XPURT_STATUS_ERROR;
    goto cleanup_instance;
  }

  fprintf(stdout, "Config:\n  graph_json = %s\n  driver     = %s\n", json_path, driver_name);
  fprintf(stdout, "Dispatch execution order (%zu nodes):\n", node_count);
  for (size_t i = 0; i < node_count; ++i) {
    size_t idx = (size_t)order[i];
    const char* key = xpurt_graph_node_key(graph, idx);
    int id = xpurt_graph_node_id(graph, idx);
    fprintf(stdout, "  %zu) %s (id=%d)\n", i + 1, key ? key : "?", id);
  }
  fflush(stdout);

  for (size_t i = 0; i < node_count; ++i) {
    size_t idx = (size_t)order[i];
    const char* vmfb_path = xpurt_graph_node_vmfb_path(graph, idx);
    if (!vmfb_path) {
      result = XPURT_STATUS_ERROR;
      goto cleanup_prepared;
    }
    status = prepare_node(instance, device, vmfb_path, 1, host_allocator, &prepared[idx]);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      result = XPURT_STATUS_ERROR;
      goto cleanup_prepared;
    }
  }

  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  for (size_t i = 0; i < node_count; ++i) {
    size_t idx = (size_t)order[i];
    status = execute_node(&prepared[idx]);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      result = XPURT_STATUS_ERROR;
      break;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  if (result == XPURT_STATUS_OK) {
    uint64_t ns = (uint64_t)(ts1.tv_sec - ts0.tv_sec) * 1000000000u
                  + (uint64_t)(ts1.tv_nsec - ts0.tv_nsec);
    double ms = (double)ns / 1000000.0;
    fprintf(stdout, "Run complete:\n  total_wall_ms=%.3f\n", ms);
  }
  fprintf(stdout, "Done.\n");
  fflush(stdout);

cleanup_prepared:
  for (size_t i = 0; i < node_count; ++i) release_node(&prepared[i]);
  free(prepared);
  iree_hal_device_release(device);
cleanup_instance:
  iree_runtime_instance_release(instance);
cleanup_graph:
  xpurt_graph_destroy(graph);
  return result;
}
