#include <inttypes.h>
#include <stdbool.h>
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
          "  %s <dispatch_graph.json> [driver] [graph_iters] [dispatch_iters] "
          "[report_every] [warmup_graph_iters]\n"
          "\n"
          "Defaults:\n"
          "  driver    = local-task\n"
          "  graph_iters    = 1\n"
          "  dispatch_iters = 1 (passed as i32 arg to each dispatch stub)\n"
          "  report_every = 0 (only print final summary)\n"
          "  warmup_graph_iters = 0 (excluded from metrics)\n"
          "\n"
          "Notes:\n"
          "  Preloads all per-dispatch benchmark VMFBs and prepares their sessions\n"
          "  once, then times only the dispatch executions in dependency order.\n",
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

//===----------------------------------------------------------------------===//
// Minimal JSON parser (only what we need for the dispatch graph format)
//===----------------------------------------------------------------------===//

typedef struct json_parser_t {
  const char* p;
  const char* end;
} json_parser_t;

static void json_skip_ws(json_parser_t* jp) {
  while (jp->p < jp->end) {
    const char c = *jp->p;
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
      ++jp->p;
      continue;
    }
    break;
  }
}

static bool json_consume(json_parser_t* jp, char expected) {
  json_skip_ws(jp);
  if (jp->p >= jp->end || *jp->p != expected) return false;
  ++jp->p;
  return true;
}

static bool json_parse_string(json_parser_t* jp, char** out_str) {
  *out_str = NULL;
  json_skip_ws(jp);
  if (jp->p >= jp->end || *jp->p != '"') return false;
  ++jp->p;
  const char* start = jp->p;
  size_t len = 0;
  bool has_escapes = false;
  while (jp->p < jp->end) {
    char c = *jp->p++;
    if (c == '"') break;
    if (c == '\\') {
      has_escapes = true;
      if (jp->p >= jp->end) return false;
      ++jp->p;  // skip escaped char
      len += 1;
      continue;
    }
    len += 1;
  }
  if (jp->p > jp->end) return false;
  const char* raw_end = jp->p - 1;
  if (!has_escapes) {
    size_t raw_len = (size_t)(raw_end - start);
    char* s = (char*)malloc(raw_len + 1);
    if (!s) return false;
    memcpy(s, start, raw_len);
    s[raw_len] = 0;
    *out_str = s;
    return true;
  }

  // Unescape minimal set used in our JSON (\", \\ , \n, \r, \t).
  char* s = (char*)malloc(len + 1);
  if (!s) return false;
  const char* r = start;
  char* w = s;
  while (r < raw_end) {
    char c = *r++;
    if (c != '\\') {
      *w++ = c;
      continue;
    }
    if (r >= raw_end) break;
    char e = *r++;
    switch (e) {
      case '"':
        *w++ = '"';
        break;
      case '\\':
        *w++ = '\\';
        break;
      case 'n':
        *w++ = '\n';
        break;
      case 'r':
        *w++ = '\r';
        break;
      case 't':
        *w++ = '\t';
        break;
      default:
        *w++ = e;
        break;
    }
  }
  *w = 0;
  *out_str = s;
  return true;
}

static bool json_parse_int(json_parser_t* jp, int* out_value) {
  *out_value = 0;
  json_skip_ws(jp);
  if (jp->p >= jp->end) return false;
  const char* s = jp->p;
  char* endptr = NULL;
  long v = strtol(s, &endptr, 10);
  if (endptr == s) return false;
  jp->p = endptr;
  *out_value = (int)v;
  return true;
}

static bool json_skip_value(json_parser_t* jp);  // fwd

static bool json_skip_array(json_parser_t* jp) {
  if (!json_consume(jp, '[')) return false;
  json_skip_ws(jp);
  if (json_consume(jp, ']')) return true;
  while (true) {
    if (!json_skip_value(jp)) return false;
    json_skip_ws(jp);
    if (json_consume(jp, ']')) return true;
    if (!json_consume(jp, ',')) return false;
  }
}

static bool json_skip_object(json_parser_t* jp) {
  if (!json_consume(jp, '{')) return false;
  json_skip_ws(jp);
  if (json_consume(jp, '}')) return true;
  while (true) {
    char* key = NULL;
    if (!json_parse_string(jp, &key)) return false;
    free(key);
    if (!json_consume(jp, ':')) return false;
    if (!json_skip_value(jp)) return false;
    json_skip_ws(jp);
    if (json_consume(jp, '}')) return true;
    if (!json_consume(jp, ',')) return false;
  }
}

static bool json_skip_value(json_parser_t* jp) {
  json_skip_ws(jp);
  if (jp->p >= jp->end) return false;
  char c = *jp->p;
  if (c == '"') {
    char* s = NULL;
    bool ok = json_parse_string(jp, &s);
    free(s);
    return ok;
  } else if (c == '{') {
    return json_skip_object(jp);
  } else if (c == '[') {
    return json_skip_array(jp);
  } else if ((c >= '0' && c <= '9') || c == '-') {
    int dummy = 0;
    return json_parse_int(jp, &dummy);
  } else if (!strncmp(jp->p, "true", 4)) {
    jp->p += 4;
    return true;
  } else if (!strncmp(jp->p, "false", 5)) {
    jp->p += 5;
    return true;
  } else if (!strncmp(jp->p, "null", 4)) {
    jp->p += 4;
    return true;
  }
  return false;
}

typedef struct dispatch_node_t {
  char* key;         // dispatch_0, dispatch_16_2, ...
  int id;            // numeric dispatch id
  char* vmfb_path;   // resolved filesystem path
  char** deps;       // array of keys this depends on
  size_t dep_count;
  invocation_stats_t stats;
  // Prepared execution state (constructed once before timing).
  iree_runtime_session_t* session;
  iree_vm_function_t entry_fn;
  iree_vm_list_t* inputs;  // contains the dispatch_iters i32
} dispatch_node_t;

static void dispatch_node_deinit(dispatch_node_t* n) {
  if (!n) return;
  iree_vm_list_release(n->inputs);
  iree_runtime_session_release(n->session);
  free(n->key);
  free(n->vmfb_path);
  for (size_t i = 0; i < n->dep_count; ++i) free(n->deps[i]);
  free(n->deps);
}

static char* path_dirname(const char* path) {
  if (!path) return NULL;
  const char* last_slash = strrchr(path, '/');
  if (!last_slash) {
    char* out = (char*)malloc(2);
    if (!out) return NULL;
    out[0] = '.';
    out[1] = 0;
    return out;
  }
  size_t len = (size_t)(last_slash - path);
  if (len == 0) len = 1;  // root '/'
  char* out = (char*)malloc(len + 1);
  if (!out) return NULL;
  memcpy(out, path, len);
  out[len] = 0;
  return out;
}

static char* path_join2(const char* a, const char* b) {
  if (!a || !a[0]) return b ? strdup(b) : NULL;
  if (!b || !b[0]) return strdup(a);
  if (b[0] == '/') return strdup(b);
  size_t la = strlen(a);
  size_t lb = strlen(b);
  bool need_slash = (la > 0 && a[la - 1] != '/');
  char* out = (char*)malloc(la + (need_slash ? 1 : 0) + lb + 1);
  if (!out) return NULL;
  memcpy(out, a, la);
  size_t pos = la;
  if (need_slash) out[pos++] = '/';
  memcpy(out + pos, b, lb);
  out[pos + lb] = 0;
  return out;
}

static bool json_parse_dependencies_array(json_parser_t* jp, char*** out_deps,
                                          size_t* out_dep_count) {
  *out_deps = NULL;
  *out_dep_count = 0;
  if (!json_consume(jp, '[')) return false;
  json_skip_ws(jp);
  if (json_consume(jp, ']')) return true;
  size_t cap = 4;
  char** deps = (char**)calloc(cap, sizeof(char*));
  if (!deps) return false;
  size_t count = 0;
  while (true) {
    char* dep = NULL;
    if (!json_parse_string(jp, &dep)) {
      for (size_t i = 0; i < count; ++i) free(deps[i]);
      free(deps);
      return false;
    }
    if (count == cap) {
      cap *= 2;
      char** nd = (char**)realloc(deps, cap * sizeof(char*));
      if (!nd) {
        free(dep);
        for (size_t i = 0; i < count; ++i) free(deps[i]);
        free(deps);
        return false;
      }
      deps = nd;
    }
    deps[count++] = dep;
    json_skip_ws(jp);
    if (json_consume(jp, ']')) break;
    if (!json_consume(jp, ',')) {
      for (size_t i = 0; i < count; ++i) free(deps[i]);
      free(deps);
      return false;
    }
  }
  *out_deps = deps;
  *out_dep_count = count;
  return true;
}

static bool json_parse_dispatches(json_parser_t* jp, const char* json_dir,
                                 dispatch_node_t** out_nodes,
                                 size_t* out_node_count) {
  *out_nodes = NULL;
  *out_node_count = 0;

  // Expect: { "dispatch_0": { ... }, ... }
  if (!json_consume(jp, '{')) return false;
  json_skip_ws(jp);
  if (json_consume(jp, '}')) return true;

  size_t cap = 32;
  dispatch_node_t* nodes =
      (dispatch_node_t*)calloc(cap, sizeof(dispatch_node_t));
  if (!nodes) return false;
  size_t count = 0;

  while (true) {
    char* dispatch_key = NULL;
    if (!json_parse_string(jp, &dispatch_key)) goto fail;
    if (!json_consume(jp, ':')) {
      free(dispatch_key);
      goto fail;
    }
    if (!json_consume(jp, '{')) {
      free(dispatch_key);
      goto fail;
    }

    dispatch_node_t node;
    memset(&node, 0, sizeof(node));
    node.key = dispatch_key;
    node.id = -1;

    json_skip_ws(jp);
    if (!json_consume(jp, '}')) {
      while (true) {
        char* field = NULL;
        if (!json_parse_string(jp, &field)) goto fail_node;
        if (!json_consume(jp, ':')) {
          free(field);
          goto fail_node;
        }
        if (!strcmp(field, "id")) {
          int v = 0;
          if (!json_parse_int(jp, &v)) {
            free(field);
            goto fail_node;
          }
          node.id = v;
        } else if (!strcmp(field, "vmfb_path")) {
          char* rel = NULL;
          if (!json_parse_string(jp, &rel)) {
            free(field);
            goto fail_node;
          }
          node.vmfb_path = path_join2(json_dir, rel);
          free(rel);
          if (!node.vmfb_path) {
            free(field);
            goto fail_node;
          }
        } else if (!strcmp(field, "dependencies")) {
          if (!json_parse_dependencies_array(jp, &node.deps, &node.dep_count)) {
            free(field);
            goto fail_node;
          }
        } else {
          if (!json_skip_value(jp)) {
            free(field);
            goto fail_node;
          }
        }
        free(field);
        json_skip_ws(jp);
        if (json_consume(jp, '}')) break;
        if (!json_consume(jp, ',')) goto fail_node;
      }
    }

    if (count == cap) {
      cap *= 2;
      dispatch_node_t* nn =
          (dispatch_node_t*)realloc(nodes, cap * sizeof(dispatch_node_t));
      if (!nn) goto fail_node;
      memset(nn + count, 0, (cap - count) * sizeof(dispatch_node_t));
      nodes = nn;
    }
    nodes[count++] = node;

    json_skip_ws(jp);
    if (json_consume(jp, '}')) break;
    if (!json_consume(jp, ',')) goto fail;
    continue;

  fail_node:
    dispatch_node_deinit(&node);
    goto fail;
  }

  *out_nodes = nodes;
  *out_node_count = count;
  return true;

fail:
  for (size_t i = 0; i < count; ++i) dispatch_node_deinit(&nodes[i]);
  free(nodes);
  return false;
}

static bool parse_dispatch_graph_json(const char* json_path,
                                     dispatch_node_t** out_nodes,
                                     size_t* out_node_count) {
  *out_nodes = NULL;
  *out_node_count = 0;

  FILE* f = fopen(json_path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (size <= 0) {
    fclose(f);
    return false;
  }
  char* buf = (char*)malloc((size_t)size + 1);
  if (!buf) {
    fclose(f);
    return false;
  }
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
    free(buf);
    fclose(f);
    return false;
  }
  buf[size] = 0;
  fclose(f);

  char* json_dir = path_dirname(json_path);
  if (!json_dir) {
    free(buf);
    return false;
  }

  json_parser_t jp;
  jp.p = buf;
  jp.end = buf + size;

  // Root object; scan until we find "dispatches": { ... }.
  bool ok = false;
  if (!json_consume(&jp, '{')) goto done;
  json_skip_ws(&jp);
  if (json_consume(&jp, '}')) goto done;
  while (true) {
    char* key = NULL;
    if (!json_parse_string(&jp, &key)) break;
    if (!json_consume(&jp, ':')) {
      free(key);
      break;
    }
    if (!strcmp(key, "dispatches")) {
      free(key);
      ok = json_parse_dispatches(&jp, json_dir, out_nodes, out_node_count);
      break;
    } else {
      free(key);
      if (!json_skip_value(&jp)) break;
    }
    json_skip_ws(&jp);
    if (json_consume(&jp, '}')) break;
    if (!json_consume(&jp, ',')) break;
  }

done:
  free(json_dir);
  free(buf);
  return ok && *out_nodes && *out_node_count > 0;
}

static int find_node_index(dispatch_node_t* nodes, size_t node_count,
                           const char* key) {
  for (size_t i = 0; i < node_count; ++i) {
    if (nodes[i].key && !strcmp(nodes[i].key, key)) return (int)i;
  }
  return -1;
}

static bool topo_sort(dispatch_node_t* nodes, size_t node_count, int** out_order,
                      size_t* out_order_count) {
  *out_order = NULL;
  *out_order_count = 0;
  if (node_count == 0) return false;

  int* order = (int*)malloc(sizeof(int) * node_count);
  int* indeg = (int*)calloc(node_count, sizeof(int));
  bool* used = (bool*)calloc(node_count, sizeof(bool));
  if (!order || !indeg || !used) {
    free(order);
    free(indeg);
    free(used);
    return false;
  }

  // Compute indegree based on dependencies.
  for (size_t i = 0; i < node_count; ++i) {
    for (size_t d = 0; d < nodes[i].dep_count; ++d) {
      int pi = find_node_index(nodes, node_count, nodes[i].deps[d]);
      if (pi < 0) {
        fprintf(stderr, "Missing dependency '%s' for node '%s'\n",
                nodes[i].deps[d], nodes[i].key);
        free(order);
        free(indeg);
        free(used);
        return false;
      }
      indeg[i] += 1;
    }
  }

  size_t out_n = 0;
  for (size_t step = 0; step < node_count; ++step) {
    int pick = -1;
    for (size_t i = 0; i < node_count; ++i) {
      if (!used[i] && indeg[i] == 0) {
        pick = (int)i;
        break;
      }
    }
    if (pick < 0) {
      fprintf(stderr, "Cycle detected in dispatch dependency graph\n");
      free(order);
      free(indeg);
      free(used);
      return false;
    }
    used[pick] = true;
    order[out_n++] = pick;

    // Decrement indegree of nodes that depend on pick.
    for (size_t j = 0; j < node_count; ++j) {
      if (used[j]) continue;
      for (size_t d = 0; d < nodes[j].dep_count; ++d) {
        if (!strcmp(nodes[j].deps[d], nodes[pick].key)) {
          indeg[j] -= 1;
        }
      }
    }
  }

  free(indeg);
  free(used);
  *out_order = order;
  *out_order_count = out_n;
  return true;
}

static iree_status_t pick_dispatch_entry_function(iree_vm_module_t* module,
                                                  iree_vm_function_t* out_fn) {
  *out_fn = (iree_vm_function_t){0};
  const iree_vm_module_signature_t sig = iree_vm_module_signature(module);
  if (sig.export_function_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "module has no exported functions");
  }

  // Prefer: a single exported function.
  if (sig.export_function_count == 1) {
    return iree_vm_module_lookup_function_by_ordinal(
        module, IREE_VM_FUNCTION_LINKAGE_EXPORT, 0, out_fn);
  }

  // Otherwise pick an export matching (i32)->() if present.
  for (iree_host_size_t i = 0; i < sig.export_function_count; ++i) {
    iree_vm_function_t fn = {0};
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &fn));
    const iree_vm_function_signature_t fsig = iree_vm_function_signature(&fn);
    const iree_string_view_t cc = fsig.calling_convention;
    if (cc.size >= 2 && cc.data[0] == '0' && cc.data[1] == 'i') {
      const char* underscore = memchr(cc.data, '_', cc.size);
      if (!underscore) {
        *out_fn = fn;
        return iree_ok_status();
      }
      const size_t upos = (size_t)(underscore - cc.data);
      if (upos == 2 && upos + 1 <= cc.size) {
        if (upos + 1 == (size_t)cc.size) {
          *out_fn = fn;
          return iree_ok_status();
        }
      }
    }
  }

  // Fallback to export ordinal 0.
  return iree_vm_module_lookup_function_by_ordinal(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT, 0, out_fn);
}

static iree_status_t prepare_dispatch_session(iree_runtime_instance_t* instance,
                                              iree_hal_device_t* device,
                                              dispatch_node_t* node,
                                              int32_t dispatch_iters,
                                              iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(node);
  IREE_ASSERT_ARGUMENT(node->vmfb_path);

  // Ensure idempotent cleanup if called on a partially prepared node.
  iree_vm_list_release(node->inputs);
  node->inputs = NULL;
  iree_runtime_session_release(node->session);
  node->session = NULL;
  node->entry_fn = (iree_vm_function_t){0};

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      instance, &session_options, device, host_allocator, &node->session));

  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      node->session, node->vmfb_path));

  iree_vm_context_t* ctx = iree_runtime_session_context(node->session);
  const iree_host_size_t module_count = iree_vm_context_module_count(ctx);
  if (module_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "session context had 0 modules");
  }
  iree_vm_module_t* module = iree_vm_context_module_at(ctx, module_count - 1);

  IREE_RETURN_IF_ERROR(pick_dispatch_entry_function(module, &node->entry_fn));

  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                          /*initial_capacity=*/1, host_allocator,
                                          &node->inputs));
  iree_vm_value_t v = iree_vm_value_make_i32(dispatch_iters);
  IREE_RETURN_IF_ERROR(iree_vm_list_push_value(node->inputs, &v));

  return iree_ok_status();
}

static iree_status_t execute_prepared_dispatch(dispatch_node_t* node) {
  IREE_ASSERT_ARGUMENT(node);
  IREE_ASSERT_ARGUMENT(node->session);
  IREE_ASSERT_ARGUMENT(node->inputs);
  return iree_runtime_session_call(node->session, &node->entry_fn, node->inputs,
                                  /*output_list=*/NULL);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  const char* json_path = argv[1];
  const char* driver_name = (argc >= 3) ? argv[2] : "local-task";
  const int graph_iters = (argc >= 4) ? parse_int_or_default(argv[3], 1) : 1;
  const int dispatch_iters =
      (argc >= 5) ? parse_int_or_default(argv[4], 1) : 1;
  const int report_every = (argc >= 6) ? parse_int_or_default(argv[5], 0) : 0;
  const int warmup_graph_iters =
      (argc >= 7) ? parse_int_or_default(argv[6], 0) : 0;

  dispatch_node_t* nodes = NULL;
  size_t node_count = 0;
  int* order = NULL;
  size_t order_count = 0;

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();

  iree_runtime_instance_t* instance = NULL;
  iree_hal_device_t* device = NULL;

  fprintf(stdout,
          "Config:\n"
          "  graph_json   = %s\n"
          "  driver       = %s\n"
          "  graph_iters  = %d\n"
          "  dispatch_iters = %d\n"
          "  report_every = %d\n"
          "  warmup_graph_iters = %d\n",
          json_path, driver_name, graph_iters, dispatch_iters, report_every,
          warmup_graph_iters);
  fflush(stdout);

  if (!parse_dispatch_graph_json(json_path, &nodes, &node_count)) {
    fprintf(stderr, "Failed to parse dispatch graph JSON: %s\n", json_path);
    return 1;
  }
  if (!topo_sort(nodes, node_count, &order, &order_count)) {
    fprintf(stderr, "Failed to topologically sort dispatch graph\n");
    for (size_t i = 0; i < node_count; ++i) dispatch_node_deinit(&nodes[i]);
    free(nodes);
    return 1;
  }

  fprintf(stdout, "Dispatch execution order (%zu nodes):\n", order_count);
  for (size_t i = 0; i < order_count; ++i) {
    dispatch_node_t* n = &nodes[order[i]];
    fprintf(stdout, "  %zu) %s (id=%d)\n", i + 1, n->key, n->id);
  }
  fflush(stdout);

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  status = iree_runtime_instance_create(&instance_options, host_allocator,
                                        &instance);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view(driver_name), &device);
  if (!iree_status_is_ok(status)) goto cleanup;

  // Prepare all sessions/functions/inputs before measuring.
  for (size_t oi = 0; oi < order_count; ++oi) {
    dispatch_node_t* n = &nodes[order[oi]];
    if (!n->vmfb_path) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "node %s missing vmfb_path", n->key);
      goto cleanup;
    }
    status = prepare_dispatch_session(instance, device, n, (int32_t)dispatch_iters,
                                     host_allocator);
    if (!iree_status_is_ok(status)) {
      fprintf(stderr, "Failed to prepare %s from %s\n", n->key, n->vmfb_path);
      goto cleanup;
    }
  }

  // Optional warmup (excluded from stats/timing).
  for (int wi = 0; wi < warmup_graph_iters; ++wi) {
    for (size_t oi = 0; oi < order_count; ++oi) {
      dispatch_node_t* n = &nodes[order[oi]];
      status = execute_prepared_dispatch(n);
      if (!iree_status_is_ok(status)) goto cleanup;
    }
  }

  const uint64_t run_start_ns = now_monotonic_ns();
  for (int gi = 0; gi < graph_iters; ++gi) {
    for (size_t oi = 0; oi < order_count; ++oi) {
      dispatch_node_t* n = &nodes[order[oi]];
      const uint64_t t0 = now_monotonic_ns();
      status = execute_prepared_dispatch(n);
      const uint64_t t1 = now_monotonic_ns();
      stats_update(&n->stats, t1 - t0);
      if (!iree_status_is_ok(status)) goto cleanup;
    }

    if (report_every > 0 && ((gi + 1) % report_every) == 0) {
      fprintf(stdout, "[graph_iter %d/%d]\n", gi + 1, graph_iters);
      for (size_t oi = 0; oi < order_count; ++oi) {
        dispatch_node_t* n = &nodes[order[oi]];
        stats_print(n->key, &n->stats);
      }
      fflush(stdout);
    }
  }
  const uint64_t run_end_ns = now_monotonic_ns();
  const uint64_t total_ns = run_end_ns - run_start_ns;
  const double total_s = (double)total_ns / 1000000000.0;
  const double graph_per_s =
      (total_s > 0.0) ? ((double)graph_iters / total_s) : 0.0;

  fprintf(stdout,
          "Run complete:\n"
          "  total_wall_ms=%.3f\n"
          "  graph_iters_per_s=%.3f\n",
          ns_to_ms(total_ns), graph_per_s);
  for (size_t oi = 0; oi < order_count; ++oi) {
    dispatch_node_t* n = &nodes[order[oi]];
    stats_print(n->key, &n->stats);
  }
  fprintf(stdout, "Done.\n");
  fflush(stdout);

cleanup:
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  if (order) free(order);
  if (nodes) {
    for (size_t i = 0; i < node_count; ++i) dispatch_node_deinit(&nodes[i]);
    free(nodes);
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}

