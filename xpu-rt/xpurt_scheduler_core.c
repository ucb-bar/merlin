#include "xpurt_scheduler_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//--- Minimal JSON parser (dispatch graph format only) ---

typedef struct json_parser_t {
  const char* p;
  const char* end;
} json_parser_t;

static void json_skip_ws(json_parser_t* jp) {
  while (jp->p < jp->end && (*jp->p == ' ' || *jp->p == '\n' || *jp->p == '\r' || *jp->p == '\t'))
    ++jp->p;
}

static int json_consume(json_parser_t* jp, char expected) {
  json_skip_ws(jp);
  if (jp->p >= jp->end || *jp->p != expected) return 0;
  ++jp->p;
  return 1;
}

static int json_parse_string(json_parser_t* jp, char** out_str) {
  *out_str = NULL;
  json_skip_ws(jp);
  if (jp->p >= jp->end || *jp->p != '"') return 0;
  ++jp->p;
  const char* start = jp->p;
  size_t len = 0;
  int has_escapes = 0;
  while (jp->p < jp->end) {
    char c = *jp->p++;
    if (c == '"') break;
    if (c == '\\') {
      has_escapes = 1;
      if (jp->p < jp->end) ++jp->p;
      len += 1;
      continue;
    }
    len += 1;
  }
  const char* raw_end = jp->p - 1;
  if (!has_escapes) {
    size_t raw_len = (size_t)(raw_end - start);
    char* s = (char*)malloc(raw_len + 1);
    if (!s) return 0;
    memcpy(s, start, raw_len);
    s[raw_len] = '\0';
    *out_str = s;
    return 1;
  }
  char* s = (char*)malloc(len + 1);
  if (!s) return 0;
  const char* r = start;
  char* w = s;
  while (r < raw_end) {
    char c = *r++;
    if (c != '\\') {
      *w++ = c;
      continue;
    }
    if (r >= raw_end) break;
    switch (*r++) {
      case '"': *w++ = '"'; break;
      case '\\': *w++ = '\\'; break;
      case 'n': *w++ = '\n'; break;
      case 'r': *w++ = '\r'; break;
      case 't': *w++ = '\t'; break;
      default: *w++ = r[-1]; break;
    }
  }
  *w = '\0';
  *out_str = s;
  return 1;
}

static int json_parse_int(json_parser_t* jp, int* out_value) {
  *out_value = 0;
  json_skip_ws(jp);
  if (jp->p >= jp->end) return 0;
  char* endptr = NULL;
  long v = strtol(jp->p, &endptr, 10);
  if (endptr == jp->p) return 0;
  jp->p = endptr;
  *out_value = (int)v;
  return 1;
}

static int json_skip_value(json_parser_t* jp);

static int json_skip_array(json_parser_t* jp) {
  if (!json_consume(jp, '[')) return 0;
  json_skip_ws(jp);
  if (json_consume(jp, ']')) return 1;
  for (;;) {
    if (!json_skip_value(jp)) return 0;
    json_skip_ws(jp);
    if (json_consume(jp, ']')) return 1;
    if (!json_consume(jp, ',')) return 0;
  }
}

static int json_skip_object(json_parser_t* jp) {
  if (!json_consume(jp, '{')) return 0;
  json_skip_ws(jp);
  if (json_consume(jp, '}')) return 1;
  for (;;) {
    char* key = NULL;
    if (!json_parse_string(jp, &key)) return 0;
    free(key);
    if (!json_consume(jp, ':')) return 0;
    if (!json_skip_value(jp)) return 0;
    json_skip_ws(jp);
    if (json_consume(jp, '}')) return 1;
    if (!json_consume(jp, ',')) return 0;
  }
}

static int json_skip_value(json_parser_t* jp) {
  json_skip_ws(jp);
  if (jp->p >= jp->end) return 0;
  if (*jp->p == '"') {
    char* s = NULL;
    int ok = json_parse_string(jp, &s);
    free(s);
    return ok;
  }
  if (*jp->p == '{') return json_skip_object(jp);
  if (*jp->p == '[') return json_skip_array(jp);
  if ((*jp->p >= '0' && *jp->p <= '9') || *jp->p == '-') {
    int dummy;
    return json_parse_int(jp, &dummy);
  }
  if (jp->p + 4 <= jp->end && strncmp(jp->p, "true", 4) == 0)  { jp->p += 4; return 1; }
  if (jp->p + 5 <= jp->end && strncmp(jp->p, "false", 5) == 0) { jp->p += 5; return 1; }
  if (jp->p + 4 <= jp->end && strncmp(jp->p, "null", 4) == 0)  { jp->p += 4; return 1; }
  return 0;
}

static char* path_dirname(const char* path) {
  if (!path) return NULL;
  const char* last = strrchr(path, '/');
  if (!last) {
    char* out = (char*)malloc(2);
    if (out) { out[0] = '.'; out[1] = '\0'; }
    return out;
  }
  size_t len = (size_t)(last - path);
  if (len == 0) len = 1;
  char* out = (char*)malloc(len + 1);
  if (!out) return NULL;
  memcpy(out, path, len);
  out[len] = '\0';
  return out;
}

static char* path_join2(const char* a, const char* b) {
  if (!a || !a[0]) return b ? strdup(b) : NULL;
  if (!b || !b[0]) return strdup(a);
  if (b[0] == '/') return strdup(b);
  size_t la = strlen(a), lb = strlen(b);
  int need_slash = (la > 0 && a[la - 1] != '/');
  char* out = (char*)malloc(la + (need_slash ? 1 : 0) + lb + 1);
  if (!out) return NULL;
  memcpy(out, a, la);
  size_t pos = la;
  if (need_slash) out[pos++] = '/';
  memcpy(out + pos, b, lb + 1);
  return out;
}

typedef struct dispatch_node_core_t {
  char* key;
  int id;
  char* vmfb_path;
  char** deps;
  size_t dep_count;
} dispatch_node_core_t;

static void node_clear(dispatch_node_core_t* n) {
  if (!n) return;
  free(n->key);
  free(n->vmfb_path);
  if (n->deps) {
    for (size_t i = 0; i < n->dep_count; ++i) free(n->deps[i]);
    free(n->deps);
  }
  n->key = NULL;
  n->vmfb_path = NULL;
  n->deps = NULL;
  n->dep_count = 0;
}

static int json_parse_deps(json_parser_t* jp, char*** out_deps, size_t* out_count) {
  *out_deps = NULL;
  *out_count = 0;
  if (!json_consume(jp, '[')) return 0;
  json_skip_ws(jp);
  if (json_consume(jp, ']')) return 1;
  size_t cap = 4, count = 0;
  char** deps = (char**)calloc(cap, sizeof(char*));
  if (!deps) return 0;
  for (;;) {
    char* dep = NULL;
    if (!json_parse_string(jp, &dep)) goto fail_deps;
    if (count >= cap) {
      cap *= 2;
      char** n = (char**)realloc(deps, cap * sizeof(char*));
      if (!n) { free(dep); goto fail_deps; }
      deps = n;
    }
    deps[count++] = dep;
    json_skip_ws(jp);
    if (json_consume(jp, ']')) break;
    if (!json_consume(jp, ',')) goto fail_deps;
  }
  *out_deps = deps;
  *out_count = count;
  return 1;
fail_deps:
  for (size_t i = 0; i < count; ++i) free(deps[i]);
  free(deps);
  return 0;
}

static int json_parse_dispatches(json_parser_t* jp, const char* json_dir,
                                 dispatch_node_core_t** out_nodes, size_t* out_count) {
  *out_nodes = NULL;
  *out_count = 0;
  if (!json_consume(jp, '{')) return 0;
  json_skip_ws(jp);
  if (json_consume(jp, '}')) return 1;
  size_t cap = 32, count = 0;
  dispatch_node_core_t* nodes = (dispatch_node_core_t*)calloc(cap, sizeof(dispatch_node_core_t));
  if (!nodes) return 0;
  for (;;) {
    char* key = NULL;
    if (!json_parse_string(jp, &key)) goto fail_nodes;
    if (!json_consume(jp, ':') || !json_consume(jp, '{')) {
      free(key);
      goto fail_nodes;
    }
    dispatch_node_core_t node;
    memset(&node, 0, sizeof(node));
    node.key = key;
    node.id = -1;
    json_skip_ws(jp);
    if (!json_consume(jp, '}')) {
      for (;;) {
        char* field = NULL;
        if (!json_parse_string(jp, &field)) goto fail_one;
        if (!json_consume(jp, ':')) { free(field); goto fail_one; }
        if (strcmp(field, "id") == 0) {
          json_parse_int(jp, &node.id);
        } else if (strcmp(field, "vmfb_path") == 0) {
          char* rel = NULL;
          if (json_parse_string(jp, &rel)) {
            node.vmfb_path = path_join2(json_dir, rel);
            free(rel);
          }
        } else if (strcmp(field, "dependencies") == 0) {
          json_parse_deps(jp, &node.deps, &node.dep_count);
        } else {
          json_skip_value(jp);
        }
        free(field);
        json_skip_ws(jp);
        if (json_consume(jp, '}')) break;
        if (!json_consume(jp, ',')) goto fail_one;
      }
    }
    if (count >= cap) {
      cap *= 2;
      dispatch_node_core_t* n = (dispatch_node_core_t*)realloc(nodes, cap * sizeof(dispatch_node_core_t));
      if (!n) goto fail_one;
      memset(n + count, 0, (cap - count) * sizeof(dispatch_node_core_t));
      nodes = n;
    }
    nodes[count++] = node;
    json_skip_ws(jp);
    if (json_consume(jp, '}')) break;
    if (!json_consume(jp, ',')) goto fail_nodes;
    continue;
fail_one:
    node_clear(&node);
    goto fail_nodes;
  }
  *out_nodes = nodes;
  *out_count = count;
  return 1;
fail_nodes:
  for (size_t i = 0; i < count; ++i) node_clear(&nodes[i]);
  free(nodes);
  return 0;
}

static int find_node_idx(dispatch_node_core_t* nodes, size_t n, const char* key) {
  for (size_t i = 0; i < n; ++i)
    if (nodes[i].key && strcmp(nodes[i].key, key) == 0) return (int)i;
  return -1;
}

static int topo_sort(dispatch_node_core_t* nodes, size_t node_count, int** out_order) {
  *out_order = NULL;
  if (node_count == 0) return 0;
  int* order = (int*)malloc((size_t)node_count * sizeof(int));
  int* indeg = (int*)calloc(node_count, sizeof(int));
  int* used = (int*)calloc(node_count, sizeof(int));
  if (!order || !indeg || !used) {
    free(order); free(indeg); free(used);
    return 0;
  }
  for (size_t i = 0; i < node_count; ++i)
    for (size_t d = 0; d < nodes[i].dep_count; ++d) {
      int pi = find_node_idx(nodes, node_count, nodes[i].deps[d]);
      if (pi < 0) {
        fprintf(stderr, "Missing dependency '%s' for node '%s'\n", nodes[i].deps[d], nodes[i].key);
        free(order); free(indeg); free(used);
        return 0;
      }
      indeg[i] += 1;
    }
  size_t out_n = 0;
  for (size_t step = 0; step < node_count; ++step) {
    int pick = -1;
    for (size_t i = 0; i < node_count; ++i)
      if (!used[i] && indeg[i] == 0) { pick = (int)i; break; }
    if (pick < 0) {
      fprintf(stderr, "Cycle in dispatch graph\n");
      free(order); free(indeg); free(used);
      return 0;
    }
    used[pick] = 1;
    order[out_n++] = pick;
    for (size_t j = 0; j < node_count; ++j) {
      if (used[j]) continue;
      for (size_t d = 0; d < nodes[j].dep_count; ++d)
        if (strcmp(nodes[j].deps[d], nodes[pick].key) == 0)
          indeg[j] -= 1;
    }
  }
  free(indeg); free(used);
  *out_order = order;
  return 1;
}

// Opaque graph = nodes + execution order
struct xpurt_graph {
  dispatch_node_core_t* nodes;
  size_t node_count;
  int* order;
};

xpurt_status_t xpurt_graph_load(const char* json_path, xpurt_graph_t** out_graph) {
  if (!json_path || !out_graph) return XPURT_STATUS_ERROR;
  *out_graph = NULL;
  FILE* f = fopen(json_path, "rb");
  if (!f) return XPURT_STATUS_ERROR;
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (size <= 0) { fclose(f); return XPURT_STATUS_ERROR; }
  char* buf = (char*)malloc((size_t)size + 1);
  if (!buf) { fclose(f); return XPURT_STATUS_ERROR; }
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
    free(buf); fclose(f); return XPURT_STATUS_ERROR;
  }
  buf[size] = '\0';
  fclose(f);
  char* json_dir = path_dirname(json_path);
  if (!json_dir) { free(buf); return XPURT_STATUS_ERROR; }
  json_parser_t jp;
  jp.p = buf;
  jp.end = buf + size;
  if (!json_consume(&jp, '{')) { free(json_dir); free(buf); return XPURT_STATUS_ERROR; }
  json_skip_ws(&jp);
  if (json_consume(&jp, '}')) { free(json_dir); free(buf); return XPURT_STATUS_ERROR; }
  int ok = 0;
  while (1) {
    char* key = NULL;
    if (!json_parse_string(&jp, &key)) break;
    if (!json_consume(&jp, ':')) { free(key); break; }
    if (strcmp(key, "dispatches") == 0) {
      free(key);
      dispatch_node_core_t* nodes = NULL;
      size_t node_count = 0;
      if (!json_parse_dispatches(&jp, json_dir, &nodes, &node_count)) break;
      int* order = NULL;
      if (!topo_sort(nodes, node_count, &order)) {
        for (size_t i = 0; i < node_count; ++i) node_clear(&nodes[i]);
        free(nodes);
        break;
      }
      xpurt_graph_t* g = (xpurt_graph_t*)malloc(sizeof(*g));
      if (!g) {
        free(order);
        for (size_t i = 0; i < node_count; ++i) node_clear(&nodes[i]);
        free(nodes);
        break;
      }
      g->nodes = nodes;
      g->node_count = node_count;
      g->order = order;
      ok = 1;
      *out_graph = g;
      break;
    }
    free(key);
    if (!json_skip_value(&jp)) break;
    json_skip_ws(&jp);
    if (json_consume(&jp, '}')) break;
    if (!json_consume(&jp, ',')) break;
  }
  free(json_dir);
  free(buf);
  return ok ? XPURT_STATUS_OK : XPURT_STATUS_ERROR;
}

void xpurt_graph_destroy(xpurt_graph_t* graph) {
  if (!graph) return;
  for (size_t i = 0; i < graph->node_count; ++i) node_clear(&graph->nodes[i]);
  free(graph->nodes);
  free(graph->order);
  free(graph);
}

size_t xpurt_graph_node_count(xpurt_graph_t* graph) {
  return graph ? graph->node_count : 0;
}

const int* xpurt_graph_execution_order(xpurt_graph_t* graph) {
  return graph ? graph->order : NULL;
}

const char* xpurt_graph_node_key(xpurt_graph_t* graph, size_t index) {
  if (!graph || index >= graph->node_count) return NULL;
  return graph->nodes[index].key;
}

int xpurt_graph_node_id(xpurt_graph_t* graph, size_t index) {
  if (!graph || index >= graph->node_count) return -1;
  return graph->nodes[index].id;
}

const char* xpurt_graph_node_vmfb_path(xpurt_graph_t* graph, size_t index) {
  if (!graph || index >= graph->node_count) return NULL;
  return graph->nodes[index].vmfb_path;
}
