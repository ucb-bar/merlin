#include "runtime_scheduler.h"

#include <inttypes.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

// --- Optional local-task topology pinning (best-effort) ---
#if defined(__has_include)
#if __has_include("iree/task/api.h") && __has_include("iree/task/topology.h") && \
    __has_include("iree/hal/drivers/local_task/driver.h") &&                  \
    __has_include("iree/base/internal/threading.h")
#define DISPATCH_GRAPH_HAS_LOCAL_TASK_PINNING 1
#include "iree/task/api.h"
#include "iree/task/topology.h"
#include "iree/hal/drivers/local_task/driver.h"
#include "iree/base/internal/threading.h"
#else
#define DISPATCH_GRAPH_HAS_LOCAL_TASK_PINNING 0
#endif
#else
#define DISPATCH_GRAPH_HAS_LOCAL_TASK_PINNING 0
#endif

namespace {

using Clock = std::chrono::steady_clock;

// ------------------------------
// Log2 histogram + running stats
// ------------------------------
struct Log2Histogram {
  static const int kBuckets = 64;
  uint64_t buckets[kBuckets];

  Log2Histogram() { Reset(); }

  void Reset() {
    for (int i = 0; i < kBuckets; ++i) buckets[i] = 0;
  }

  static int BucketForUs(uint64_t us) {
    if (us == 0) return 0;
    int b = 0;
    while (us) {
      ++b;
      us >>= 1;
      if (b >= kBuckets - 1) return kBuckets - 1;
    }
    return b;
  }

  void Add(uint64_t us) { buckets[BucketForUs(us)]++; }

  uint64_t Count() const {
    uint64_t c = 0;
    for (int i = 0; i < kBuckets; ++i) c += buckets[i];
    return c;
  }

  uint64_t ApproxPercentile(double pct) const {
    if (pct <= 0.0) return 0;
    if (pct >= 1.0) pct = 1.0;

    const uint64_t total = Count();
    if (total == 0) return 0;
    const uint64_t target = (uint64_t)((double)total * pct);

    uint64_t run = 0;
    for (int b = 0; b < kBuckets; ++b) {
      run += buckets[b];
      if (run >= target) {
        if (b == 0) return 0;
        if (b >= 63) return (uint64_t)(-1);
        return (1ull << b);  // upper bound of bucket
      }
    }
    return (1ull << 63);
  }
};

struct RunningStats {
  uint64_t count = 0;
  uint64_t sum_us = 0;
  uint64_t min_us = UINT64_MAX;
  uint64_t max_us = 0;
  Log2Histogram hist;

  void Reset() {
    count = 0;
    sum_us = 0;
    min_us = UINT64_MAX;
    max_us = 0;
    hist.Reset();
  }

  void Add(uint64_t us) {
    ++count;
    sum_us += us;
    if (us < min_us) min_us = us;
    if (us > max_us) max_us = us;
    hist.Add(us);
  }

  double AvgMs() const {
    if (count == 0) return 0.0;
    return ((double)sum_us / 1000.0) / (double)count;
  }

  double MinMs() const { return (count == 0) ? 0.0 : ((double)min_us / 1000.0); }
  double MaxMs() const { return (double)max_us / 1000.0; }

  double P50Ms() const { return (double)hist.ApproxPercentile(0.50) / 1000.0; }
  double P90Ms() const { return (double)hist.ApproxPercentile(0.90) / 1000.0; }
  double P99Ms() const { return (double)hist.ApproxPercentile(0.99) / 1000.0; }
};

// ------------------------------
// Shared fatal state helpers
// ------------------------------
struct SharedState {
  std::atomic<int> fatal_code{IREE_STATUS_OK};
};

static bool HasFatal(const SharedState* s) {
  return s->fatal_code.load(std::memory_order_relaxed) != IREE_STATUS_OK;
}

static void SetFatalOnce(SharedState* s, iree_status_t st, const char* tag) {
  if (iree_status_is_ok(st)) return;
  const int code = (int)iree_status_code(st);
  int expected = IREE_STATUS_OK;
  if (s->fatal_code.compare_exchange_strong(expected, code,
                                            std::memory_order_relaxed)) {
    if (tag && tag[0]) fprintf(stderr, "%s\n", tag);
    iree_status_fprint(stderr, st);
  }
  iree_status_ignore(st);
}

// ------------------------------
// Minimal JSON parser (supports what your dispatch graph JSON uses)
// We parse:
// - root.dispatch_vmfb_dir (optional string)
// - root.dispatches (object)
// Dependencies come from dispatches[*].dependencies.
// ------------------------------
struct JsonParser {
  const char* p = nullptr;
  const char* end = nullptr;

  void SkipWs() {
    while (p < end) {
      const char c = *p;
      if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
        ++p;
        continue;
      }
      break;
    }
  }

  bool Consume(char expected) {
    SkipWs();
    if (p >= end || *p != expected) return false;
    ++p;
    return true;
  }

  bool ParseString(std::string* out) {
    out->clear();
    SkipWs();
    if (p >= end || *p != '"') return false;
    ++p;
    const char* start = p;
    bool has_escapes = false;
    while (p < end) {
      char c = *p++;
      if (c == '"') break;
      if (c == '\\') {
        has_escapes = true;
        if (p >= end) return false;
        ++p;  // skip escaped char
      }
    }
    if (p > end) return false;
    const char* raw_end = p - 1;

    if (!has_escapes) {
      out->assign(start, raw_end - start);
      return true;
    }

    // Minimal unescape (\", \\ , \n, \r, \t).
    out->reserve((size_t)(raw_end - start));
    const char* r = start;
    while (r < raw_end) {
      char c = *r++;
      if (c != '\\') {
        out->push_back(c);
        continue;
      }
      if (r >= raw_end) break;
      char e = *r++;
      switch (e) {
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case 'n': out->push_back('\n'); break;
        case 'r': out->push_back('\r'); break;
        case 't': out->push_back('\t'); break;
        default: out->push_back(e); break;
      }
    }
    return true;
  }

  bool ParseInt(int* out) {
    *out = 0;
    SkipWs();
    if (p >= end) return false;
    const char* s = p;
    char* endptr = nullptr;
    long v = strtol(s, &endptr, 10);
    if (endptr == s) return false;
    p = endptr;
    *out = (int)v;
    return true;
  }

  bool SkipValue();
  bool SkipArray();
  bool SkipObject();
};

bool JsonParser::SkipArray() {
  if (!Consume('[')) return false;
  SkipWs();
  if (Consume(']')) return true;
  while (true) {
    if (!SkipValue()) return false;
    SkipWs();
    if (Consume(']')) return true;
    if (!Consume(',')) return false;
  }
}

bool JsonParser::SkipObject() {
  if (!Consume('{')) return false;
  SkipWs();
  if (Consume('}')) return true;
  while (true) {
    std::string key;
    if (!ParseString(&key)) return false;
    if (!Consume(':')) return false;
    if (!SkipValue()) return false;
    SkipWs();
    if (Consume('}')) return true;
    if (!Consume(',')) return false;
  }
}

bool JsonParser::SkipValue() {
  SkipWs();
  if (p >= end) return false;
  char c = *p;
  if (c == '"') {
    std::string s;
    return ParseString(&s);
  } else if (c == '{') {
    return SkipObject();
  } else if (c == '[') {
    return SkipArray();
  } else if ((c >= '0' && c <= '9') || c == '-') {
    int dummy = 0;
    return ParseInt(&dummy);
  } else if (!strncmp(p, "true", 4)) {
    p += 4;
    return true;
  } else if (!strncmp(p, "false", 5)) {
    p += 5;
    return true;
  } else if (!strncmp(p, "null", 4)) {
    p += 4;
    return true;
  }
  return false;
}

static std::string PathDirname(const std::string& path) {
  const size_t pos = path.find_last_of('/');
  if (pos == std::string::npos) return ".";
  if (pos == 0) return "/";
  return path.substr(0, pos);
}

static std::string PathJoin2(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  if (!b.empty() && b[0] == '/') return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

static bool StartsWithDirPrefix(const std::string& s, const std::string& dir) {
  if (dir.empty()) return false;
  if (s.size() < dir.size()) return false;
  if (s.compare(0, dir.size(), dir) != 0) return false;
  if (s.size() == dir.size()) return true;
  return s[dir.size()] == '/';
}

static std::string StripLeadingDir(const std::string& s, const std::string& dir) {
  if (!StartsWithDirPrefix(s, dir)) return s;
  if (s.size() == dir.size()) return std::string();
  // skip "dir/"
  return s.substr(dir.size() + 1);
}

static bool FileReadable(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

// Dispatch node parsed from JSON
struct DispatchNode {
  std::string key;                 // "dispatch_16_2"
  int id = -1;                     // 16
  int ordinal = 0;                 // optional
  int total = 0;                   // optional

  std::string vmfb_path_json;      // as in JSON
  std::string vmfb_path_resolved;  // resolved actual path used

  std::vector<std::string> deps;   // keys

  RunningStats stats;
};

static bool ParseDependenciesArray(JsonParser* jp, std::vector<std::string>* out) {
  out->clear();
  if (!jp->Consume('[')) return false;
  jp->SkipWs();
  if (jp->Consume(']')) return true;
  while (true) {
    std::string dep;
    if (!jp->ParseString(&dep)) return false;
    out->push_back(std::move(dep));
    jp->SkipWs();
    if (jp->Consume(']')) break;
    if (!jp->Consume(',')) return false;
  }
  return true;
}

static bool ParseDispatchesObject(JsonParser* jp,
                                 std::vector<DispatchNode>* out_nodes) {
  out_nodes->clear();
  if (!jp->Consume('{')) return false;
  jp->SkipWs();
  if (jp->Consume('}')) return true;

  while (true) {
    std::string dispatch_key;
    if (!jp->ParseString(&dispatch_key)) return false;
    if (!jp->Consume(':')) return false;
    if (!jp->Consume('{')) return false;

    DispatchNode node;
    node.key = std::move(dispatch_key);

    jp->SkipWs();
    if (!jp->Consume('}')) {
      while (true) {
        std::string field;
        if (!jp->ParseString(&field)) return false;
        if (!jp->Consume(':')) return false;

        if (field == "id") {
          int v = 0;
          if (!jp->ParseInt(&v)) return false;
          node.id = v;
        } else if (field == "ordinal") {
          int v = 0;
          if (!jp->ParseInt(&v)) return false;
          node.ordinal = v;
        } else if (field == "total") {
          int v = 0;
          if (!jp->ParseInt(&v)) return false;
          node.total = v;
        } else if (field == "vmfb_path") {
          std::string rel;
          if (!jp->ParseString(&rel)) return false;
          node.vmfb_path_json = std::move(rel);
        } else if (field == "dependencies") {
          if (!ParseDependenciesArray(jp, &node.deps)) return false;
        } else {
          if (!jp->SkipValue()) return false;
        }

        jp->SkipWs();
        if (jp->Consume('}')) break;
        if (!jp->Consume(',')) return false;
      }
    }

    out_nodes->push_back(std::move(node));

    jp->SkipWs();
    if (jp->Consume('}')) break;
    if (!jp->Consume(',')) return false;
  }
  return true;
}

static bool ParseDispatchGraphJson(const std::string& json_path,
                                  std::vector<DispatchNode>* out_nodes,
                                  std::string* out_dispatch_vmfb_dir) {
  out_nodes->clear();
  if (out_dispatch_vmfb_dir) out_dispatch_vmfb_dir->clear();

  std::ifstream f(json_path, std::ios::binary);
  if (!f) return false;

  std::string buf((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
  if (buf.empty()) return false;

  JsonParser jp;
  jp.p = buf.data();
  jp.end = buf.data() + buf.size();

  if (!jp.Consume('{')) return false;
  jp.SkipWs();
  if (jp.Consume('}')) return false;

  bool saw_dispatches = false;

  while (true) {
    std::string key;
    if (!jp.ParseString(&key)) break;
    if (!jp.Consume(':')) break;

    if (key == "dispatches") {
      if (!ParseDispatchesObject(&jp, out_nodes)) return false;
      saw_dispatches = true;
    } else if (key == "dispatch_vmfb_dir") {
      if (out_dispatch_vmfb_dir) {
        std::string v;
        if (!jp.ParseString(&v)) return false;
        *out_dispatch_vmfb_dir = std::move(v);
      } else {
        if (!jp.SkipValue()) return false;
      }
    } else {
      if (!jp.SkipValue()) break;
    }

    jp.SkipWs();
    if (jp.Consume('}')) break;
    if (!jp.Consume(',')) break;
  }

  return saw_dispatches && !out_nodes->empty();
}

// ------------------------------
// Topo sort
// Deterministic tie-break: smaller (id, ordinal, key) first.
// ------------------------------
static bool TopoSort(const std::vector<DispatchNode>& nodes,
                     std::vector<int>* out_order,
                     std::vector<std::vector<int>>* out_dependents) {
  out_order->clear();
  out_dependents->assign(nodes.size(), {});

  std::unordered_map<std::string, int> index_of;
  index_of.reserve(nodes.size() * 2);
  for (int i = 0; i < (int)nodes.size(); ++i) {
    index_of[nodes[i].key] = i;
  }

  std::vector<int> indeg(nodes.size(), 0);
  for (int i = 0; i < (int)nodes.size(); ++i) {
    indeg[i] = (int)nodes[i].deps.size();
    for (const auto& dep_key : nodes[i].deps) {
      auto it = index_of.find(dep_key);
      if (it == index_of.end()) {
        fprintf(stderr, "Missing dependency '%s' for node '%s'\n",
                dep_key.c_str(), nodes[i].key.c_str());
        return false;
      }
      (*out_dependents)[it->second].push_back(i);
    }
  }

  auto better = [&](int a, int b) {
    const auto& A = nodes[a];
    const auto& B = nodes[b];
    if (A.id != B.id) return A.id < B.id;
    if (A.ordinal != B.ordinal) return A.ordinal < B.ordinal;
    return A.key < B.key;
  };

  std::vector<int> ready;
  ready.reserve(nodes.size());
  for (int i = 0; i < (int)nodes.size(); ++i) {
    if (indeg[i] == 0) ready.push_back(i);
  }

  while (!ready.empty()) {
    int best_i = 0;
    for (int i = 1; i < (int)ready.size(); ++i) {
      if (better(ready[i], ready[best_i])) best_i = i;
    }
    int pick = ready[best_i];
    ready.erase(ready.begin() + best_i);

    out_order->push_back(pick);

    for (int child : (*out_dependents)[pick]) {
      indeg[child]--;
      if (indeg[child] == 0) ready.push_back(child);
    }
  }

  if (out_order->size() != nodes.size()) {
    fprintf(stderr, "Cycle detected in dispatch dependency graph\n");
    return false;
  }

  return true;
}

// ------------------------------
// Entry function picking + signature handling
// Supports 0 args or 1 i32 arg.
// ------------------------------
static iree_status_t PickEntryFunction(iree_vm_module_t* module,
                                       iree_vm_function_t* out_fn,
                                       int* out_arity,
                                       bool* out_first_is_i32) {
  *out_fn = (iree_vm_function_t){0};
  *out_arity = 0;
  *out_first_is_i32 = false;

  const iree_vm_module_signature_t sig = iree_vm_module_signature(module);
  if (sig.export_function_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "module has no exports");
  }

  auto compute_arity = [&](const iree_vm_function_t& fn, int* arity, bool* first_i32) {
    const iree_vm_function_signature_t fsig = iree_vm_function_signature(&fn);
    const iree_string_view_t cc = fsig.calling_convention;
    if (cc.size == 0 || cc.data[0] != '0') {
      *arity = 0;
      *first_i32 = false;
      return;
    }
    const void* u = memchr(cc.data, '_', cc.size);
    size_t upos = u ? (size_t)((const char*)u - cc.data) : (size_t)cc.size;
    if (upos < 1) upos = 1;
    const size_t n_in = (upos >= 1) ? (upos - 1) : 0;
    *arity = (int)n_in;
    *first_i32 = (n_in >= 1 && cc.data[1] == 'i');
  };

  if (sig.export_function_count == 1) {
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        module, IREE_VM_FUNCTION_LINKAGE_EXPORT, 0, out_fn));
    compute_arity(*out_fn, out_arity, out_first_is_i32);
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < sig.export_function_count; ++i) {
    iree_vm_function_t fn = {0};
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &fn));
    int arity = 0;
    bool first_i32 = false;
    compute_arity(fn, &arity, &first_i32);
    if (arity == 1 && first_i32) {
      *out_fn = fn;
      *out_arity = arity;
      *out_first_is_i32 = first_i32;
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT, 0, out_fn));
  compute_arity(*out_fn, out_arity, out_first_is_i32);
  return iree_ok_status();
}

// ------------------------------
// Optional local-task pinned device
// ------------------------------
static iree_hal_device_t* CreatePinnedLocalTaskDeviceIfAvailable(
    iree_allocator_t host_allocator,
    uint64_t core_mask,
    iree_status_t* out_status) {
  *out_status = iree_ok_status();

#if DISPATCH_GRAPH_HAS_LOCAL_TASK_PINNING
  if (core_mask == 0) return nullptr;

  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);

  for (int core_id = 0; core_id < 64; ++core_id) {
    if (((core_mask >> core_id) & 1ull) == 0) continue;

    iree_task_topology_group_t group;
    iree_task_topology_group_initialize(topology.group_count, &group);

    group.processor_index = core_id;
    memset(&group.ideal_thread_affinity, 0, sizeof(group.ideal_thread_affinity));
    iree_thread_affinity_set_bit(&group.ideal_thread_affinity, core_id);

    iree_status_t st = iree_task_topology_push_group(&topology, &group);
    if (!iree_status_is_ok(st)) {
      iree_task_topology_deinitialize(&topology);
      *out_status = st;
      return nullptr;
    }
  }

  iree_task_executor_options_t exec_opts = iree_task_executor_options_default();
  exec_opts.worker_local_memory_size = 64 * 1024;

  iree_task_executor_t* executor = nullptr;
  iree_status_t st =
      iree_task_executor_create(exec_opts, &topology, host_allocator, &executor);
  iree_task_topology_deinitialize(&topology);
  if (!iree_status_is_ok(st)) {
    *out_status = st;
    return nullptr;
  }

  iree_hal_task_device_params_t params = iree_hal_task_device_params_default();

  iree_hal_device_t* device = nullptr;
  st = iree_hal_task_device_create(
      iree_make_cstring_view("pinned_local_task"),
      &params, executor,
      /*queue_count=*/1,
      /*queues=*/nullptr,
      host_allocator,
      &device);

  iree_task_executor_release(executor);

  *out_status = st;
  if (!iree_status_is_ok(st)) return nullptr;
  return device;
#else
  (void)host_allocator;
  (void)core_mask;
  return nullptr;
#endif
}

// ------------------------------
// Module/session cache
// One session per VMFB is safest (avoids symbol/name collisions).
// ------------------------------
struct CachedModule {
  std::string vmfb_path;
  iree_runtime_session_t* session = nullptr;
  iree_vm_function_t entry_fn = {0};
  int arity = 0;
  bool first_is_i32 = false;

  std::mutex mu;  // serialize calls on this session
};

static void CachedModuleRelease(CachedModule* m) {
  if (!m) return;
  iree_runtime_session_release(m->session);
  m->session = nullptr;
}

static iree_status_t LoadModuleCached(iree_runtime_instance_t* instance,
                                      iree_hal_device_t* device,
                                      iree_allocator_t host_alloc,
                                      const std::string& vmfb_path,
                                      CachedModule* out) {
  out->vmfb_path = vmfb_path;

  iree_runtime_session_options_t session_opts;
  iree_runtime_session_options_initialize(&session_opts);

  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      instance, &session_opts, device,
      iree_runtime_instance_host_allocator(instance),
      &out->session));

  IREE_RETURN_IF_ERROR(
      iree_runtime_session_append_bytecode_module_from_file(out->session, vmfb_path.c_str()));

  iree_vm_context_t* ctx = iree_runtime_session_context(out->session);
  const iree_host_size_t module_count = iree_vm_context_module_count(ctx);
  if (module_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "session context had 0 modules");
  }
  iree_vm_module_t* module = iree_vm_context_module_at(ctx, module_count - 1);

  IREE_RETURN_IF_ERROR(PickEntryFunction(module, &out->entry_fn,
                                        &out->arity, &out->first_is_i32));

  if (!(out->arity == 0 || (out->arity == 1 && out->first_is_i32))) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "entry function arity=%d (first_i32=%d) not supported; "
                            "supported: 0 args or 1 i32 arg",
                            out->arity, out->first_is_i32 ? 1 : 0);
  }

  (void)host_alloc;
  return iree_ok_status();
}

static iree_status_t CallCachedModule(CachedModule* m,
                                      int32_t dispatch_iters,
                                      iree_allocator_t host_alloc) {
  std::lock_guard<std::mutex> lock(m->mu);

  iree_status_t st = iree_ok_status();
  iree_vm_list_t* inputs = nullptr;

  st = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                           /*initial_capacity=*/(iree_host_size_t)m->arity,
                           host_alloc, &inputs);
  if (!iree_status_is_ok(st)) goto cleanup;

  if (m->arity == 1) {
    iree_vm_value_t v = iree_vm_value_make_i32(dispatch_iters);
    st = iree_vm_list_push_value(inputs, &v);
    if (!iree_status_is_ok(st)) goto cleanup;
  }

  st = iree_runtime_session_call(m->session, &m->entry_fn, inputs,
                                 /*output_list=*/nullptr);

cleanup:
  iree_vm_list_release(inputs);
  return st;
}

// ------------------------------
// Trace writer (timeline reconstruction)
// ------------------------------
struct TraceWriter {
  std::mutex mu;
  FILE* f = nullptr;
  bool wrote_header = false;

  bool Open(const char* path) {
    if (!path || !path[0]) return false;
    f = fopen(path, "wb");
    if (!f) return false;
    return true;
  }

  void Close() {
    if (f) fclose(f);
    f = nullptr;
  }

  void WriteHeaderOnce() {
    if (!f || wrote_header) return;
    fprintf(f, "graph_iter,dispatch_key,dispatch_id,ordinal,start_us,duration_us,thread\n");
    wrote_header = true;
  }

  void WriteEvent(int graph_iter,
                  const DispatchNode& n,
                  uint64_t start_us,
                  uint64_t dur_us) {
    if (!f) return;
    const uint64_t tid = (uint64_t)std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::lock_guard<std::mutex> lock(mu);
    WriteHeaderOnce();
    fprintf(f, "%d,%s,%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
            graph_iter,
            n.key.c_str(),
            n.id,
            n.ordinal,
            start_us,
            dur_us,
            tid);
    fflush(f);
  }
};

// ------------------------------
// Parallel executor for one graph iteration
// ------------------------------
struct WorkQueue {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<int> ready;
  int remaining = 0;
  bool stop = false;
};

// ------------------------------
// Output helpers (JSON + DOT)
// ------------------------------
static void JsonWriteEscaped(FILE* f, const std::string& s) {
  fputc('"', f);
  for (char c : s) {
    switch (c) {
      case '\\': fputs("\\\\", f); break;
      case '"':  fputs("\\\"", f); break;
      case '\n': fputs("\\n", f); break;
      case '\r': fputs("\\r", f); break;
      case '\t': fputs("\\t", f); break;
      default:
        fputc(c, f);
        break;
    }
  }
  fputc('"', f);
}

static std::string DotEscapeLabel(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 16);
  for (char c : s) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else out.push_back(c);
  }
  return out;
}

static bool WriteSummaryJson(const char* path,
                             const dispatch_graph_config_t* cfg,
                             const std::string& json_dispatch_vmfb_dir,
                             const std::vector<DispatchNode>& nodes,
                             const std::vector<int>& topo_order) {
  if (!path || !path[0]) return true;
  FILE* f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "Failed to open out_json: %s\n", path);
    return false;
  }

  fprintf(f, "{\n");

  // config
  fprintf(f, "  \"config\": {\n");
  fprintf(f, "    \"graph_json_path\": "); JsonWriteEscaped(f, cfg->graph_json_path ? cfg->graph_json_path : ""); fprintf(f, ",\n");
  fprintf(f, "    \"driver\": "); JsonWriteEscaped(f, cfg->driver_name ? cfg->driver_name : ""); fprintf(f, ",\n");
  fprintf(f, "    \"graph_iters\": %d,\n", cfg->graph_iters);
  fprintf(f, "    \"dispatch_iters\": %d,\n", cfg->dispatch_iters);
  fprintf(f, "    \"report_every\": %d,\n", cfg->report_every);
  fprintf(f, "    \"parallelism\": %d,\n", cfg->parallelism);
  fprintf(f, "    \"core_mask\": \"0x%016" PRIx64 "\",\n", cfg->core_mask);
  fprintf(f, "    \"vmfb_root_dir\": "); JsonWriteEscaped(f, cfg->vmfb_root_dir ? cfg->vmfb_root_dir : ""); fprintf(f, ",\n");
  fprintf(f, "    \"json_dispatch_vmfb_dir\": "); JsonWriteEscaped(f, json_dispatch_vmfb_dir); fprintf(f, "\n");
  fprintf(f, "  },\n");

  // nodes
  fprintf(f, "  \"nodes\": [\n");
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& n = nodes[i];
    fprintf(f, "    {\n");
    fprintf(f, "      \"key\": "); JsonWriteEscaped(f, n.key); fprintf(f, ",\n");
    fprintf(f, "      \"id\": %d,\n", n.id);
    fprintf(f, "      \"ordinal\": %d,\n", n.ordinal);
    fprintf(f, "      \"total\": %d,\n", n.total);
    fprintf(f, "      \"vmfb_path_json\": "); JsonWriteEscaped(f, n.vmfb_path_json); fprintf(f, ",\n");
    fprintf(f, "      \"vmfb_path_resolved\": "); JsonWriteEscaped(f, n.vmfb_path_resolved); fprintf(f, ",\n");

    // deps
    fprintf(f, "      \"dependencies\": [");
    for (size_t d = 0; d < n.deps.size(); ++d) {
      if (d) fprintf(f, ", ");
      JsonWriteEscaped(f, n.deps[d]);
    }
    fprintf(f, "],\n");

    // stats
    fprintf(f, "      \"stats\": {\n");
    fprintf(f, "        \"count\": %" PRIu64 ",\n", n.stats.count);
    fprintf(f, "        \"avg_ms\": %.6f,\n", n.stats.AvgMs());
    fprintf(f, "        \"p50_ms\": %.6f,\n", n.stats.P50Ms());
    fprintf(f, "        \"p90_ms\": %.6f,\n", n.stats.P90Ms());
    fprintf(f, "        \"p99_ms\": %.6f,\n", n.stats.P99Ms());
    fprintf(f, "        \"min_ms\": %.6f,\n", n.stats.MinMs());
    fprintf(f, "        \"max_ms\": %.6f\n", n.stats.MaxMs());
    fprintf(f, "      }\n");

    fprintf(f, "    }%s\n", (i + 1 < nodes.size()) ? "," : "");
  }
  fprintf(f, "  ],\n");

  // edges reconstructed from deps
  fprintf(f, "  \"edges\": [\n");
  bool first_edge = true;
  for (const auto& n : nodes) {
    for (const auto& dep : n.deps) {
      if (!first_edge) fprintf(f, ",\n");
      first_edge = false;
      fprintf(f, "    {\"from\": ");
      JsonWriteEscaped(f, dep);
      fprintf(f, ", \"to\": ");
      JsonWriteEscaped(f, n.key);
      fprintf(f, "}");
    }
  }
  if (!first_edge) fprintf(f, "\n");
  fprintf(f, "  ],\n");

  // topo order
  fprintf(f, "  \"topo_order\": [");
  for (size_t i = 0; i < topo_order.size(); ++i) {
    if (i) fprintf(f, ", ");
    JsonWriteEscaped(f, nodes[(size_t)topo_order[i]].key);
  }
  fprintf(f, "]\n");

  fprintf(f, "}\n");
  fclose(f);
  return true;
}

static bool WriteDotGraph(const char* path,
                          const std::vector<DispatchNode>& nodes) {
  if (!path || !path[0]) return true;
  FILE* f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "Failed to open out_dot: %s\n", path);
    return false;
  }

  fprintf(f, "digraph dispatch_graph {\n");
  fprintf(f, "  rankdir=LR;\n");
  fprintf(f, "  node [shape=box];\n");

  // nodes
  for (const auto& n : nodes) {
    const std::string label =
        n.key +
        "\\n" + "avg=" + std::to_string(n.stats.AvgMs()) + "ms" +
        "\\n" + "p50=" + std::to_string(n.stats.P50Ms()) + "ms" +
        "\\n" + "p99=" + std::to_string(n.stats.P99Ms()) + "ms";
    fprintf(f, "  \"%s\" [label=\"%s\"];\n",
            n.key.c_str(),
            DotEscapeLabel(label).c_str());
  }

  // edges from deps
  for (const auto& n : nodes) {
    for (const auto& dep : n.deps) {
      fprintf(f, "  \"%s\" -> \"%s\";\n", dep.c_str(), n.key.c_str());
    }
  }

  fprintf(f, "}\n");
  fclose(f);
  return true;
}

// ------------------------------
// VMFB path resolution
// ------------------------------
static std::string ResolveVmfbPath(const dispatch_graph_config_t* cfg,
                                  const std::string& json_dir,
                                  const std::string& json_dispatch_vmfb_dir,
                                  const std::string& vmfb_path_json) {
  if (vmfb_path_json.empty()) return vmfb_path_json;
  if (!vmfb_path_json.empty() && vmfb_path_json[0] == '/') {
    return vmfb_path_json;  // absolute
  }

  // If vmfb_root_dir provided, treat it as the directory containing VMFBs.
  // If JSON vmfb_path begins with dispatch_vmfb_dir (like "dispatches/..."),
  // strip it before joining to avoid ".../dispatches/dispatches/...".
  if (cfg && cfg->vmfb_root_dir && cfg->vmfb_root_dir[0]) {
    std::string rel = vmfb_path_json;
    if (!json_dispatch_vmfb_dir.empty()) {
      rel = StripLeadingDir(rel, json_dispatch_vmfb_dir);
    } else {
      // common fallback: strip "dispatches" if present in path
      rel = StripLeadingDir(rel, "dispatches");
    }
    return PathJoin2(cfg->vmfb_root_dir, rel);
  }

  // Default: relative to JSON directory (classic behavior).
  return PathJoin2(json_dir, vmfb_path_json);
}

// ------------------------------
// Entry point
// ------------------------------
}  // namespace

extern "C" int dispatch_graph_run(const dispatch_graph_config_t* cfg) {
  if (!cfg || !cfg->graph_json_path || !cfg->graph_json_path[0]) {
    fprintf(stderr, "dispatch_graph_run: missing graph_json_path\n");
    return 1;
  }

  const char* driver = (cfg->driver_name && cfg->driver_name[0]) ? cfg->driver_name
                                                                 : "local-task";
  const int graph_iters = (cfg->graph_iters > 0) ? cfg->graph_iters : 1;
  const int dispatch_iters = (cfg->dispatch_iters > 0) ? cfg->dispatch_iters : 1;
  const int report_every = (cfg->report_every >= 0) ? cfg->report_every : 0;
  const uint64_t core_mask = cfg->core_mask;
  const int parallelism = (cfg->parallelism > 0) ? cfg->parallelism : 1;

  fprintf(stdout,
          "Dispatch-graph runner:\n"
          "  json          = %s\n"
          "  driver        = %s\n"
          "  graph_iters   = %d\n"
          "  dispatch_iters= %d\n"
          "  report_every  = %d\n"
          "  core_mask     = 0x%016" PRIx64 "\n"
          "  parallelism   = %d\n"
          "  vmfb_root_dir = %s\n"
          "  out_json      = %s\n"
          "  out_dot       = %s\n"
          "  trace_csv     = %s\n",
          cfg->graph_json_path, driver, graph_iters, dispatch_iters, report_every,
          core_mask, parallelism,
          (cfg->vmfb_root_dir ? cfg->vmfb_root_dir : ""),
          (cfg->out_json_path ? cfg->out_json_path : ""),
          (cfg->out_dot_path ? cfg->out_dot_path : ""),
          (cfg->trace_csv_path ? cfg->trace_csv_path : ""));
  fflush(stdout);

  // Parse graph
  std::vector<DispatchNode> nodes;
  std::string json_dispatch_vmfb_dir;
  if (!ParseDispatchGraphJson(cfg->graph_json_path, &nodes, &json_dispatch_vmfb_dir)) {
    fprintf(stderr, "Failed to parse dispatch graph JSON: %s\n", cfg->graph_json_path);
    return 1;
  }

  const std::string json_dir = PathDirname(cfg->graph_json_path);

  // Resolve vmfb paths now (using cfg->vmfb_root_dir override if provided).
  for (auto& n : nodes) {
    n.vmfb_path_resolved = ResolveVmfbPath(cfg, json_dir, json_dispatch_vmfb_dir, n.vmfb_path_json);
    if (!FileReadable(n.vmfb_path_resolved)) {
      fprintf(stderr,
              "VMFB not readable for node %s:\n"
              "  vmfb_path_json     = %s\n"
              "  vmfb_path_resolved = %s\n"
              "Hint: pass --vmfb_dir=<DIR_CONTAINING_VMFBS>\n",
              n.key.c_str(),
              n.vmfb_path_json.c_str(),
              n.vmfb_path_resolved.c_str());
      return 1;
    }
  }

  std::vector<int> topo_order;
  std::vector<std::vector<int>> dependents;
  if (!TopoSort(nodes, &topo_order, &dependents)) {
    fprintf(stderr, "Failed to topo-sort dispatch graph\n");
    return 1;
  }

  fprintf(stdout, "Dispatch topo order (%zu nodes):\n", topo_order.size());
  for (size_t i = 0; i < topo_order.size(); ++i) {
    const auto& n = nodes[(size_t)topo_order[i]];
    fprintf(stdout, "  %zu) %s (id=%d ord=%d/%d)\n",
            i + 1, n.key.c_str(), n.id, n.ordinal, n.total);
  }
  fflush(stdout);

  SharedState shared;
  iree_allocator_t host_alloc = iree_allocator_system();
  iree_status_t st = iree_ok_status();

  iree_runtime_instance_t* instance = nullptr;
  iree_hal_device_t* device = nullptr;

  // Trace
  TraceWriter trace;
  if (cfg->trace_csv_path && cfg->trace_csv_path[0]) {
    if (!trace.Open(cfg->trace_csv_path)) {
      fprintf(stderr, "Failed to open trace_csv: %s\n", cfg->trace_csv_path);
      return 1;
    }
  }

  // Create instance
  {
    iree_runtime_instance_options_t opts;
    iree_runtime_instance_options_initialize(&opts);
    iree_runtime_instance_options_use_all_available_drivers(&opts);
    st = iree_runtime_instance_create(&opts, host_alloc, &instance);
    if (!iree_status_is_ok(st)) {
      iree_status_fprint(stderr, st);
      iree_status_ignore(st);
      trace.Close();
      return 1;
    }
  }

  // Create device (pinned local-task if requested)
  if (core_mask != 0) {
    iree_status_t pin_st = iree_ok_status();
    iree_hal_device_t* pinned =
        CreatePinnedLocalTaskDeviceIfAvailable(host_alloc, core_mask, &pin_st);
    if (pinned) {
      device = pinned;
      fprintf(stdout,
              "[dispatch] Using pinned local-task device (core_mask=0x%016" PRIx64 ")\n",
              core_mask);
      fflush(stdout);
    } else {
      if (!iree_status_is_ok(pin_st)) {
        fprintf(stderr, "[dispatch] Pinning requested but failed; falling back.\n");
        iree_status_fprint(stderr, pin_st);
        iree_status_ignore(pin_st);
      } else {
#if !DISPATCH_GRAPH_HAS_LOCAL_TASK_PINNING
        fprintf(stderr, "[dispatch] Pinning requested but task headers unavailable; falling back.\n");
#endif
      }
    }
  }

  if (!device) {
    st = iree_runtime_instance_try_create_default_device(
        instance, iree_make_cstring_view(driver), &device);
    if (!iree_status_is_ok(st)) {
      iree_status_fprint(stderr, st);
      iree_status_ignore(st);
      iree_runtime_instance_release(instance);
      trace.Close();
      return 1;
    }
  }

  // Cache modules by resolved vmfb path
  std::unordered_map<std::string, CachedModule*> cache;
  cache.reserve(nodes.size() * 2);
  std::vector<CachedModule*> owned_modules;
  owned_modules.reserve(nodes.size());

  auto get_cached = [&](const std::string& vmfb_path) -> CachedModule* {
    auto it = cache.find(vmfb_path);
    if (it != cache.end()) return it->second;

    CachedModule* cm = new (std::nothrow) CachedModule();
    if (!cm) {
      fprintf(stderr, "OOM allocating CachedModule\n");
      return nullptr;
    }

    iree_status_t load_st =
        LoadModuleCached(instance, device, host_alloc, vmfb_path, cm);
    if (!iree_status_is_ok(load_st)) {
      SetFatalOnce(&shared, load_st, "[dispatch] load module failed");
      CachedModuleRelease(cm);
      delete cm;
      return nullptr;
    }

    cache.emplace(vmfb_path, cm);
    owned_modules.push_back(cm);
    return cm;
  };

  // Pre-resolve cached module pointer per node
  std::vector<CachedModule*> node_module(nodes.size(), nullptr);
  for (size_t i = 0; i < nodes.size(); ++i) {
    node_module[i] = get_cached(nodes[i].vmfb_path_resolved);
    if (HasFatal(&shared) || !node_module[i]) break;
  }
  if (HasFatal(&shared)) {
    for (auto* m : owned_modules) { CachedModuleRelease(m); delete m; }
    iree_hal_device_release(device);
    iree_runtime_instance_release(instance);
    trace.Close();
    return 1;
  }

  const auto run_t0 = Clock::now();

  auto us_since_start = [&](Clock::time_point t) -> uint64_t {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t - run_t0).count();
  };

  // Sequential topo-order runner
  auto run_sequential_iter = [&](int graph_iter) {
    for (int idx : topo_order) {
      if (HasFatal(&shared)) return;

      const auto t0 = Clock::now();
      iree_status_t call_st =
          CallCachedModule(node_module[(size_t)idx], (int32_t)dispatch_iters, host_alloc);
      const auto t1 = Clock::now();

      if (!iree_status_is_ok(call_st)) {
        SetFatalOnce(&shared, call_st, "[dispatch] call failed");
        return;
      }

      const uint64_t dur_us =
          (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      nodes[(size_t)idx].stats.Add(dur_us);

      trace.WriteEvent(graph_iter, nodes[(size_t)idx], us_since_start(t0), dur_us);
    }
  };

  // Parallel runner (dependency-driven)
  auto run_parallel_iter = [&](int graph_iter) {
    std::vector<int> indeg(nodes.size(), 0);
    for (size_t i = 0; i < nodes.size(); ++i) indeg[i] = (int)nodes[i].deps.size();

    WorkQueue wq;
    wq.remaining = (int)nodes.size();

    for (int i = 0; i < (int)nodes.size(); ++i) {
      if (indeg[(size_t)i] == 0) wq.ready.push_back(i);
    }

    auto worker = [&]() {
      while (true) {
        int node_idx = -1;
        {
          std::unique_lock<std::mutex> lock(wq.mu);
          wq.cv.wait(lock, [&]() {
            return wq.stop || HasFatal(&shared) || !wq.ready.empty() || wq.remaining == 0;
          });

          if (wq.stop || HasFatal(&shared) || wq.remaining == 0) return;

          node_idx = wq.ready.front();
          wq.ready.pop_front();
        }

        const auto t0 = Clock::now();
        iree_status_t call_st =
            CallCachedModule(node_module[(size_t)node_idx], (int32_t)dispatch_iters, host_alloc);
        const auto t1 = Clock::now();

        if (!iree_status_is_ok(call_st)) {
          SetFatalOnce(&shared, call_st, "[dispatch] call failed");
          std::lock_guard<std::mutex> lk(wq.mu);
          wq.stop = true;
          wq.cv.notify_all();
          return;
        }

        const uint64_t dur_us =
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        nodes[(size_t)node_idx].stats.Add(dur_us);

        trace.WriteEvent(graph_iter, nodes[(size_t)node_idx], us_since_start(t0), dur_us);

        {
          std::lock_guard<std::mutex> lock(wq.mu);
          for (int child : dependents[(size_t)node_idx]) {
            indeg[(size_t)child]--;
            if (indeg[(size_t)child] == 0) wq.ready.push_back(child);
          }
          wq.remaining--;
        }
        wq.cv.notify_all();
      }
    };

    std::vector<std::thread> threads;
    threads.reserve((size_t)parallelism);
    for (int i = 0; i < parallelism; ++i) threads.emplace_back(worker);

    {
      std::lock_guard<std::mutex> lk(wq.mu);
      wq.cv.notify_all();
    }

    for (auto& t : threads) t.join();
  };

  // Run loop
  for (int gi = 0; gi < graph_iters && !HasFatal(&shared); ++gi) {
    if (parallelism <= 1) {
      run_sequential_iter(gi);
    } else {
      run_parallel_iter(gi);
    }

    if (report_every > 0 && ((gi + 1) % report_every) == 0 && !HasFatal(&shared)) {
      fprintf(stdout, "[graph_iter %d/%d]\n", gi + 1, graph_iters);
      for (int idx : topo_order) {
        const auto& n = nodes[(size_t)idx];
        fprintf(stdout,
                "  %s: count=%" PRIu64 " avg=%.3fms p50=%.3fms p90=%.3fms p99=%.3fms min=%.3fms max=%.3fms\n",
                n.key.c_str(),
                n.stats.count,
                n.stats.AvgMs(), n.stats.P50Ms(), n.stats.P90Ms(), n.stats.P99Ms(),
                n.stats.MinMs(), n.stats.MaxMs());
      }
      fflush(stdout);
    }
  }

  const auto run_t1 = Clock::now();
  const double total_s =
      std::chrono::duration_cast<std::chrono::duration<double>>(run_t1 - run_t0).count();
  const double graphs_per_s = (total_s > 0.0) ? ((double)graph_iters / total_s) : 0.0;

  // Final report
  if (!HasFatal(&shared)) {
    fprintf(stdout,
            "Run complete:\n"
            "  total_wall_ms=%.3f\n"
            "  graph_iters_per_s=%.3f\n",
            total_s * 1000.0, graphs_per_s);
    for (int idx : topo_order) {
      const auto& n = nodes[(size_t)idx];
      fprintf(stdout,
              "  %s: count=%" PRIu64 " avg=%.3fms p50=%.3fms p90=%.3fms p99=%.3fms min=%.3fms max=%.3fms\n",
              n.key.c_str(),
              n.stats.count,
              n.stats.AvgMs(), n.stats.P50Ms(), n.stats.P90Ms(), n.stats.P99Ms(),
              n.stats.MinMs(), n.stats.MaxMs());
    }
    fprintf(stdout, "Done.\n");
    fflush(stdout);
  }

  // Outputs for reconstruction/plotting
  bool ok_write = true;
  ok_write = ok_write && WriteSummaryJson(cfg->out_json_path, cfg, json_dispatch_vmfb_dir, nodes, topo_order);
  ok_write = ok_write && WriteDotGraph(cfg->out_dot_path, nodes);
  if (!ok_write) {
    fprintf(stderr, "Warning: failed writing one or more outputs\n");
  }

  // Cleanup
  trace.Close();
  for (auto* m : owned_modules) {
    CachedModuleRelease(m);
    delete m;
  }
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);

  return HasFatal(&shared) ? 1 : 0;
}