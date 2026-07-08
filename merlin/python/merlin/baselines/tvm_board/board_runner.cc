// Board-local relax-VM runner: loads a TVM-exported .so, builds the relax VM, sets inputs from raw
// bins (per a manifest), runs "main", writes output bytes. NO tvm_rpc / tracker / RPC session.
// Cross-built on the host with the SpacemiT clang, linked against the cross-built libtvm_runtime.so.
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/device_api.h>
#include <dlpack/dlpack.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

using namespace tvm::runtime;

static std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
  std::streamsize n = f.tellg(); f.seekg(0);
  std::vector<char> buf(n);
  f.read(buf.data(), n);
  return buf;
}

int main(int argc, char** argv) {
  // argv: 1=so_path 2=manifest 3=indir 4=out_path 5=n_iters(optional)
  if (argc < 5) { fprintf(stderr, "usage: runner so manifest indir out [iters]\n"); return 1; }
  std::string so_path = argv[1], manifest = argv[2], indir = argv[3], out_path = argv[4];
  int iters = argc > 5 ? atoi(argv[5]) : 1;

  Module exec = Module::LoadFromFile(so_path);
  PackedFunc load = exec.GetFunction("vm_load_executable");
  if (load == nullptr) { fprintf(stderr, "no vm_load_executable\n"); return 3; }
  Module vm = load();
  PackedFunc init = vm.GetFunction("vm_initialization");
  init(int(kDLCPU), 0, 2 /*kPooled*/);

  // Parse manifest: one line per input "dtypecode dtypebits ndim d0 d1 ... filename"
  std::ifstream mf(manifest);
  std::vector<NDArray> inputs;
  std::string line;
  DLDevice dev{kDLCPU, 0};
  while (std::getline(mf, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    int code, bits, ndim; ss >> code >> bits >> ndim;
    std::vector<int64_t> shape(ndim);
    for (int i = 0; i < ndim; i++) ss >> shape[i];
    std::string fname; ss >> fname;
    DLDataType dt{(uint8_t)code, (uint8_t)bits, 1};
    NDArray arr = NDArray::Empty(ShapeTuple(shape.begin(), shape.end()), dt, dev);
    auto bytes = read_file(indir + "/" + fname);
    arr.CopyFromBytes(bytes.data(), bytes.size());
    inputs.push_back(arr);
  }
  fprintf(stderr, "loaded %zu inputs\n", inputs.size());

  PackedFunc set_input = vm.GetFunction("set_input");
  PackedFunc invoke = vm.GetFunction("invoke_stateful");
  PackedFunc get_output = vm.GetFunction("get_output");
  PackedFunc get_output_arity = vm.GetFunction("get_output_arity");

  // Build packed args for set_input: [ "main", in0, in1, ... ]
  std::vector<TVMValue> vals(1 + inputs.size());
  std::vector<int> codes(1 + inputs.size());
  std::string fname = "main";
  vals[0].v_str = fname.c_str(); codes[0] = kTVMStr;
  for (size_t i = 0; i < inputs.size(); i++) {
    vals[1 + i].v_handle = const_cast<DLTensor*>(inputs[i].operator->());
    codes[1 + i] = kTVMDLTensorHandle;
  }
  TVMRetValue rv_set;
  set_input.CallPacked(TVMArgs(vals.data(), codes.data(), (int)vals.size()), &rv_set);

  // Warmup (output exists only after invoke), then query arity: -1 => single leaf tensor
  // (get_output with NO index); else a tuple => take element 0.
  invoke("main");
  int top_arity = get_output_arity("main");
  bool single = (top_arity == -1);
  fprintf(stderr, "output arity=%d (%s)\n", top_arity, single ? "single tensor" : "tuple->[0]");
  auto fetch = [&]() -> NDArray {
    return single ? get_output("main").operator NDArray() : get_output("main", 0).operator NDArray();
  };
  NDArray out = fetch();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < iters; it++) {
    set_input.CallPacked(TVMArgs(vals.data(), codes.data(), (int)vals.size()), &rv_set);
    invoke("main");
    out = fetch();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / (double)iters;
  fprintf(stderr, "E2E_NS %.0f\n", ns);

  // Write output bytes + a shape line
  int64_t nb = 1; for (int i = 0; i < out->ndim; i++) nb *= out->shape[i];
  nb *= (out->dtype.bits / 8);
  std::vector<char> obuf(nb);
  out.CopyToBytes(obuf.data(), nb);
  std::ofstream of(out_path, std::ios::binary); of.write(obuf.data(), nb); of.close();
  // shape/dtype metadata
  std::ofstream om(out_path + ".meta");
  om << (int)out->dtype.code << " " << (int)out->dtype.bits << " " << out->ndim;
  for (int i = 0; i < out->ndim; i++) om << " " << out->shape[i];
  om << "\n"; om.close();
  fprintf(stderr, "wrote %ld bytes\n", (long)nb);
  return 0;
}
