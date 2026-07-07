#include "Backend.h"
#include "Dialects.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Program model extracted from the iface/gemmini ops.
//===----------------------------------------------------------------------===//

struct LeafTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype;
  std::string role;
};

struct Job {
  std::string lhsName;
  int64_t M = 0, K = 0;       // lhs is M x K
  std::string outName;
  int64_t outM = 0, outN = 0; // out is M x N
  std::vector<std::string> epilogue;
  std::string outDtype = "i32";
  double accScale = 1.0;
  bool hasScale = false;
  std::string accHandle;
};

struct ConvRecipe {
  bool present = false;
  std::string source;  // IFM
  std::string target;  // derived im2col P
  int kh = 0, kw = 0, ci = 0;
  std::vector<int64_t> stride{1, 1};
  std::vector<int64_t> padding{0, 0, 0, 0};
  std::vector<int64_t> dilation{1, 1};
  std::string layout = "nhwc";
};

struct Program {
  bool isMovement = false;
  // movement
  std::string mvSrc, mvDst, mvDtype;
  std::vector<int64_t> mvShape;
  // matmul / conv
  std::string weightName;
  int64_t K = 0, N = 0;       // weight is K x N
  std::vector<Job> jobs;
  ConvRecipe conv;
  std::vector<LeafTensor> tensors; // for cb tensors table (leaves + derived + outputs)
};

static std::string opSuffix(Operation *op) {
  StringRef n = op->getName().getStringRef();
  size_t dot = n.rfind('.');
  return (dot == StringRef::npos) ? n.str() : n.substr(dot + 1).str();
}

static std::string dtypeStr(Type t) {
  if (auto it = dyn_cast<IntegerType>(t)) {
    return ("i" + std::to_string(it.getWidth()));
  }
  return "i8";
}

static bool tensorShape(Value v, std::vector<int64_t> &shape, std::string &dtype) {
  auto rt = dyn_cast<RankedTensorType>(v.getType());
  if (!rt)
    return false;
  shape.assign(rt.getShape().begin(), rt.getShape().end());
  dtype = dtypeStr(rt.getElementType());
  return true;
}

static std::string strAttr(Operation *op, StringRef name, StringRef dflt = "") {
  if (auto a = op->getAttrOfType<StringAttr>(name))
    return a.getValue().str();
  return dflt.str();
}

static std::vector<std::string> strArrayAttr(Operation *op, StringRef name) {
  std::vector<std::string> out;
  if (auto a = op->getAttrOfType<ArrayAttr>(name))
    for (Attribute e : a)
      if (auto s = dyn_cast<StringAttr>(e))
        out.push_back(s.getValue().str());
  return out;
}

static std::vector<int64_t> intArrayAttr(Operation *op, StringRef name) {
  std::vector<int64_t> out;
  if (auto a = op->getAttrOfType<ArrayAttr>(name))
    for (Attribute e : a)
      if (auto i = dyn_cast<IntegerAttr>(e))
        out.push_back(i.getInt());
  return out;
}

// Extract a Program by walking the module body in order (works for either dialect).
static LogicalResult extractProgram(ModuleOp m, Program &prog) {
  DenseMap<Value, std::string> valName;
  DenseMap<Value, Value> residentWeight; // resident result -> weight tensor value
  DenseMap<Value, Operation *> mmByAcc;   // matmul acc result -> matmul op
  bool haveWeight = false;

  for (Operation &op : *m.getBody()) {
    std::string suf = opSuffix(&op);
    if (suf == "tensor") {
      std::vector<int64_t> shape;
      std::string dtype;
      tensorShape(op.getResult(0), shape, dtype);
      std::string name = strAttr(&op, "name");
      std::string role = strAttr(&op, "role", "input");
      valName[op.getResult(0)] = name;
      prog.tensors.push_back({name, shape, dtype, role});
    } else if (suf == "resident_pack") {
      residentWeight[op.getResult(0)] = op.getOperand(0);
    } else if (suf == "matmul") {
      mmByAcc[op.getResult(0)] = &op;
      // record weight (once)
      Value resV = op.getOperand(1);
      if (residentWeight.count(resV) && !haveWeight) {
        Value w = residentWeight[resV];
        std::vector<int64_t> shape;
        std::string dtype;
        tensorShape(w, shape, dtype);
        if (shape.size() == 2) {
          prog.weightName = valName.count(w) ? valName[w] : std::string("W");
          prog.K = shape[0];
          prog.N = shape[1];
          haveWeight = true;
        }
      }
    } else if (suf == "commit") {
      Operation *mm = mmByAcc.lookup(op.getOperand(0));
      if (!mm)
        return failure();
      Job j;
      Value lhs = mm->getOperand(0);
      std::vector<int64_t> ls;
      std::string ld;
      tensorShape(lhs, ls, ld);
      j.lhsName = valName.count(lhs) ? valName[lhs] : std::string("A");
      if (ls.size() == 2) {
        j.M = ls[0];
        j.K = ls[1];
      }
      std::vector<int64_t> os;
      std::string od;
      tensorShape(op.getResult(0), os, od);
      j.outName = strAttr(&op, "name", "Y");
      if (os.size() == 2) {
        j.outM = os[0];
        j.outN = os[1];
      }
      j.epilogue = strArrayAttr(&op, "epilogue");
      j.outDtype = strAttr(&op, "output_dtype", "i32");
      if (auto sa = op.getAttrOfType<FloatAttr>("acc_scale")) {
        j.accScale = sa.getValueAsDouble();
        j.hasScale = true;
      }
      j.accHandle = "__acc" + std::to_string(prog.jobs.size());
      prog.jobs.push_back(j);
      prog.tensors.push_back({j.outName, {j.outM, j.outN}, j.outDtype, "output"});
    } else if (suf == "conv2d") {
      // weight from resident
      Value resV = op.getOperand(1);
      Value w = residentWeight.lookup(resV);
      std::vector<int64_t> ws;
      std::string wd;
      tensorShape(w, ws, wd);
      if (ws.size() == 2 && !haveWeight) {
        prog.weightName = valName.count(w) ? valName[w] : std::string("W");
        prog.K = ws[0];
        prog.N = ws[1];
        haveWeight = true;
      }
      // ifm
      Value ifm = op.getOperand(0);
      std::vector<int64_t> is;
      std::string id;
      tensorShape(ifm, is, id);
      std::string ifmName = valName.count(ifm) ? valName[ifm] : std::string("IFM");
      // output
      std::vector<int64_t> os;
      std::string od;
      tensorShape(op.getResult(0), os, od);
      std::string outName = strAttr(&op, "name", "Y0");
      // kernel = [kh, kw, ci, co]
      auto kernel = intArrayAttr(&op, "kernel");
      ConvRecipe &c = prog.conv;
      c.present = true;
      c.source = ifmName;
      c.target = "__im2col";
      c.kh = kernel.size() > 0 ? (int)kernel[0] : 0;
      c.kw = kernel.size() > 1 ? (int)kernel[1] : 0;
      c.ci = kernel.size() > 2 ? (int)kernel[2] : 0;
      auto st = intArrayAttr(&op, "stride");
      if (st.size() == 2) c.stride = st;
      auto pd = intArrayAttr(&op, "padding");
      if (pd.size() == 4) c.padding = pd;
      auto dl = intArrayAttr(&op, "dilation");
      if (dl.size() == 2) c.dilation = dl;
      c.layout = strAttr(&op, "layout", "nhwc");
      // job: lhs = derived im2col P (M x K), out = Y0 (M x N)
      Job j;
      j.lhsName = c.target;
      j.M = os.size() == 2 ? os[0] : 0; // = N*Ho*Wo
      j.K = prog.K;                     // = kh*kw*ci
      j.outName = outName;
      j.outM = j.M;
      j.outN = os.size() == 2 ? os[1] : 0;
      j.epilogue = strArrayAttr(&op, "epilogue");
      j.outDtype = strAttr(&op, "output_dtype", "i32");
      if (auto sa = op.getAttrOfType<FloatAttr>("acc_scale")) {
        j.accScale = sa.getValueAsDouble();
        j.hasScale = true;
      }
      j.accHandle = "__acc0";
      prog.jobs.push_back(j);
      // tensors: ifm already added as leaf; add derived P and output
      prog.tensors.push_back({c.target, {j.M, j.K}, "i8", "input"});
      prog.tensors.push_back({outName, {j.outM, j.outN}, j.outDtype, "output"});
    } else if (suf == "movement") {
      prog.isMovement = true;
      Value src = op.getOperand(0);
      std::vector<int64_t> shape;
      std::string dtype;
      tensorShape(src, shape, dtype);
      prog.mvSrc = valName.count(src) ? valName[src] : std::string("X");
      prog.mvShape = shape;
      prog.mvDtype = dtype;
      prog.mvDst = strAttr(&op, "name", "Y0");
      prog.tensors.push_back({prog.mvDst, shape, dtype, "output"});
    }
    // evict: ignore
  }
  return success();
}

//===----------------------------------------------------------------------===//
// JSON helpers
//===----------------------------------------------------------------------===//

static std::string jShape(const std::vector<int64_t> &s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(s[i]);
  }
  out += "]";
  return out;
}

static std::string fmtDouble(double d) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.9g", d);
  return std::string(buf);
}

} // namespace

//===----------------------------------------------------------------------===//
// convert-iface-to-gemmini
//===----------------------------------------------------------------------===//

LogicalResult backend::convertIfaceToGemmini(ModuleOp m) {
  // If already gemmini (no merlin_iface ops), nothing to do.
  bool hasIface = false;
  for (Operation &op : *m.getBody())
    if (op.getName().getDialectNamespace() == "merlin_iface") {
      hasIface = true;
      break;
    }
  if (!hasIface)
    return success();

  OpBuilder b(m.getContext());
  IRMapping map;
  SmallVector<Operation *> toErase;
  for (Operation &op : *m.getBody()) {
    if (op.getName().getDialectNamespace() != "merlin_iface")
      continue;
    std::string newName = "gemmini." + opSuffix(&op);
    b.setInsertionPoint(&op);
    OperationState st(op.getLoc(), newName);
    SmallVector<Value> operands;
    for (Value v : op.getOperands())
      operands.push_back(map.lookupOrDefault(v));
    st.addOperands(operands);
    st.addTypes(op.getResultTypes());
    st.addAttributes(op.getAttrs());
    Operation *ng = b.create(st);
    for (auto it : llvm::zip(op.getResults(), ng->getResults()))
      map.map(std::get<0>(it), std::get<1>(it));
    toErase.push_back(&op);
  }
  for (Operation *op : llvm::reverse(toErase))
    op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// emit-command-buffer
//===----------------------------------------------------------------------===//

LogicalResult backend::emitCommandBuffer(ModuleOp m, StringRef path) {
  Program prog;
  if (failed(extractProgram(m, prog)))
    return failure();

  std::string out;
  out += "{\n";
  out += "  \"abi_version\": \"0.1\",\n";
  out += "  \"target\": \"gemmini\",\n";
  out += "  \"backend\": \"gemmini_oot_cpp_v0\",\n";

  // tensors table (dedup by name)
  out += "  \"tensors\": {\n";
  {
    DenseMap<StringRef, bool> seen;
    bool first = true;
    for (const LeafTensor &t : prog.tensors) {
      if (t.name.empty()) continue;
      if (seen.count(t.name)) continue;
      seen[t.name] = true;
      if (!first) out += ",\n";
      first = false;
      out += "    \"" + t.name + "\": {\"shape\": " + jShape(t.shape) +
             ", \"dtype\": \"" + t.dtype + "\", \"role\": \"" + t.role + "\"}";
    }
    out += "\n  },\n";
  }

  // commands
  out += "  \"commands\": [\n";
  if (prog.isMovement) {
    out += "    {\"opcode\": \"VECTOR_MAP\", \"operands\": {\"lhs\": \"" + prog.mvSrc +
           "\", \"dst\": \"" + prog.mvDst +
           "\"}, \"attributes\": {\"combine\": \"identity\"}}\n";
  } else {
    std::vector<std::string> cmds;
    cmds.push_back("    {\"opcode\": \"RES_PACK\", \"operands\": {\"src\": \"" +
                   prog.weightName + "\", \"dst\": \"__res\"}, \"attributes\": "
                   "{\"layout\": \"packed_rhs\"}}");
    for (const Job &j : prog.jobs) {
      cmds.push_back("    {\"opcode\": \"MATMUL_RESIDENT\", \"operands\": {\"lhs\": \"" +
                     j.lhsName + "\", \"rhs\": \"__res\", \"dst\": \"" + j.accHandle +
                     "\"}}");
      std::string epi = "[";
      for (size_t i = 0; i < j.epilogue.size(); ++i) {
        if (i) epi += ", ";
        epi += "\"" + j.epilogue[i] + "\"";
      }
      epi += "]";
      std::string attrs = "\"epilogue\": " + epi + ", \"output_dtype\": \"" +
                          j.outDtype + "\"";
      if (j.hasScale)
        attrs += ", \"acc_scale\": " + fmtDouble(j.accScale);
      cmds.push_back("    {\"opcode\": \"COMMIT\", \"operands\": {\"src\": \"" +
                     j.accHandle + "\", \"dst\": \"" + j.outName +
                     "\"}, \"attributes\": {" + attrs + "}}");
    }
    cmds.push_back("    {\"opcode\": \"EVICT\", \"operands\": {\"handle\": \"__res\"}}");
    for (size_t i = 0; i < cmds.size(); ++i) {
      out += cmds[i];
      if (i + 1 < cmds.size()) out += ",";
      out += "\n";
    }
  }
  out += "  ]";

  // params (conv im2col recipe)
  if (prog.conv.present) {
    const ConvRecipe &c = prog.conv;
    out += ",\n  \"params\": {\n    \"im2col_recipes\": [\n      {";
    out += "\"source\": \"" + c.source + "\", \"target\": \"" + c.target + "\", ";
    out += "\"kh\": " + std::to_string(c.kh) + ", \"kw\": " + std::to_string(c.kw) +
           ", \"ci\": " + std::to_string(c.ci) + ", ";
    out += "\"stride\": " + jShape(c.stride) + ", \"padding\": " + jShape(c.padding) +
           ", \"dilation\": " + jShape(c.dilation) + ", ";
    out += "\"layout\": \"" + c.layout + "\"}";
    out += "\n    ]\n  }";
  }
  out += "\n}\n";

  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec)
    return failure();
  os << out;
  os.close();
  return success();
}

//===----------------------------------------------------------------------===//
// convert-gemmini-to-llvm-rocc
//===----------------------------------------------------------------------===//

namespace {

constexpr int64_t DIM = 16;
constexpr uint64_t ACC_BIT = 0x80000000ULL;
constexpr uint64_t ACCUM_BIT = 0x40000000ULL;
constexpr uint64_t FULLC_BIT = 0x20000000ULL;
constexpr uint64_t GARBAGE = 0xFFFFFFFFULL;

static int64_t ceil16(int64_t x) { return ((x + 15) / 16) * 16; }
static int64_t tiles(int64_t x) { return ceil16(x) / 16; }

static uint32_t f32bits(double d) {
  float f = (float)d;
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

struct KernelEmitter {
  OpBuilder &b;
  Location loc;
  Block *entry;
  Type i64;

  Value cst(int64_t v) {
    return b.create<LLVM::ConstantOp>(loc, i64, b.getI64IntegerAttr(v));
  }
  // pointer arg + byte offset -> i64 address
  Value addr(unsigned argIdx, int64_t byteOff) {
    Value p = b.create<LLVM::PtrToIntOp>(loc, i64, entry->getArgument(argIdx));
    if (byteOff == 0)
      return p;
    return b.create<LLVM::AddOp>(loc, p, cst(byteOff));
  }
  void insn(int funct, Value rs1, Value rs2) {
    std::string asmStr = ".insn r 0x7b, 0x3, " + std::to_string(funct) +
                         ", x0, $0, $1";
    b.create<LLVM::InlineAsmOp>(
        loc, TypeRange{}, ValueRange{rs1, rs2}, asmStr, "r,r,~{memory}",
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        LLVM::tailcallkind::TailCallKind::None, LLVM::AsmDialectAttr(),
        ArrayAttr());
  }
  void fence() {
    b.create<LLVM::InlineAsmOp>(
        loc, TypeRange{}, ValueRange{}, "fence", "~{memory}",
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        LLVM::tailcallkind::TailCallKind::None, LLVM::AsmDialectAttr(),
        ArrayAttr());
  }
  // packed dim|addr field: (rows<<48)|(cols<<32)|addr
  Value packed(uint64_t addr) {
    return cst((int64_t)((16ULL << 48) | (16ULL << 32) | (addr & 0xFFFFFFFFULL)));
  }
};

} // namespace

LogicalResult backend::lowerToLlvmRocc(ModuleOp m) {
  Program prog;
  if (failed(extractProgram(m, prog)))
    return failure();

  MLIRContext *ctx = m.getContext();
  // strip merlin_iface.* module attributes so a downstream (re)parse that does not
  // register the merlin_iface dialect does not choke on them.
  {
    SmallVector<StringRef> drop;
    for (NamedAttribute a : m->getAttrs())
      if (a.getName().strref().starts_with("merlin_iface"))
        drop.push_back(a.getName().strref());
    for (StringRef n : drop)
      m->removeAttr(n);
  }
  // erase existing body ops
  SmallVector<Operation *> toErase;
  for (Operation &op : *m.getBody())
    toErase.push_back(&op);
  for (Operation *op : llvm::reverse(toErase))
    op->erase();

  OpBuilder b(ctx);
  b.setInsertionPointToStart(m.getBody());
  Location loc = m.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  Type i64 = b.getI64Type();

  // arg count
  unsigned nArgs = prog.isMovement ? 2 : (1 + prog.jobs.size() * 2);
  SmallVector<Type> argTys(nArgs, ptrTy);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto fnTy = LLVM::LLVMFunctionType::get(voidTy, argTys, /*isVarArg=*/false);
  auto func = b.create<LLVM::LLVMFuncOp>(loc, "gemmini_kernel", fnTy);
  Block *entry = func.addEntryBlock(b);
  b.setInsertionPointToStart(entry);

  KernelEmitter E{b, loc, entry, i64};

  E.fence();
  // FLUSH(0)
  E.insn(7, E.cst(0), E.cst(0));

  if (prog.isMovement) {
    // movement: mvin X -> spad 0, mvout spad 0 -> Y
    int64_t R = prog.mvShape.size() > 0 ? prog.mvShape[0] : DIM;
    int64_t C = prog.mvShape.size() > 1 ? prog.mvShape[1] : DIM;
    int64_t rp = ceil16(R), cp = ceil16(C);
    int64_t Rt = tiles(R), Ct = tiles(C);
    // CONFIG_EX (harmless; satisfies nothing required but fine)
    // CONFIG_LD stride = cp (bytes, i8)
    E.insn(0, E.cst((int64_t)((0x3F800000ULL << 32) | (16ULL << 16) | (1ULL << 8) | 1ULL)),
           E.cst(cp));
    unsigned spad = 0;
    DenseMap<uint64_t, unsigned> spadOf;
    for (int64_t i = 0; i < Rt; ++i)
      for (int64_t j = 0; j < Ct; ++j) {
        unsigned sp = (unsigned)((i * Ct + j) * DIM);
        int64_t off = (i * DIM * cp + j * DIM); // i8
        E.insn(2, E.addr(0, off), E.packed(sp)); // MVIN
      }
    (void)spad;
    // CONFIG_ST stride = cp bytes, no act, identity scale
    E.insn(0, E.cst((int64_t)((0ULL << 2) | 2ULL)),
           E.cst((int64_t)(((uint64_t)0x3F800000ULL << 32) | (uint64_t)cp)));
    for (int64_t i = 0; i < Rt; ++i)
      for (int64_t j = 0; j < Ct; ++j) {
        unsigned sp = (unsigned)((i * Ct + j) * DIM);
        int64_t off = (i * DIM * cp + j * DIM);
        E.insn(3, E.addr(1, off), E.packed(sp)); // MVOUT (spad readout)
      }
    E.fence();
    b.create<LLVM::ReturnOp>(loc, ValueRange{});
    return success();
  }

  // matmul / conv
  int64_t K = prog.K, N = prog.N;
  int64_t Kt = tiles(K), Nt = tiles(N);
  int64_t kp = ceil16(K), np = ceil16(N);

  // CONFIG_EX (weight stationary)
  uint64_t exRs1 = ((uint64_t)0x3F800000ULL << 32) | (1ULL << 16) | (1ULL << 2);
  uint64_t exRs2 = (1ULL << 48);
  E.insn(0, E.cst((int64_t)exRs1), E.cst((int64_t)exRs2));

  // weight mvin: CONFIG_LD(stride = np), then mvin B tiles. weight = arg0.
  uint64_t ldRs1 = ((uint64_t)0x3F800000ULL << 32) | (16ULL << 16) | (1ULL << 8) | 1ULL;
  E.insn(0, E.cst((int64_t)ldRs1), E.cst(np));
  const int64_t B_BASE = 0;
  for (int64_t k = 0; k < Kt; ++k)
    for (int64_t j = 0; j < Nt; ++j) {
      unsigned sp = (unsigned)(B_BASE + (k * Nt + j) * DIM);
      int64_t off = (k * DIM * np + j * DIM); // i8
      E.insn(2, E.addr(0, off), E.packed(sp));
    }

  for (size_t jobIdx = 0; jobIdx < prog.jobs.size(); ++jobIdx) {
    const Job &job = prog.jobs[jobIdx];
    int64_t M = job.M;
    int64_t Mt = tiles(M);
    int64_t mp = ceil16(M);
    (void)mp;
    unsigned lhsArg = 1 + (unsigned)jobIdx;
    unsigned outArg = 1 + (unsigned)prog.jobs.size() + (unsigned)jobIdx;
    bool fullC = (job.outDtype != "i8");
    int outElem = fullC ? 4 : 1;
    int64_t A_BASE = 2048 + (int64_t)jobIdx * 2048;

    // CONFIG_LD(stride = kp) for A
    E.insn(0, E.cst((int64_t)ldRs1), E.cst(kp));
    for (int64_t i = 0; i < Mt; ++i)
      for (int64_t k = 0; k < Kt; ++k) {
        unsigned sp = (unsigned)(A_BASE + (i * Kt + k) * DIM);
        int64_t off = (i * DIM * kp + k * DIM); // i8
        E.insn(2, E.addr(lhsArg, off), E.packed(sp));
      }

    // CONFIG_ST: acc_act + acc_scale + stride
    bool relu = false;
    for (auto &s : job.epilogue)
      if (s == "relu") relu = true;
    double scale = job.hasScale ? job.accScale : 1.0;
    uint64_t stRs1 = ((uint64_t)(relu ? 1ULL : 0ULL) << 2) | 2ULL;
    uint64_t stRs2 = ((uint64_t)f32bits(scale) << 32) | (uint64_t)(np * outElem);
    E.insn(0, E.cst((int64_t)stRs1), E.cst((int64_t)stRs2));

    uint64_t fc = fullC ? FULLC_BIT : 0ULL;
    for (int64_t i = 0; i < Mt; ++i)
      for (int64_t j = 0; j < Nt; ++j) {
        uint64_t accOff = (uint64_t)((i * Nt + j) * DIM);
        for (int64_t k = 0; k < Kt; ++k) {
          unsigned bsp = (unsigned)(B_BASE + (k * Nt + j) * DIM);
          unsigned asp = (unsigned)(A_BASE + (i * Kt + k) * DIM);
          uint64_t cAddr = ACC_BIT | fc | accOff | (k > 0 ? ACCUM_BIT : 0ULL);
          // PRELOAD: rs1 = weight spad (BD), rs2 = C acc addr
          E.insn(6, E.packed(bsp), E.packed(cAddr));
          // COMPUTE_PRELOADED: rs1 = A spad, rs2 = garbage
          E.insn(4, E.packed(asp),
                 E.cst((int64_t)((16ULL << 48) | (16ULL << 32) | GARBAGE)));
        }
        // MVOUT
        uint64_t readAddr = ACC_BIT | fc | accOff;
        int64_t off = (i * DIM * np + j * DIM) * outElem;
        E.insn(3, E.addr(outArg, off), E.packed(readAddr));
      }
  }

  E.fence();
  b.create<LLVM::ReturnOp>(loc, ValueRange{});
  return success();
}
