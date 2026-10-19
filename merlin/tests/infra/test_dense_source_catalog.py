"""Neutral source-bound catalog for exercising ordinary compilation.

The byte-memory implementation below is a test execution model, not a target
backend, hardware capability, private reference compiler or timing model.
"""

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, schedule_source_dense
from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.sched.isa import InstrDef, InstructionSet, Operand
from merlin.transforms.dense_planning import DenseResources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation(
    result,
    problem,
    storage,
    dtypes,
    symbol,
    *,
    lose_continuation=False,
    primitive_entry=None,
    bridge_override=None,
    readout_body=None,
    output_dtype=None,
):
    """Independent byte-addressed synchronous execution of actual emitted calls."""
    from test_compact_dense import instances

    views = (storage.lhs, storage.rhs, storage.accumulator)
    sizes = {
        backing: max((v.base + v.capacity) * v.unit_bytes for v in views if v.backing == backing)
        for backing in {v.backing for v in views}
    }
    arrays = {backing: f"bank_{i}" for i, backing in enumerate(sorted(sizes))}
    declarations = "".join(f"static unsigned char {arrays[b]}[{n + 16}];\n" for b, n in sizes.items())
    left, right, sums = [arrays[v.backing] + "+8" for v in views]
    ct = dict(i8="int8_t", i16="int16_t", i32="int32_t")
    output_dtype = dtypes[2] if output_dtype is None else output_dtype
    preamble = f"""
#include <stdint.h>
{declarations}
static volatile uint64_t command_count, load_count, store_count;
static int64_t weight_row, sum_row, continuation, columns;
static void put(unsigned char *p,uint32_t value,int bytes) {{
 for(int b=0;b<bytes;b++) p[b]=(unsigned char)(value>>(8*b));
}}
static int32_t get(const unsigned char *p,int bytes) {{
 uint32_t value=0;for(int b=0;b<bytes;b++) value|=(uint32_t)p[b]<<(8*b);
 if(bytes==1)return (int8_t)value;if(bytes==2)return (int16_t)value;return (int32_t)value;
}}
static void do_lhs_load(int64_t a,int64_t b,int64_t r,int64_t c,int64_t rows,int64_t cols,
 int64_t dst,int64_t step,{ct[dtypes[0]]} *src) {{
 command_count++;load_count+=rows*cols*sizeof(*src);
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)
  put(({left})+(dst+i)*{storage.lhs.unit_bytes}+j*sizeof(*src),
      (uint32_t)src[(r+i*step)*{problem.k}+c+j],sizeof(*src));
}}
static void do_weight_load(int64_t a,int64_t b,int64_t r,int64_t c,int64_t rows,int64_t cols,
 int64_t dst,int64_t step,{ct[dtypes[1]]} *src) {{
 command_count++;load_count+=rows*cols*sizeof(*src);
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)
  put(({right})+(dst+i)*{storage.rhs.unit_bytes}+j*sizeof(*src),
      (uint32_t)src[(r+i*step)*{problem.n}+c+j],sizeof(*src));
}}
static void do_preload(int64_t w,int64_t k,int64_t n,int64_t acc,int64_t cont,
 int64_t rows,int64_t cols,int64_t depth) {{
 command_count++;weight_row=w;sum_row=acc;continuation={"0" if lose_continuation else "cont"};columns=cols;
}}
static void do_compute(int64_t lhs,int64_t m,int64_t k,int64_t rows,int64_t depth) {{
 command_count++;
 for(int i=0;i<rows;i++)for(int j=0;j<columns;j++){{
  unsigned char *target=({sums})+(sum_row+i)*{storage.accumulator.unit_bytes}+j*4;
  uint32_t value=continuation?(uint32_t)get(target,4):0;
  for(int t=0;t<depth;t++){{
   int32_t x=get(({left})+(lhs+i)*{storage.lhs.unit_bytes}+t*{problem.lhs_bytes},{problem.lhs_bytes});
   int32_t y=get(({right})+(weight_row+t)*{storage.rhs.unit_bytes}+j*{problem.rhs_bytes},{problem.rhs_bytes});
   value+=(uint32_t)x*(uint32_t)y;
  }}
  put(target,value,4);
 }}
}}
static void do_store(int64_t r,int64_t c,int64_t rows,int64_t cols,int64_t acc,int32_t *dst) {{
 command_count++;store_count+=rows*cols*4;
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)
  dst[(r+i)*{problem.n}+c+j]=get(({sums})+(acc+i)*{storage.accumulator.unit_bytes}+j*4,4);
}}
uint64_t {symbol}_count(int which) {{return which==0?command_count:which==1?load_count:store_count;}}
void {symbol}_reset(void) {{command_count=load_count=store_count=0;}}
int {symbol}_guards(void) {{
"""
    for backing, size in sizes.items():
        valid = " || ".join(
            f"(i>={8 + v.base * v.unit_bytes} && i<{8 + (v.base + v.capacity) * v.unit_bytes})"
            for v in views
            if v.backing == backing
        )
        preamble += f"for(int i=0;i<{size + 16};i++) if(!({valid}) && {arrays[backing]}[i]) return 0;\n"
    preamble += "return 1;}\n"
    if readout_body is not None:
        preamble += f"""
static void do_store_i8(int64_t r,int64_t c,int64_t rows,int64_t cols,int64_t acc,int8_t *dst) {{
 command_count++;store_count+=rows*cols;
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++){{
  int32_t value=get(({sums})+(acc+i)*{storage.accumulator.unit_bytes}+j*4,4);
  {readout_body}
  dst[(r+i)*{problem.n}+c+j]=(int8_t)value;
 }}
}}
"""
    if primitive_entry is not None:
        operands = {
            "lhs_load": ("a", "b", "r", "c", "rows", "cols", "dst", "step"),
            "weight_load": ("a", "b", "r", "c", "rows", "cols", "dst", "step"),
            "preload": ("w", "k", "n", "acc", "cont", "rows", "cols", "depth"),
            "compute": ("lhs", "m", "k", "rows", "depth"),
            "store": ("r", "c", "rows", "cols", "acc"),
        }
        if readout_body is not None:
            operands["store_i8"] = operands["store"]
        for name, names in operands.items():
            position = preamble.index("{", preamble.index("static void do_" + name + "(")) + 1
            preamble = preamble[:position] + "\n " + primitive_entry(name, names) + preamble[position:]
    definitions = {}
    for call, _ in instances(result.compiled.kernel):
        if call.instr in definitions:
            continue
        suffix = (
            ", lhs"
            if call.instr == "lhs_load"
            else ", rhs"
            if call.instr == "weight_load"
            else (", output" if call.instr in ("store", "store_i8") else "")
        )
        definitions[call.instr] = InstrDef(
            call.instr,
            tuple(Operand(key, "int") for key, _ in call.args),
            render_c=lambda args, op=call.instr, end=suffix: f"do_{op}({', '.join(args)}{end});",
        )
    instruction_set = InstructionSet(
        "neutral-synchronous-byte-machine", definitions, ct, "/* synchronous completion */"
    )
    body = result.emit_c(instruction_set, symbol=symbol + "_implementation")
    body += f"\nvoid {symbol}(void *lhs,void *rhs,void *out) {{{symbol}_implementation(lhs,rhs,out);}}\n"
    bridge = f"""
typedef struct {{void *allocated,*aligned;int64_t offset,sizes[2],strides[2];}} neutral_descriptor;
void _mlir_ciface_borrowed_{symbol}(const neutral_descriptor *a,const neutral_descriptor *b,
                                   const neutral_descriptor *c) {{
 {symbol}(({ct[dtypes[0]]}*)a->aligned+a->offset,
          ({ct[dtypes[1]]}*)b->aligned+b->offset,
          ({ct[output_dtype]}*)c->aligned+c->offset);
}}
"""
    return preamble + body + (bridge if bridge_override is None else bridge_override)


class Catalog:
    """Original test provider; production does not discover this class or its helpers."""

    def __init__(self, model, *, streaming=False, lose_continuation=False):
        self.selected, _, self.storage, adapter = model
        self.adapter = replace(adapter, qualification_identity="neutral-source-catalog-byte-machine-v1")
        self.streaming, self.lose_continuation = streaming, lose_continuation
        self.results, self.commands = [], []
        self.manifest = None

    def __call__(self, prepared, work):
        from merlin.llvmlower.toolchain import clang
        from merlin.runtime.backends.spike import gcc_path

        work = Path(work)
        work.mkdir(parents=True)
        source_digest = digest(prepared)
        (work / "admitted-source.mlir").write_bytes(Path(prepared).read_bytes())
        module = parse_mlir_file(prepared)
        bindings, kernels, objects = [], [], []
        for ordinal, op in enumerate(module.walk()):
            if op.name != "linalg.generic":
                continue
            symbol = f"neutral_kernel_{len(kernels)}"
            resources = (
                DenseResources(1 << 20, 1 << 20, 1, 1, 2, partial_output_tiles=True, partial_output_blocks=True)
                if self.streaming
                else None
            )
            try:
                result = schedule_source_dense(
                    module,
                    op,
                    selected_contracts=self.selected,
                    storage=self.storage,
                    adapter=self.adapter,
                    name=symbol,
                    static_budget=80,
                    external_buffers_disjoint=True,
                    verification_timeout_ms=20_000,
                    resources=resources,
                )
            except DenseSourceUnavailable:
                continue
            if result is None:
                raise ValueError("neutral provider needs an explicit host fallback for unavailable tiling")
            # Recover dimensions from the admitted current source, not a benchmark table.
            m, k = op.inputs[0].type.get_shape()
            _, n = op.inputs[1].type.get_shape()
            problem = self.selected.dense.problem(m, k, n)
            source = work / f"{symbol}.c"
            source.write_text(
                implementation(
                    result,
                    problem,
                    self.storage,
                    self.selected.dense.dtypes,
                    symbol,
                    lose_continuation=self.lose_continuation,
                )
            )
            obj = work / f"{symbol}.o"
            self.run(
                [
                    str(clang()),
                    "--target=riscv64-unknown-elf",
                    "-march=rv64gc",
                    "-mabi=lp64d",
                    "-mcmodel=medany",
                    "-O2",
                    "-ffreestanding",
                    "-fno-builtin",
                    "-c",
                    str(source),
                    "-o",
                    str(obj),
                ],
                obj,
            )
            objects.append(obj)
            tag = getattr(op.attributes.get("prov.region_id"), "data", "")
            if not tag:
                raise ValueError("source-bound catalog needs an explicit original region identity")
            bindings.append(
                dict(
                    region=tag,
                    symbol=symbol,
                    source_operation_ordinal=ordinal,
                    tensor_types=[str(v.type) for v in (*op.inputs, op.results[0])],
                )
            )
            kernels.append(dict(symbol=symbol, dimensions=dict(batch=1, m=m, n=n, k=k), batched=False))
            self.results.append((result, problem, source, symbol))
        if not kernels:
            raise ValueError("no supported source contractions for the neutral catalog")
        obj = work / "catalog.o"
        ld = gcc_path().with_name("riscv64-unknown-elf-ld")
        self.run([str(ld), "-r", *[str(o) for o in objects], "-o", str(obj)], obj)
        for result, *_ in self.results:
            result.revalidate()
        self.selected.revalidate()
        if digest(prepared) != source_digest:
            raise ValueError("prepared source changed during neutral catalog compilation")
        self.manifest = work / "catalog.json"
        self.manifest.write_text(
            json.dumps(
                dict(
                    source_sha256=source_digest,
                    coverage_complete=True,
                    covered_contractions=len(bindings),
                    matched_contractions=len(bindings),
                    unique_kernels=len(kernels),
                    bindings=bindings,
                    kernels=kernels,
                    abi=dict(
                        argument_order=["lhs", "rhs", "out"],
                        pointee_layout="dense_row_major",
                        batch_call="whole_batch",
                        dtypes=list(self.selected.dense.dtypes),
                    ),
                    compilation=dict(object_sha256=digest(obj)),
                ),
                indent=2,
            )
        )
        (work / "provider-commands.json").write_text(json.dumps(self.commands, indent=2))
        (work / "scheduling.json").write_text(
            json.dumps(
                [
                    dict(
                        source_stamp=asdict(r.source_stamp),
                        source_passes=[asdict(step) for step in r.source_passes],
                        realization=r.compiled.realization,
                    )
                    for r, *_ in self.results
                ],
                indent=2,
            )
        )
        return self.manifest, obj

    def fresh_writers(self, prepared, sidecar, work):
        """Explicit test-qualified full-write/borrowed effects, not name inference."""
        from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
        from merlin.transforms.certificates import ir_text

        data = json.loads(Path(sidecar).read_text())
        bindings = {row["region"]: row["symbol"] for row in json.loads(self.manifest.read_text())["bindings"]}
        contracts = []
        for row in data["routed"]:
            original = row["symbol"]
            kernel = bindings[row["source_region"]]
            contracts.append(
                FreshTensorWriterContract(
                    original,
                    2,
                    (2,),
                    "borrowed_" + kernel,
                    64,
                    declaration_abi="expanded_memref",
                    wrapper_symbol=original + "_fresh",
                    allow_initialized_writers=True,
                )
            )
        module = parse_mlir_file(prepared)
        report = rewrite_fresh_tensor_writers(module, contracts)
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        result = work / "fresh-writers.mlir"
        result.write_text(ir_text(module))
        (work / "fresh-writers.json").write_text(json.dumps(report, indent=2))
        return result

    def run(self, argv, output):
        started = time.perf_counter()
        inputs = [
            dict(path=str(path), sha256=digest(path))
            for arg in argv[1:]
            if (path := Path(arg)).suffix in (".c", ".o") and path != output and path.is_file()
        ]
        result = subprocess.run(argv, check=True, capture_output=True)
        if any(digest(item["path"]) != item["sha256"] for item in inputs):
            raise ValueError("neutral provider compiler inputs changed during compilation")
        self.commands.append(
            dict(
                inputs=inputs,
                argv=argv,
                returncode=result.returncode,
                output=str(output),
                sha256=digest(output),
                wall_seconds=time.perf_counter() - started,
            )
        )


def qualify_native(provider, w0, w1, data, state, position, directory):
    """Execute the exact emitted target C separately; counters are requested traffic."""
    import ctypes
    import shutil

    import numpy as np

    cc = shutil.which("cc")
    if cc is None:
        raise RuntimeError("independent native compiler required for byte-machine qualification")
    library_path = Path(directory) / "qualification.so"
    sources = [str(item[2]) for item in provider.results]
    subprocess.run([cc, "-shared", "-fPIC", "-O2", *sources, "-o", str(library_path)], check=True, capture_output=True)
    library = ctypes.CDLL(str(library_path))
    kernels = []
    for result, p, source, symbol in provider.results:
        operation = getattr(library, symbol)
        operation.argtypes = [ctypes.c_void_p] * 3
        operation.restype = None
        counter = getattr(library, symbol + "_count")
        counter.argtypes = [ctypes.c_int]
        counter.restype = ctypes.c_uint64
        reset = getattr(library, symbol + "_reset")
        reset()
        kernels.append((operation, counter, getattr(library, symbol + "_guards")))
    observed = []
    original = [a.copy() for a in (w0, w1, data)]
    for step in range(3):
        a = np.add(data, state, dtype=state.dtype)
        first = np.full((data.shape[0], w0.shape[1]), 0x7F7F7F7F, np.int32)
        before_a = a.copy()
        want = (a.astype(np.int64) @ w0.astype(np.int64)).astype(np.int32)
        kernels[0][0](a.ctypes.data, w0.ctypes.data, first.ctypes.data)
        np.testing.assert_array_equal(a, before_a)
        np.testing.assert_array_equal(first, want)
        activation = np.maximum(first, 0).astype(state.dtype)
        second = np.full(state.shape, 0x7F7F7F7F, np.int32)
        before_activation = activation.copy()
        want = (activation.astype(np.int64) @ w1.astype(np.int64)).astype(np.int32)
        kernels[1][0](activation.ctypes.data, w1.ctypes.data, second.ctypes.data)
        np.testing.assert_array_equal(activation, before_activation)
        np.testing.assert_array_equal(second, want)
        state = np.add(second, np.int32(position), dtype=np.int32).astype(state.dtype)
        position += 1
        observed.append(state.copy())
        assert all(guard() for _, _, guard in kernels)
    for after, before in zip((w0, w1, data), original, strict=True):
        np.testing.assert_array_equal(after, before)
    counts = [
        dict(commands=counter(0), requested_load_bytes=counter(1), requested_store_bytes=counter(2))
        for _, counter, _ in kernels
    ]
    return (
        dict(
            counters=counts,
            calls=6,
            guards=True,
            complete_outputs=True,
            implementation_sources=[dict(path=str(p), sha256=digest(p)) for p in sources],
            scope="exact emitted target C in native execution; host model glue independently interpreted",
            traffic_unit="requested bytes; not physical bus traffic",
        ),
        np.stack(observed),
        state,
        position,
    )


def compilation_memory():
    """Context manager: sampled Linux process-tree RSS, including shared pages."""
    import os
    import threading
    from contextlib import contextmanager

    @contextmanager
    def measured():
        stop = threading.Event()
        result = dict(
            peak_tree_rss_bytes_sampled_100ms=0,
            samples=0,
            scope="same-process RSS plus descendants; summed shared pages; short peaks may be missed",
        )

        def sample():
            pending = [os.getpid()]
            seen = set()
            rss = 0
            while pending:
                pid = pending.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                try:
                    rss += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                    for task in Path(f"/proc/{pid}/task").iterdir():
                        try:
                            pending.extend(int(p) for p in (task / "children").read_text().split())
                        except (FileNotFoundError, PermissionError, ProcessLookupError):
                            pass
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
            result["peak_tree_rss_bytes_sampled_100ms"] = max(result["peak_tree_rss_bytes_sampled_100ms"], rss)
            result["samples"] += 1

        def run():
            while not stop.is_set():
                sample()
                stop.wait(0.1)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            yield result
        finally:
            stop.set()
            thread.join()
            sample()

    return measured()
