"""Explicit LLVM/guarded-host/RVV functional proof; no performance claim.

Requires Merlin/xDSL in PYTHONPATH. Tool paths are explicit and byte-bound in
report.json. Run with a fresh --output directory; no existing result is edited.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from im2col_fixture import _module
from merlin.common import mlir_query as mq

from rvv.packing import im2col_stream as ip

RUNTIME = r"""
#include <stdint.h>
#include <stddef.h>
#ifndef BAREMETAL
#include <stdlib.h>
#else
static unsigned char heap[4*1024*1024] __attribute__((aligned(64)));
static size_t heap_pos;
void *malloc(size_t n){n=(n+63)&~(size_t)63;if(heap_pos+n>sizeof(heap))return 0;void*p=heap+heap_pos;heap_pos+=n;return p;}
void free(void*p){(void)p;}
void *memcpy(void*d,const void*s,size_t n){for(size_t i=0;i<n;i++)((char*)d)[i]=((const char*)s)[i];return d;}
void *memset(void*d,int x,size_t n){for(size_t i=0;i<n;i++)((char*)d)[i]=x;return d;}
#endif
static uint64_t max_copy_bytes;
struct unranked{int64_t rank;void*descriptor;};
struct ranked{void*allocated;char*aligned;int64_t offset;};
void memrefCopy(int64_t bytes,struct unranked*src,struct unranked*dst){
 struct ranked*s=src->descriptor,*d=dst->descriptor;
 int64_t*ss=(int64_t*)(s+1),*ds=(int64_t*)(d+1);
 int64_t count=1;for(int i=0;i<src->rank;i++)count*=ss[i];
 if((uint64_t)(count*bytes)>max_copy_bytes)max_copy_bytes=count*bytes;
 for(int64_t linear=0;linear<count;linear++){
  int64_t q=linear,so=s->offset,doff=d->offset;
  for(int j=src->rank-1;j>=0;j--){int64_t ix=q%ss[j];q/=ss[j];so+=ix*ss[src->rank+j];doff+=ix*ds[dst->rank+j];}
  for(int64_t b=0;b<bytes;b++)d->aligned[doff*bytes+b]=s->aligned[so*bytes+b];
 }
}
uint64_t get_max_copy_bytes(void){return max_copy_bytes;}
"""
START = r"""
.section .text.init
.globl _start
_start:
 csrr a0,mhartid
 bnez a0,park
 la sp,stack_top
 la t0,trap
 csrw mtvec,t0
 li t0,0x6600
 csrs mstatus,t0
 csrwi fcsr,0
 call main
 slli a0,a0,1
 ori a0,a0,1
 la t0,tohost
 sd a0,0(t0)
park:
 wfi
 j park
trap:
 li t1,511
 la t0,tohost
 sd t1,0(t0)
 j park
.section .tohost,"aw",@progbits
.balign 64
.globl tohost
tohost: .dword 0
.balign 64
.globl fromhost
fromhost: .dword 0
.section .bss
.balign 64
stack_bottom: .skip 1048576
stack_top:
"""
LINK = """ENTRY(_start)
SECTIONS { . = 0x80000000; .text : { *(.text.init) *(.text*) } .rodata : { *(.rodata*) } .data : { *(.data*) *(.sdata*) } .tohost : { *(.tohost) } .bss : { *(.bss*) *(.sbss*) *(COMMON) } }
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, directory, name, timeout=60):
    r = subprocess.run(
        [str(x) for x in argv], capture_output=True, timeout=timeout, check=False
    )
    (directory / (name + ".stdout")).write_bytes(r.stdout)
    (directory / (name + ".stderr")).write_bytes(r.stderr)
    if r.returncode:
        raise RuntimeError(f"{name} exited {r.returncode}: {r.stderr[-2000:]!r}")
    return r


def allocation_audit(text, symbol, max_output_panel_bytes):
    body = text.split("define void @" + symbol + "(", 1)[1].split("\n}", 1)[0]
    blocks = {"entry": []}
    current = "entry"
    for line in body.splitlines()[1:]:
        label = re.match(r"^([A-Za-z0-9_.]+):", line)
        if label:
            current = label[1]
            blocks[current] = []
        blocks[current].append(line)
    edges = {
        key: re.findall(r"label %([A-Za-z0-9_.]+)", "\n".join(lines))
        for key, lines in blocks.items()
    }

    def cyclic(start):
        pending = list(edges[start])
        visited = set()
        while pending:
            at = pending.pop()
            if at == start:
                return True
            if at not in visited:
                visited.add(at)
                pending.extend(edges.get(at, ()))
        return False

    loop_blocks = {key for key in blocks if cyclic(key)}
    allocations = []
    loop_copy_bytes = []
    scoped_stack_blocks = []
    for key, lines in blocks.items():
        block = "\n".join(lines)
        sizes = re.findall(r"call ptr @malloc\(i64 ([0-9]+)\)", block)
        if sizes:
            assert key not in loop_blocks, "heap allocation on loop backedge"
            allocations.extend(int(v) for v in sizes)
        if key in loop_blocks and " = alloca " in block:
            first = block.index(" = alloca ")
            last = block.rindex(" = alloca ")
            assert "@llvm.stacksave" in block[:first], (
                "unscoped stack allocation in loop"
            )
            assert "@llvm.stackrestore" in block[last:], (
                "unscoped stack allocation in loop"
            )
            scoped_stack_blocks.append(key)
        if key in loop_blocks:
            for line in lines:
                if "call void @llvm.memcpy" in line:
                    size = re.search(r", i64 ([0-9]+), i1 false", line)
                    assert size is not None, "unbounded dynamic copy on loop backedge"
                    assert int(size[1]) <= max_output_panel_bytes, (
                        "copy exceeds output panel"
                    )
                    loop_copy_bytes.append(int(size[1]))
    return {
        "no_heap_allocation_on_backedge": True,
        "no_copy_larger_than_output_panel_on_backedge": True,
        "loop_copy_bytes": loop_copy_bytes,
        "loop_stack_allocations_have_save_restore": True,
        "scoped_stack_blocks": scoped_stack_blocks,
        "constant_heap_requests": allocations,
        "activation_budget_excludes_allocator_slack_output_and_vector_scratch": True,
    }


def c_array(name, value, type_):
    return (
        f"static const {type_} {name}[]={{"
        + ",".join(str(int(v)) for v in value.flat)
        + "};\n"
    )


def wrapper(shape, f, k, m, symbol):
    n, c, h, w = shape
    types = [
        "void*",
        "void*",
        *["int64_t"] * 9,
        "void*",
        "void*",
        *["int64_t"] * 5,
        "void*",
        "void*",
        *["int64_t"] * 5,
    ]
    args = [
        "x",
        "x",
        "0",
        n,
        c,
        h,
        w,
        c * h * w,
        h * w,
        w,
        1,
        "w",
        "w",
        0,
        f,
        k,
        k,
        1,
        "y",
        "y",
        0,
        f,
        m,
        m,
        1,
    ]
    return f"extern void {symbol}({','.join(types)});\nvoid run_{symbol}(void*x,void*w,void*y){{{symbol}({','.join(map(str, args))});}}\n"


def expected(inp, weight, p):
    col = np.empty(
        (p["c"] * p["kh"] * p["kw"], p["n"] * p["oh"] * p["ow"]), dtype=np.int64
    )
    for c in range(p["c"]):
        for kh in range(p["kh"]):
            for kw in range(p["kw"]):
                for n in range(p["n"]):
                    for oh in range(p["oh"]):
                        for ow in range(p["ow"]):
                            col[
                                (c * p["kh"] + kh) * p["kw"] + kw,
                                (n * p["oh"] + oh) * p["ow"] + ow,
                            ] = inp[
                                n,
                                c,
                                oh * p["sh"] + kh * p["dh"],
                                ow * p["sw"] + kw * p["dw"],
                            ]
    return (weight.astype(np.int64) @ col).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--spike", type=Path, required=True)
    ap.add_argument("--linker", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    tools = {
        name: a.llvm_bin / name for name in ("mlir-opt", "mlir-translate", "clang-23")
    }
    report = {
        "schema": "rvv.im2col_stream_proof.v1",
        "timing_certified": False,
        "hardware_performance_measured": False,
        "tools": {str(p): sha(p) for p in [*tools.values(), a.spike, a.linker]},
        "sources": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path(ip.__file__),
                Path(ip.__file__).parents[1] / "policy/panel_geometry.py",
            ]
        },
        "cases": [],
    }
    base = {
        "c": 2,
        "kh": 3,
        "kw": 3,
        "n": 1,
        "oh": 4,
        "ow": 32,
        "f": 8,
        "sh": 1,
        "sw": 1,
        "dh": 1,
        "dw": 1,
    }
    cases = [
        {"n": 2, "oh": 3, "ow": 131, "f": 5, "sh": 2, "sw": 2, "dh": 2, "dw": 2},
        {"c": 1, "kh": 1, "kw": 1, "oh": 1, "ow": 7, "f": 3},
        {"c": 1, "kh": 2, "kw": 2, "oh": 2, "ow": 37, "f": 4, "sh": 2, "dw": 2},
    ]
    for vlen in (128, 256, 512, 1024):
        d = a.output / f"vlen{vlen}"
        d.mkdir()
        objects = []
        wrappers = []
        arrays = []
        checks = []
        common = [
            "-target",
            "riscv64",
            "-march=rv64gcv_zvl" + str(vlen) + "b",
            "-mabi=lp64d",
            "-mcmodel=medany",
            "-O2",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ]
        for index, change in enumerate(cases):
            p = base | change
            symbol = f"forward_{index}"
            module = mq.parse(_module(**p).replace("@forward", f"@{symbol}"))
            r = ip.rewrite_module(
                module,
                ip.PanelTarget(vlen, 64, 8, 32, 2 if index == 2 else 1, 65536),
                enabled=True,
            )
            assert r.streamed == 1
            module.verify()
            src = d / f"case{index}.mlir"
            src.write_text(str(module))
            sched = d / f"case{index}.schedule.mlir"
            sched.write_text(ip.lowering_schedule(r))
            lowered = d / f"case{index}.llvm.mlir"
            run(
                [
                    tools["mlir-opt"],
                    src,
                    "--pass-pipeline=builtin.module("
                    + ip.lowering_pipeline(sched)
                    + ")",
                    "-o",
                    lowered,
                ],
                d,
                f"lower{index}",
            )
            ll = d / f"case{index}.ll"
            run(
                [tools["mlir-translate"], "--mlir-to-llvmir", lowered, "-o", ll],
                d,
                f"translate{index}",
            )
            ll.write_text(
                ll.read_text().replace("@dealloc_helper", f"@dealloc_helper_{index}")
            )
            obj = d / f"case{index}.o"
            objects.append(obj)
            run([tools["clang-23"], *common, "-c", ll, "-o", obj], d, f"compile{index}")
            assembly = d / f"case{index}.s"
            run(
                [tools["clang-23"], *common, "-S", ll, "-o", assembly],
                d,
                f"assembly{index}",
            )
            assert any(
                op in assembly.read_text()
                for op in ("vmacc.", "vwmacc.", "vmadd.", "vmul.", "vwmul.")
            ), "consumer silently scalarized"
            optimized = d / f"case{index}.optimized.ll"
            run(
                [tools["clang-23"], *common, "-S", "-emit-llvm", ll, "-o", optimized],
                d,
                f"optimize{index}",
            )
            shape = (
                p["n"],
                p["c"],
                (p["oh"] - 1) * p["sh"] + (p["kh"] - 1) * p["dh"] + 1,
                (p["ow"] - 1) * p["sw"] + (p["kw"] - 1) * p["dw"] + 1,
            )
            k = p["c"] * p["kh"] * p["kw"]
            m = p["n"] * p["oh"] * p["ow"]
            inp = (
                ((np.arange(np.prod(shape), dtype=np.int64) * 71 + 13) % 256 - 128)
                .astype(np.int8)
                .reshape(shape)
            )
            weight = (
                ((np.arange(p["f"] * k, dtype=np.int64) * 37 + 5) % 256 - 128)
                .astype(np.int8)
                .reshape(p["f"], k)
            )
            gold = expected(inp, weight, p)
            wrap = wrapper(shape, p["f"], k, m, symbol)
            wrappers.append(wrap)
            wc = d / f"case{index}.wrapper.c"
            wc.write_text(RUNTIME + wrap)
            library = d / f"case{index}.so"
            run(
                [tools["clang-23"], "-O2", "-shared", "-fPIC", ll, wc, "-o", library],
                d,
                f"hostcompile{index}",
            )
            lib = ctypes.CDLL(str(library))
            fn = getattr(lib, "run_" + symbol)
            fn.argtypes = [ctypes.c_void_p] * 3
            buf = np.full(gold.size + 32, 0x5A17C03D, dtype=np.int32)
            fn(inp.ctypes.data, weight.ctypes.data, buf[16:-16].ctypes.data)
            np.testing.assert_array_equal(buf[16:-16], gold.flat)
            assert (buf[:16] == 0x5A17C03D).all() and (buf[-16:] == 0x5A17C03D).all()
            lib.get_max_copy_bytes.restype = ctypes.c_uint64
            copy = lib.get_max_copy_bytes()
            assert copy <= p["f"] * r.entries[0]["geometry"]["panel_width"] * 4
            arrays.extend(
                [
                    c_array("input_" + str(index), inp, "int8_t"),
                    c_array("weight_" + str(index), weight, "int8_t"),
                    c_array("expected_" + str(index), gold, "int32_t"),
                ]
            )
            checks.append(
                f"{{static int32_t buf[{gold.size + 32}];for(size_t i=0;i<{gold.size + 32};i++)buf[i]=0x5a17c03d;run_{symbol}((void*)input_{index},(void*)weight_{index},buf+16);for(size_t i=0;i<{gold.size};i++)if(buf[i+16]!=expected_{index}[i])return {index + 1};for(int i=0;i<16;i++)if(buf[i]!=0x5a17c03d||buf[{gold.size + 16}+i]!=0x5a17c03d)return {index + 10};}}"
            )
            report["cases"].append(
                {
                    "vlen": vlen,
                    "params": p,
                    "pass_host": True,
                    "full_output_words": gold.size,
                    "expected_sha256": hashlib.sha256(
                        gold.astype("<i4").tobytes()
                    ).hexdigest(),
                    "max_panel_copy_bytes": copy,
                    "rewrite": r.to_dict(),
                    "llvm_sha256": sha(ll),
                    "assembly_sha256": sha(assembly),
                    "rvv_mac_present": True,
                    "allocation_audit": allocation_audit(
                        optimized.read_text(),
                        symbol,
                        p["f"] * r.entries[0]["geometry"]["panel_width"] * 4,
                    ),
                    "optimized_llvm_sha256": sha(optimized),
                }
            )
        native = d / "main.c"
        native.write_text(
            "#define BAREMETAL\n"
            + RUNTIME
            + "\n".join(wrappers + arrays)
            + f'\nint main(void){{unsigned long vl;__asm__ volatile("csrr %0,vlenb":"=r"(vl));if(vl!={vlen // 8})return 200;'
            + "".join(checks)
            + "return 0;}\n"
        )
        start = d / "start.S"
        start.write_text(START)
        link = d / "link.ld"
        link.write_text(LINK)
        elf = d / "test.elf"
        run(
            [
                tools["clang-23"],
                *common,
                "-fno-builtin",
                "-nostdlib",
                "--ld-path=" + str(a.linker),
                "-Wl,-T," + str(link),
                native,
                start,
                *objects,
                "-o",
                elf,
            ],
            d,
            "link",
        )
        isa = "rv64gcv_zvl" + str(vlen) + "b"
        run(
            [
                a.spike,
                "--iris-vme",
                "--isa=" + isa,
                "-p1",
                "-m0x80000000:0x8000000",
                elf,
            ],
            d,
            "native",
            timeout=60,
        )
        for row in report["cases"]:
            if row["vlen"] == vlen:
                row.update(pass_native=True, native_elf_sha256=sha(elf), native_isa=isa)
    report["pass"] = all(
        r.get("pass_native") and r["pass_host"] for r in report["cases"]
    )
    report["artifact_sha256"] = {
        str(p.relative_to(a.output)): sha(p) for p in a.output.rglob("*") if p.is_file()
    }
    (a.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": report["pass"],
                "cases": len(report["cases"]),
                "output": str(a.output),
            }
        )
    )


if __name__ == "__main__":
    main()
