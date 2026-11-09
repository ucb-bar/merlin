"""Independent byte-oracle proof of strided copies, including overlapping views."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import prove_im2col_stream as corpus

from rvv.transforms import static_copy

PIPELINE = "canonicalize,cse,convert-scf-to-cf,expand-strided-metadata,lower-affine,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts,canonicalize,cse"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--compiler-python", type=Path, required=True)
    p.add_argument("--linker", type=Path, required=True)
    p.add_argument("--spike", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    tools = {x: a.llvm_bin / x for x in ("mlir-opt", "mlir-translate", "clang-23")}
    common = [
        "-target",
        "riscv64",
        "-march=rv64gcv_zvl128b",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-fno-vectorize",
        "-fno-slp-vectorize",
    ]
    objects = []
    declarations = []
    checks = []
    records = []
    cases = [
        ("disjoint", 16, 128, (9, 1), (11, 1)),
        ("forward_overlap", 16, 17, (9, 1), (9, 1)),
        ("backward_overlap", 17, 16, (9, 1), (9, 1)),
        ("same_view", 16, 16, (9, 1), (9, 1)),
        ("transposed_overlap", 16, 20, (9, 1), (1, 8)),
        ("negative_source", 50, 18, (-9, 1), (9, 1)),
        ("broadcast_source", 16, 18, (0, 1), (9, 1)),
        ("broadcast_destination", 18, 16, (9, 1), (0, 1)),
        ("negative_destination", 16, 50, (9, 1), (-9, 1)),
    ]
    for bits in (8, 16, 32, 64):
        for extent in (3, 0):
            shape = (extent, 7)
            host = {}
            for mode in ("before", "after"):
                name = f"copy{bits}_{extent}_{mode}"
                typ = f"memref<{extent}x7xi{bits}, strided<[?, ?], offset: ?>>"
                source = a.output / (name + ".mlir")
                source.write_text(
                    f"module {{func.func @{name}(%s: {typ}, %d: {typ}) {{memref.copy %s, %d : {typ} to {typ} return}}}}"
                )
                report = static_copy.lower_file(
                    source,
                    pipeline=PIPELINE,
                    mlir_opt=tools["mlir-opt"],
                    compiler_python=a.compiler_python,
                    work=a.output / (name + "-lower"),
                    enabled=mode == "after",
                )
                if mode == "after":
                    assert report["copy_report"]["rewritten"] == (1 if extent else 0)
                ll = a.output / (name + ".ll")
                corpus.run(
                    [
                        tools["mlir-translate"],
                        "--mlir-to-llvmir",
                        report["llvm_dialect"],
                        "-o",
                        ll,
                    ],
                    a.output,
                    name + "-translate",
                )
                obj = a.output / (name + ".o")
                objects.append(obj)
                corpus.run(
                    [tools["clang-23"], *common, "-c", ll, "-o", obj],
                    a.output,
                    name + "-object",
                )
                library = a.output / (name + ".so")
                support = a.output / (name + ".c")
                support.write_text(corpus.RUNTIME)
                corpus.run(
                    [
                        tools["clang-23"],
                        "-O2",
                        "-shared",
                        "-fPIC",
                        ll,
                        support,
                        "-o",
                        library,
                    ],
                    a.output,
                    name + "-host",
                )
                lib = ctypes.CDLL(str(library))
                fn = getattr(lib, name)
                fn.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    *([ctypes.c_int64] * 5),
                ] * 2
                host[mode] = (fn, lib)
                types = ["void*", "void*", *(["int64_t"] * 5)] * 2
                declarations.append(f"extern void {name}({','.join(types)});")
            for case, so, do, ss, ds in (
                cases if extent else [("zero_extent", 16, 17, (9, 1), (9, 1))]
            ):
                word = bits // 8
                initial = (
                    (np.arange((256 + 32) * word, dtype=np.uint64) * 71 + 13) % 256
                ).astype(np.uint8)
                gold = initial.copy()
                for i, j in itertools.product(range(extent), range(7)):
                    source_index = (16 + so + i * ss[0] + j * ss[1]) * word
                    destination_index = (16 + do + i * ds[0] + j * ds[1]) * word
                    gold[destination_index : destination_index + word] = gold[
                        source_index : source_index + word
                    ].copy()
                for mode in ("before", "after"):
                    data = initial.copy()
                    base = data.ctypes.data + 16 * word
                    host[mode][0](
                        base, base, so, extent, 7, *ss, base, base, do, extent, 7, *ds
                    )
                    np.testing.assert_array_equal(data, gold)
                    name = f"copy{bits}_{extent}_{mode}"
                    case_id = len(records)
                    init_name = "initial" + str(case_id)
                    expected = "expected" + str(case_id)
                    declarations.append(
                        corpus.c_array(init_name, initial, "unsigned char")
                    )
                    declarations.append(corpus.c_array(expected, gold, "unsigned char"))
                    checks.append(
                        f"{{static unsigned char data[{len(initial)}] __attribute__((aligned(64)));memcpy(data,{init_name},sizeof(data));void*b=data+{16 * word};{name}(b,b,{so},{extent},7,{ss[0]},{ss[1]},b,b,{do},{extent},7,{ds[0]},{ds[1]});for(size_t i=0;i<sizeof(data);i++)if(data[i]!={expected}[i])return {case_id + 1};}}"
                    )
                    records.append(
                        {
                            "element_bits": bits,
                            "shape": list(shape),
                            "case": case,
                            "mode": mode,
                            "pass_host": True,
                            "compared_storage_bytes": len(initial),
                            "expected_sha256": hashlib.sha256(
                                gold.tobytes()
                            ).hexdigest(),
                        }
                    )
    main = a.output / "main.c"
    main.write_text(
        "#define BAREMETAL\n"
        + corpus.RUNTIME
        + "\n".join(declarations)
        + "\nint main(void){"
        + "".join(checks)
        + "return 0;}\n"
    )
    start = a.output / "start.S"
    start.write_text(corpus.START)
    link = a.output / "link.ld"
    link.write_text(corpus.LINK)
    elf = a.output / "aliases.elf"
    corpus.run(
        [
            tools["clang-23"],
            *common,
            "-fno-builtin",
            "-nostdlib",
            "--ld-path=" + str(a.linker),
            "-Wl,-T," + str(link),
            main,
            start,
            *objects,
            "-o",
            elf,
        ],
        a.output,
        "link",
    )
    corpus.run(
        [
            a.spike,
            "--iris-vme",
            "--isa=rv64gcv_zvl128b",
            "-p1",
            "-m0x80000000:0x8000000",
            elf,
        ],
        a.output,
        "native",
        timeout=60,
    )
    for row in records:
        row["pass_native"] = True
    report = {
        "schema": "rvv.static_copy_alias_proof.v1",
        "pass": True,
        "cases": records,
        "native_elf_sha256": corpus.sha(elf),
        "independent_oracle": "byte-exact immediate lexicographic source-read/destination-write over full overlapping storage",
        "alias_assumptions_added": False,
        "physical_timing_claimed": False,
        "input_sha256": {
            str(x): corpus.sha(x)
            for x in [
                Path(__file__),
                Path(static_copy.__file__),
                Path(corpus.__file__),
                *tools.values(),
                a.spike,
                a.linker,
            ]
        },
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pass": True,
                "cases": len(records),
                "compared_bytes": sum(r["compared_storage_bytes"] for r in records),
            }
        )
    )


if __name__ == "__main__":
    main()
