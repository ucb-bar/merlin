"""Check explicit math-library placement and execute the actual RV64 calls."""

import copy
import hashlib
import json
import os
import random
import subprocess
from dataclasses import replace
from math import prod
from pathlib import Path

import pytest

from merlin.common.paths import runtime_dir
from merlin.compile.c_unary_math import OPERATIONS, emit_unary_math_c, validate_policy
from merlin.llvmlower.link_supplier_trace import observe_link_suppliers, trace_symbol_flags
from merlin.semantic_compiler.constant_math_backend import build_helper
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.targetgen.host_linkage_contract import merge_linkage_requirements


def policy(digest="0" * 64):
    return {
        "schema": "merlin.compiled_unary_math.v1",
        "policy": "selected-libm-rne-finite-v1",
        "isa": "rv64gc",
        "abi": "lp64d",
        "effects": "enabled-fpu-rne-unobserved-flags-and-errno-v1",
        "operations": sorted(OPERATIONS),
        "linkage": {
            "schema": "merlin.host_linkage_contract.v1",
            "supplier": "selected_math_archive",
            "archive_sha256": digest,
            "symbols": ["cos", "cosf", "sin", "sinf"],
        },
    }


def request(operation, shape=(2, 3)):
    dtype, symbol = OPERATIONS[operation]
    type_ = TensorType(shape, dtype, "raw-bit-pattern")
    arg = SemanticNode("arg", "input", (), type_, effect="input")
    out = SemanticNode(
        "out",
        operation,
        (arg.id,),
        type_,
        (("numeric_semantics", f"source-{dtype}-{symbol.rstrip('f')}-unqualified-v1"),),
        index_maps=pointwise_identity_maps(len(shape), 1),
    )
    return KernelRequest((arg, out), (out.id,), ("external",), ((arg.id, "external"),), "synthetic")


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_source_signature_and_library_symbol_are_checked(operation):
    source = request(operation)
    record, files = emit_unary_math_c(source, "unary", policy())
    assert record["host_linkage"]["symbols"] == [OPERATIONS[operation][1]]
    assert record["framework_equivalence"] == "not_established"
    assert record["host_abi"] == "lp64d" and "frrm" in files["unary.c"]
    for wrong in (replace(source.nodes[-1], attrs=()), replace(source.nodes[-1], index_maps=())):
        with pytest.raises(ValueError):
            emit_unary_math_c(replace(source, nodes=(source.nodes[0], wrong)), "wrong", policy())
    for change in ({"abi": "lp64"}, {"isa": "rv32gc"}, {"effects": "unknown"}, {"operations": []}):
        with pytest.raises(ValueError):
            validate_policy({**policy(), **change})
    changed = copy.deepcopy(policy())
    changed["linkage"]["symbols"] = ["sin"]
    with pytest.raises(ValueError, match="symbols"):
        validate_policy(changed)


def test_linkage_merges_same_archive_and_rejects_ambiguous_suppliers():
    a, b = policy()["linkage"], {**policy()["linkage"], "symbols": ["sin"]}
    assert merge_linkage_requirements([a, b, a]) == [a]
    assert merge_linkage_requirements([]) == []
    with pytest.raises(ValueError, match="different math archives"):
        merge_linkage_requirements([a, {**b, "archive_sha256": "1" * 64}])


@pytest.fixture(scope="module")
def toolchain(tmp_path_factory):
    gcc, spike = os.environ.get("MERLIN_RISCV_GCC"), os.environ.get("MERLIN_SPIKE")
    if not gcc or not spike:
        pytest.skip("select the RV64 compiler and Spike for execution qualification")
    gcc, spike = Path(gcc), Path(spike)
    libraries = [
        Path(subprocess.check_output([str(gcc), flag], text=True).strip()).resolve()
        for flag in ("-print-file-name=libm.a", "-print-file-name=libc.a", "-print-libgcc-file-name")
    ]
    helper, _ = build_helper(tmp_path_factory.mktemp("math-reference") / "helper")
    return gcc, spike, libraries, helper


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_actual_rv64_library_call_matches_bounded_reference_and_rejects_bad_invocations(tmp_path, toolchain, operation):
    run_source_case(tmp_path, toolchain, request(operation, (548,)))


def run_source_case(tmp_path, toolchain, source):
    """The same bounded checker accepts captured regions without changing their IR."""
    if source.constants:
        raise ValueError("sampled unary-math qualification needs a runtime-input region")
    gcc, spike, libraries, reference = toolchain
    operation = source.nodes[-1].op
    dtype, symbol = OPERATIONS[operation]
    width, fraction = (32, 23) if dtype == "f32" else (64, 52)
    sign, inf = 1 << (width - 1), ((1 << (width - 1)) - 1) - ((1 << fraction) - 1)
    # Compile the function before evaluator values exist; no LTO is used later.
    count = prod(source.nodes[-1].type.shape)
    if not 1 <= count <= 65536:
        raise ValueError("bounded unary-math probe supports 1..65536 tensor elements")
    selected = policy(hashlib.sha256(libraries[0].read_bytes()).hexdigest())
    record, files = emit_unary_math_c(source, "unary", selected)
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    flags = [
        "-march=rv64gcv",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fno-builtin",
        "-fno-tree-vectorize",
        "-fno-tree-loop-distribute-patterns",
        "-ffreestanding",
    ]
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    compile_command = [
        str(gcc),
        *flags,
        "-Wall",
        "-Wextra",
        "-Werror",
        "-c",
        str(tmp_path / "unary.c"),
        "-o",
        str(tmp_path / "unary.o"),
    ]
    subprocess.run(compile_command, env=env, check=True, capture_output=True)
    rng = random.Random(width)
    # Source encodings include both zeros, subnormal boundaries and huge angles.
    values = [0, sign, 1, sign | 1, (1 << fraction) - 1, 1 << fraction, inf - 1, sign | (inf - 1)]
    values = values[:count]
    values += [(rng.randrange(inf) | (sign if rng.randrange(2) else 0)) for _ in range(count - len(values))]
    byte_values = b"".join(value.to_bytes(width // 8, "little") for value in values)
    bad = inf.to_bytes(width // 8, "little")
    nan = (inf | 1).to_bytes(width // 8, "little")
    harness = f"""#include <stdint.h>
#include <string.h>
#include "htif.h"
#include "unary.h"
static unsigned char input[{len(byte_values)}]={{{",".join(map(str, byte_values))}}};
static unsigned char output[{len(byte_values)}+16];
static int untouched(void) {{for(unsigned i=0;i<sizeof(output);++i) if(output[i]!=0xcd) return 0;return 1;}}
int main(long hart) {{
 if(hart) for(;;){{}}
 console_init();const void *inputs[]={{input}};void *outputs[]={{output+8}};memset(output,0xcd,sizeof(output));
 if(unary(0,inputs,outputs)) htif_exit(1);
 unsigned initial_rounding;__asm__ volatile("frrm %0":"=r"(initial_rounding));
 if(initial_rounding) htif_exit(7);
 htif_line_flush(0);
 for(unsigned i=0;i<{count}u;++i) {{uint64_t x=0;for(unsigned j=0;j<{width // 8}u;++j)
   x|=(uint64_t)output[8+{width // 8}u*i+j]<<(8u*j);htif_puthex(x);htif_putc(' ');}}
 htif_putc('\\n');htif_line_flush(1);
 for(unsigned i=0;i<8;++i) if(output[i]!=0xcd || output[sizeof(output)-1-i]!=0xcd) htif_exit(2);
 for(unsigned mode=1;mode<=3;++mode) {{memset(output,0xcd,sizeof(output));
   __asm__ volatile("fsrm %0"::"r"(mode));if(unary(0,inputs,outputs)!=-3 || !untouched()) htif_exit(3);}}
 __asm__ volatile("fsrm zero");
 const unsigned char nonfinite[]={{{",".join(map(str, bad))}}};
 memcpy(input+sizeof(input)-sizeof(nonfinite),nonfinite,sizeof(nonfinite));
 if(unary(0,inputs,outputs)!=-2 || !untouched()) htif_exit(4);
 const unsigned char nan[]={{{','.join(map(str,nan))}}};
 memcpy(input+sizeof(input)-sizeof(nan),nan,sizeof(nan));
 if(unary(0,inputs,outputs)!=-2 || !untouched()) htif_exit(8);
 outputs[0]=input;if(unary(0,inputs,outputs)!=-1) htif_exit(5);
 if(unary(0,0,0)!=-1) htif_exit(6);
 htif_puts("DONE\\n");htif_exit(0);
}}
"""
    (tmp_path / "main.c").write_text(harness)
    runtime = runtime_dir() / "baremetal/spike"
    trace = trace_symbol_flags([symbol])
    command = [
        str(gcc),
        *flags,
        "-nostdlib",
        "-nostartfiles",
        "-Wl,--no-relax",
        "-I",
        str(runtime),
        "-T",
        str(runtime / "link.ld"),
        str(runtime / "crt.S"),
        str(runtime / "htif.c"),
        str(runtime / "libc_min.c"),
        str(tmp_path / "main.c"),
        str(tmp_path / "unary.o"),
        *trace,
        *map(str, libraries),
        "-o",
        str(tmp_path / "test.elf"),
    ]
    linked = subprocess.run(command, env=env, capture_output=True, text=True)
    (tmp_path / "link.log").write_text(linked.stdout + linked.stderr)
    assert linked.returncode == 0, linked.stderr
    identity = {
        "path": str(libraries[0]),
        "bytes": libraries[0].stat().st_size,
        "sha256": selected["linkage"]["archive_sha256"],
    }
    observation = observe_link_suppliers(
        argv=command, input_identities=[identity], expected_suppliers={symbol: libraries[0]}, stderr=linked.stderr
    )
    run_command = [str(spike), "--isa=rv64gcv", "-p1", str(tmp_path / "test.elf")]
    executed = subprocess.run(run_command, env=env, capture_output=True, text=True, timeout=30)
    (tmp_path / "execution.log").write_text(executed.stdout + executed.stderr)
    assert executed.returncode == 0 and "DONE" in executed.stdout, executed.stdout + executed.stderr
    observed = [int(token, 16) for token in executed.stdout.splitlines()[0].split()]
    assert len(observed) == len(values)
    expected = [reference.unary_reference(width, value, symbol.rstrip("f"))[0] for value in values]

    def ordered(bits):
        return sign - (bits & (sign - 1)) if bits & sign else sign + bits

    ulps = [abs(ordered(a) - ordered(b)) for a, b in zip(observed, expected, strict=True)]
    # Frozen bounded regression tolerance; no all-input libm accuracy proof.
    assert max(ulps) <= 2
    assert observed[:2] == expected[:2]  # Preserve sin(-0) and cos(±0).
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "operation": operation,
                "elements": count,
                "max_ulp": max(ulps),
                "bitwise_equal": sum(n == 0 for n in ulps),
                "supplier": observation,
                "compile_command": compile_command,
                "link_command": command,
                "run_command": run_command,
                "region_requirements": record,
            },
            indent=2,
        )
        + "\n"
    )
