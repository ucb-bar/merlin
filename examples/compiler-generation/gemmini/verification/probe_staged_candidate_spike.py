#!/usr/bin/env python3
"""Execute one byte-bound staged kernel and its generated descriptor shim on Spike.

This checks one finite input vector and its valid output window. It does not
review the software specification, attest RTL execution, or qualify whole-model
offload. The two objects consumed here are exactly those in the staged receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from merlin.common.digest import sha256_text
from merlin.common.paths import out_dir
from merlin.llvmlower.device_build import _nm, verify_object_symbol_binding
from merlin.llvmlower.exact_offload import _package_sha256
from merlin.runtime.backends.base import get_backend, harness_build_recipe
from merlin.targetgen.contract.resident_interface_abi import bind_single_resident_matmul
from merlin.targetgen.operation_numerics import integer_partial_sum_bound
from merlin.targetgen.runtime_build import derived_link_script


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_file(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or digest(path) != expected:
        raise ValueError(f"missing or changed staged file: {path}")


def run_command(argv: list[str], *, timeout: int) -> None:
    process = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f"build command failed: {process.stderr[-1200:]}")


def _inputs(m: int, n: int, k: int) -> tuple[list[int], list[int], list[int]]:
    lhs = [((row * 17 + depth * 7) % 255) - 127 for row in range(m) for depth in range(k)]
    weight = [((depth * 19 + col * 11) % 255) - 127 for depth in range(k) for col in range(n)]
    expected = [
        sum(lhs[row * k + depth] * weight[depth * n + col] for depth in range(k))
        for row in range(m)
        for col in range(n)
    ]
    return lhs, weight, expected


def _harness(m: int, n: int, k: int, lhs: list[int], weight: list[int]) -> str:
    return f"""#include <stdint.h>
#include <stdio.h>
typedef struct {{
  void *allocated, *aligned;
  intptr_t offset, sizes[2], strides[2];
}} merlin_memref_2d;
extern merlin_memref_2d merlin_staged_kernel(
    void *, void *, intptr_t, intptr_t, intptr_t, intptr_t, intptr_t,
    void *, void *, intptr_t, intptr_t, intptr_t, intptr_t, intptr_t,
    void *, void *, intptr_t, intptr_t, intptr_t, intptr_t, intptr_t);
static int8_t lhs[{m * k}] = {{{",".join(map(str, lhs))}}};
static int8_t weight[{k * n}] = {{{",".join(map(str, weight))}}};
static int32_t output[{m * n}];
int main(void) {{
  merlin_memref_2d rejected = merlin_staged_kernel(
      lhs, lhs, 0, {m + 1}, {k}, {k}, 1,
      weight, weight, 0, {k}, {n}, {n}, 1,
      output, output, 0, {m}, {n}, {n}, 1);
  if (rejected.allocated || rejected.aligned || rejected.offset ||
      rejected.sizes[0] || rejected.sizes[1] || rejected.strides[0] || rejected.strides[1])
    return 3;
  merlin_memref_2d result = merlin_staged_kernel(
      lhs, lhs, 0, {m}, {k}, {k}, 1,
      weight, weight, 0, {k}, {n}, {n}, 1,
      output, output, 0, {m}, {n}, {n}, 1);
  if (result.allocated != output || result.aligned != output || result.offset != 0 ||
      result.sizes[0] != {m} || result.sizes[1] != {n} ||
      result.strides[0] != {n} || result.strides[1] != 1)
    return 2;
  printf("OUT Y0 {m} {n}");
  for (int i = 0; i < {m * n}; ++i) printf(" %d", output[i]);
  printf("\\nDONE\\n");
  return 0;
}}
"""


def _executed_package_instructions(
    backend, elf: Path, output: Path, kernel_symbol: str, console: str, timeout: int
) -> dict:
    objdump = backend.gcc_path().with_name("riscv64-unknown-elf-objdump")
    disassembly = subprocess.run(
        [str(objdump), "-d", str(elf)], capture_output=True, text=True, timeout=timeout, check=True
    ).stdout
    (output / "disassembly.txt").write_text(disassembly, encoding="utf-8")
    expected = {}
    in_kernel = False
    for line in disassembly.splitlines():
        if f"<{kernel_symbol}>:" in line:
            in_kernel = True
            continue
        if in_kernel and line.strip().endswith(">:"):
            break
        address, separator, instruction = line.partition(":")
        if in_kernel and separator and ".insn" in instruction:
            fields = instruction.split()
            if fields and len(fields[0]) == 8:
                expected[int(address.strip(), 16)] = int(fields[0], 16)
    if not expected:
        raise ValueError("linked package kernel has no disassembled target instructions")
    trace = output / "spike_full.trace"
    flags, extension_dir = backend.spike_extension()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(extension_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
    process = subprocess.run(
        [str(backend.spike_path()), "-l", f"--log={trace}", "--instructions=250000", *flags, str(elf)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if process.returncode or process.stdout != console:
        raise ValueError("bounded Spike trace did not repeat the numerical output")
    if trace.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("bounded Spike trace exceeded 16 MiB")
    observed = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "core" and fields[1].endswith(":"):
            try:
                address = int(fields[2], 16)
                word = int(fields[3].strip("()"), 16)
            except ValueError:
                continue
            if expected.get(address) == word:
                observed.append(line)
    trace.unlink()
    if {int(line.split()[2], 16) for line in observed} != set(expected):
        raise ValueError("Spike did not execute every disassembled package instruction")
    excerpt = output / "spike_package_instructions.trace"
    excerpt.write_text("\n".join(observed) + "\n", encoding="utf-8")
    return {
        "static_pcs": len(expected),
        "executed_count": len(observed),
        "trace_excerpt_sha256": digest(excerpt),
        "disassembly_sha256": digest(output / "disassembly.txt"),
    }


def run(args: argparse.Namespace) -> dict:
    if args.target != "gemmini":
        raise ValueError("this target-owned diagnostic accepts only gemmini")
    stage_path, build, package, facts, support = (
        path.resolve(strict=True)
        for path in (args.stage_report, args.build_dir, args.package, args.facts, args.support)
    )
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or not output.resolve().is_relative_to(out_dir().resolve()):
        raise ValueError("output must be a fresh directory under the configured out root")
    stage = json.loads(stage_path.read_bytes())
    receipt_path = build / "candidate-build-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    binding = receipt["exact_binding"]
    if (
        receipt.get("status") != "built_unverified"
        or receipt.get("whole_model_offload_verified") is not False
        or stage.get("exact_binding") != binding
        or binding.get("target") != args.target
        or binding.get("rtl_facts_sha256") != digest(facts)
        or binding.get("package_sha256") != _package_sha256(package)
    ):
        raise ValueError("staged report, build receipt, package, target, or RTL facts disagree")
    interface = stage["selected_interface_mlir"]
    if sha256_text(interface) != binding["interface_sha256"]:
        raise ValueError("selected interface bytes changed")
    resident = bind_single_resident_matmul(interface, target=args.target)
    m, n, k = resident.m, resident.n, resident.k
    if resident.dtypes != ("i8", "i8", "i32") or min(m, n, k) <= 0 or max(m, n, k) > 64 or m * n * k > 4096:
        raise ValueError("this finite signed-i8 diagnostic cannot safely represent the selected extents")
    object_rows = {row["path"]: row for row in receipt["objects"]}
    if set(object_rows) != {"merlin_staged_kernel.o", "device_shim.o"}:
        raise ValueError("staged receipt does not name exactly the kernel and shim objects")
    for name, row in object_rows.items():
        checked_file(build / name, row["sha256"])
    checked_file(build / "merlin_staged_kernel.device.mlir", receipt["codegen"]["target_artifact"]["sha256"])
    checked_file(build / "device_shim.c", receipt["codegen"]["shim_source"]["sha256"])
    checked_file(build / "merlin_staged_kernel.iface.mlir", binding["interface_sha256"])
    symbol_binding = verify_object_symbol_binding(
        build / "merlin_staged_kernel.o",
        build / "device_shim.o",
        entry_symbol="merlin_staged_kernel",
        kernel_symbol=receipt["kernel_symbol"],
        original_kernel_symbol=resident.kernel_symbol,
        timeout=args.timeout,
    )

    lhs, weight, expected = _inputs(m, n, k)
    selected_facts = json.loads(facts.read_bytes())["facts"]
    compute = {row["name"]: row for row in selected_facts["compute_datapaths"]}
    storage = {row["name"]: row for row in selected_facts["storage_datapaths"]}
    operand_bits = compute["input"]["elem_bits"]
    mac_bits = compute["accumulator"]["elem_bits"]
    if (
        compute["input"].get("declared_carrier", {}).get("signedness") != "signed"
        or compute["accumulator"].get("declared_carrier", {}).get("signedness") != "signed"
        or storage["accumulator"].get("dtype") != resident.dtypes[2]
    ):
        raise ValueError("selected RTL facts do not establish the diagnostic's signed arithmetic")
    partial_sum = integer_partial_sum_bound(
        {"internal_arithmetic": {"signed_operand_bits": operand_bits, "mac_result_bits": mac_bits}},
        reduction_extent=k,
        lhs_values=lhs,
        rhs_values=weight,
    )
    if partial_sum["status"] != "proven_safe":
        raise ValueError("selected stimulus may overflow the RTL-derived internal accumulator")
    backend = get_backend(args.target)
    backend_source = Path(backend.run_elf.__code__.co_filename).resolve(strict=True)
    if not backend_source.is_relative_to(support):
        raise ValueError("selected backend source is outside the explicitly chosen support package")
    provenance_path = support / "provenance.json"
    provenance = json.loads(provenance_path.read_bytes()) if provenance_path.is_file() else {}
    declared_backend_sha = next(
        (
            row.get("sha256")
            for row in provenance.get("files", ())
            if row.get("path") == str(backend_source.relative_to(support))
        ),
        None,
    )
    if not backend.available("spike"):
        raise RuntimeError("selected target Spike backend is unavailable")
    recipe = harness_build_recipe(args.target)
    support_sources = []
    for source in recipe.support_sources:
        selected = source.resolve(strict=True)
        if not selected.is_relative_to(support):
            raise ValueError("selected harness source is outside the explicitly chosen support package")
        support_sources.append({"path": str(selected.relative_to(support)), "sha256": digest(selected)})
    spike = backend.spike_path().resolve(strict=True)
    _flags, extension_dir = backend.spike_extension()
    extension = (extension_dir / "libgemmini.so").resolve(strict=True)
    nm_executable = _nm()
    if nm_executable is None:
        raise ValueError("a readable nm tool is required for the execution receipt")
    nm = Path(nm_executable).resolve(strict=True)
    tools = {
        "backend_source": digest(backend_source),
        "spike": digest(spike),
        "spike_extension": digest(extension),
        "gcc": digest(recipe.compiler.resolve(strict=True)),
        "link_script_template": digest(recipe.link_script.resolve(strict=True)),
        "nm": digest(nm),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs.json").write_text(
        json.dumps({"lhs": lhs, "weight": weight}, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "expected.json").write_text(json.dumps(expected) + "\n", encoding="utf-8")
    source = output / "harness.c"
    source.write_text(_harness(m, n, k, lhs, weight), encoding="utf-8")
    objects = []
    for index, source_path in enumerate((source, *recipe.support_sources)):
        obj = output / f"support_{index}.o"
        run_command(recipe.compile_command(source=source_path, output=obj), timeout=args.timeout)
        objects.append(obj)
    linker_script = derived_link_script(recipe.load_address, recipe.link_script, output)
    elf = output / "candidate.elf"
    run_command(
        recipe.link_command(
            objects=[objects[0], build / "merlin_staged_kernel.o", build / "device_shim.o", *objects[1:]],
            output=elf,
            link_script=linker_script,
        ),
        timeout=args.timeout,
    )
    console = backend.run_elf(elf, simulator="spike", timeout=args.timeout)
    (output / "spike.log").write_text(console, encoding="utf-8")
    lines = [line for line in console.splitlines() if line.startswith("OUT Y0 ")]
    if len(lines) != 1 or "DONE" not in console.splitlines():
        raise ValueError("Spike did not produce exactly one complete staged output")
    fields = lines[0].split()
    if fields[:4] != ["OUT", "Y0", str(m), str(n)] or len(fields) != m * n + 4:
        raise ValueError("Spike returned an incomplete or wrong-shaped output")
    observed = [int(value) for value in fields[4:]]
    if observed != expected:
        raise ValueError("staged shim and kernel differ from independent signed-i8 scalar matmul")
    (output / "observed.json").write_text(json.dumps(observed) + "\n", encoding="utf-8")
    instruction_trace = _executed_package_instructions(
        backend, elf, output, receipt["kernel_symbol"], console, args.timeout
    )
    result = {
        "status": "incomplete_support_provenance",
        "numerical_observation": {
            "status": "matched",
            "matched_elements": m * n,
            "total_elements": m * n,
            "engine": "spike",
        },
        "qualification": (
            "one finite Spike vector through exact staged objects and descriptor shim; "
            "support backend provenance or package-to-support binding is unqualified; "
            "no RTL, SW review, or whole-model proof"
        ),
        "exact_binding": binding,
        "build_receipt_sha256": digest(receipt_path),
        "stage_report_sha256": digest(stage_path),
        "symbol_binding": symbol_binding,
        "package_instruction_trace": instruction_trace,
        "execution_sources": {
            "support_root": str(support),
            "backend_source": str(backend_source),
            "support_provenance_sha256": digest(provenance_path) if provenance_path.is_file() else None,
            "support_sources": support_sources,
            "backend_declared_sha256": declared_backend_sha,
            "backend_snapshot_digest_matches": declared_backend_sha == digest(backend_source),
            "package_support_binding": "not_established",
            "spike": str(spike),
            "spike_extension": str(extension),
            "gcc": str(recipe.compiler.resolve(strict=True)),
            "nm": str(nm),
            "link_script_template": str(recipe.link_script.resolve(strict=True)),
            "sha256": tools,
        },
        "shape": {"M": m, "N": n, "K": k},
        "partial_sum_safety": partial_sum,
        "input_sha256": digest(output / "inputs.json"),
        "expected_sha256": digest(output / "expected.json"),
        "observed_sha256": digest(output / "observed.json"),
        "harness_sha256": digest(source),
        "diagnostic_source_sha256": digest(Path(__file__)),
        "wrong_shape_descriptor_guard": "returned_null_descriptor_before_valid_call",
        "elf_sha256": digest(elf),
        "spike_log_sha256": digest(output / "spike.log"),
        "whole_model_offload_verified": False,
    }
    (output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--stage-report", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 180:
        parser.error("--timeout must be between 1 and 180 seconds")
    try:
        result = run(args)
    except Exception as exc:
        output = args.output.absolute()
        if output.is_dir() and output.resolve().is_relative_to(out_dir().resolve()):
            stage = (
                "compile_or_link"
                if not (output / "candidate.elf").is_file()
                else "spike_execution"
                if not (output / "spike.log").is_file()
                else "numerics_or_instruction_trace"
            )
            incomplete = {
                "status": "incomplete",
                "failure_stage": stage,
                "failure_type": type(exc).__name__,
                "reason_tail": str(exc)[-500:],
                "stage_report_sha256": digest(args.stage_report) if args.stage_report.is_file() else None,
                "build_receipt_sha256": digest(args.build_dir / "candidate-build-receipt.json")
                if (args.build_dir / "candidate-build-receipt.json").is_file()
                else None,
                "elf_sha256": digest(output / "candidate.elf") if (output / "candidate.elf").is_file() else None,
                "whole_model_offload_verified": False,
            }
            (output / "receipt.json").write_text(
                json.dumps(incomplete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
