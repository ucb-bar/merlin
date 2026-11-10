#!/usr/bin/env python3
"""Finite signed Gemmini matmul diagnostic against a selected Phase 0 corpus.

The selected MacUnit receipt checks i20 wrapping separately. This probe uses
bounded operands, so its mathematical i32 golden does not depend on wrapping.
Neither result reviews the software spec or qualifies arbitrary model inputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import subprocess
from pathlib import Path

from probe_headline_kernel import _scaled_i8_reference
from probe_native_kernel import checked_declared_inputs, checked_source_binding, phase0_manifest_path

from merlin.development.testgen.cell_probe import verify_characterization
from merlin.runtime.backends.base import get_backend
from merlin.targetgen.contract.compile import compile_lowered_to_elf
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.operation_numerics import integer_partial_sum_bound

REPO = Path(__file__).resolve().parents[4]
CAPSULE = "SY_contraction_i8_partial"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def signed_operands(m: int, k: int, n: int) -> dict:
    """Distinguish both signs and exceed the i8 output range in both directions."""
    return {
        "A0": [[-128 if row % 2 == 0 else 127 for _ in range(k)] for row in range(m)],
        "W": [[-128 if col % 2 == 0 else 127 for col in range(n)] for _ in range(k)],
    }


def scalar_and_partial(inputs: dict, m: int, k: int, n: int) -> tuple[list[list[int]], int]:
    result, largest = [], 0
    for row in range(m):
        output_row = []
        for col in range(n):
            partial = 0
            for depth in range(k):
                partial += inputs["A0"][row][depth] * inputs["W"][depth][col]
                largest = max(largest, abs(partial))
            output_row.append(partial)
        result.append(output_row)
    return result, largest


def traced_rocc_pcs(backend: object, elf: Path, case_dir: Path, disassembly: str, console: str, timeout: int) -> dict:
    """Require dynamically executed custom-3 PCs, retaining only a tiny trace excerpt."""
    pattern = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+\.insn\s+4,\s+0x[0-9a-f]*7b\b", re.M)
    expected = {int(pc, 16): int(word, 16) for pc, word in pattern.findall(disassembly)}
    if not expected:
        raise AssertionError("ELF has no disassembled custom-3 instructions")
    trace_path = case_dir / "spike_full.trace"
    flags, libdir = backend.spike_extension()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(libdir) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [
        str(backend.spike_path()),
        "-l",
        f"--log={trace_path}",
        "--instructions=250000",
        *flags,
        str(elf),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0 or proc.stdout != console:
        raise AssertionError("bounded Spike trace run did not repeat the matched numerical console")
    if trace_path.stat().st_size > 16 * 1024 * 1024:
        raise AssertionError("bounded Spike trace exceeded 16 MiB")
    observed = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^core\s+\d+:\s+0x([0-9a-f]+)\s+\(0x([0-9a-f]+)\)", line)
        if match and expected.get(int(match[1], 16)) == int(match[2], 16):
            observed.append(line)
    trace_path.unlink()
    if {int(re.search(r"0x([0-9a-f]+)", line)[1], 16) for line in observed} != set(expected):
        raise AssertionError("Spike did not dynamically execute every disassembled custom-3 PC")
    excerpt = case_dir / "spike_rocc.trace"
    excerpt.write_text("\n".join(observed) + "\n", encoding="utf-8")
    return {
        "executed_custom3_count": len(observed),
        "executed_custom3_distinct_pcs": len(expected),
        "trace_excerpt_sha256": sha(excerpt),
        "trace_instruction_limit": 250000,
    }


def run(args: argparse.Namespace) -> dict:
    root = args.output.resolve()
    if root.exists() or root.is_symlink() or not root.is_relative_to(REPO / "out"):
        raise ValueError("output must be a fresh path under this checkout's out/")
    corpus, source, facts_root = (path.resolve(strict=True) for path in (args.corpus, args.source, args.facts))
    manifest_path = phase0_manifest_path(corpus)
    manifest = json.loads(manifest_path.read_bytes())
    selected = checked_source_binding(manifest, source, facts_root)
    support = Path(os.environ["MERLIN_TARGET_PATH"]).resolve(strict=True)
    contract_path = support / "contracts/target_contract.yaml"
    selected_inputs = checked_declared_inputs(
        manifest, REPO / "examples/compiler-generation/gemmini/target/software-spec.yaml", contract_path
    )
    facts = json.loads((facts_root / "facts.json").read_bytes())["facts"]
    compute = {row["name"]: row for row in facts["compute_datapaths"]}
    storage = {row["name"]: row for row in facts["storage_datapaths"]}
    if (
        compute["input"]["elem_bits"] != 8
        or compute["input"]["declared_carrier"]["signedness"] != "signed"
        or compute["accumulator"]["elem_bits"] != 20
        or compute["accumulator"]["declared_carrier"]["signedness"] != "signed"
        or storage["accumulator"]["dtype"] != "i32"
    ):
        raise ValueError("selected RTL facts do not establish signed i8, signed i20, and i32 storage")
    cell = verify_characterization(args.cell_receipt, source_sha256=selected["core_hw"]["sha256"])
    if (
        cell.get("vector_count") != 77056
        or cell.get("domain", {}).get("expected") != "signed multiply-add wrapping to 20 bits"
    ):
        raise ValueError("selected cell receipt does not cover the expected i20 wrap domain")

    capsule_dir = corpus / "isa" / CAPSULE
    capsule_path = capsule_dir / "capsule.yaml"
    interface_path = capsule_dir / "capsule.interface.mlir"
    cb = parse_interface_mlir(interface_path.read_text(encoding="utf-8"))
    if [row["opcode"] for row in cb["commands"]] != ["RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT"]:
        raise ValueError("selected capsule is not the one resident matmul under test")
    m, k = cb["tensors"]["A0"]["shape"]
    wk, n = cb["tensors"]["W"]["shape"]
    if k != wk or cb["tensors"]["A0"]["dtype"] != "i8" or cb["tensors"]["W"]["dtype"] != "i8":
        raise ValueError("selected capsule is not a signed i8 matmul")
    if cb["commands"][2]["attributes"] != {"epilogue": [], "output_dtype": "i32"}:
        raise ValueError("selected capsule does not have an unscaled full i32 readout")
    inputs = signed_operands(m, k, n)
    scalar, largest = scalar_and_partial(inputs, m, k, n)
    semantics = {"internal_arithmetic": {"signed_operand_bits": 8, "mac_result_bits": 20}}
    bound = integer_partial_sum_bound(
        semantics,
        reduction_extent=k,
        lhs_values=[value for row in inputs["A0"] for value in row],
        rhs_values=[value for row in inputs["W"] for value in row],
    )
    if bound["status"] != "proven_safe" or largest > bound["bound"]:
        raise ValueError("selected signed operands can overflow the i20 partial sum")
    if not (
        any(value < -128 for row in scalar for value in row) and any(value > 127 for row in scalar for value in row)
    ):
        raise ValueError("signed stimulus does not distinguish both i8 saturation directions")

    backend = get_backend("gemmini")
    if not backend.available("spike") or (args.rtl and not backend.available("verilator")):
        raise RuntimeError("requested prebuilt Gemmini oracle is unavailable")
    spike_flags, spike_libdir = backend.spike_extension()
    if spike_flags != ("--extension=gemmini",):
        raise ValueError("this diagnostic expects the selected Gemmini Spike extension")
    spike_extension = spike_libdir / "libgemmini.so"
    if not spike_extension.is_file():
        raise ValueError("selected Gemmini Spike extension library is missing")
    emitter = importlib.import_module(f"{backend.__package__}.gemmini_codegen_mlir")
    output = {
        "schema": "merlin.gemmini.signed_i8_diagnostic.v1",
        "status": "incomplete",
        "selected": {
            "probe": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
            "phase0_manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)},
            "source": selected,
            "software_and_support": selected_inputs,
            "capsule": {"path": str(capsule_path), "sha256": sha(capsule_path)},
            "interface": {"path": str(interface_path), "sha256": sha(interface_path)},
            "cell_characterization": {
                "path": str(args.cell_receipt.resolve()),
                "sha256": sha(args.cell_receipt),
                "vectors": cell["vector_count"],
                "status": cell["status"],
            },
            "emitter": {"path": emitter.__file__, "sha256": sha(Path(emitter.__file__))},
            "backend": {"path": backend.__file__, "sha256": sha(Path(backend.__file__))},
            "backend_implementation": {
                "path": str(support / "backend/gemmini.py"),
                "sha256": sha(support / "backend/gemmini.py"),
            },
            "compiler_python": {
                "path": str(Path(os.environ["MERLIN_M2M_VENV"]) / "bin/python"),
                "sha256": sha(Path(os.environ["MERLIN_M2M_VENV"]) / "bin/python"),
            },
            "clang": {"path": os.environ["MERLIN_CLANG"], "sha256": sha(Path(os.environ["MERLIN_CLANG"]))},
            "spike": {"path": str(backend.spike_path()), "sha256": sha(backend.spike_path())},
            "spike_extension": {"path": str(spike_extension), "sha256": sha(spike_extension)},
            "verilator": {"path": str(backend.verilator_path()), "sha256": sha(backend.verilator_path())},
        },
        "input_shape": {"A0": [m, k], "W": [k, n]},
        "input_extrema": [-128, 127],
        "partial_sum": {"sufficient_bound": bound, "observed_maximum_absolute": largest},
        "limitations": [
            "The cell wrap check and whole-kernel checks are separate finite observations.",
            "Spike is a functional model. A Verilator run, when requested, is not a source-build attestation.",
            "This synthetic operand vector does not establish physical layout, tails, aliasing, or broad SW review.",
        ],
        "cases": {},
    }
    root.mkdir(parents=True, exist_ok=False)
    save(root / "inputs.json", inputs)
    output["inputs_sha256"] = sha(root / "inputs.json")
    for label, scale in (("full_i32", None), ("saturating_i8", 0.5)):
        case_cb = copy.deepcopy(cb)
        expected = scalar
        if scale is not None:
            case_cb["commands"][2]["attributes"] = {
                "epilogue": ["acc_scale"],
                "output_dtype": "i8",
                "acc_scale": scale,
            }
            expected = _scaled_i8_reference(scalar, scale)
        case_dir = root / label
        case_dir.mkdir()
        save(case_dir / "command_buffer.json", case_cb)
        save(case_dir / "expected.json", {"Y0": expected})
        lowered, _ = emitter.emit_kernel_mlir(case_cb, native_conv=False)
        elf = compile_lowered_to_elf(case_cb, lowered, case_dir, target="gemmini", inputs=inputs)
        objdump = Path(os.environ["MERLIN_CHIPYARD"]) / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-objdump"
        disassembly = subprocess.run([str(objdump), "-d", str(elf)], capture_output=True, text=True, check=True).stdout
        (case_dir / "disassembly.txt").write_text(disassembly, encoding="utf-8")
        rocc_count = len(re.findall(r"\.insn\s+4,\s+0x[0-9a-f]*7b\b", disassembly))
        if not rocc_count:
            raise AssertionError(f"{label}: ELF contains no custom-3 RoCC instruction")
        case_receipt = {
            "status": "incomplete",
            "readout_dtype": "i32" if scale is None else "i8",
            "acc_scale": scale,
            "same_elf_sha256": sha(elf),
            "rocc_instructions": rocc_count,
            "command_buffer_sha256": sha(case_dir / "command_buffer.json"),
            "expected_sha256": sha(case_dir / "expected.json"),
            "lowered_mlir_sha256": sha(case_dir / "model.mlir"),
            "harness_sha256": sha(case_dir / "harness.c"),
            "disassembly_sha256": sha(case_dir / "disassembly.txt"),
            "engines": {},
        }
        output["cases"][label] = case_receipt
        engines = case_receipt["engines"]
        for engine in ("spike", "verilator") if args.rtl else ("spike",):
            try:
                console = backend.run_elf(elf, simulator=engine, timeout=args.timeout)
            except subprocess.TimeoutExpired as exc:
                timeout_evidence = {"status": "timeout", "wall_limit_seconds": args.timeout}
                for stream in ("stdout", "stderr"):
                    captured = getattr(exc, stream) or b""
                    data = captured.encode("utf-8") if isinstance(captured, str) else captured
                    retained = data[-65536:]
                    path = case_dir / f"{engine}.{stream}.tail"
                    path.write_bytes(retained)
                    timeout_evidence[stream] = {
                        "captured_bytes": len(data),
                        "retained_bytes": len(retained),
                        "tail_sha256": sha(path),
                        "truncated": len(data) > len(retained),
                    }
                engines[engine] = timeout_evidence
                case_receipt["status"] = "rtl_timeout_incomplete"
                output["status"] = "rtl_timeout_incomplete"
                save(root / "receipt.json", output)
                raise RuntimeError(f"{label}: {engine} timed out; incomplete receipt saved at {root}") from exc
            (case_dir / f"{engine}.log").write_text(console, encoding="utf-8")
            observed, metrics = backend.parse_output(console)
            if observed.get("Y0") != expected:
                raise AssertionError(f"{label}: {engine} differs from independent scalar reference")
            if (
                metrics.get("cycle_window_gemmini_region") != 1
                or type(metrics.get("cycles")) is not int
                or metrics["cycles"] <= 0
            ):
                raise AssertionError(f"{label}: {engine} did not report a measured Gemmini target region")
            engines[engine] = {
                "status": "matched",
                "console_sha256": sha(case_dir / f"{engine}.log"),
                "cycles": metrics.get("cycles"),
                "output_elements": m * n,
            }
            if engine == "spike":
                engines["spike"].update(traced_rocc_pcs(backend, elf, case_dir, disassembly, console, args.timeout))
        case_receipt["status"] = "spike_and_verilator_matched" if args.rtl else "spike_matched_rtl_not_run"
    output["status"] = "spike_and_verilator_passed" if args.rtl else "spike_passed_rtl_not_run"
    save(root / "receipt.json", output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--cell-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rtl", action="store_true", help="run the same ELFs on the selected Verilator binary")
    parser.add_argument("--timeout", type=int, default=180, help="wall seconds per simulator execution")
    args = parser.parse_args()
    if args.timeout < 1 or args.timeout > 180:
        parser.error("--timeout must be between 1 and 180 seconds")
    result = run(args)
    print(json.dumps({"status": result["status"], "cases": list(result["cases"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
