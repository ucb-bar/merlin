"""Reproduce bounded generated Gemmini kernel checks into ignored out/ evidence.

The scalar oracle consumes literal input arrays from the compiled C program. It
does not call Merlin's reference evaluator or simulator for expected outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
from pathlib import Path

import yaml

from merlin.development.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.runtime.backends.base import get_backend

REPO = Path(__file__).resolve().parents[4]
NAMES = ("SY_contraction_i8_aligned", "SY_contraction_i8_partial")
TILE = 16
ORACLE_WALL_S = 180


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def literal_array(c_source: str, name: str) -> list[int]:
    match = re.search(
        rf"static const elem_t T_{re.escape(name)}\[(\d+)\] row_align\(1\) = \{{([^}}]+)\}};",
        c_source,
    )
    if match is None:
        raise ValueError(f"compiled program lacks literal {name} input array")
    values = [int(value.strip()) for value in match.group(2).split(",")]
    if len(values) != int(match.group(1)) or any(not -128 <= value <= 127 for value in values):
        raise ValueError(f"invalid compiled {name} input array")
    return values


def rounded(value: int, tile: int = TILE) -> int:
    return ((value + tile - 1) // tile) * tile


def inputs_and_scalar(c_source: str, m: int, k: int, n: int, *, tile: int = TILE):
    a_flat = literal_array(c_source, "A0")
    w_flat = literal_array(c_source, "W")
    a_pitch, w_pitch = rounded(k, tile), rounded(n, tile)
    if len(a_flat) != rounded(m, tile) * a_pitch or len(w_flat) != rounded(k, tile) * w_pitch:
        raise ValueError("compiled input-array lengths differ from tile-padded capsule geometry")
    # Device loads the tile-padded arrays. Preserve exact row strides and require
    # every padding byte to be zero; a wrong stride/tail cannot be hidden by a
    # separately generated mathematical input.
    if any(a_flat[i * a_pitch + j] for i in range(rounded(m, tile)) for j in range(a_pitch) if i >= m or j >= k):
        raise ValueError("nonzero A padding")
    if any(w_flat[i * w_pitch + j] for i in range(rounded(k, tile)) for j in range(w_pitch) if i >= k or j >= n):
        raise ValueError("nonzero W padding")
    a = [[a_flat[i * a_pitch + t] for t in range(k)] for i in range(m)]
    w = [[w_flat[t * w_pitch + j] for j in range(n)] for t in range(k)]
    scalar = [[sum(a[i][t] * w[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
    return a, w, scalar, a_flat, w_flat


def i8_bytes(rows: list[list[int]]) -> bytes:
    return bytes(value & 0xFF for row in rows for value in row)


def i32_bytes(rows: list[list[int]]) -> bytes:
    return b"".join(struct.pack("<i", value) for row in rows for value in row)


def rejected_cycle_limit_flags(backend, elf: Path) -> dict:
    """Retain evidence that this selected binary cannot enforce an emulated-cycle cap."""
    simulator = str(backend.verilator_path())
    probes = {}
    for label, prefix, expected in (
        ("long", ["--max-cycles=20000000"], "unrecognized option"),
        ("short", ["-m", "20000000"], "invalid option"),
        ("plusarg", ["+max-cycles=20000000"], "could not open +max-cycles"),
    ):
        argv = [simulator, *prefix, str(elf.resolve())]
        try:
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=3, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"cycle-limit form {label} is ambiguous, not an established rejection") from exc
        if completed.returncode == 0 or expected not in completed.stderr:
            raise RuntimeError(f"cycle-limit form {label} did not exhibit expected rejection")
        probes[label] = {
            "argv": argv,
            "returncode": completed.returncode,
            "stderr": completed.stderr,
        }
    return {"status": "unavailable", "probes": probes}


def checked_output_root(path: Path) -> Path:
    """Refuse writes outside the checkout's ignored generated-output subtree."""
    output_root = path.resolve()
    if not output_root.is_relative_to(REPO / "out"):
        raise ValueError(f"--output-root must resolve beneath {REPO / 'out'}")
    tracked = subprocess.run(
        ["git", "ls-files", "--", str(output_root)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    if tracked.stdout.strip():
        raise ValueError("--output-root contains tracked files")
    for relative in ("receipt.json", f"{NAMES[0]}/spike/main.c"):
        candidate = output_root / relative
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(candidate)],
            cwd=REPO,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError(f"generated evidence path is not ignored: {candidate}")
    return output_root


def phase0_manifest_path(corpus: Path) -> Path:
    """Resolve the frozen controller layout or the retained standalone-corpus layout."""
    candidates = [corpus / "_evidence/evidence-manifest.json"]
    if corpus.name == "capsules":
        candidates.append(corpus.parent / "evidence-manifest.json")
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(f"Phase 0 corpus has {'ambiguous' if existing else 'no'} evidence manifest: {corpus}")
    return existing[0]


def checked_source_binding(phase0_manifest: dict, source_evidence: Path, facts_evidence: Path | None = None) -> dict:
    """Bind selected RTL, current facts, and their validation before executing a kernel."""
    facts_root = facts_evidence if facts_evidence is not None else source_evidence
    selection_path = source_evidence / "source-selection.json"
    core_hw_path = source_evidence / "core.hw.mlir"
    facts_path = facts_root / "facts.json"
    validation_path = facts_root / "validation.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    selection_sha = sha(selection_path)
    facts_sha = sha(facts_path)
    core_hw_sha = sha(core_hw_path)
    if selection["hierarchy_correspondence"]["status"] != "verified" or validation["status"] != "verified":
        raise RuntimeError("selected source record is not structurally verified")
    if phase0_manifest["raw_facts_sha256"] != facts_sha or not any(
        artifact.get("sha256") == selection_sha for artifact in phase0_manifest["artifacts"].values()
    ):
        raise RuntimeError("generated corpus does not bind selected Gemmini source/facts bytes")
    if facts_evidence is not None:
        if validation.get("facts_sha256") != facts_sha or validation.get("source_selection_sha256") != selection_sha:
            raise RuntimeError("facts validation does not bind selected Gemmini source/facts bytes")
        if selection["sources"]["core_hw"]["sha256"] != core_hw_sha:
            raise RuntimeError("selected core HW bytes differ from source selection")
        for role, expected in (
            ("rtl-facts", facts_sha),
            ("rtl-source:source_bundle_path", selection_sha),
            ("rtl-source:core_hw_path", core_hw_sha),
        ):
            recorded = [row["sha256"] for row in phase0_manifest["sources"] if row["role"] == role]
            if recorded != [expected]:
                raise RuntimeError(f"generated corpus does not bind {role} bytes")
    return {
        "selection": {"path": str(selection_path), "sha256": selection_sha},
        "facts": {"path": str(facts_path), "sha256": facts_sha},
        "core_hw": {"path": str(core_hw_path), "sha256": core_hw_sha},
        "validation": {"path": str(validation_path), "sha256": sha(validation_path)},
        "validation_status": validation["status"],
        "hierarchy_correspondence_status": selection["hierarchy_correspondence"]["status"],
    }


def checked_declared_inputs(phase0_manifest: dict, software_spec: Path, support_contract: Path) -> dict:
    """Require selected software and OOT contract bytes in the frozen Phase 0 source list."""
    binding = {}
    for name, role, path in (
        ("software_spec", "software-spec", software_spec),
        ("support_contract", "support-source", support_contract),
    ):
        actual_sha = sha(path)
        recorded = [
            row["sha256"] for row in phase0_manifest["sources"] if row["role"] == role and row["source"] == str(path)
        ]
        if recorded != [actual_sha]:
            raise RuntimeError(f"generated corpus does not bind {name} bytes")
        binding[name] = {"path": str(path), "sha256": actual_sha, "manifest_role": role}
    return binding


def require_two_engine_backend(backend) -> None:
    """A successful receipt must represent functional and RTL execution."""
    for simulator in ("spike", "verilator"):
        if not backend.available(simulator):
            raise RuntimeError(f"selected native Gemmini {simulator} unavailable; two-engine qualification not run")


def run(
    *,
    corpus: Path,
    source_evidence: Path,
    output_root: Path,
    facts_evidence: Path | None = None,
    audit_existing: bool = False,
) -> None:
    corpus = corpus.resolve()
    source_evidence = source_evidence.resolve()
    facts_evidence = facts_evidence.resolve() if facts_evidence is not None else None
    output_root = checked_output_root(output_root)
    if any(
        output_root.is_relative_to(input_root) or input_root.is_relative_to(output_root)
        for input_root in (corpus, source_evidence, facts_evidence)
        if input_root is not None
    ):
        raise ValueError("--output-root must not overlap an input evidence bundle")
    corpus_isa = corpus / "isa"
    support = Path(os.environ["MERLIN_TARGET_PATH"]).resolve()
    chipyard = Path(os.environ["MERLIN_CHIPYARD"]).resolve()
    backend = get_backend("gemmini")
    require_two_engine_backend(backend)
    manifest_path = phase0_manifest_path(corpus)
    phase0_manifest = json.loads(manifest_path.read_text())
    selected_source = checked_source_binding(phase0_manifest, source_evidence, facts_evidence)
    declared_inputs = (
        checked_declared_inputs(
            phase0_manifest,
            REPO / "examples/compiler-generation/gemmini/target/software-spec.yaml",
            support / "contracts/target_contract.yaml",
        )
        if facts_evidence is not None
        else None
    )
    objdump = chipyard / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-objdump"
    tool_files = {
        "spike": backend.spike_path(),
        "libgemmini": backend.libgemmini_dir() / "libgemmini.so",
        "riscv_gcc": backend.gcc_path(),
        "verilator": backend.verilator_path(),
        "objdump": objdump,
    }
    tools = {name: {"path": str(path), "sha256": sha(path)} for name, path in tool_files.items() if path.is_file()}
    common = {
        "schema": "merlin.gemmini-kernel-numerical-qualification.v1",
        "probe_source": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
        "evidence_roots": {
            "corpus": str(corpus),
            "source_evidence": str(source_evidence),
            **({"facts_evidence": str(facts_evidence)} if facts_evidence is not None else {}),
            "output_root": str(output_root),
        },
        "phase0_evidence_manifest": {
            "path": str(manifest_path),
            "sha256": sha(manifest_path),
            "status": phase0_manifest["status"],
        },
        "selected_source": selected_source,
        "software_spec": {
            "path": str(REPO / "examples/compiler-generation/gemmini/target/software-spec.yaml"),
            "sha256": sha(REPO / "examples/compiler-generation/gemmini/target/software-spec.yaml"),
            **({"phase0_source_role": "software-spec"} if declared_inputs is not None else {}),
        },
        "support": {
            "path": str(support),
            "contract_sha256": sha(support / "contracts/target_contract.yaml"),
            **({"contract_phase0_source_role": "support-source"} if declared_inputs is not None else {}),
            "backend_sha256": sha(support / "backend/gemmini.py"),
            "codegen_sha256": sha(support / "backend/gemmini_codegen.py"),
        },
        "toolchain": tools,
        "qualification": (
            "fixed two-case exact-output observation on Gemmini Spike and Verilator; not a universal proof"
        ),
        "limits": {
            "oracle_wall_seconds": ORACLE_WALL_S,
            "verilator_cycle_limit": "unavailable: selected binary rejects --max-cycles, -m, and +max-cycles",
        },
        "limitations": [
            "The selected generated capsules are diagnostic and their software_screen is unknown; "
            "this receipt does not admit them to Phase 1/2.",
            "Spike/libgemmini is a functional model, not RTL. Verilator is an RTL simulator, "
            "but this receipt does not prove its binary was built from the exact selected source bytes.",
            "Only i8 operands in the generated 0..3 stimulus range, zero initial accumulator, "
            "no epilogue, and the two recorded shapes are exercised.",
            "The independent scalar oracle is an executable numerical check, "
            "not a quantified SMT proof of every input.",
        ],
    }
    previous = json.loads((output_root / "receipt.json").read_text()) if audit_existing else None
    results = {}
    for name in NAMES:
        capsule_dir = corpus_isa / name
        capsule = yaml.safe_load((capsule_dir / "capsule.yaml").read_text())
        phase0_golden = yaml.safe_load((capsule_dir / "golden.yaml").read_text())
        if capsule["operation"]["op"] != "matmul" or capsule["numeric_policy"]["compare"] != "exact_int":
            raise ValueError(f"unexpected generated capsule semantics: {name}")
        leaves = {item["name"]: item for item in capsule["inputs"]}
        m, k = leaves["A0"]["shape"]
        wk, n = leaves["W"]["shape"]
        if k != wk:
            raise ValueError("invalid contraction geometry")
        interface_path = capsule_dir / "capsule.interface.mlir"
        cb = parse_interface_mlir(interface_path.read_text())
        case = output_root / name
        spike_dir = case / "spike"
        spike_dir.mkdir(parents=True, exist_ok=True)
        save_json(case / "command_buffer.json", cb)
        if audit_existing:
            # Re-execute the previously emitted exact ELF; no program rebuild can
            # silently invalidate a saved RTL console's program hash.
            elf = spike_dir / "merlin_gemmini_c0.elf"
            spike_console = backend.run_elf(elf, simulator="spike", timeout=ORACLE_WALL_S)
            spike_outputs, spike_metrics = backend.parse_output(spike_console)
            spike = {
                "outputs": spike_outputs,
                "raw_metrics": spike_metrics,
                "oracle": {"kind": "spike_gemmini_functional", "derived_from_rtl": False},
                "console": spike_console,
                "elf": str(elf),
            }
        else:
            spike = backend.run_command_buffer(cb, workdir=spike_dir, simulator="spike", timeout=ORACLE_WALL_S)
        (case / "spike_console.log").write_text(spike["console"], encoding="utf-8")
        c_path = spike_dir / "main.c"
        elf = Path(spike["elf"])
        if name == NAMES[0]:
            common["cycle_limit_probe"] = rejected_cycle_limit_flags(backend, elf)
        a, w, scalar, a_padded, w_padded = inputs_and_scalar(c_path.read_text(), m, k, n)
        expected = phase0_golden["outputs"]["Y0"]
        observed = spike["outputs"]["Y0"]
        if scalar != expected or observed != scalar or (not audit_existing and not spike["correct"]):
            raise AssertionError(f"Spike/independent scalar/Phase0 golden mismatch for {name}")
        disassembly = subprocess.run(
            [str(objdump), "-d", str(elf)], capture_output=True, text=True, timeout=20, check=True
        ).stdout
        (case / "spike_disassembly.txt").write_text(disassembly, encoding="utf-8")
        custom3_count = len(re.findall(r"\.insn\s+4,\s+0x[0-9a-f]*7b\b", disassembly))
        if custom3_count < 1:
            raise AssertionError("compiled ELF has no custom-3 RoCC opcode")
        save_json(case / "inputs.json", {"A0": a, "W": w})
        save_json(case / "outputs.json", {"Y0": observed})
        results[name] = {
            "status": "passed",
            "shape": {"A0": [m, k], "W": [k, n], "Y0": [m, n]},
            "dtype": {"input": "i8", "output": "i32"},
            "output_elements": m * n,
            "compared": ["compiled-literal-input scalar matmul", "Phase0 golden", "Gemmini Spike output"],
            "max_absolute_partial_sum_bound": capsule["integer_partial_sum_bound"]["bound"],
            "capsule_sha256": sha(capsule_dir / "capsule.yaml"),
            "interface_sha256": sha(interface_path),
            "phase0_golden_sha256": sha(capsule_dir / "golden.yaml"),
            "command_buffer_sha256": sha(case / "command_buffer.json"),
            "compiled_c_sha256": sha(c_path),
            "elf_sha256": sha(elf),
            "spike_console_sha256": sha(case / "spike_console.log"),
            "disassembly_sha256": sha(case / "spike_disassembly.txt"),
            "custom3_rocc_instruction_count": custom3_count,
            "inputs": {
                "A0_logical_i8_sha256": digest(i8_bytes(a)),
                "W_logical_i8_sha256": digest(i8_bytes(w)),
                "A0_padded_i8_sha256": digest(bytes(value & 0xFF for value in a_padded)),
                "W_padded_i8_sha256": digest(bytes(value & 0xFF for value in w_padded)),
                "input_document_sha256": sha(case / "inputs.json"),
            },
            "output_i32_le_sha256": digest(i32_bytes(observed)),
            "output_document_sha256": sha(case / "outputs.json"),
            "spike_cycles": spike["raw_metrics"].get("cycles"),
            "spike_oracle": spike["oracle"],
            "phase0_software_screen": capsule["software_screen"]["status"],
        }
        # Crucially, execute the SAME already-hashed ELF on both engines.
        # A separate Verilator rebuild could silently change the program.
        prior_case = ((previous or {}).get("cases") or {}).get(name, {})
        prior_rtl = prior_case.get("verilator") or {}
        reused = bool(
            audit_existing
            and prior_rtl.get("status") == "passed"
            and prior_rtl.get("same_elf_sha256") == sha(elf)
            and (case / "verilator_console.log").is_file()
            and prior_rtl.get("console_sha256") == sha(case / "verilator_console.log")
        )
        rtl_console = (
            (case / "verilator_console.log").read_text(encoding="utf-8")
            if reused
            else backend.run_elf(elf, simulator="verilator", timeout=ORACLE_WALL_S)
        )
        (case / "verilator_console.log").write_text(rtl_console, encoding="utf-8")
        rtl_outputs, rtl_metrics = backend.parse_output(rtl_console)
        if rtl_outputs.get("Y0") != scalar:
            raise AssertionError(f"Verilator/independent scalar mismatch for {name}")
        results[name]["compared"].append("same-ELF Gemmini Verilator output")
        results[name]["verilator"] = {
            "status": "passed",
            "engine": "rtl_verilator",
            "reused_prior_console_for_identical_elf": reused,
            "cycle_limit": "unavailable",
            "wall_timeout_seconds": ORACLE_WALL_S,
            "same_elf_sha256": sha(elf),
            "console_sha256": sha(case / "verilator_console.log"),
            "output_i32_le_sha256": digest(i32_bytes(rtl_outputs["Y0"])),
            "cycles": rtl_metrics.get("cycles"),
        }
    common["cases"] = results
    common["summary"] = {
        "cases": len(results),
        "output_elements": sum(row["output_elements"] for row in results.values()),
        "status": "passed",
    }
    save_json(output_root / "receipt.json", common)
    print(json.dumps(common["summary"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="generated Phase 0 corpus root containing isa/")
    parser.add_argument("--source-evidence", required=True, type=Path, help="selected source evidence bundle root")
    parser.add_argument(
        "--facts-evidence", type=Path, help="separate verified facts/validation root bound to the selected source"
    )
    parser.add_argument("--output-root", required=True, type=Path, help="ignored checkout out/ evidence directory")
    parser.add_argument(
        "--audit-existing",
        action="store_true",
        help="re-execute existing ELF on Spike and reuse a prior hashed Verilator console only for an identical ELF",
    )
    args = parser.parse_args()
    run(
        corpus=args.corpus,
        source_evidence=args.source_evidence,
        output_root=args.output_root,
        facts_evidence=args.facts_evidence,
        audit_existing=args.audit_existing,
    )
