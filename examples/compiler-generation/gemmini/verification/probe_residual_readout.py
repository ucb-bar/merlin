#!/usr/bin/env python3
"""Qualify Gemmini's direct-load residual sum and narrowing ReLU readout.

This is a target-local numerical witness, not a Phase 0 review decision or a
compiler test.  The C program uses CONFIG, two ordinary MVINs, MVOUT and FLUSH;
it never calls Gemmini's LOOP_WS residual helper.  The same ELF is run on Spike
and the selected Rocket/Gemmini Verilator.  ``--campaign full`` covers every
ordered pair of signed i8 operands at one group-derived pair of scales.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import yaml

from merlin.common.paths import repo_root
from merlin.runtime.backends.base import get_backend
from merlin.targetgen.plugins import load_declared

CONFIG = "GemminiRocketConfig"
SMOKE_VALUES = (-128, -127, -122, -85, -64, -33, -1, 0, 1, 31, 63, 85, 119, 122, 126, 127)
SCHEMA = "merlin.gemmini-direct-residual-readout-witness.v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def select_group(report: dict, name: str | None = None) -> dict:
    rows = [
        row
        for row in report.get("entries", ())
        if row.get("entry", {}).get("op") == "residual_add"
        and row.get("entry", {}).get("epilogue") == ["relu"]
        and not row.get("raw_of")
        and (name is None or row.get("name") == name)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one selected residual-add/ReLU group, found {len(rows)}")
    entry = rows[0]["entry"]
    if entry.get("operand_dtype") != "int8" or not isinstance(entry.get("bound_lsb"), int):
        raise ValueError("selected group is not an i8 bounded residual add")
    lhs, rhs = f32(float(entry["lhs_scale"])), f32(float(entry["rhs_scale"]))
    if not (0 < lhs < float("inf") and 0 < rhs < float("inf") and max(lhs, rhs) > 1):
        raise ValueError("this witness requires positive finite scales with gain greater than one")
    return {
        "name": rows[0]["name"],
        "lhs": lhs,
        "rhs": rhs,
        "bound": entry["bound_lsb"],
        "source_role": entry.get("source_role"),
        "source_reference": entry.get("source_reference"),
    }


def selected_evidence(phase0: Path, vendor: Path) -> tuple[dict, dict]:
    manifest_path = phase0 / "evidence-manifest.json"
    facts_path = phase0 / "hardware/circt/facts.json"
    facets_path = phase0 / "hardware/effective-views/readout-facets.json"
    inputs_path = phase0 / "hardware/effective-views/readout-inputs.json"
    manifest, facts = json.loads(manifest_path.read_text()), json.loads(facts_path.read_text())
    facets, inputs = json.loads(facets_path.read_text()), json.loads(inputs_path.read_text())
    for path in (facts_path, facets_path, inputs_path):
        rel = path.relative_to(phase0).as_posix()
        if manifest["artifacts"][rel]["sha256"] != sha(path):
            raise ValueError(f"Phase 0 manifest does not bind {rel}")
    if manifest.get("status") != "verified" or facts.get("source_consistency", {}).get("status") != "verified":
        raise ValueError("selected Phase 0 facts/source are not verified")
    if facts["facts"]["source"]["config"] != CONFIG:
        raise ValueError("selected facts are not for GemminiRocketConfig")
    if manifest.get("raw_facts_sha256") != sha(facts_path):
        raise ValueError("manifest raw fact hash differs from selected CIRCT facts")
    if len(facets) != 1 or facets[0].get("unit") != "systolic_mesh":
        raise ValueError("selected facts have no unique Gemmini mesh readout")
    facet = facets[0]
    operand_sum = facet.get("operand_sum") or {}
    if (
        facet.get("accumulator_kind") != "addressable"
        or facet.get("accumulator_dtype") != "i32"
        or operand_sum.get("operands") != 2
        or operand_sum.get("operand_dtype") != "i8"
        or operand_sum.get("operand_rounding") != "half_even"
        or operand_sum.get("operand_saturates") is not True
        or facet.get("scale", {}).get("granularities") != ["tensor"]
    ):
        raise ValueError("selected operand-sum/readout contract differs from this witness")
    readouts = {row["selector"]: set(row["applies"]) for row in facet.get("readouts", ())}
    if not {"relu", "acc_scale"} <= readouts.get("i8", set()) or readouts.get("i32") != set():
        raise ValueError("selected i8/i32 readout stage claims differ from this witness")
    header = vendor / "include/gemmini_params.h"
    for key in ("operand_sum", "scalar_abi"):
        if inputs[key]["provenance"]["params_header_sha256"] != sha(header):
            raise ValueError(f"selected {key} ABI is not this vendor header")
    source = {
        "phase0_manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)},
        "facts": {"path": str(facts_path), "sha256": sha(facts_path)},
        "readout_facets": {"path": str(facets_path), "sha256": sha(facets_path)},
        "readout_inputs": {"path": str(inputs_path), "sha256": sha(inputs_path)},
        "params_header": {"path": str(header), "sha256": sha(header)},
    }
    return source, facts


def campaign_values(campaign: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if campaign == "smoke":
        return SMOKE_VALUES, SMOKE_VALUES
    if campaign == "full":
        domain = tuple(range(-128, 128))
        return domain, domain
    if campaign == "boundary_rows":
        return SMOKE_VALUES, tuple(range(-128, 128))
    if campaign == "boundary_cols":
        return tuple(range(-128, 128)), SMOKE_VALUES
    raise ValueError(f"unknown campaign {campaign!r}")


def scales(group: dict) -> dict[str, float]:
    factor = max(1.0, group["lhs"], group["rhs"])
    return {"lhs_load": f32(group["lhs"] / factor), "rhs_load": f32(group["rhs"] / factor), "readout": f32(factor)}


def expected_outputs(group: dict, campaign: str) -> dict:
    aa, bb = campaign_values(campaign)
    gain = scales(group)
    reference, unit = [], []
    for a in aa:
        reference_row, unit_row = [], []
        for b in bb:
            one_round = round(f32(f32(a * group["lhs"]) + f32(b * group["rhs"])))
            reference_row.append(min(127, max(0, one_round)))
            a_load = min(127, max(-128, round(f32(a * gain["lhs_load"]))))
            b_load = min(127, max(-128, round(f32(b * gain["rhs_load"]))))
            narrowed = round(f32(max(0, a_load + b_load) * gain["readout"]))
            unit_row.append(min(127, max(0, narrowed)))
        reference.append(reference_row)
        unit.append(unit_row)
    return {"reference": reference, "unit_model": unit, "gain": gain}


def _c_float(value: float) -> str:
    digits = format(value, ".9g")
    return (digits if "." in digits or "e" in digits else digits + ".0") + "f"


def render_source(group: dict, campaign: str) -> str:
    aa, bb = campaign_values(campaign)
    gain = scales(group)
    if len(aa) % 16 or len(bb) % 16:
        raise ValueError("campaign extents must be full Gemmini tiles")
    values = (
        ("static const int vals[16] = {" + ", ".join(str(v) for v in SMOKE_VALUES) + "};\n  ")
        if campaign != "full"
        else ""
    )
    row_value = "vals[i]" if len(aa) == 16 else "i-128"
    col_value = "vals[j]" if len(bb) == 16 else "j-128"
    init = (
        values
        + "for (int i=0;i<ROWS;i++) for (int j=0;j<COLS;j++) "
        + "{ A[i][j]="
        + row_value
        + "; B[i][j]="
        + col_value
        + "; }"
    )
    if campaign != "smoke":
        # Simulated UART output dominates a 65,536-element RTL campaign. Embed
        # the independently computed expectation and compare every output on
        # the Rocket, emitting only a bounded diagnostic transcript.
        unit = expected_outputs(group, campaign)["unit_model"]
        table = (
            "static const int8_t expected[ROWS][COLS] = {\n"
            + "\n".join("  {" + ", ".join(str(value) for value in row) + "}," for row in unit)
            + "\n};\n"
        )
        report = """int mismatches = 0;
  for (int i=0;i<ROWS;i++) for (int j=0;j<COLS;j++) {
    if (C[i][j] != expected[i][j]) {
      if (mismatches < 8) printf("BAD %d %d %d %d\\n", i, j, (int)C[i][j], (int)expected[i][j]);
      mismatches++;
    }
  }
  printf("CHECK %d\\n", mismatches);"""
    else:
        table = ""
        report = """for (int i=0;i<ROWS;i++) {
    printf("ROW %d", i);
    for (int j=0;j<COLS;j++) printf(" %d", (int)C[i][j]);
    printf("\\n");
  }"""
    return f"""/* Generated from one selected residual-add group; direct commands, no LOOP_WS. */
#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#define ROWS {len(aa)}
#define COLS {len(bb)}
static elem_t A[ROWS][COLS] row_align(1);
static elem_t B[ROWS][COLS] row_align(1);
static elem_t C[ROWS][COLS] row_align(1);
{table}
int main(void) {{
  {init}
  gemmini_flush(0);
  const uint32_t acc = (uint32_t)1 << (ADDR_LEN - 1);
  const uint32_t accumulate = (uint32_t)1 << (ADDR_LEN - 2);
  for (int i=0;i<ROWS;i+=DIM) for (int j=0;j<COLS;j+=DIM) {{
    gemmini_extended4_config_ld(COLS*sizeof(elem_t), {_c_float(gain["lhs_load"])}, true, DIM, 0);
    gemmini_extended_mvin(&A[i][j], acc, DIM, DIM);
    gemmini_fence();
    gemmini_extended4_config_ld(COLS*sizeof(elem_t), {_c_float(gain["rhs_load"])}, true, DIM, 0);
    gemmini_extended_mvin(&B[i][j], acc | accumulate, DIM, DIM);
    gemmini_fence();
    gemmini_extended_config_st(COLS*sizeof(elem_t), RELU, {_c_float(gain["readout"])});
    gemmini_extended_mvout(&C[i][j], acc, DIM, DIM);
    gemmini_fence();
  }}
  {report}
  printf("DONE\\n");
  return 0;
}}
"""


def parse_rows(console: str, shape: tuple[int, int]) -> list[list[int]]:
    out: dict[int, list[int]] = {}
    for line in console.splitlines():
        cells = line.split()
        if len(cells) >= 2 and cells[0] == "ROW":
            index = int(cells[1])
            if index in out or not 0 <= index < shape[0]:
                raise ValueError(f"duplicate or invalid output row {index}")
            out[index] = [int(value) for value in cells[2:]]
    if "DONE" not in console.splitlines() or sorted(out) != list(range(shape[0])):
        raise ValueError("incomplete native output transcript")
    rows = [out[index] for index in range(shape[0])]
    if any(len(row) != shape[1] or any(not -128 <= value <= 127 for value in row) for row in rows):
        raise ValueError("native output has wrong dimensions or dtype")
    return rows


def parse_full_check(console: str, pairs: int = 65536) -> int:
    checks = [line.split() for line in console.splitlines() if line.startswith("CHECK ")]
    if len(checks) != 1 or len(checks[0]) != 2 or "DONE" not in console.splitlines():
        raise ValueError("incomplete full-domain native comparison transcript")
    mismatches = int(checks[0][1])
    if mismatches < 0 or mismatches > pairs:
        raise ValueError("invalid full-domain mismatch count")
    return mismatches


def decode_custom(disassembly: str, facts: dict) -> list[dict]:
    table = next(row for row in facts["facts"]["interfaces"] if row.get("name") == "funct_decode_table")
    opcode = table.get("custom_opcode")
    if not isinstance(opcode, int):
        raise ValueError("selected facts did not derive custom opcode")
    seen = []
    for line in disassembly.splitlines():
        _, sep, remainder = line.partition(":")
        if not sep:
            continue
        tokens = remainder.split()
        if not tokens or len(tokens[0]) != 8 or any(ch not in "0123456789abcdefABCDEF" for ch in tokens[0]):
            continue
        word = int(tokens[0], 16)
        if word & 0x7F == opcode:
            funct = (word >> 25) & 0x7F
            seen.append({"word": tokens[0], "funct": funct, "class": table["names"].get(str(funct))})
    classes = {row["class"] for row in seen}
    needed = {"CONFIG_CMD", "LOAD_CMD", "STORE_CMD", "FLUSH_CMD"}
    if not needed <= classes or classes - needed:
        raise ValueError(f"direct command proof failed: classes={sorted(str(value) for value in classes)}")
    return seen


def compare(actual: list[list[int]], expected: dict, bound: int) -> dict:
    differences = [abs(x - y) for row, gold in zip(actual, expected["reference"]) for x, y in zip(row, gold)]
    model_diff = [abs(x - y) for row, gold in zip(actual, expected["unit_model"]) for x, y in zip(row, gold)]
    return {
        "max_reference_error": max(differences),
        "n_over_bound": sum(d > bound for d in differences),
        "max_unit_model_error": max(model_diff),
        "n_unit_model_mismatch": sum(d != 0 for d in model_diff),
    }


def bind_simulator_attestation(path: Path, *, phase0: Path, numerical: dict) -> dict:
    """Recheck retained exact-rebuild bytes against this execution and CIRCT source."""
    attestation = json.loads(path.read_text(encoding="utf-8"))
    if (
        attestation.get("schema") != "merlin.gemmini-verilator-provenance.v1"
        or attestation.get("status") != "reproduced_exact_binary"
    ):
        raise ValueError("simulator attestation is not an exact-binary rebuild record")
    facts = json.loads((phase0 / "hardware/circt/facts.json").read_text(encoding="utf-8"))
    selected = facts["facts"]["source"]
    source_selection = Path(selected["declared_by"])
    firrtl = Path(selected["fir_path"])
    executable = Path(numerical["tools"]["verilator"]["path"])
    binary_sha = sha(executable)
    if (
        sha(source_selection) != attestation["source_selection_sha256"]
        or sha(firrtl) != selected["fir_sha256"]
        or selected["fir_sha256"] != attestation["selected_firrtl_sha256"]
        or binary_sha != numerical["tools"]["verilator"]["sha256"]
        or binary_sha != attestation["kernel_tested_binary_sha256"]
        or binary_sha != attestation["rebuilt_binary_sha256"]
        or binary_sha != sha(path.parent / "simulator-rebuild")
    ):
        raise ValueError("attested selected FIRRTL/source/simulator bytes differ from this run")
    hashes = attestation.get("core_rtl_sha256") or {}
    if len(hashes) < 100:
        raise ValueError("attestation lacks the selected Gemmini core RTL closure")
    selected_rtl = firrtl.parent / "gen-collateral"
    retained_rtl = path.parent / "firtool/gen-collateral"
    for name, digest in hashes.items():
        if Path(name).name != name or sha(selected_rtl / name) != digest or sha(retained_rtl / name) != digest:
            raise ValueError(f"attested Gemmini RTL file changed: {name}")
    return {
        "receipt_sha256": sha(path),
        "source_selection_sha256": sha(source_selection),
        "selected_firrtl_sha256": sha(firrtl),
        "simulator_binary_sha256": binary_sha,
        "retained_core_rtl_files_checked": len(hashes),
        "historical_procedure_limit": "The archived prior rebuild receipt records exact firtool/Verilator reproduction; "
        "its temporary FIRRTL copy and annotations were not retained in the archive.",
    }


def bind_existing_run(args: argparse.Namespace) -> dict:
    """Bind identical current Phase 0 group bytes to a completed numerical run.

    This is a provenance-equivalence record, not a rerun, SW-spec review, or
    assertion that the selected source model was compiled by the ELF.
    """
    root = repo_root()
    output, phase0 = args.output.resolve(), args.phase0.resolve()
    if output.exists() or not output.is_relative_to(root / "out"):
        raise ValueError("--output must be a new directory under this checkout's out/")
    old_path, current_path = args.rebind_receipt.resolve(), args.groups.resolve()
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if old.get("schema") != SCHEMA or old.get("status") != "passed":
        raise ValueError("the prior numerical receipt must be complete and passed")
    if set(old.get("observations", {})) != {"spike", "verilator"}:
        raise ValueError("the prior numerical receipt lacks one independent engine")
    if any(row.get("status") != "passed" for row in old["observations"].values()):
        raise ValueError("the prior numerical receipt has an unpassed engine")
    old_dir = old_path.parent
    for key, filename in (
        ("source", "direct_residual_readout.c"),
        ("expected_sha256", "expected.json"),
        ("disassembly_sha256", "disassembly.txt"),
        ("elf_sha256", "direct_residual_readout.elf"),
    ):
        recorded = old[key]["sha256"] if key == "source" else old[key]
        if sha(old_dir / filename) != recorded:
            raise ValueError(f"prior receipt does not bind {filename}")
    for engine, row in old["observations"].items():
        if sha(old_dir / f"{engine}.console.log") != row["console_sha256"]:
            raise ValueError(f"prior receipt does not bind {engine} transcript")
    old_report_path = Path(old["groups"]["path"])
    if sha(old_report_path) != old["groups"]["sha256"]:
        raise ValueError("prior group report changed since the numerical run")
    previous, current = (json.loads(path.read_text(encoding="utf-8")) for path in (old_report_path, current_path))
    group = select_group(current, args.group)
    if group != old["group"]:
        raise ValueError("current and previously simulated numerical groups differ")
    prior_row = next(row for row in previous["entries"] if row["name"] == group["name"])
    current_row = next(row for row in current["entries"] if row["name"] == group["name"])
    if prior_row["entry"] != current_row["entry"] or prior_row["groups"] != current_row["groups"]:
        raise ValueError("group entry or source group identity changed")
    if (old_dir / "direct_residual_readout.c").read_text(encoding="utf-8") != render_source(group, old["campaign"]):
        raise ValueError("current group does not render the simulated source bytes")
    expected = expected_outputs(group, old["campaign"])
    expected_bytes = json.dumps(expected, separators=(",", ":")) + "\n"
    if (old_dir / "expected.json").read_text(encoding="utf-8") != expected_bytes:
        raise ValueError("current group does not reproduce the numerical expectation bytes")
    unit_digest = hashlib.sha256(bytes(value & 0xFF for row in expected["unit_model"] for value in row)).hexdigest()
    if old["campaign"] != "smoke" and old.get("embedded_expected_i8_sha256") != unit_digest:
        raise ValueError("full-domain run does not bind the embedded i8 expectation table")
    manifest = json.loads((phase0 / "evidence-manifest.json").read_text(encoding="utf-8"))
    report_inputs = current["inputs"]
    capture = Path(report_inputs["capture"]["path"]).resolve()
    if (
        not capture.is_relative_to(phase0 / "capsules/model")
        or capture.name != "capsule.interface.mlir"
        or report_inputs["capture"]["sha256"] != sha(capture)
    ):
        raise ValueError("current report is not bound to a frozen model interface")
    model = yaml.safe_load((capture.parent / "capsule.yaml").read_text(encoding="utf-8"))
    qualified = (model.get("model_qualification") or {}).get("group_capsules") or []
    if not any(row.get("capsule") == f"layers/{group['name']}" for row in qualified):
        raise ValueError("frozen model does not name this isolated numerical group")
    source_capture_sha = model["materialized_capture"]["capture_sha256"]
    role = f"application-capture:{model['materialized_capture']['workload_id']}"
    if not any(row.get("role") == role and row.get("sha256") == source_capture_sha for row in manifest["sources"]):
        raise ValueError("frozen source model is absent from Phase 0 evidence")
    facts = phase0 / "hardware/circt/facts.json"
    if report_inputs["rtl_facts"]["sha256"] != sha(facts) or manifest["raw_facts_sha256"] != sha(facts):
        raise ValueError("current group plan did not use the frozen CIRCT facts")
    contract_sha = report_inputs["capability_contract"]["sha256"]
    if not any(
        row.get("role") == "target-contract" and row.get("sha256") == contract_sha for row in manifest["sources"]
    ):
        raise ValueError("current group plan did not use the frozen target contract")
    if old["selected_evidence"]["phase0_manifest"]["sha256"] != sha(phase0 / "evidence-manifest.json"):
        raise ValueError("numerical run used a different Phase 0 manifest")
    record = {
        "schema": "merlin.gemmini-direct-residual-provenance-equivalence.v1",
        "status": "passed_numeric_provenance_equivalence",
        "phase0_admission": "unchanged_unreviewed",
        "scope": "same numerical group, generated source, expectation and ELF run; no compiler lowering claim",
        "phase0_manifest_sha256": sha(phase0 / "evidence-manifest.json"),
        "source_capture_sha256": source_capture_sha,
        "frozen_interface_sha256": sha(capture),
        "selected_facts_sha256": sha(facts),
        "selected_contract_sha256": contract_sha,
        "current_group_report_sha256": sha(current_path),
        "previous_group_report_sha256": sha(old_report_path),
        "numerical_receipt_sha256": sha(old_path),
        "simulated_source_sha256": old["source"]["sha256"],
        "simulated_elf_sha256": old["elf_sha256"],
        "independent_expected_i8_sha256": unit_digest,
        "pairs": old["pairs"],
        "group": group,
        "observations": old["observations"],
    }
    if args.attestation is not None:
        record["simulator_attestation"] = bind_simulator_attestation(
            args.attestation.resolve(), phase0=phase0, numerical=old
        )
    output.mkdir(parents=True)
    (output / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def run(args: argparse.Namespace) -> dict:
    root = repo_root()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "out"):
        raise ValueError("--output must be a new directory under this checkout's out/")
    phase0 = args.phase0.resolve()
    groups_path = args.groups.resolve()
    if output.is_relative_to(phase0) or output.is_relative_to(groups_path.parent):
        raise ValueError("--output must not overlap selected inputs")
    backend = get_backend("gemmini")
    plugin = load_declared("gemmini", "reference_programs")
    vendor = backend.rocc_tests_dir()
    evidence, facts = selected_evidence(phase0, vendor)
    group = select_group(json.loads(groups_path.read_text()), args.group)
    expected = expected_outputs(group, args.campaign)
    model_check = compare(expected["unit_model"], expected, group["bound"])
    if model_check["n_over_bound"]:
        raise ValueError("selected scales fail declared bound even in the independent software model")
    output.mkdir(parents=True)
    source = render_source(group, args.campaign)
    elf = plugin.build(source, "direct_residual_readout", output)
    chipyard = Path(args.chipyard).resolve()
    objdump = chipyard / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-objdump"
    disassembly = subprocess.run(
        [str(objdump), "-d", str(elf)], capture_output=True, text=True, check=True, timeout=30
    ).stdout
    disasm_path = output / "disassembly.txt"
    disasm_path.write_text(disassembly, encoding="utf-8")
    commands = decode_custom(disassembly, facts)
    expected_path = output / "expected.json"
    expected_path.write_text(json.dumps(expected, separators=(",", ":")) + "\n", encoding="utf-8")
    unit_bytes = bytes(value & 0xFF for row in expected["unit_model"] for value in row)
    observations = {}
    values = campaign_values(args.campaign)
    for simulator in ("spike", "verilator"):
        if not backend.available(simulator):
            observations[simulator] = {"status": "unavailable"}
            continue
        console = backend.run_elf(elf, simulator=simulator, timeout=args.timeout)
        transcript = output / f"{simulator}.console.log"
        transcript.write_text(console, encoding="utf-8")
        if args.campaign != "smoke":
            mismatches = parse_full_check(console, len(values[0]) * len(values[1]))
            check = {
                **model_check,
                "n_unit_model_mismatch": mismatches,
                "max_unit_model_error": 0 if mismatches == 0 else None,
                "native_comparison": "exact_per_element_against_embedded_independent_unit_model",
            }
        else:
            actual = parse_rows(console, (len(values[0]), len(values[1])))
            check = compare(actual, expected, group["bound"])
        observations[simulator] = {
            "status": "passed" if check["n_over_bound"] == 0 and check["n_unit_model_mismatch"] == 0 else "failed",
            "console_sha256": sha(transcript),
            **check,
        }
    receipt = {
        "schema": SCHEMA,
        "status": "passed" if all(row["status"] == "passed" for row in observations.values()) else "incomplete",
        "campaign": args.campaign,
        "pairs": len(values[0]) * len(values[1]),
        "operand_domain": {"lhs_values": list(values[0]), "rhs_values": list(values[1])},
        "group": group,
        "scales": expected["gain"],
        "model_check": model_check,
        "selected_evidence": evidence,
        "groups": {"path": str(groups_path), "sha256": sha(groups_path)},
        "source": {
            "path": str(output / "direct_residual_readout.c"),
            "sha256": sha(output / "direct_residual_readout.c"),
        },
        "elf_sha256": sha(elf),
        "disassembly_sha256": sha(disasm_path),
        "expected_sha256": sha(expected_path),
        "embedded_expected_i8_sha256": hashlib.sha256(unit_bytes).hexdigest(),
        "custom_instruction_classes": commands,
        "tools": {
            name: {"path": str(path), "sha256": sha(path)}
            for name, path in (
                ("gcc", backend.gcc_path()),
                ("objdump", objdump),
                ("spike", backend.spike_path()),
                ("libgemmini", backend.libgemmini_dir() / "libgemmini.so"),
                ("verilator", backend.verilator_path()),
            )
            if path.is_file()
        },
        "observations": observations,
        "limits": [
            "Selected scales and one readout mode only; no compiler code generation is exercised.",
            "Verilator binary hash is recorded, but its RTL build identity needs a separate attestation.",
            "Finite simulation is not a universal proof of transform correctness.",
        ],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0", required=True, type=Path, help="frozen phase0/ directory with verified facts")
    parser.add_argument("--groups", required=True, type=Path, help="selected group_capsules.json")
    parser.add_argument("--group", help="exact residual/ReLU group name when report has more than one")
    parser.add_argument("--chipyard", required=True, type=Path)
    parser.add_argument("--campaign", choices=("smoke", "boundary_rows", "boundary_cols", "full"), default="smoke")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--rebind-receipt",
        type=Path,
        help="bind an existing passed numerical run to this frozen group if all bytes match",
    )
    parser.add_argument(
        "--attestation", type=Path, help="with --rebind-receipt, recheck a retained exact Verilator rebuild attestation"
    )
    args = parser.parse_args()
    if args.rebind_receipt is not None:
        result = bind_existing_run(args)
        print(json.dumps({key: result[key] for key in ("status", "pairs", "observations")}, indent=2))
        return 0
    if args.attestation is not None:
        parser.error("--attestation requires --rebind-receipt")
    result = run(args)
    print(json.dumps({key: result[key] for key in ("status", "campaign", "pairs", "observations")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
