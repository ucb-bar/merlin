#!/usr/bin/env python3
"""A/B the reviewed Atlas weight-buffer erratum on saved independent/chained cases."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path[:0] = [str(REPO / "merlin/python"), str(ROOT), str(ROOT / "submission")]

from merlin.common.paths import ext_path  # noqa: E402
from merlin.targetgen.dram_facts import dram_base_for  # noqa: E402
from merlin.targetgen.isa_model import _reviewed_errata  # noqa: E402
from merlin.targetgen.program_oracle import (  # noqa: E402
    _FUNC_ERRATA_HELPER,
    _bundle_preload,
    _decode_output,
    _func_program_helper,
    _model_venv_python,
    _out_nbytes,
    _resolve_out_specs,
    emit_bundle,
)
from mlir_oot.cmdbuf import build_command_buffer  # noqa: E402
from mlir_oot.frontend import parse_verified  # noqa: E402
from run_integration import fixture  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_arm(case: str, *, include_weight_layout: bool, workdir: Path) -> dict:
    case_root = ROOT / "cases" / case
    workdir.mkdir(parents=True)
    cb = build_command_buffer(parse_verified(
        (case_root / "two_matmuls.mlir").read_text(encoding="utf-8")
    ))
    inputs, expected = fixture(case, cb)
    bundle = emit_bundle(
        model_ext="npu_model", kernel_s=case_root / "kernel.S", inputs=inputs,
        fix_itype_rd=True, workdir=workdir, timeout=120,
    )
    specs = _resolve_out_specs("atlas", cb, bundle)
    request = {
        "words": [int(word) & 0xFFFFFFFF for word in bundle["words"]],
        "preload": [
            {"base": int(base), "b64": base64.b64encode(raw).decode()}
            for base, raw in _bundle_preload(bundle, cb)
        ],
        "out_bases": [[int(spec["base"]), _out_nbytes(spec)] for spec in specs.values()],
        "max_cycles": 200000,
        "dram_base": int(dram_base_for("atlas")),
        "reviewed_isa_errata": {
            name: row for name, row in _reviewed_errata("atlas").items()
            if str(row.get("authoritative") or "").lower() == "rtl"
            and row.get("declared") not in (None, "")
            and row.get("hardware") not in (None, "")
        },
        "functional_model_isa_module": "npu_model.configs.isa_definition",
    }
    registry = yaml.safe_load(
        (REPO / "merlin/contract/functional_model_errata.yaml").read_text()
    )["errata"]["atlas"]["corrections"]
    request["reviewed_functional_model_errata"] = {
        name: row for name, row in registry.items()
        if include_weight_layout or name != "weight_buffer_lane_order"
    }
    infile, outfile = workdir / "func_in.json", workdir / "func_out.json"
    infile.write_text(json.dumps(request), encoding="utf-8")
    process = subprocess.run(
        [str(_model_venv_python("npu_model")), str(_FUNC_ERRATA_HELPER),
         "--runner", str(_func_program_helper("atlas")),
         "--in", str(infile), "--out", str(outfile)],
        cwd=ext_path("npu_model"), capture_output=True, text=True, timeout=120,
    )
    if process.returncode:
        raise RuntimeError(process.stderr[-500:])
    raw = json.loads(outfile.read_text())
    if not raw.get("halted"):
        raise RuntimeError(f"{case} did not halt")
    comparisons = {}
    for name, reference in expected.items():
        spec = specs[name]
        actual = np.asarray(_decode_output(
            base64.b64decode(raw["outputs"][str(spec["base"])]), spec["shape"],
            spec["dtype"], spec["physical"],
        ), dtype=np.float32)
        comparisons[name] = {
            "elements": int(reference.size),
            "mismatches": int(np.count_nonzero(actual != reference)),
            "max_abs_error": float(np.max(np.abs(actual - reference))),
        }
    return {
        "halted": True,
        "cycles": int(raw["cycles"]),
        "comparisons": comparisons,
        "all_outputs_bit_exact": all(row["mismatches"] == 0 for row in comparisons.values()),
        "weight_layout_erratum_applied": any(
            row["name"] == "weight_buffer_lane_order"
            for row in raw["reviewed_functional_model_errata_applied"]
        ),
    }


def main() -> int:
    cases = {}
    with tempfile.TemporaryDirectory(prefix="atlas-weight-layout-") as temp:
        temp_root = Path(temp)
        for case in ("independent", "chained"):
            cases[case] = {
                "without_weight_layout_erratum": run_arm(
                    case, include_weight_layout=False, workdir=temp_root / f"{case}-before"
                ),
                "with_weight_layout_erratum": run_arm(
                    case, include_weight_layout=True, workdir=temp_root / f"{case}-after"
                ),
                "retained_rtl": json.loads(
                    (ROOT / "cases" / case / "gsim_result.json").read_text()
                )["comparisons"],
            }

    model = ext_path("npu_model") / "npu_model/configs/isa_definition.py"
    atlas = ext_path("npu_model").parent
    rtl_sources = [
        atlas / "src/main/scala/atlas/mxu/WeightBuffers.scala",
        atlas / "src/main/scala/atlas/mxu/sa/SystolicArrayTop.scala",
        atlas / "src/main/scala/atlas/mxu/ipt/InnerProductTrees.scala",
    ]
    report = {
        "schema": "atlas_functional_weight_layout_diagnosis_v1",
        "status": "rtl_authoritative_erratum_qualified",
        "claim": "process-local functional-model correction; not RTL execution or performance evidence",
        "diagnosis": (
            "functional read_wb_fp8 exposes [output_lane,reduction], but _vmatmul "
            "consumed it as [reduction,output_lane]"
        ),
        "ruled_out": [
            "VTRPOSE_XLU: model and RTL both transpose the 32x32 MRF",
            "VMATPUSH_WEIGHT: model and RTL both copy each MRF row to one output-lane slot",
            "output DMA layout: the correction is internal to matmul and preserves the chained store",
        ],
        "cases": cases,
        "source_evidence": {
            "functional_model": {
                "path": "npu_model/configs/isa_definition.py",
                "sha256": digest(model),
            },
            "rtl": [{"path": path.relative_to(atlas).as_posix(), "sha256": digest(path)}
                    for path in rtl_sources],
        },
        "admission_rule": (
            "independent must become bit-exact and chained must remain bit-exact; "
            "both arms must halt"
        ),
    }
    out = ROOT / "evidence/functional_weight_layout_diagnosis.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if (
        cases["independent"]["with_weight_layout_erratum"]["all_outputs_bit_exact"]
        and cases["chained"]["with_weight_layout_erratum"]["all_outputs_bit_exact"]
        and not cases["independent"]["without_weight_layout_erratum"]["all_outputs_bit_exact"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
