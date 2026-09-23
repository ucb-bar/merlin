#!/usr/bin/env python3
"""The published-GEMM comparison table, every row measured ON HARDWARE and admitted on its digest.

``autocomp_exo_gemm_table.py`` answers this question on GSIM. GSIM is a simulator, and a claim about a
device needs the device: this script assembles the same table out of SEALED FireSim batches produced by
``firesim_gemm_windows.py``, and reports the simulator's error against the hardware wherever an arm has
both. Nothing here measures anything -- it reads receipts -- so a number it prints came from a queue job
whose UART was parsed by ``merlin.perf.firesim_receipt`` and whose every window matched the off-device
integer oracle.

WHAT IT REFUSES, AND WHY EACH REFUSAL IS THE POINT.

* An UNSEALED batch. A cycle count whose receipt was refused is not a measurement.
* A window whose printed digest is not the oracle's. FireSim job 730 cleared two declared cycle
  thresholds, printed a success marker byte-identical to a correct run's, and was wrong; a window that
  clears its cycle count and misses its digest is a FAILURE, not an incomplete, and never a row.
* A batch whose order-effect control diverged past its declared bound. The control repeats window 0 at
  the end; past the bound, nothing distinguishes "this arm is faster" from "this arm ran first", and
  every window in the batch is invalidated, including the ones that looked fine.
* ARMS OF ONE SHAPE THAT DISAGREE ON THEIR OUTPUT. Ratios between arms mean something only while the
  arms compute one function. Two arms with different digests are two questions.

    .venv/bin/python merlin/experiments/gemmini_perf_bench/scripts/autocomp_hardware_table.py \\
        --run-dir <a sealed firesim_gemm_windows run dir> --run-dir <another> \\
        --gsim-table out/artifacts/probes/<probe>/autocomp_exo_gemm_table.json \\
        --ratio ours_package/exo_opt --ratio ours_schedule/gemmini_baseline
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

#: A batch names an arm for how the window ISSUED it (``schedule`` is this repo's own renderer); a
#: probe and a paper name it for WHOSE it is. Both vocabularies are real and neither is wrong, so the
#: translation is stated once here rather than guessed wherever the two meet. An arm with no entry
#: keeps its own name.
REPORT_NAME = {"schedule": "ours_schedule", "library": "ours_vendor", "package": "ours_package"}
#: Reading order for the columns: ours first, then the published arms in the figure's own order.
ARM_ORDER = (
    "ours_schedule",
    "ours_vendor",
    "ours_package",
    "gemmini_baseline",
    "exo_baseline",
    "exo_opt",
    "autocomp_generated",
)


def geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def sealed_windows(run_dir: Path) -> dict[str, Any]:
    """Every measured window of one SEALED batch, keyed by label, with the run's own provenance.

    The refusals live here rather than in the caller because this is the only place that can see the
    receipt: a run dir whose ``firesim`` block is missing, unsealed, or whose admission refused a
    window has no rows to contribute, and saying so is more useful than a table with a hole in it.
    """
    document = json.loads((run_dir / "firesim_gemm_windows.json").read_text(encoding="utf-8"))
    firesim = document.get("firesim")
    if not firesim:
        raise SystemExit(f"{run_dir}: this run was built but never submitted; it holds no hardware number")
    if not firesim.get("sealed"):
        raise SystemExit(f"{run_dir}: the batch was NOT sealed ({firesim.get('refusal', 'no reason recorded')})")
    admission = firesim.get("admission") or {}
    refused = [window for window in admission.get("windows", []) if window.get("status") not in (None, "pass")]
    if refused:
        names = ", ".join(f"{w.get('label')}={w.get('status')}" for w in refused)
        raise SystemExit(f"{run_dir}: the batch holds a window that did not pass: {names}")
    declared = {row["label"]: row for row in document["windows"]}
    measured: dict[str, dict[str, Any]] = {}
    for window in firesim["windows"]:
        row = declared[window["label"]]
        measured[window["label"]] = {
            "cycles": int(window["cycles"]),
            "is_order_control": bool(window.get("is_order_control")),
            "shape": row["shape"],
            "arm": REPORT_NAME.get(row["arm"], row["arm"]),
            "macs": int(row["macs"]),
            "digest_expected": int(row["digest_expected"]),
        }
    return {
        "run_dir": str(run_dir),
        "job_id": firesim["job_id"],
        "batch_id": document["batch_id"],
        "elf_sha256": document["elf_sha256"],
        "blob_sha256": document["blob_sha256"],
        "order_effect_ppm": firesim.get("order_effect_ppm"),
        "uartlog_sha256": firesim["uartlog_sha256"],
        "device": document["device"],
        "spec": {key: document.get(key) for key in ("scale", "relu", "elem_mode", "bias_span", "contract_digest")},
        "windows": measured,
    }


def gsim_rows(path: Path) -> dict[str, dict[str, int]]:
    """``shape -> arm -> cycles`` from a GSIM arm table, admitting only its digest-exact rows."""
    document = json.loads(path.read_text(encoding="utf-8"))
    table: dict[str, dict[str, int]] = {}
    for row in document["rows"]:
        if row.get("numerics") != "exact" or row.get("cycles") is None:
            continue
        table.setdefault(row["sig"], {})[row["arm"]] = int(row["cycles"])
    return table


def collect(runs: list[dict[str, Any]], *, order_effect_bound_ppm: int) -> dict[str, dict[str, dict[str, Any]]]:
    """``shape -> arm -> {cycles, job, ...}`` over every sealed batch, refusing a disagreeing digest.

    An arm measured in two batches keeps BOTH numbers: they are a cross-job control on the comparison,
    not a duplicate to be silently collapsed to one.
    """
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        drift = run.get("order_effect_ppm")
        if drift is not None and abs(float(drift)) > order_effect_bound_ppm:
            raise SystemExit(
                f"job {run['job_id']}: the order-effect control diverged {drift} ppm, past the declared "
                f"{order_effect_bound_ppm} ppm; every window in this batch is invalidated"
            )
        for label, window in run["windows"].items():
            if window["is_order_control"]:
                continue
            shape, arm = window["shape"], window["arm"]
            slot = table.setdefault(shape, {}).setdefault(arm, {"cycles": [], "jobs": [], "digest": set()})
            slot["cycles"].append(window["cycles"])
            slot["jobs"].append(run["job_id"])
            slot["by_job"] = {**slot.get("by_job", {}), run["job_id"]: window["cycles"]}
            slot["digest"].add(window["digest_expected"])
            slot["macs"] = window["macs"]
    for shape, arms in table.items():
        digests = {next(iter(slot["digest"])) for slot in arms.values()}
        if len(digests) > 1:
            raise SystemExit(
                f"{shape}: the arms were admitted against DIFFERENT oracles {sorted(digests)}; they do not "
                "compute one function and no ratio between them means anything"
            )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", type=Path, required=True, help="a sealed run dir; repeatable")
    parser.add_argument("--gsim-table", action="append", type=Path, default=None, help="an arm table; repeatable")
    parser.add_argument("--ratio", action="append", default=None, help="numerator/denominator arm pair; repeatable")
    parser.add_argument("--order-effect-bound-ppm", type=int, default=10_000)
    parser.add_argument("--shape", action="append", default=None, help="shape order; default: as measured")
    parser.add_argument(
        "--pin",
        action="append",
        default=None,
        help="hardware pin to verify and record; repeatable. A table of hardware numbers that does not "
        "name the revision they came from is unattributable, so this is verified, not asserted.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--out-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)

    runs = [sealed_windows(Path(run_dir)) for run_dir in arguments.run_dir]
    table = collect(runs, order_effect_bound_ppm=arguments.order_effect_bound_ppm)
    simulated: dict[str, dict[str, int]] = {}
    for path in arguments.gsim_table or []:
        for shape, arms in gsim_rows(Path(path)).items():
            simulated.setdefault(shape, {}).update(arms)

    shapes = arguments.shape or list(table)
    present = {arm for arms in table.values() for arm in arms}
    arms = [arm for arm in ARM_ORDER if arm in present] + sorted(present - set(ARM_ORDER))
    ratios = [tuple(spec.split("/", 1)) for spec in (arguments.ratio or [])]

    rows: list[dict[str, Any]] = []
    for shape in shapes:
        measured = table.get(shape, {})
        row: dict[str, Any] = {"shape_MxNxK": shape}
        for arm, slot in measured.items():
            row[f"hw_{arm}_cycles"] = min(slot["cycles"])
            if len(slot["cycles"]) > 1:
                row[f"hw_{arm}_repeats"] = slot["cycles"]
                spread = (max(slot["cycles"]) - min(slot["cycles"])) / min(slot["cycles"])
                row[f"hw_{arm}_cross_job_ppm"] = round(1e6 * spread, 1)
            row[f"hw_{arm}_jobs"] = sorted(set(slot["jobs"]))
            if shape in simulated and arm in simulated[shape]:
                gsim = simulated[shape][arm]
                row[f"gsim_{arm}_cycles"] = gsim
                row[f"gsim_error_{arm}_pct"] = round(
                    100.0 * (gsim - row[f"hw_{arm}_cycles"]) / row[f"hw_{arm}_cycles"], 4
                )
        for numerator, denominator in ratios:
            # WITHIN ONE BATCH where a batch measured both arms. Two arms measured in different jobs
            # sit behind a different boot, a different image and a different DRAM history, and the
            # cross-job spread below is the size of that: quoting a ratio across jobs would fold that
            # spread into the comparison while claiming it came from the device's own behaviour.
            above, below = measured.get(numerator), measured.get(denominator)
            if not above or not below:
                continue
            shared = sorted(set(above.get("by_job", {})) & set(below.get("by_job", {})))
            if shared:
                job = shared[0]
                row[f"{denominator}_over_{numerator}"] = round(below["by_job"][job] / above["by_job"][job], 4)
                row[f"{denominator}_over_{numerator}_job"] = job
            else:
                row[f"{denominator}_over_{numerator}"] = round(
                    row[f"hw_{denominator}_cycles"] / row[f"hw_{numerator}_cycles"], 4
                )
                row[f"{denominator}_over_{numerator}_job"] = "ACROSS JOBS"
        rows.append(row)

    summary: dict[str, Any] = {}
    for numerator, denominator in ratios:
        key = f"{denominator}_over_{numerator}"
        values = [row[key] for row in rows if key in row]
        if values:
            summary[key] = {
                "geomean": round(geomean(values), 4),
                "min": min(values),
                "max": max(values),
                "n": len(values),
                "per_shape": {row["shape_MxNxK"]: row[key] for row in rows if key in row},
                "measured_in": sorted({str(row[f"{key}_job"]) for row in rows if f"{key}_job" in row}),
            }
    errors = [abs(row[f"gsim_error_{arm}_pct"]) for row in rows for arm in arms if f"gsim_error_{arm}_pct" in row]
    if errors:
        summary["gsim_error_vs_hardware_pct"] = {
            "max": round(max(errors), 4),
            "mean": round(statistics.fmean(errors), 4),
            "windows": len(errors),
            "signed_mean": round(
                statistics.fmean(
                    row[f"gsim_error_{arm}_pct"] for row in rows for arm in arms if f"gsim_error_{arm}_pct" in row
                ),
                4,
            ),
        }

    header = "| shape (MxNxK) | " + " | ".join(f"{arm} HW" for arm in arms) + " | "
    header += " | ".join(f"{d}/{n}" for n, d in ratios) + " |"
    lines = [header, "|---|" + "---:|" * (len(arms) + len(ratios))]
    for row in rows:
        cells = [f"{row[f'hw_{arm}_cycles']:,}" if f"hw_{arm}_cycles" in row else "-" for arm in arms]
        cells += [f"{row[f'{d}_over_{n}']:.3f}x" if f"{d}_over_{n}" in row else "-" for n, d in ratios]
        lines.append(f"| `{row['shape_MxNxK']}` | " + " | ".join(cells) + " |")
    geomean_cells = []
    for numerator, denominator in ratios:
        stats = summary.get(f"{denominator}_over_{numerator}")
        geomean_cells.append(f"**{stats['geomean']:.3f}x**" if stats else "-")
    if geomean_cells:
        lines.append("| **geomean** | " + " | ".join("" for _ in arms) + " | " + " | ".join(geomean_cells) + " |")
        lines.append("")
        for numerator, denominator in ratios:
            stats = summary.get(f"{denominator}_over_{numerator}")
            if stats:
                lines.append(
                    f"- `{denominator}/{numerator}`: geomean **{stats['geomean']:.3f}x** over {stats['n']} "
                    f"shapes, spread {stats['min']:.3f}x to {stats['max']:.3f}x."
                )
    if "gsim_error_vs_hardware_pct" in summary:
        error = summary["gsim_error_vs_hardware_pct"]
        lines.append("")
        lines.append(
            f"- GSIM vs this hardware over {error['windows']} compared windows: at most "
            f"**{error['max']:.2f}%**, mean {error['mean']:.2f}%, signed mean {error['signed_mean']:+.2f}% "
            "(negative means GSIM understates the hardware)."
        )
        # PER ARM, because one number over all of them hides the thing a reader needs. The simulator's
        # error is a property of the program it is simulating, not a constant of the simulator, so an
        # arm whose error is an order of magnitude larger than the rest must not be covered by a mean.
        lines.append("")
        lines.append("| arm | windows | max abs error | mean signed error |")
        lines.append("|---|---:|---:|---:|")
        for arm in arms:
            errors = [row[f"gsim_error_{arm}_pct"] for row in rows if f"gsim_error_{arm}_pct" in row]
            if errors:
                lines.append(
                    f"| `{arm}` | {len(errors)} | {max(abs(e) for e in errors):.2f}% | "
                    f"{statistics.fmean(errors):+.2f}% |"
                )
    repeated = [
        (row["shape_MxNxK"], arm, row[f"hw_{arm}_repeats"], row[f"hw_{arm}_cross_job_ppm"])
        for row in rows
        for arm in arms
        if f"hw_{arm}_repeats" in row
    ]
    if repeated:
        lines.append("")
        lines.append(
            "Cross-job control: an arm measured in more than one batch, which is the size of the "
            "difference a different boot, image and DRAM history makes. Ratios above are taken WITHIN "
            "one batch, so this spread does not enter them."
        )
        lines.append("")
        lines.append("| shape | arm | cycles per job | spread |")
        lines.append("|---|---|---|---:|")
        for shape, arm, values, ppm in repeated:
            lines.append(f"| `{shape}` | `{arm}` | {', '.join(f'{v:,}' for v in values)} | {ppm:.0f} ppm |")
    table_md = "\n".join(lines)
    print(table_md)
    print(json.dumps(summary, indent=1))

    from merlin.common.artifacts import new_measurement

    # The measurement dir is minted even when the caller names one, so the manifest that says WHICH
    # substrate, revision and run these files belong to is written either way. A directory of numbers
    # with no manifest beside it is the shape a later reader cannot attribute.
    substrate = f"firesim_{runs[0]['device']['hw_config']}"
    measurement = new_measurement(
        substrate,
        "gemm_shapes",
        "published_arms",
        notes=arguments.notes or "published GEMM arms on hardware",
        update_latest=arguments.out_dir is None,
    )
    out_dir = arguments.out_dir or measurement.path
    out_dir.mkdir(parents=True, exist_ok=True)
    if arguments.out_dir is not None and measurement.path != out_dir:
        measurement.path.rmdir()
        measurement = replace(measurement, path=out_dir, manifest_path=out_dir / "manifest.yaml")
    from merlin.common import provenance as provenance_api

    pins = {name: provenance_api.require(name) for name in (arguments.pin or [])}
    document = {
        "schema": "autocomp_hardware_table_v1",
        "provenance": provenance_api.record(
            pins=pins,
            extra={
                "bitstream": {
                    "artifact_name": runs[0]["device"].get("bitstream_artifact_name"),
                    "tar_sha256": runs[0]["device"].get("bitstream_tar_sha256"),
                    "why_not_the_config_string": runs[0]["device"].get("config_string_is_not_an_identity"),
                },
                "jobs": [run["job_id"] for run in runs],
                "uartlog_sha256": {run["job_id"]: run["uartlog_sha256"] for run in runs},
            },
        ),
        "generated_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "arms": arms,
        "shapes": shapes,
        "rows": rows,
        "summary": summary,
        "runs": [{key: run[key] for key in run if key != "windows"} for run in runs],
        "gsim_tables": [str(path) for path in (arguments.gsim_table or [])],
    }
    (out_dir / "measurement.json").write_text(json.dumps(document, indent=1, sort_keys=True), encoding="utf-8")
    (out_dir / "table.md").write_text(table_md + "\n", encoding="utf-8")
    for name in ("measurement.json", "table.md"):
        measurement.add_artifact(name)
    measurement.write_manifest()
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
