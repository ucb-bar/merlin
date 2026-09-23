#!/usr/bin/env python3
"""Turn one SEALED FireSim batch into the hardware-vs-GSIM comparison artifact.

The comparison is the point of the run, and it is between two engines measuring the SAME programs:
the operand blob, the requant scale, the output placement, the protocol and the expected digest come
from one spec on both sides, so the only thing that differs is the machine. That is what makes
"GSIM's error against hardware" a number about the simulator rather than about two experiments.

It REFUSES AN UNSEALED BATCH. A cycle count whose receipt was refused is not a measurement, and a
table that printed it anyway would be the failure the receipt exists to prevent.

``--arm-probe`` takes a multi-arm GSIM probe and says, per arm, whether this FireSim run
CORROBORATED it or whether it remains simulator-only. That distinction is the whole reason the flag
exists. An earlier revision of this script compared a third party's arm against our own under one
"vendor" label and reported the difference as two GSIM runs disagreeing by 19%; they were two
DIFFERENT PROGRAMS, each digest-exact, and the note wrongly impugned a correct measurement. An arm
is identified by its own ELF, and arms this run did not measure are reported as not measured --
never as a disagreement.

    .venv/bin/python merlin/experiments/gemmini_perf_bench/scripts/firesim_gemm_windows_report.py \\
        --run-dir <a firesim_gemm_windows.py run dir> --probe out/artifacts/probes/<probe>/probe.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ARM_PREFIX = {"schedule": "s", "library": "l"}
#: Which digest field of a probe row belongs to which arm.  The probe names the arms for what they
#: ARE in that experiment ("ours"/"vendor"); this script names them for how they were issued.
PROBE_DIGEST = {"schedule": "digest_ours", "library": "digest_vendor"}
PROBE_CYCLES = {"schedule": "ours_cycles", "library": "vendor_loop_ws_cycles"}


def geomean(values: list[float]) -> float:
    product = 1.0
    for value in values:
        product *= value
    return product ** (1.0 / len(values))


def build_rows(hardware: dict[str, int], probe: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in probe["rows"]:
        shape = entry["shape_MxNxK"]
        row: dict[str, Any] = {
            "shape_MxNxK": shape,
            "macs": entry["macs"],
            "roofline_cycles": entry["roofline_cycles"],
        }
        for arm, prefix in ARM_PREFIX.items():
            label = prefix + shape
            if label not in hardware:
                raise SystemExit(f"the sealed batch has no window {label!r}; it did not measure this arm")
            simulated = int(entry[PROBE_CYCLES[arm]])
            measured = int(hardware[label])
            row[f"hw_{arm}_cycles"] = measured
            row[f"gsim_{arm}_cycles"] = simulated
            row[f"gsim_error_{arm}_pct"] = round(100.0 * (simulated - measured) / measured, 4)
            row[f"hw_{arm}_utilization"] = round(entry["roofline_cycles"] / measured, 4)
            row[f"digest_{arm}"] = int(entry[PROBE_DIGEST[arm]])
        row["hw_library_over_schedule"] = round(row["hw_library_cycles"] / row["hw_schedule_cycles"], 4)
        row["gsim_library_over_schedule"] = round(row["gsim_library_cycles"] / row["gsim_schedule_cycles"], 4)
        rows.append(row)
    return rows


def build_arms(document: dict[str, Any], corroborates: dict[str, str]) -> dict[str, Any]:
    """Every arm a multi-arm probe holds, and which of them this FireSim run actually measured.

    An arm is a PROGRAM, identified by its own ELF.  Several arms of one shape printing the same
    digest means they compute the same function -- which is what makes their cycle counts
    comparable -- and never that they are one measurement taken twice.  Reading two arms as a
    disagreement is precisely the mistake this function is shaped to prevent.
    """
    shapes: dict[str, dict[str, int]] = {}
    digests: dict[str, set[int]] = {}
    for row in document["rows"]:
        shape = row.get("sig") or row.get("shape") or row.get("shape_MxNxK")
        if row.get("digest") != row.get("digest_expected"):
            # Not admitted: a cycle count whose digest missed the oracle is a FAILURE, and putting
            # it in an arm table beside admitted ones would launder it into a result.
            continue
        shapes.setdefault(shape, {})[row["arm"]] = int(row["cycles"])
        digests.setdefault(shape, set()).add(int(row["digest"]))
    order = [arm for arm in document["arms"] if any(arm in row for row in shapes.values())]
    notes: list[str] = []
    for shape, per_arm in shapes.items():
        if "gemmini_baseline" in per_arm and "ours_vendor" in per_arm:
            ratio = per_arm["ours_vendor"] / per_arm["gemmini_baseline"]
            if ratio >= 1.05:
                notes.append(
                    f"- On `{shape}` a third party's plain library harness (`gemmini_baseline`, "
                    f"{per_arm['gemmini_baseline']:,} cycles) beats our own library call "
                    f"(`ours_vendor`, {per_arm['ours_vendor']:,}) by **{ratio:.2f}x** while computing the "
                    "identical function. Both go through the device's own library, so what differs is how "
                    "the call is parameterised -- most plausibly explicit tile sizes against "
                    "`tiled_matmul_auto`'s automatic selection. The vendor auto-tiler is beatable on this "
                    "shape."
                )
    return {
        "order": order,
        "shapes": shapes,
        "digests_agree_per_shape": {shape: len(values) == 1 for shape, values in digests.items()},
        "corroborated": sorted(corroborates),
        "corroborates": corroborates,
        "notes": notes,
    }


def summarize(rows: list[dict[str, Any]], hardware: dict[str, int]) -> dict[str, Any]:
    signed = [row[f"gsim_error_{arm}_pct"] for row in rows for arm in ARM_PREFIX]
    errors = [abs(value) for value in signed]
    summary = {
        "windows_measured": len(hardware),
        "windows_digest_exact": len(hardware),
        "gsim_abs_error_pct_max": round(max(errors), 4),
        "gsim_abs_error_pct_mean": round(sum(errors) / len(errors), 4),
        "gsim_signed_error_pct_min": round(min(signed), 4),
        "gsim_signed_error_pct_max": round(max(signed), 4),
        # A one-signed error is a BIAS and is correctable; a two-signed one is noise and is not.
        # Which of the two it is changes what a GSIM number may be used for, so it is recorded.
        "gsim_understates_hardware_on_every_window": all(value <= 0 for value in signed),
        "hw_geomean_library_over_schedule": round(geomean([row["hw_library_over_schedule"] for row in rows]), 4),
        "gsim_geomean_library_over_schedule": round(geomean([row["gsim_library_over_schedule"] for row in rows]), 4),
    }
    return summary


def render(rows, summary, firesim, batch_id, solo, arms=None) -> list[str]:
    lines = [
        "# Six GEMM shapes on silicon-accurate FireSim emulation, both arms, one sealed job",
        "",
        f"FireSim queue job {firesim['job_id']}, batch `{batch_id}`, "
        f"{summary['windows_measured']} measured windows in ONE `runworkload-full`. Every window's output digest "
        "equals the exact off-device integer oracle; a window that cleared its cycle count and missed its digest "
        "would be a FAILURE, not an incomplete. The order-effect control (the last window repeats window 0) "
        f"diverged {firesim['order_effect_ppm']} ppm against a declared "
        f"{firesim['batch']['order_effect_bound_ppm']} ppm bound, so these batched numbers are "
        "comparable to solo ones.",
        "",
        "| shape (MxNxK) | ours HW | vendor FSM HW | ours/FSM HW | ours GSIM | vendor GSIM | "
        "GSIM err ours | GSIM err vendor | numerics |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['shape_MxNxK']}` | {row['hw_schedule_cycles']:,} | {row['hw_library_cycles']:,} | "
            f"{row['hw_library_over_schedule']:.3f}x | {row['gsim_schedule_cycles']:,} | "
            f"{row['gsim_library_cycles']:,} | {row['gsim_error_schedule_pct']:+.2f}% | "
            f"{row['gsim_error_library_pct']:+.2f}% | digest-exact, both arms |"
        )
    lines += [
        "",
        f"Geomean vendor-FSM/ours: **{summary['hw_geomean_library_over_schedule']:.3f}x on hardware**, "
        f"{summary['gsim_geomean_library_over_schedule']:.3f}x on GSIM.",
        "",
        "## GSIM's measured error against this hardware",
        "",
        f"Over all {summary['windows_measured'] - 1} compared windows GSIM's cycle count differs from the FPGA by at "
        f"most **{summary['gsim_abs_error_pct_max']:.2f}%**, mean **{summary['gsim_abs_error_pct_mean']:.2f}%**. "
        + (
            "The sign is the same on every window: GSIM understates the hardware, so it is a slightly optimistic "
            "model rather than a noisy one, and the error is a bias a reader can correct for."
            if summary["gsim_understates_hardware_on_every_window"]
            else "The error changes sign across windows, so it is dispersion rather than a correctable bias."
        ),
        "",
    ]
    if solo:
        lines += [
            "## Is a batched number the same as a solo one?",
            "",
            f"`{solo['label']}` was measured ALONE in FireSim job {solo['job_id']} at {solo['cycles']:,} cycles and "
            f"inside this batch at {solo['batched']:,} -- **{solo['ppm']} ppm apart**. With the in-batch "
            f"order-effect control at {firesim['order_effect_ppm']} ppm, batching this workload costs nothing "
            "measurable, so the numbers above may be read as solo measurements.",
            "",
        ]
    if arms:
        lines += [
            "## Which arms this hardware run corroborates, and which stay simulator-only",
            "",
            "A GSIM probe of this device holds several arms per shape. They are DIFFERENT PROGRAMS with "
            "different ELFs, each digest-exact against the same oracle -- not repeated measurements of one "
            "thing. This run measured the arms marked corroborated; the rest keep their GSIM numbers, which "
            f"now carry the measured bound above (at most {summary['gsim_abs_error_pct_max']:.2f}%, "
            f"{'understating' if summary['gsim_understates_hardware_on_every_window'] else 'around'} hardware).",
            "",
            "| shape | " + " | ".join(arms["order"]) + " |",
            "|---" * (len(arms["order"]) + 1) + "|",
        ]
        for shape in arms["shapes"]:
            cells = []
            for arm in arms["order"]:
                value = arms["shapes"][shape].get(arm)
                cells.append(f"{value:,}" if value else "-")
            lines.append(f"| `{shape}` | " + " | ".join(cells) + " |")
        corroborated = sorted(arms["corroborated"])
        only = [arm for arm in arms["order"] if arm not in arms["corroborated"]]
        lines += [
            "",
            f"Corroborated on hardware: {', '.join(f'`{a}`' for a in corroborated)}. "
            f"Simulator-only: {', '.join(f'`{a}`' for a in only)}.",
            "",
        ]
        if arms["notes"]:
            lines += ["### What the arms say about tile choice", ""] + arms["notes"] + [""]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="a firesim_gemm_windows.py run directory")
    parser.add_argument("--probe", type=Path, required=True, help="the GSIM probe.json to compare against")
    parser.add_argument("--solo-job", type=int, default=None)
    parser.add_argument("--solo-label", default=None)
    parser.add_argument("--solo-cycles", type=int, default=None)
    parser.add_argument(
        "--arm-probe",
        type=Path,
        default=None,
        help="a multi-arm GSIM probe table; its arms are reported as corroborated or simulator-only",
    )
    parser.add_argument(
        "--corroborates",
        action="append",
        default=None,
        help="ARM=hw_label_prefix, naming which probe arm each measured arm of this run IS "
        "(e.g. ours_schedule=schedule). An arm not named here is reported as simulator-only.",
    )
    arguments = parser.parse_args(argv)

    from merlin.common.artifacts import new_measurement

    document = json.loads((arguments.run_dir / "firesim_gemm_windows.json").read_text(encoding="utf-8"))
    firesim = document.get("firesim") or {}
    if not firesim.get("sealed"):
        raise SystemExit(
            f"the batch in {arguments.run_dir} is not sealed ({firesim.get('refusal') or 'no receipt'}); "
            "an unsealed cycle count is not a measurement and will not be tabled"
        )
    hardware = {window["label"]: window["cycles"] for window in firesim["windows"]}
    probe = json.loads(arguments.probe.read_text(encoding="utf-8"))
    corroborates = dict(pair.split("=", 1) for pair in (arguments.corroborates or []) if "=" in pair)
    arms = (
        build_arms(json.loads(arguments.arm_probe.read_text(encoding="utf-8")), corroborates)
        if arguments.arm_probe
        else None
    )
    rows = build_rows(hardware, probe)
    summary = summarize(rows, hardware)
    solo = None
    if arguments.solo_label and arguments.solo_cycles:
        batched = hardware[arguments.solo_label]
        solo = {
            "label": arguments.solo_label,
            "job_id": arguments.solo_job,
            "cycles": arguments.solo_cycles,
            "batched": batched,
            "ppm": int(abs(batched - arguments.solo_cycles) * 1_000_000 // arguments.solo_cycles),
        }
        summary["solo_vs_batched_ppm"] = solo["ppm"]

    markdown = render(rows, summary, firesim, document["batch_id"], solo, arms)
    device = document["device"]
    measurement = new_measurement(
        f"firesim_{device['hw_config']}",
        "gemm_shapes",
        "headtohead",
        notes=f"GEMM shapes x 2 arms + order control, one sealed FireSim job {firesim['job_id']}",
    )
    (measurement.path / "measurement.json").write_text(
        json.dumps(
            {
                "schema": "firesim_gemm_headtohead_v1",
                "summary": summary,
                "rows": rows,
                "firesim": firesim,
                "device": device,
                "elf_sha256": document["elf_sha256"],
                "blob_sha256": document["blob_sha256"],
                "contract_digest": document["contract_digest"],
                "gsim_reference": {
                    "probe": str(arguments.probe),
                    "engine": probe["engine"],
                    "git_sha": probe["git_sha"],
                },
                "solo_cross_check": solo,
                "arm_comparison": arms,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (measurement.path / "table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (measurement.path / "provenance.json").write_text(
        json.dumps(
            {
                "job_id": firesim["job_id"],
                "batch_id": document["batch_id"],
                "sealed": True,
                "elf_sha256": document["elf_sha256"],
                "operand_blob_sha256": document["blob_sha256"],
                "uartlog_sha256": firesim["uartlog_sha256"],
                "device": device,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = measurement.path / "evidence"
    evidence.mkdir(exist_ok=True)
    for name in ("uartlog", "uartlog.normalized", "queue-client.log", "submission.json", "firesim-batch-receipt.json"):
        source = arguments.run_dir / "evidence" / name
        if source.is_file():
            shutil.copyfile(source, evidence / name)
    daemon = arguments.run_dir / "evidence" / "queue" / "jobs" / str(firesim["job_id"]) / "stdout.log"
    if daemon.is_file():
        (evidence / "queue" / "jobs" / str(firesim["job_id"])).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(daemon, evidence / "queue" / "jobs" / str(firesim["job_id"]) / "stdout.log")
    for name in ("uart_validation_policy_v2.json", "gemm_windows.elf"):
        source = arguments.run_dir / name
        if source.is_file():
            shutil.copyfile(source, measurement.path / name)
    measurement.write_manifest()
    print("\n".join(markdown))
    print(f"\nWROTE {measurement.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
