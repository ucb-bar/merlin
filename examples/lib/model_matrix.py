#!/usr/bin/env python3
"""Model regression matrix: every model x hart count x leg, built for one board and gated on spike.

    model_matrix.py --board B --models m1,m2 [--harts 1,2]
                    [--matrix-unit U --matrix-config C --matrix-support-target T [--matrix-kernel K]]
                    [--legs rvv,matrix] [--jobs 2] [--rerun] [--timeout S]

A model is a capture bundle, ``recaptures/<model>_int8_full`` (``chip_flow.py capture`` writes one). For
each model and hart count the matrix builds, with ``merlin-compile``:

  rvv     the plain RVV image, run on spike at the board's VLEN and CPU count, graded against the
          bundle's W8A8 golden;
  matrix  (with the matrix flags) the image that routes contractions to the matrix unit. Its stand-in
          twin (same lowering, scalar tile body) is graded on spike; when the support target declares a
          spike model of its unit, the SHIPPED image runs on that model too, is graded, and must match
          the twin bit for bit (``chip_flow.matrix_on_spike``). Its ``routing_report.json`` gives the
          routed contractions, the unit's MAC share and why the rest was not routed.

Then, per model and leg, every hart count's output must be bit-identical to the smallest one's: a
multicore split computes the same thing, so any difference is a lost or overlapping work split.

Writes ``out/artifacts/probes/model_matrix/<board>/``: one ``<model>/h<N>_<leg>.json`` per cell (kept, so
a rerun only builds the cells that have none, unless ``--rerun``), ``matrix.json`` and ``MATRIX.md``; with
``--matrix-kernel K``, ``<board>+kernel-<K>/`` instead, so cells of different kernels never mix. The
process exits non-zero unless every cell passed. Nothing here names a model or a chip: the board, the unit
and the models are arguments, and a cell's verdict comes only from its own gates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chip_flow  # noqa: E402  (the shared compile + matrix_on_spike steps)

LEGS = ("rvv", "matrix")


def out_root(board: str, kernel: str | None = None) -> Path:
    """The board's table; a selected matrix kernel gets its own (``<board>+kernel-<name>``), so cells built
    with one kernel are never reused, or tabulated, as cells of another."""
    from merlin.common.paths import artifacts_dir

    tag = "" if not kernel else "+kernel-" + "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(kernel).name)
    return artifacts_dir() / "probes" / "model_matrix" / f"{board}{tag}"


def _cell_args(a, harts: int) -> SimpleNamespace:
    return SimpleNamespace(
        board=a.board,
        harts=harts,
        matrix_unit=a.matrix_unit,
        matrix_config=a.matrix_config,
        matrix_support_target=a.matrix_support_target,
        matrix_kernel=a.matrix_kernel,
        timeout=a.timeout,
    )


def _verdict_terms(verify: dict) -> dict:
    """The W8A8 tier's numbers, which decide the gate on an int8 image (the fp32 tier is the fallback)."""
    keep = ("gate_ok", "w8a8_cos", "w8a8_rel", "w8a8_max_rel", "w8a8_argmax", "fp32_cos", "fp32_argmax")
    return {k: verify.get(k) for k in keep if k in verify}


def run_cell(a, model: str, harts: int, leg: str) -> dict:
    """Build and gate one image (two simulations for a matrix cell: twin, then shipped on the unit model)."""
    args = _cell_args(a, harts)
    started = time.monotonic()
    res = chip_flow._compile(model, args, matrix=(leg == "matrix"))
    cell: dict = {"model": model, "harts": harts, "leg": leg, "board": a.board, "compile": res}
    if leg == "matrix" and res.get("binary"):
        cell["on_unit_model"] = chip_flow.matrix_on_spike(model, args, res)
    cell["wall_s"] = round(time.monotonic() - started, 1)
    return summarize(cell)


def summarize(cell: dict) -> dict:
    """The cell's own verdict and the numbers the table shows, derived from what its gates returned."""
    res = cell["compile"]
    problems: list[str] = []
    row: dict = {"status_compile": res.get("status")}
    if res.get("status") == "not_routed":
        # No image to gate: the selector declined every contraction, and the reason is the result.
        m = res.get("matrix") or {}
        report = _load(m.get("routing_report"))
        cell["routing"] = {
            "routed_contractions": 0,
            "opu_mac_share": m.get("opu_mac_share"),
            "not_routed_by_reason": m.get("not_routed_by_reason") or {},
            "top_not_routed": Counter(str(r.get("why")) for r in report.get("not_routed") or []).most_common(3),
            "routing_report": m.get("routing_report"),
        }
        cell.update(gate={}, cycles=None, identity=None, problems=[f"nothing routed: {res.get('reason')}"])
        cell["status"] = "NO-ROUTE"
        return cell
    if res.get("status") not in ("verified", "run_mismatch"):
        problems.append(f"{res.get('status')}: {str(res.get('error') or res.get('reason') or '')[:300]}")
    gate = _verdict_terms(res.get("verify") or {})
    row["gate"] = gate
    if not gate.get("gate_ok"):
        problems.append("gate failed" + (f" ({_why(gate)})" if gate else ""))
    if cell["leg"] == "rvv":
        row["cycles"] = res.get("cycles")
        row["identity"] = res.get("output_identity")
    else:
        m = res.get("matrix") or {}
        report = _load(m.get("routing_report"))
        row["routing"] = {
            "routed_contractions": m.get("routed_contractions"),
            "opu_mac_share": m.get("opu_mac_share"),
            "not_routed_by_reason": m.get("not_routed_by_reason") or {},
            # The reasons above are categories; the most frequent concrete ones say which shapes they hit.
            "top_not_routed": Counter(str(r.get("why")) for r in report.get("not_routed") or []).most_common(3),
            "not_routed_contractions": report.get("not_routed_contractions"),
            "macs_routed": report.get("macs_routed"),
            "macs_not_routed": report.get("macs_not_routed"),
            "unit_instruction_counts": m.get("unit_instruction_counts"),
            "routing_report": m.get("routing_report"),
        }
        row["twin_cycles"] = m.get("twin_spike_cycles")
        row["identity"] = m.get("twin_output_identity")
        on = cell.get("on_unit_model") or {}
        row["shipped"] = {
            "status": on.get("status"),
            "reason": on.get("reason") or on.get("error"),
            "gate": _verdict_terms(on.get("verify") or {}),
            "bit_exact_vs_twin": on.get("bit_exact_vs_twin"),
            "spike_cycles": on.get("spike_cycles"),
            "identity": on.get("output_identity"),
        }
        row["cycles"] = on.get("spike_cycles")
        if m.get("provenance"):
            row["unit_provenance"] = m["provenance"]
        if on.get("status") != "verified":
            why = on.get("reason") or on.get("error") or ""
            if on.get("status") == "run_mismatch":
                why = f"bit_exact_vs_twin={on.get('bit_exact_vs_twin')}; {_why(row['shipped']['gate'])}"
            problems.append(f"shipped image on the unit model: {on.get('status')} {str(why)[:300]}".rstrip())
    # The image's memory budget: why it fits the board (rvv: the image; matrix: the shipped image).
    cell["memory"] = res.get("memory") or (res.get("matrix") or {}).get("ship_memory")
    cell.update(row)
    cell["problems"] = problems
    cell["status"] = "PASS" if not problems else "FAIL"
    return cell


def _memory_cell(memory: dict | None) -> str:
    """``sum > arena + heap / region`` in MiB: the buffers' sum, what the plan seats them in, the region."""
    if not memory:
        return "n/a"

    def mib(v) -> str:
        return "?" if v is None else f"{int(v) / 2**20:.0f}"

    heap = (memory.get("heap_residue") or {}).get("bytes")
    return (
        f"{mib(memory.get('allocation_sum_bytes'))} > {mib(memory.get('arena_bytes'))}+{mib(heap)} "
        f"/ {mib(memory.get('region_bytes'))}"
    )


def _why(gate: dict) -> str:
    parts = []
    for key in ("w8a8_cos", "w8a8_rel", "w8a8_max_rel"):
        if gate.get(key) is not None:
            parts.append(f"{key.removeprefix('w8a8_')} {gate[key]:.6g}")
    if gate.get("w8a8_argmax") is False:
        parts.append("top-1 differs")
    return ", ".join(parts)


def _load(path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return {}


def harts_identity(cells: list[dict]) -> None:
    """Every hart count's output against the smallest one's, per model and leg (both images for matrix)."""
    from merlin.runtime.backends import output_identity

    by_key: dict = {}
    for c in cells:
        by_key.setdefault((c["model"], c["leg"]), []).append(c)
    for group in by_key.values():
        group.sort(key=lambda c: c["harts"])
        base = group[0]
        for c in group[1:]:
            if "NO-ROUTE" in (base["status"], c["status"]):
                continue  # no image was built, so there is no output to compare
            same = output_identity.same_output(base.get("identity"), c.get("identity"))
            if c["leg"] == "matrix":
                shipped = output_identity.same_output(
                    (base.get("shipped") or {}).get("identity"), (c.get("shipped") or {}).get("identity")
                )
                same = None if None in (same, shipped) else (same and shipped)
            c["identical_to"] = {"harts": base["harts"], "bit_identical": same}
            if same is not True:
                c["problems"].append(
                    f"output vs {base['harts']}-hart: "
                    + ("NOT bit-identical" if same is False else "identity unknown (an image printed no digest)")
                )
                c["status"] = "FAIL"


def _fmt(x, spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


def table(cells: list[dict], board: str) -> str:
    lines = [
        f"# Model matrix on {board}",
        "",
        "Every cell is gated on spike against the bundle's W8A8 golden (`cos`, `rel` = max abs error / max "
        "|ref|, `max_rel` = per-element). `cycles` are spike's (functional: instructions retired, not "
        "time); for `matrix` they are the shipped image's on the unit's functional model, which does not "
        "model the unit's timing. `share` is the matrix unit's share of the model's contraction MACs. "
        "`memory` is MiB: the sum of the model's buffers > the static arena they are planned into + the heap "
        "residue / the region the image is linked for.",
        "",
        "| model | harts | leg | status | cos | rel | max_rel | cycles | = h1 | share | routed | memory "
        "| not routed (why) |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for c in sorted(cells, key=lambda c: (c["model"], c["harts"], LEGS.index(c["leg"]))):
        g = c.get("gate") or {}
        ident = (c.get("identical_to") or {}).get("bit_identical")
        r = c.get("routing") or {}
        why = "; ".join(
            f"{n}x {k}" for k, n in sorted((r.get("not_routed_by_reason") or {}).items(), key=lambda kv: -kv[1])
        )
        if r.get("top_not_routed"):
            why += " (most often: " + "; ".join(f"{n}x {w}" for w, n in r["top_not_routed"][:2]) + ")"
        lines.append(
            f"| {c['model']} | {c['harts']} | {c['leg']} | {c['status']} | {_fmt(g.get('w8a8_cos'), '.7f')} "
            f"| {_fmt(g.get('w8a8_rel'), '.4g')} | {_fmt(g.get('w8a8_max_rel'), '.4g')} | {_fmt(c.get('cycles'), ',')} "
            f"| {'-' if 'identical_to' not in c else ('yes' if ident else ('NO' if ident is False else '?'))} "
            f"| {_fmt(r.get('opu_mac_share'), '.4f') if c['leg'] == 'matrix' else ''} "
            f"| {r.get('routed_contractions', '') if c['leg'] == 'matrix' else ''} | {_memory_cell(c.get('memory'))} "
            f"| {why} |"
        )
    lines += ["", "## Problems", ""]
    any_problem = False
    for c in cells:
        for p in c["problems"]:
            any_problem = True
            lines.append(f"- {c['model']} h{c['harts']} {c['leg']}: {p}")
    if not any_problem:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--board", required=True)
    ap.add_argument("--models", required=True, help="comma-separated bundle names (recaptures/<name>_int8_full)")
    ap.add_argument("--harts", default="1,2", help="comma-separated hart counts")
    ap.add_argument("--legs", default=None, help=f"subset of {','.join(LEGS)} (default: rvv, plus matrix when given)")
    ap.add_argument("--matrix-unit")
    ap.add_argument("--matrix-config")
    ap.add_argument("--matrix-support-target")
    ap.add_argument("--matrix-kernel", help="the matrix kernel (registered name or C file); default: the package's")
    ap.add_argument("--jobs", type=int, default=2, help="cells built and simulated at once (each is a whole model)")
    ap.add_argument("--rerun", action="store_true", help="rebuild cells that already have a result")
    ap.add_argument("--timeout", type=int, default=4 * 3600, help="per-simulation limit, s")
    a = ap.parse_args(argv)
    matrix = (a.matrix_unit, a.matrix_config, a.matrix_support_target)
    if any(matrix) and not all(matrix):
        ap.error("--matrix-unit, --matrix-config and --matrix-support-target are given together")
    legs = a.legs.split(",") if a.legs else ["rvv"] + (["matrix"] if all(matrix) else [])
    if "matrix" in legs and not all(matrix):
        ap.error("the matrix leg needs --matrix-unit, --matrix-config and --matrix-support-target")
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    harts = sorted({int(h) for h in a.harts.split(",")})
    for m in models:
        if not (chip_flow.bundle_dir(m) / "golden_w8a8.npy").is_file():
            raise SystemExit(
                f"{m}: no {chip_flow.bundle_dir(m)}/golden_w8a8.npy; capture it first (chip_flow.py capture)"
            )

    root = out_root(a.board, a.matrix_kernel)
    todo, cells = [], []
    for m in models:
        for h in harts:
            for leg in legs:
                f = root / m / f"h{h}_{leg}.json"
                if f.is_file() and not a.rerun:
                    cells.append(summarize(json.loads(f.read_text(encoding="utf-8"))))
                else:
                    todo.append((m, h, leg, f))

    def _one(item):
        m, h, leg, f = item
        print(f"[matrix] {m} h{h} {leg}: building and simulating", flush=True)
        try:
            cell = run_cell(a, m, h, leg)
        except Exception as exc:  # noqa: BLE001 -- a cell that crashes is a FAIL row, not a lost matrix
            cell = summarize(
                {"model": m, "harts": h, "leg": leg, "board": a.board,
                 "compile": {"status": "error", "error": f"{type(exc).__name__}: {exc}"}}
            )  # fmt: skip
        f.parent.mkdir(parents=True, exist_ok=True)
        # The routing report lives in merlin-compile's scratch directory; keep it beside the cell.
        report = Path((cell.get("routing") or {}).get("routing_report") or "")
        if report.is_file():
            kept = f.with_name(f"h{h}_{leg}.routing_report.json")
            shutil.copy2(report, kept)
            cell["routing"]["routing_report"] = cell["compile"]["matrix"]["routing_report"] = str(kept)
        f.write_text(json.dumps(cell, indent=2, default=str), encoding="utf-8")
        print(f"[matrix] {m} h{h} {leg}: {cell['status']} {'; '.join(cell['problems'])[:300]}", flush=True)
        return cell

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        cells += list(pool.map(_one, todo))
    mine = {(c["model"], c["harts"], c["leg"]) for c in cells}
    # The board's table covers every cell on disk, so models matrixed in separate runs share one table.
    everything = cells + [
        summarize(json.loads(f.read_text(encoding="utf-8")))
        for f in sorted(root.glob("*/h*_*.json"))
        if not f.name.endswith(".routing_report.json") and _cell_key(f) not in mine
    ]
    harts_identity(everything)
    root.mkdir(parents=True, exist_ok=True)
    (root / "matrix.json").write_text(json.dumps(everything, indent=2, default=str), encoding="utf-8")
    text = table(everything, a.board)
    (root / "MATRIX.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"written: {root}")
    return 0 if all(c["status"] == "PASS" for c in cells) else 1


def _cell_key(f: Path) -> tuple:
    harts, _, leg = f.stem.partition("_")
    return (f.parent.name, int(harts.removeprefix("h")), leg)


if __name__ == "__main__":
    raise SystemExit(main())
