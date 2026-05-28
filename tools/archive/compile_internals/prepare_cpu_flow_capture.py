#!/usr/bin/env python3
"""Prepare a strict CPU data-flow capture run for dispatch inputs.

The output is a small XPU-RT-compatible schedule + manifest pair. It covers
the CPU ancestor subgraph required to produce the real input arenas for a set
of HTA islands, and points each dispatch at a locally compiled CPU wrapper
VMFB. `XPU-RT/scripts/run_on_board_flow.py` can then run the plan on board
with `--strict-binding-sources` and `--capture-dispatches`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from collections.abc import Iterable
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _dispatch_id(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def _source_dispatches(hta_manifest: dict[str, Any]) -> list[str]:
    out = []
    for row in hta_manifest.get("dispatches", {}).values():
        cell = row.get("qnn_hta")
        if isinstance(cell, dict) and cell.get("source_dispatch"):
            out.append(str(cell["source_dispatch"]))
    return sorted(set(out), key=_dispatch_id)


def _sort_dispatch_names(names: Iterable[str]) -> list[str]:
    return sorted(set(names), key=lambda name: (_dispatch_id(name), name))


def _ancestor_closure(graph: dict[str, Any], roots: Iterable[str]) -> list[str]:
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        for dep in graph.get(name, {}).get("dependencies", []):
            visit(str(dep))
        seen.add(name)

    for root in roots:
        visit(root)
    return sorted(seen, key=_dispatch_id)


def _matrix_key(dispatch_name: str) -> str:
    return f"main_graph$async_{dispatch_name}"


def _canonical_dispatch(name: str, graph: dict[str, Any]) -> str:
    row = graph.get(name, {})
    if isinstance(row, dict) and row.get("canonical_dispatch"):
        return str(row["canonical_dispatch"])
    return name.split("_call_", 1)[0]


def _load_graph(matrix: dict[str, Any], graph_json: pathlib.Path | None) -> dict[str, Any]:
    if graph_json is None:
        return matrix.get("dispatch_graph", {})
    payload = json.loads(graph_json.read_text())
    return payload.get("dispatch_graph", payload)


def _available_cpu_dispatches(matrix: dict[str, Any]) -> set[str]:
    available: set[str] = set()
    for key, row in matrix.get("dispatches", {}).items():
        if not key.startswith("main_graph$async_dispatch_"):
            continue
        cell = row.get("cpu")
        if not isinstance(cell, dict):
            continue
        if cell.get("feasible") and cell.get("mlir"):
            available.add(key.removeprefix("main_graph$async_"))
    return available


def _split_capturable_roots(
    *,
    graph: dict[str, Any],
    roots: list[str],
    available: set[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    included: list[str] = []
    excluded: list[str] = []
    reports: list[dict[str, Any]] = []
    for root in roots:
        closure = _ancestor_closure(graph, [root])
        missing_graph = [name for name in closure if name not in graph]
        missing_wrapper = [name for name in closure if name not in available]
        report = {
            "root": root,
            "closure_dispatches": len(closure),
            "missing_graph": missing_graph,
            "missing_cpu_wrapper": missing_wrapper,
        }
        if missing_graph or missing_wrapper:
            excluded.append(root)
            report["status"] = "excluded-undumpable-ancestor"
        else:
            included.append(root)
            report["status"] = "included"
        reports.append(report)
    return included, excluded, reports


def _compile_cpu_wrapper(
    *,
    mlir: pathlib.Path,
    out_dir: pathlib.Path,
    build_dir: str,
    target: str,
) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    vmfbs = list(out_dir.glob("*.vmfb"))
    if vmfbs:
        return vmfbs[0]
    cmd = [
        str(REPO_ROOT / "merlin"),
        "compile",
        str(mlir),
        "--target",
        target,
        "--build-dir",
        build_dir,
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    vmfbs = list(out_dir.glob("*.vmfb"))
    if not vmfbs:
        raise RuntimeError(f"compile produced no VMFB in {out_dir}")
    return vmfbs[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-matrix", type=pathlib.Path, required=True)
    parser.add_argument("--hta-manifest", type=pathlib.Path)
    parser.add_argument(
        "--dispatch-graph-json",
        type=pathlib.Path,
        help="Optional call-site graph from " "tools/extract_flow_dispatch_call_graph.py.",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--build-dir", default="host-merlin-release-qrb")
    parser.add_argument("--target", default="qrb5165_aarch64")
    parser.add_argument(
        "--dispatch", action="append", default=[], help="Capture roots; defaults to all HTA island source dispatches."
    )
    parser.add_argument(
        "--all-graph-dispatches", action="store_true", help="Capture every dispatch node in the selected graph."
    )
    parser.add_argument(
        "--binding-sources-json",
        type=pathlib.Path,
        help="Optional explicit binding-source map. Defaults "
        "to the map embedded in --dispatch-graph-json, "
        "when present.",
    )
    parser.add_argument(
        "--require-all-roots",
        action="store_true",
        help="Fail if any requested HTA root has an ancestor "
        "without a real CPU wrapper. By default those "
        "roots are excluded and reported.",
    )
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args(argv)

    matrix = json.loads(args.cpu_matrix.read_text())
    hta_manifest = json.loads(args.hta_manifest.read_text()) if args.hta_manifest else {}
    graph_payload = json.loads(args.dispatch_graph_json.read_text()) if args.dispatch_graph_json else {}
    graph = _load_graph(matrix, args.dispatch_graph_json)
    if args.all_graph_dispatches:
        requested_roots = _sort_dispatch_names(graph)
    else:
        default_roots = _source_dispatches(hta_manifest) if hta_manifest else []
        requested_roots = _sort_dispatch_names(args.dispatch or default_roots)
    if not requested_roots:
        raise RuntimeError("no capture roots requested")
    available = _available_cpu_dispatches(matrix)
    roots, excluded_roots, root_reports = _split_capturable_roots(
        graph=graph,
        roots=requested_roots,
        available={name for name in graph if _canonical_dispatch(name, graph) in available},
    )
    if excluded_roots and args.require_all_roots:
        missing = ", ".join(excluded_roots)
        raise RuntimeError(f"requested roots have undumpable ancestors: {missing}")
    if not roots:
        raise RuntimeError("no requested HTA roots have complete CPU wrapper coverage")
    required = _ancestor_closure(graph, roots)

    vmfb_root = args.out_dir / "cpu_vmfbs"
    schedule = {
        "machines": ["CPU"],
        "dispatches": {
            name: {
                "hardware_target": "CPU",
                "start_us": float(i),
            }
            for i, name in enumerate(required)
        },
    }
    manifest: dict[str, Any] = {
        "dispatches": {},
        "dispatch_graph": {name: {"dependencies": graph.get(name, {}).get("dependencies", [])} for name in required},
    }
    binding_sources: dict[str, list[str]] = {}
    if args.binding_sources_json:
        binding_sources = json.loads(args.binding_sources_json.read_text())
    elif isinstance(graph_payload.get("binding_sources"), dict):
        binding_sources = graph_payload["binding_sources"]

    compiled_by_canonical: dict[str, pathlib.Path] = {}
    for name in required:
        canonical = _canonical_dispatch(name, graph)
        key = _matrix_key(canonical)
        cell = matrix["dispatches"][key]["cpu"]
        mlir = pathlib.Path(cell["mlir"])
        out_dir = vmfb_root / canonical
        if args.skip_compile:
            vmfbs = list(out_dir.glob("*.vmfb"))
            if not vmfbs:
                raise RuntimeError(f"--skip-compile but no VMFB in {out_dir}")
            vmfb = vmfbs[0]
        elif canonical in compiled_by_canonical:
            vmfb = compiled_by_canonical[canonical]
        else:
            vmfb = _compile_cpu_wrapper(
                mlir=mlir,
                out_dir=out_dir,
                build_dir=args.build_dir,
                target=args.target,
            )
            compiled_by_canonical[canonical] = vmfb
        manifest["dispatches"][name] = {
            "cpu": {
                "feasible": True,
                "func": cell["func"],
                "binding_byte_sizes": cell["binding_byte_sizes"],
                "vmfb": str(vmfb),
                "canonical_dispatch": canonical,
            }
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "schedule.json").write_text(json.dumps(schedule, indent=2))
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out_dir / "capture_dispatches.txt").write_text(",".join(roots) + "\n")
    if binding_sources:
        selected_sources = {name: binding_sources[name] for name in required if name in binding_sources}
        (args.out_dir / "binding_sources.json").write_text(json.dumps(selected_sources, indent=2))
    (args.out_dir / "capture_root_coverage.json").write_text(
        json.dumps(
            {
                "requested_roots": requested_roots,
                "included_roots": roots,
                "excluded_roots": excluded_roots,
                "available_cpu_wrappers": len(available),
                "root_reports": root_reports,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "requested_roots": len(requested_roots),
                "included_roots": len(roots),
                "excluded_roots": len(excluded_roots),
                "required_dispatches": len(required),
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
