"""Synthetic records for the richer dashboard views, in the owners' own schemas.

* :func:`codex_stream` writes ``rounds/round_NN.codex_events.timestamped.jsonl`` in the provider's
  wrapper (``{"seq", "arrived_at", "event"}``) with the event vocabulary of
  :mod:`merlin_experiments.phase1.providers.codex_agent`;
* :func:`phase1_run` builds a Phase 1 run with grades, an agent stream, a self-check log, a job channel,
  cost/timing records, a real ``oot/`` snapshot history (through :mod:`merlin.common.oot_repo`) and a freeze;
* :func:`phase0_generation` builds a generation run: MANIFEST cohorts, capsules (one hidden), coverage.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

TARGET = "toy"
HOUR = 3600.0
T0 = 1_790_000_000.0


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def stamp(epoch: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(epoch))


def write(path: Path, document) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = document if isinstance(document, str) else json.dumps(document, indent=1)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- codex events
def command(item_id: str, cmd: str, *, exit_code: int | None, output: str = "") -> list[dict]:
    """A started/completed command pair (completed only when ``exit_code`` is not None)."""
    started = {
        "type": "item.started",
        "item": {
            "id": item_id,
            "type": "command_execution",
            "command": f"/bin/bash -lc '{cmd}'",
            "aggregated_output": "",
            "exit_code": None,
            "status": "in_progress",
        },
    }
    if exit_code is None:
        return [started]
    done = {
        "type": "item.completed",
        "item": {
            "id": item_id,
            "type": "command_execution",
            "command": f"/bin/bash -lc '{cmd}'",
            "aggregated_output": output,
            "exit_code": exit_code,
            "status": "completed" if exit_code == 0 else "failed",
        },
    }
    return [started, done]


def codex_stream(
    run_dir: Path,
    round_number: int,
    t0: float,
    events: list[dict],
    *,
    step: float = 30.0,
    partial_tail: str | None = None,
) -> Path:
    """Write ``events`` stamped ``step`` seconds apart; ``partial_tail`` appends an unfinished line."""
    path = run_dir / "rounds" / f"round_{round_number:02d}.codex_events.timestamped.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"seq": i + 1, "arrived_at": iso(t0 + i * step), "event": e}) for i, e in enumerate(events)]
    text = "\n".join(lines) + "\n" + (partial_tail or "")
    path.write_text(text, encoding="utf-8")
    return path


def agent_round(t0: float, *, running: bool = False) -> list[dict]:
    events = [
        {"type": "thread.started", "thread_id": "01a01161-dead-beef-0000-000000000001"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "r1", "type": "reasoning", "text": "Plan: lower linalg.matmul to the tile op first."},
        },
        *command("c1", "ls submission", exit_code=0, output="manifest.yaml\n"),
        {
            "type": "item.completed",
            "item": {
                "id": "f1",
                "type": "file_change",
                "status": "completed",
                "changes": [
                    {"path": "submission/lib/Lower.cpp", "kind": "update"},
                    {"path": "submission/manifest.yaml", "kind": "add"},
                ],
            },
        },
        *command("c2", "cmake --build submission/build", exit_code=2, output="error: undefined reference to tile_mm"),
        *command("c3", "cmake --build submission/build", exit_code=0),
        *command(
            "c4",
            "python3 agent_selfcheck.py --submission submission --sim spike --capsules cap_mm",
            exit_code=1,
            output="cap_mm: numeric mismatch",
        ),
        {
            "type": "item.updated",
            "item": {
                "id": "t1",
                "type": "todo_list",
                "items": [
                    {"text": "lower matmul", "completed": True},
                    {"text": "fix the epilogue scale", "completed": False},
                ],
            },
        },
        {
            "type": "item.completed",
            "item": {"id": "m1", "type": "agent_message", "text": "Working on the epilogue scale for cap_mm."},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 36767,
                "cached_input_tokens": 28160,
                "cache_write_input_tokens": 0,
                "output_tokens": 2030,
                "reasoning_output_tokens": 900,
            },
        },
        {"type": "turn.started"},
        *command(
            "c5",
            "python3 agent_selfcheck.py --submission submission --sim spike --capsules all",
            exit_code=None if running else 0,
        ),
    ]
    if not running:
        events.append(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 1000, "cached_input_tokens": 500, "output_tokens": 300},
            }
        )
    return events


# --------------------------------------------------------------------------- phase 1
LOWER_CPP = 'struct LowerPass {\n  StringRef getArgument() const final {\n    return "toy-lower";\n  }\n};\n'
SUBMISSIONS = [
    {
        "manifest.yaml": yaml.safe_dump(
            {
                "artifact_type": "mlir_oot_target_backend",
                "package_id": "toy_v0",
                "language": "cpp",
                "entrypoints": {"tool": "bin/toy-opt"},
                "commands": {"parse": {"argv": ["x"]}, "emit_command_buffer": {"argv": ["y"]}},
            }
        ),
        "include/Toy/ToyOps.td": 'def Toy_Dialect : Dialect {\n  let name = "toy";\n}\n'
        'def Toy_TileMatmulOp : Toy_Op<"tile_matmul", [Pure]> {\n}\n',
        "lib/Lower.cpp": LOWER_CPP,
    },
    {
        "manifest.yaml": yaml.safe_dump(
            {
                "artifact_type": "mlir_oot_target_backend",
                "package_id": "toy_v0",
                "language": "cpp",
                "entrypoints": {"tool": "bin/toy-opt"},
                "commands": {
                    "parse": {"argv": ["x"]},
                    "emit_command_buffer": {"argv": ["y"]},
                    "lower_interface_to_target": {"argv": ["z"]},
                },
                "optimization_surfaces": [{"id": "tile-size"}],
            }
        ),
        "include/Toy/ToyOps.td": 'def Toy_Dialect : Dialect {\n  let name = "toy";\n}\n'
        'def Toy_TileMatmulOp : Toy_Op<"tile_matmul", [Pure]> {\n}\n'
        'def Toy_MvinOp : Toy_Op<"mvin"> {\n}\n'
        'def ToyFuse : Pass<"toy-fuse-epilogue", "ModuleOp"> {\n}\n',
        "lib/Lower.cpp": LOWER_CPP + "// more lowering\nint helper() { return 1; }\n",
        "python/passes.py": 'class TileScheduler(ModulePass):\n    name = "toy-tile-schedule"\n',
    },
]


def capsule_yaml(name: str, op: str, family: str, *, label: str = "public", **extra) -> dict:
    return {
        "name": name,
        "kind": "layer",
        "label": label,
        "inputs": [
            {"name": "A0", "role": "input", "shape": [16, 32], "dtype": "i8"},
            {"name": "W", "role": "weight", "shape": [32, 8], "dtype": "i8"},
        ],
        "operation": {"op": op, "attributes": {"epilogue": ["acc_scale"], **extra.pop("attributes", {})}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i8"},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        "semantic": {"semantic_family": family, "generalization_axis": "application"},
        **extra,
    }


def phase1_run(root: Path, *, t0: float = T0, running: bool = False, workspace: bool = True) -> Path:
    """A Phase 1 run: grades, an agent stream, self-checks, a job channel, cost/time, snapshots, freeze."""
    from merlin.common import oot_repo

    run_dir = root / "runs" / TARGET / "phase1" / f"{stamp(t0)}_fixture_abc1234"
    ws = root / "build" / "agent-workspaces" / "ws"
    capsules = {
        "cap_add": ("add", "elementwise_map"),
        "cap_mm": ("matmul", "contraction"),
        "cap_conv": ("conv2d", "contraction"),
    }
    for name, (op, family) in capsules.items():
        write(ws / "capsules" / "layers" / name / "capsule.yaml", yaml.safe_dump(capsule_yaml(name, op, family)))
    grades = [
        (t0 + 0.5 * HOUR, {"cap_add": "pass", "cap_mm": "fail", "cap_conv": "fail"}),
        (t0 + 1.5 * HOUR, {"cap_add": "pass", "cap_mm": "pass", "cap_conv": "fail"}),
    ]
    for index, (at, statuses) in enumerate(grades):
        write(
            run_dir / "qa_history" / f"verdict_round_{index:02d}.json",
            {
                "graded_at": iso(at),
                "n_passed": sum(s == "pass" for s in statuses.values()),
                "n_capsules": len(statuses),
                "all_pass": False,
                "highest_tier": "L3",
                "first_failure_planes": {"numeric_golden": sum(s != "pass" for s in statuses.values())},
                "per_capsule": [
                    {
                        "capsule": n,
                        "label": "public",
                        "status": s,
                        "tiers": {"L0": "pass", "L1": "pass" if s == "pass" else "fail"},
                        **({} if s == "pass" else {"failure_plane": "numeric_golden"}),
                    }
                    for n, s in statuses.items()
                ],
            },
        )
    repo = oot_repo.init(run_dir / "oot")
    rows = []
    for index, files in enumerate(SUBMISSIONS):
        package = root / "packages" / f"p{index}"
        for rel, text in files.items():
            write(package / rel, text)
        record = oot_repo.commit_candidate(
            repo, package, label=f"round {index:02d}", when=int(grades[index][0]), run_id=run_dir.name
        )
        rows.append(
            {
                "label": "round",
                "key": f"{index:02d}",
                "oot": record.as_record(),
                "all_pass": False,
                "n_passed": index + 1,
                "n_capsules": 3,
            }
        )
    write(run_dir / "oot_commits.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    codex_stream(run_dir, 0, t0, agent_round(t0), step=60.0)
    codex_stream(run_dir, 1, t0 + 2 * HOUR, agent_round(t0 + 2 * HOUR, running=running), step=60.0)
    write(
        run_dir / "selfcheck_log.jsonl",
        "".join(
            json.dumps(r) + "\n"
            for r in [
                {
                    "wall_offset_s": 600,
                    "sim": "spike",
                    "capsules": "cap_mm",
                    "n_passed": 0,
                    "n_capsules": 1,
                    "all_pass": False,
                    "build_failed": False,
                    "failing": ["cap_mm"],
                },
                {
                    "wall_offset_s": 4000,
                    "sim": "spike",
                    "capsules": "all",
                    "n_passed": 2,
                    "n_capsules": 3,
                    "all_pass": False,
                    "build_failed": False,
                    "failing": ["cap_conv"],
                },
            ]
        ),
    )
    channel = ws / ".qa_channel"
    write(
        channel / "simreq_100_1.json",
        {"sim": "gsim", "capsules": "cap_mm", "workers": 1, "submitted_at": int(t0 + 1000)},
    )
    write(channel / "simrun_100_1", "running")
    write(channel / "simdone_100_1", "")
    write(channel / "simresp_100_1.json", {"all_pass": True})
    write(
        channel / "simreq_100_2.json",
        {"sim": "verilator", "capsules": "cap_conv", "workers": 1, "submitted_at": int(t0 + 2000)},
    )
    write(channel / "simrun_100_2", "running")
    write(
        channel / "req_abc.json",
        {
            "protocol": 3,
            "request_id": "abc",
            "sim": "spike",
            "capsules": "all",
            "requested_at_unix_ns": int((t0 + 3000) * 1e9),
        },
    )
    write(channel / "progress_abc.json", {"status": "running", "started_at_unix_ns": int((t0 + 3010) * 1e9)})
    write(
        run_dir / "environment.yaml",
        yaml.safe_dump(
            {
                "task_scope": {"target": TARGET},
                "workspace_path": str(ws) if workspace else None,
                "started_at": iso(t0),
                "model": "fixture-model",
                "driver": "codex",
            }
        ),
    )
    write(
        run_dir / "qa_loop_state.yaml",
        yaml.safe_dump(
            {
                "cumulative": {"started_at": iso(t0), "active_wall_s": 5000},
                "rounds": [
                    {
                        "round": 0,
                        "n_passed": 1,
                        "n_capsules": 3,
                        "tool_calls": 6,
                        "tokens_input": 36767,
                        "tokens_output": 2030,
                        "agent_rc": 0,
                    }
                ],
            }
        ),
    )
    write(
        run_dir / "cost_time_toolcalls.yaml",
        yaml.safe_dump(
            {
                "model": "fixture-model",
                "wall_time_seconds": 7200,
                "tokens_input": 37767,
                "tokens_cached": 28660,
                "tokens_output": 2330,
                "tokens_total": 40097,
                "tool_calls": 12,
                "billing_mode": "subscription_notional",
                "subscription_notional_usd": 1.25,
                "usage_complete": True,
            }
        ),
    )
    write(
        run_dir / "timing_detailed.json",
        {
            "method": "arrival_stamps",
            "think_generate_s": 3000.0,
            "tool_and_wait_s": 4200.0,
            "think_pct": 41.7,
            "tool_calls_matched": 10,
            "tool_calls_unterminated": 1,
            "tools": {"by_tool": {"command_execution": {"calls_started": 10, "errors": 2, "duration_p50_s": 4.0}}},
        },
    )
    if not running:
        write(
            run_dir / "freeze.json",
            {
                "frozen_at": stamp(t0 + 4 * HOUR),
                "submission_sha256": "5" * 64,
                "oot": {"frozen_commit": rows[-1]["oot"]["commit"]},
            },
        )
    return run_dir


# --------------------------------------------------------------------------- phase 0
def phase0_generation(root: Path) -> Path:
    """A generation run: requirements, MANIFEST cohorts, written capsules (one hidden), coverage."""
    run = root / "p0run"
    derive = run / "derive"
    write(
        derive / "requirements.yaml",
        yaml.safe_dump(
            {
                "target": TARGET,
                "boundaries": {"tile_edge": 16},
                "cells": [
                    {"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8", "n_regions": 4},
                    {"cell": "contraction/i8/partial", "family": "contraction", "dtype": "i8", "n_regions": 2},
                    {"cell": "movement/i8/aligned", "family": "movement", "dtype": "i8", "n_regions": 3},
                ],
                "shape_geometry": {
                    "required": [
                        {
                            "class": "tall_skinny",
                            "family": "contraction",
                            "M": 256,
                            "K": 72,
                            "N": 8,
                            "n_regions": 2,
                            "observed_in": ["app_a"],
                        },
                        {
                            "class": "gemv_like",
                            "family": "contraction",
                            "M": 1,
                            "K": 48,
                            "N": 80,
                            "n_regions": 1,
                            "observed_in": ["app_b"],
                        },
                    ]
                },
                "conv_geometry": {
                    "required": [
                        {"signature": "k3x3/s1x1/d1x1/pad1x1", "n_regions": 3, "sources": ["app_a"]},
                        {"signature": "k7x7/s2x2/d1x1/pad3x3", "n_regions": 1, "sources": ["app_a"]},
                    ]
                },
                "epilogue": {"required": [{"stage": "acc_scale", "family": "elementwise_map"}]},
                "scope": {
                    "required": [{"signature": "movement -> contraction", "occurrences": 5, "observed_in": ["app_a"]}],
                    "typed_required_instances": {
                        "status": "typed_source_only",
                        "instances": [
                            {
                                "application": "app_a",
                                "signature": "movement -> contraction",
                                "regions": [{"semantic_family": "movement"}, {"semantic_family": "contraction"}],
                            }
                        ],
                    },
                    "performance": {
                        "status": "resolved",
                        "required": [{}],
                        "excluded": [],
                        "unresolved": [],
                        "forms": {
                            "classes": [
                                {
                                    "class_id": "c1",
                                    "label": "conv2d_k3x3_bias",
                                    "key": {"placement": "systolic_mesh"},
                                    "max_share": 0.5,
                                    "members": [{}, {}],
                                    "share_by_application": {"app_a": 0.5},
                                },
                                {
                                    "class_id": "c2",
                                    "label": "matmul_m1_raw",
                                    "key": {"placement": "systolic_mesh"},
                                    "max_share": 0.1,
                                    "members": [{}],
                                    "share_by_application": {"app_b": 0.1},
                                },
                            ]
                        },
                    },
                },
                "heldout_layer_guard": {
                    "heldout_layer_shapes_sha256": "e" * 64,
                    "networks": 3,
                    "gemm_shapes": 62,
                    "conv_windows": 24,
                    "status": "no_member_reproduces_a_heldout_layer",
                },
                "diagnostics": {"families_observed": {"contraction": 6, "movement": 3}},
            }
        ),
    )
    write(derive / "derivation.json", {"status": "diagnostic", "qualification": "derivation only"})
    capsules = run / "phase0" / "capsules"
    entries = {
        "layers/MF_mm_tall": capsule_yaml("MF_mm_tall", "matmul", "contraction"),
        "layers/SY_conv_k3": capsule_yaml(
            "SY_conv_k3",
            "conv2d",
            "contraction",
            attributes={"kh": 3, "kw": 3, "stride": [1, 1], "padding": [1, 1, 1, 1]},
        ),
        "isa/SY_move": capsule_yaml("SY_move", "movement", "movement"),
        "_perf/PW00_conv": capsule_yaml(
            "PW00_conv",
            "conv2d",
            "contraction",
            label="dev",
            performance={"family": "PW", "form": {"label": "conv2d_k3x3_bias", "geometry": "squareish_gemm"}},
        ),
        "hidden/HE00_secret_layer": capsule_yaml("HE00_secret_layer", "matmul", "contraction", label="hidden"),
    }
    for rel, doc in entries.items():
        write(capsules / rel / "capsule.yaml", yaml.safe_dump(doc))
    write(
        capsules / "MANIFEST.yaml",
        yaml.safe_dump(
            {
                "generated": [k for k in entries if not k.startswith("hidden/")],
                "hand_authored": [],
                "held_out": {"n_generated": 1, "n_hand_authored": 0},
                "phase_corpora": {
                    TARGET: {
                        "phase1": {
                            "purpose": "functional_conformance",
                            "generated_members": ["layers/MF_mm_tall", "layers/SY_conv_k3", "isa/SY_move"],
                        },
                        "phase2": {"purpose": "performance_optimization", "generated_members": ["_perf/PW00_conv"]},
                        "diagnostic": {"purpose": "phase0_diagnostic", "generated_members": []},
                    }
                },
                "performance_generation": {
                    TARGET: {"counts": {"by_family": {"PW": {"admitted_members": 1, "written_members": 1}}}}
                },
            }
        ),
    )
    coverage = run / "phase0" / "coverage"
    write(
        coverage / "generation.json",
        {
            "schema": "merlin.phase0_generation.v1",
            "evidence_status": "complete",
            "capsules_written": 5,
            "omitted": [{"capsule": "layers/SY_never", "status": "not_built", "reason": "no emitter"}],
            "failures": [{"capsule": "hidden/HE01_other", "reason": "golden failed"}],
            "qualification": "not_established",
            "hidden_disjointness": {
                "status": "disjoint",
                "hidden_capsules": ["hidden/HE00_secret_layer"],
                "overlapping_hidden_capsules": [],
            },
            "cohort_coverage": {"phase1": {"status": "incomplete", "n_capsules": 3}},
        },
    )
    write(
        coverage / "phase1-capsule-coverage.json",
        {
            "schema": "merlin.phase0.coverage_commitment.v2",
            "phase": "phase1",
            "status": "incomplete",
            "cohort": {"n_capsules": 3},
            "conformance": {
                "n_required": 3,
                "n_covered": 2,
                "uncovered": ["movement/i8/aligned"],
                "corpus_cells": ["contraction/i8/aligned", "contraction/i8/partial"],
                "shape_geometry": {
                    "status": "ok",
                    "uncovered": ["gemv_like"],
                    "covered_by": {"tall_skinny": ["MF_mm_tall"]},
                },
                "conv_geometry": {
                    "status": "ok",
                    "uncovered": ["k7x7/s2x2/d1x1/pad3x3"],
                    "covered_by": {"k3x3/s1x1/d1x1/pad1x1": ["SY_conv_k3"]},
                },
            },
            "applications": {
                "app_a": {
                    "operations": [
                        {"id": "o1", "status": "covered", "role": "compute_placement"},
                        {"id": "o2", "status": "missing", "role": "compute_placement"},
                    ]
                }
            },
            "blockers": [{"component": "conformance", "reason": "finite admitted-cohort cell coverage is incomplete"}],
        },
    )
    write(
        coverage / "phase2-capsule-coverage.json",
        {
            "phase": "phase2",
            "status": "incomplete",
            "cohort": {"n_capsules": 1},
            "form_perf_coverage": {
                "status": "incomplete",
                "threshold": {"min_predicted_cycle_share": 0.02},
                "missing": ["c2"],
                "source_windows_without_form": [],
                "classes": [
                    {
                        "class_id": "c1",
                        "label": "conv2d_k3x3_bias",
                        "status": "covered",
                        "required": True,
                        "capsules": ["PW00_conv"],
                    },
                    {
                        "class_id": "c2",
                        "label": "matmul_m1_raw",
                        "status": "missing",
                        "required": True,
                        "capsules": [],
                        "geometry_strata": {"unrepresented": ["gemv_like"]},
                    },
                ],
            },
            "blockers": [{"component": "form_perf_coverage", "reason": "required forms are not represented"}],
        },
    )
    return run
