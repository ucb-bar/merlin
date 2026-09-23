"""Capsules generated from the compute groups a captured model forms on a target.

The compute-group pass decides what a model asks of a unit: this contraction, with these stages
fused into its readout, at these extents, with this multiplier. A backend is only ever shown work
as a capsule, so that decision is worth exactly as much as the capsules that carry it. This module
closes the loop: every accelerator group of a capture is restated as a device program
(:mod:`~merlin.xdsl_dialects.lowering.group_command`), deduplicated, and built by the SAME generator
the graded corpus uses -- same builder, same golden, same oracle adapters. Nothing here is a second
harness.

Each grouped capsule gets a ``raw`` sibling: the same contraction committing its bare accumulator,
with no stage fused. That is the device half of what a model route emits when the group does not
close (the other half being per-element host loops), so the pair measures what closing the group
costs or saves on the unit itself, on one oracle, under one protocol.

A group the capsule vocabulary cannot state is counted with its reason and never dropped: it is a
finding about the vocabulary, not about the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .group_capsule_entries import _IDENTITY_DROPS as _IDENTITY_DROPS
from .group_capsule_entries import SCHEMA as SCHEMA
from .group_capsule_entries import SOURCE_ROLE as SOURCE_ROLE
from .group_capsule_entries import _element_range as _element_range
from .group_capsule_entries import _identity as _identity
from .group_capsule_entries import _label as _label
from .group_capsule_entries import entries as entries


def _work(entry: Mapping[str, Any]) -> int:
    """Multiply-accumulates of one entry: what its golden and its functional-tier run cost."""
    if entry["op"] == "conv2d":
        pad, stride = entry["padding"], entry["stride"]
        rows = (entry["Himg"] + pad[0] + pad[2] - entry["kh"]) // stride[0] + 1
        cols = (entry["Wimg"] + pad[1] + pad[3] - entry["kw"]) // stride[1] + 1
        return rows * cols * entry["kh"] * entry["kw"] * entry["ci"] * entry["N"]
    return int(entry["M"]) * int(entry["K"]) * int(entry["N"])


def covering_subset(stated: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One program per (op, stage combination), the cheapest that has it, with its bare sibling.

    Every distinct way the model fuses stages onto a contraction is represented at the model's own
    extents, and the choice within a combination is by work, because each capsule's golden is
    computed by an integer engine written for clarity and its cost is what a backend author waits
    on. Deterministic: ties break on the name.
    """
    rows = [row for row in stated["entries"] if not row.get("raw_of")]
    best: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: (_work(r["entry"]), r["name"])):
        best.setdefault((row["entry"]["op"], tuple(row["entry"].get("epilogue") or ())), row)
    chosen = {row["name"] for row in best.values()}
    return [row for row in stated["entries"] if row["name"] in chosen or row.get("raw_of") in chosen]


def _experiment(target: str, declaration=None):
    from merlin_experiments.phase0.declarations import for_target

    from merlin.targetgen.target_experiment import load_target_experiment

    selected = declaration if declaration is not None else for_target(target)
    experiment = load_target_experiment(selected.descriptor)
    if selected.target != target or experiment.target != target:
        raise ValueError("selected derivation declaration/descriptor does not match the requested target")
    return experiment, selected


def promote(target: str, stated: Mapping[str, Any], *, declaration=None) -> dict[str, Any]:
    """Write the covering subset into the target's corpus, under the ungraded model-layer category."""
    from merlin_experiments.phase0 import profiles

    from merlin.targetgen import corpus_spec
    from merlin.targetgen.phase_policy import _CYCLE_ACCURATE_TIERS  # noqa: PLC2701
    from merlin.targetgen.target_experiment import MODEL_LAYERS_CATEGORY

    experiment, selected = _experiment(target, declaration)
    # Model-shape capsules can screen at a functional rung without claiming unaffordable
    # cycle-accurate self-certification. Use the same selected declaration/profile binding as write().
    profile = profiles.load_profile(selected.profile, include_holdouts=False, **selected.profile_inputs())
    binding = corpus_spec.derive_binding(experiment, profile.get("datapath", {}))
    tiers = [str(t) for t in (getattr(binding, "tiers", None) or ())]
    screenable = [t for t in tiers if t not in _CYCLE_ACCURATE_TIERS]
    loop_tier = screenable[-1] if screenable and len(screenable) < len(tiers) else None
    subset = []
    for row in covering_subset(stated):
        entry = {**row["entry"], "cat": MODEL_LAYERS_CATEGORY, "label": "dev"}
        if loop_tier and not entry.get("max_oracle_tier"):
            entry["max_oracle_tier"] = loop_tier
        subset.append({**row, "entry": entry})
    return write(target, {**stated, "entries": subset}, Path(experiment.capsule_corpus).parent, declaration=selected)


def write(target: str, stated: Mapping[str, Any], out_root: Path, *, declaration=None) -> dict[str, Any]:
    """Build every entry with the corpus generator. A builder that refuses one is reported by name."""
    from merlin_experiments.phase0 import profiles, writer

    from merlin.targetgen import corpus_spec

    experiment, selected = _experiment(target, declaration)
    profile = profiles.load_profile(selected.profile, include_holdouts=False, **selected.profile_inputs())
    binding = corpus_spec.derive_binding(experiment, profile.get("datapath", {}))
    built: dict[str, str] = {}
    refused: dict[str, str] = {}
    for row in stated["entries"]:
        try:
            built[row["name"]] = str(writer._write_capsule(dict(row["entry"]), binding, Path(out_root)))
        except Exception as error:  # noqa: BLE001 -- reported per capsule, never swallowed
            refused[row["name"]] = f"{type(error).__name__}: {str(error)[:300]}"
    return {"built": built, "refused_by_generator": refused}


def run(
    target: str,
    built: Mapping[str, str],
    package: Path,
    runs_root: Path,
    *,
    timeout: int = 3600,
    max_tier: str | None = None,
    declaration=None,
) -> dict[str, Any]:
    """Grade each built capsule against ``package`` on the target's own oracle adapters.

    ``max_tier`` withholds every deeper oracle adapter, and each row names the tiers it withheld. A
    real layer's extents can cost an elaborated-RTL simulator hours; the FPGA is the tier that
    answers for them. A row graded this way is a SCREEN, and says so: ``status`` is the ladder's own
    verdict, which is not a pass while a mandatory tier did not run.
    """
    import yaml

    from merlin.targetgen import capsule_runner, tier_policy

    experiment, _ = _experiment(target, declaration)
    adapters = capsule_runner.oracle_adapters(target, experiment.sim_via) or {}
    withheld: list[str] = []
    if max_tier:
        depth = tier_policy.tier_depth_order([*adapters, str(max_tier)])
        withheld = [tier for tier in adapters if depth.index(tier) > depth.index(str(max_tier))]
        adapters = {tier: adapter for tier, adapter in adapters.items() if tier not in withheld}
    rows: dict[str, Any] = {}
    for name, directory in built.items():
        capsule = yaml.safe_load((Path(directory) / "capsule.yaml").read_text(encoding="utf-8"))
        capsule["__dir__"] = str(directory)
        try:
            result = capsule_runner.run_capsule(
                capsule,
                Path(package),
                runs_root=Path(runs_root),
                oracle_adapters=adapters,
                target=target,
                timeout=timeout,
            )
        except Exception as error:  # noqa: BLE001
            rows[name] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
            continue
        result = result or {}
        rows[name] = {
            "status": result.get("status"),
            "failure": result.get("failure"),
            "numeric": result.get("numeric"),
            "tiers_withheld": list(withheld),
            # A real layer is where a backend reaches for loop commands, and a pass at a tier blind
            # to them is a screen. Read from the program's own decoded trace.
            "blind_spots": _blind_spot_record(target, capsule, result, Path(runs_root)),
            "tiers": {
                tier: {k: record.get(k) for k in ("status", "cycles", "detail", "engine")}
                for tier, record in (result.get("tiers") or {}).items()
                if isinstance(record, Mapping)
            },
        }
    return rows


def _blind_spot_record(target: str, capsule: Mapping[str, Any], result: Mapping[str, Any], runs_root: Path):
    import json as _json

    from merlin.targetgen import capsule_runner, oracle_blind_spots

    try:
        spots = oracle_blind_spots.for_target(target)
    except oracle_blind_spots.BlindSpotError:
        raise
    except Exception:  # noqa: BLE001 -- no contract for this name: nothing is declared
        return None
    if not spots:
        return None
    trace_path = (
        runs_root / "runs" / capsule_runner.suite_for(target) / str(capsule.get("name")) / "generated"
    ) / "instruction_trace.json"
    trace = None
    if trace_path.is_file():
        try:
            trace = _json.loads(trace_path.read_text(encoding="utf-8"))
        except ValueError:
            trace = None
    return oracle_blind_spots.judge_result(spots, capsule, result.get("tiers") or {}, trace)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Attach the shared corpus-group options without importing execution dependencies."""
    parser.add_argument("--target", required=True)
    parser.add_argument("--definition", type=Path, help="explicit derivation definition; disambiguates target reuse")
    parser.add_argument("--capture", required=True, help="the captured model's linalg MLIR")
    parser.add_argument("--manifest", help="the capture's weights manifest (says which arguments are stored)")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", required=True, help="directory the capsules and the report are written under")
    parser.add_argument("--package", help="a backend package to grade the capsules against")
    parser.add_argument("--only", action="append", default=[], help="build and run only these capsule names")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="write a covering subset (one program per stage combination, the cheapest) into the "
        "target's corpus under its ungraded model-layer category, instead of under --out",
    )
    parser.add_argument(
        "--max-tier", help="withhold every oracle tier deeper than this one (a screen, recorded as one)"
    )


def run_from_args(args: argparse.Namespace) -> int:
    """Derive/build the requested corpus groups; never review or seal their outputs."""
    from merlin_experiments.phase0.declarations import for_target, from_definition

    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import stream_plan

    definition = getattr(args, "definition", None)
    selected = from_definition(definition) if definition is not None else for_target(args.target)
    _experiment(args.target, selected)  # refuse an incompatible declaration before writing output

    weights = None
    if args.manifest:
        weights = stream_plan.weight_args_of(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    stated = entries(args.target, mq.parse(args.capture), weight_args=weights, model=args.model)
    if args.only:
        stated["entries"] = [row for row in stated["entries"] if row["name"] in set(args.only)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.promote:
        report: dict[str, Any] = {**stated, **promote(args.target, stated, declaration=selected)}
        report["entries"] = [row for row in stated["entries"] if row["name"] in report["built"]]
    else:
        report = {**stated, **write(args.target, stated, out / "capsules", declaration=selected)}
    if args.package:
        report["graded"] = run(
            args.target,
            report["built"],
            Path(args.package),
            out / "runs",
            timeout=args.timeout,
            max_tier=args.max_tier,
            declaration=selected,
        )
    (out / "group_capsules.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{stated['stated']} of {stated['accelerator_groups']} accelerator group(s) stated as "
        f"{stated['distinct']} distinct program(s); built {len(report['built'])}, "
        f"refused by the generator {len(report['refused_by_generator'])}, unstated {stated['unstated'] or 'none'}"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    configure_parser(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
