"""Connect a complete shared-RHS region to checked scheduling and the normal pass interface.

An authored ``tools/shared_rhs_owner.py`` supplies ``context(package)``,
``eligible(source, package=...)`` and ``compile(source, package=..., scheduler=...)``.
The compiler calls this SAME ``prepare`` path for its declared emit command.
The owner retains source parsing, alias guards, numeric/readout qualification,
selected laws, target encoding and the complete program's output ABI.

``scheduler=None`` means the incumbent. The alternative receives the existing
checked ``schedule_dense_shared_rhs`` callable; it must pass the actual current
source identity and schedule.plan and consume the returned result in emission.
The public recipe supplies no target facts, encoder, measured winner or law.

    python shared_rhs_epoch_recipe.py --describe
    python shared_rhs_epoch_recipe.py --package submission --source source.mlir \
        --mechanism shared_rhs_epochs --output emitted.mlir --receipt application.json

Use ``ab-library-pass/scripts/apply_library_pass.py`` with a tiny authored adapter
that re-exports ``prepare``. A command which ignores the selection is rejected
there as DISCONNECTED_ADAPTER. A changed emission still needs normal paired
numerical and timing checks over the complete grouped region.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

MECHANISM = "shared_rhs_epochs"
LEVEL = "original_source_region"
OWNER = "tools/shared_rhs_owner.py"


def _owner(package):
    path = package / OWNER
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(package):
        raise ValueError("construction required: supply the authored shared-RHS owner inside the package")
    name = "shared_rhs_owner_" + hashlib.sha256(str(path).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if any(not callable(getattr(module, field, None)) for field in ("context", "eligible", "compile")):
        raise ValueError("construction required: owner needs context, eligible and compile callbacks")
    return module


def prepare(source, *, package, mechanism):
    """Existing author-library-pass ABI; both routes use one original emitter."""
    from merlin.transforms.pipeline_policy import (
        PipelinePolicy,
        PipelineVariant,
        PolicyFacts,
        PropertyPredicate,
        RegisteredPass,
    )
    from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession

    if mechanism not in (None, MECHANISM):
        raise ValueError("unsupported shared-RHS mechanism selection")
    package = Path(package).resolve()
    owner = _owner(package)
    text = Path(source).read_text()
    context = owner.context(package)
    if type(context) is not PipelineContext:
        raise ValueError("owner must supply the original typed pipeline context")

    def verify(value):
        if type(value) is not str or not value:
            raise ValueError("the complete original source must remain intact")

    session = PipelineSession(text, context=context, ir_level=LEVEL,
        fingerprint=lambda value: hashlib.sha256(value.encode()).hexdigest(), verify_ir=verify)
    eligible = owner.eligible(text, package=package)
    if eligible is not None and type(eligible) is not bool:
        raise ValueError("owner eligibility must be true, false or unknown")
    facts = PolicyFacts(session.stamp(), operations=frozenset({"shared_rhs_region"}), capabilities=(
        ("shared_rhs_epochs_eligible", eligible),
    ))

    def construct(current, selected):
        scheduler = None
        if selected:
            from merlin.compile.scheduling.dense_shared_rhs import schedule_dense_shared_rhs
            scheduler = schedule_dense_shared_rhs
        product = owner.compile(current.module, package=package, scheduler=scheduler)
        if not isinstance(product, dict) or not isinstance(product.get("emitted"), (str, bytes)):
            raise ValueError("owner must return its actual emitted product and source-bound receipts")
        return {"compilation": product}

    def emit(current):
        product = current.analysis("compilation")["emitted"]
        if not product:
            raise ValueError("empty shared-RHS compiler product")
        return {"emitted": product}

    registry = {
        "shared_rhs_incumbent": RegisteredPass(PassSpec("shared_rhs_incumbent", 0,
            produces=("compilation",), mutates_ir=False, ir_level=LEVEL),
            lambda current: construct(current, False)),
        "shared_rhs_epochs": RegisteredPass(PassSpec("shared_rhs_epochs", 0,
            produces=("compilation",), mutates_ir=False, ir_level=LEVEL),
            lambda current: construct(current, True)),
        "emit_shared_rhs_program": RegisteredPass(PassSpec("emit_shared_rhs_program", 1,
            requires=("compilation",), produces=("emitted",), mutates_ir=False, ir_level=LEVEL), emit),
    }
    variants = [PipelineVariant("incumbent", ("shared_rhs_incumbent", "emit_shared_rhs_program"))]
    if mechanism == MECHANISM:
        variants.append(PipelineVariant(MECHANISM, ("shared_rhs_epochs", "emit_shared_rhs_program"), (
            PropertyPredicate("capability", "shared_rhs_epochs_eligible"),
        )))
    policy = PipelinePolicy(tuple(variants), fallback="incumbent")
    return dict(session=session, facts=facts, registry=registry, policy=policy, output="emitted")


def describe():
    return dict(mechanism=MECHANISM, prepare="shared_rhs_epoch_recipe:prepare", owner=OWNER,
        scheduler="merlin.compile.scheduling.dense_shared_rhs:schedule_dense_shared_rhs",
        pass_interface="merlin.transforms.pipeline_policy:execute_policy",
        obligations=["complete source group and live-outs", "selected original resource/retention/epoch laws",
            "actual schedule.plan consumed", "source alias guard or independently qualified custody",
            "bound fused output numerics", "actual emitted product and complete-region paired checks"],
        timing="UNKNOWN", winner="unranked", missing_owner="CONSTRUCTION_REQUIRED")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mechanism", choices=(MECHANISM,))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if args.describe or args.package is None and args.source is None:
        print(json.dumps(describe(), indent=2))
        return 0
    if args.package is None or args.source is None or args.output is None:
        parser.error("--package, --source and --output are required for an application")
    from merlin.transforms.pipeline_policy import execute_policy
    binding = prepare(args.source, package=args.package, mechanism=args.mechanism)
    result = execute_policy(**binding)
    raw = result.artifact.encode() if isinstance(result.artifact, str) else result.artifact
    args.output.write_bytes(raw)
    receipt = dict(result.receipt, emitted_sha256=hashlib.sha256(raw).hexdigest(),
        numerical_assessment="NOT_CHECKED", timing="UNKNOWN")
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
