"""Stage exact public development loaders for the original sealed-capture route.

This small operator command prepares no compiler or numerical policy. A positive
runtime preflight is not a seal: the original capture-selection owner must still
verify the full runtime namespace, execute, replay and attest it independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from merlin.common.cpu_pool import shared_compute_pool

from . import development_capture_intake as INTAKE
from . import development_cohort as COHORT

BATCH_LOADER = """import importlib.util
from pathlib import Path
from m2m.capture.external_runtime import ExternalRuntimeProgram,make_external_runtime_session
class Batch:
    def external_runtime_session(self):
        programs=[]
        for ordinal in ORDINALS:
            path=Path(__file__).resolve().parent / ('member_'+str(ordinal)+'.py')
            spec=importlib.util.spec_from_file_location('public_member_'+str(ordinal),path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            model,inputs=module.get_model_and_inputs()
            programs.append(ExternalRuntimeProgram(name='member_'+str(ordinal),module=model,inputs=tuple(inputs),steps=1))
        metadata=dict(kind='public_independent_development_batch',paper_ready=False,stages=[p.name for p in programs],
            stage_schedule=[dict(name=p.name,steps=1,execution='external_runtime',timed=True) for p in programs],
            quality_program=programs[0].name,quality=dict(key='output0',output_index=0),steps=1,
            states=[],streams=[],bindings=[],parameters={},provenance=dict(scope='explicit_public_development_batch'))
        return make_external_runtime_session(version=2,programs=tuple(programs),metadata=metadata)
def get_model_and_inputs():return Batch(),()
"""


def prepare(root, *, manifest_sha256, ordinals, output, limits, supplements=()):
    """Copy unchanged public loaders and bind their complete existing references."""
    root, output = Path(root), Path(output)
    manifest, references = INTAKE.references(root, manifest_sha256=manifest_sha256, supplements=supplements)
    if (
        type(ordinals) is not tuple
        or not 1 <= len(ordinals) <= 8
        or any(type(n) is not int or not 0 <= n < len(manifest["members"]) for n in ordinals)
        or len(set(ordinals)) != len(ordinals)
        or output.exists()
        or output.is_symlink()
        or output.resolve() != output
        or not output.parent.is_dir()
    ):
        raise ValueError("sealed public batch needs distinct original ordinals and a fresh canonical output")
    intake = INTAKE.plan(manifest, references, **limits)
    payload, selected = {}, []
    for ordinal in ordinals:
        member = manifest["members"][ordinal]
        if intake["members"][ordinal]["status"] != "capture_cost_screen_passed_not_admission":
            raise ValueError("original public member capture cost or numerical contract refused")
        description = member["description"]
        if any(
            type(description[key]) is not int or description[key] < 1 for key in ("contractions", "scalar_products")
        ):
            raise ValueError("sealed development batch requires declared nonzero accelerator work")
        if any(row["dtype"] != "int8" for row in description["inputs"]) or any(
            row["dtype"] not in {"int8", "int32"} for row in description["outputs"]
        ):
            raise ValueError("native integer batch preserves only original signed integer boundaries")
        source = root / member["loader"]
        raw = source.read_bytes()
        if (
            source.is_symlink()
            or dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) != member["loader_identity"]
        ):
            raise ValueError("original public loader identity changed")
        INTAKE.load_outputs(member, references[ordinal])
        payload[f"member_{ordinal}.py"] = raw
        selected.append(
            dict(
                ordinal=ordinal,
                request_sha256=member["request_sha256"],
                source=member["loader_identity"],
                reference=references[ordinal],
            )
        )
    payload["loader.py"] = BATCH_LOADER.replace("ORDINALS", repr(list(ordinals))).encode()
    # Original inventory and reference leaves are checked again before copying.
    INTAKE.references(root, manifest_sha256=manifest_sha256, supplements=supplements)
    output.mkdir(mode=0o700)
    workload = output / "workload"
    workload.mkdir(mode=0o700)
    for name, raw in payload.items():
        (workload / name).write_bytes(raw)
    report = dict(
        schema="merlin.public_development_sealed_batch_preparation.v1",
        source_manifest_sha256=manifest_sha256,
        members=selected,
        workload=str(workload),
        workload_files={
            name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name, raw in payload.items()
        },
        source_mode="native_integer_v1",
        capture_intake=intake,
        source_sealed=False,
        writer_admitted=False,
        performance_objective_changed=False,
        timing="UNKNOWN",
    )
    (output / "preparation.json").write_bytes(COHORT._bytes(report))
    return report


def runtime_preflight(path, *, sha256):
    """Refuse missing selected storage early; positive results grant no reuse."""
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 1 << 20:
        raise ValueError("original runtime selection exceeds its bound or is indirect")
    raw = path.read_bytes()
    if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("original runtime selection bytes changed")
    from merlin.common import strict_json

    selection = strict_json.loads(raw, max_bytes=1 << 20)
    if selection.get("schema") not in {
        "merlin.sealed_m2m_readonly_runtime.v1",
        "merlin.sealed_m2m_readonly_runtime.v2",
        "merlin.sealed_m2m_readonly_runtime.v3",
    }:
        raise ValueError("explicit original readonly runtime selection is required")
    entry = Path(selection["entry"])
    present = entry.is_dir() and not entry.is_symlink() and entry.resolve() == entry
    return dict(
        schema="merlin.public_development_runtime_preflight.v1",
        status="ORIGINAL_OWNER_REQUIRED" if present else "REFUSED_MISSING_SELECTED_RUNTIME",
        selected_entry=str(entry),
        source_cas_identity=selection["source_cas_identity"],
        effective_runtime_identity=selection.get("effective_runtime_identity", selection["source_cas_identity"]),
        runtime_qualified=False,
        required_next="original capture_selection select/issue/attest; complete namespace and independent replay",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, help="read an existing preparation; execute nothing")
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--ordinals", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-selection", type=Path)
    parser.add_argument("--runtime-selection-sha256")
    args = parser.parse_args(argv)
    if args.status:
        from merlin.common import strict_json

        records = {}
        for name in ("preparation.json", "runtime-preflight.json"):
            path = args.status / name
            if path.stat().st_size > 16 << 20:
                raise ValueError("prepared batch status exceeds its bounded metadata namespace")
            records[name] = strict_json.loads(path.read_bytes(), max_bytes=16 << 20)
        print(json.dumps(dict(recorded_preparation_only=True, **records)))
        return 0
    if any(value is None for key, value in vars(args).items() if key != "status"):
        parser.error("preparation needs every explicit source, member, output and runtime selection")
    with shared_compute_pool(timeout_s=120):
        prepare(
            args.sources,
            manifest_sha256=args.manifest_sha256,
            ordinals=tuple(args.ordinals),
            output=args.output,
            limits=dict(max_work=1_000_000, max_elements=1_000_000, max_seed_bytes=16 << 20, max_contractions=1),
        )
        status = runtime_preflight(args.runtime_selection, sha256=args.runtime_selection_sha256)
    (args.output / "runtime-preflight.json").write_bytes(COHORT._bytes(status))
    print(json.dumps(dict(preparation=str(args.output / "preparation.json"), **status)))
    return 0 if status["status"] == "ORIGINAL_OWNER_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
