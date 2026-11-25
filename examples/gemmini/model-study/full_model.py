#!/usr/bin/env python3
"""Build and check one captured forward through ordinary model_check APIs.

The selected host-lowering Python and frozen candidate Python are separate.
This component diagnostic retains all outputs; it does not submit FPGA jobs or
claim complete recurrent-model, hardware, or performance qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sibling(name):
    spec = importlib.util.spec_from_file_location("model_study_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_bytes(path, prepare):
    """Parse exactly the bytes that the retained selected-file pin describes."""
    before = prepare.pin(path)
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != before["sha256"] or prepare.pin(path) != before:
        raise ValueError("selected input changed while reading its bytes")
    return payload, before


def candidate_wrapper(settings, bootstrap, configuration):
    """Select isolation only for the actual manifest's literal compiler entrypoint."""
    return (
        "#!/usr/bin/python3\nimport os,sys\nfrom pathlib import Path\n"
        + "selection=" + repr(settings) + "\n"
        + "env=dict(os.environ)\nargs=sys.argv[1:]\npython=selection['outer_python']\n"
        + "if args and not args[0].startswith('-') and str(Path(args[0]).resolve()) == selection['entrypoint']:\n"
        + " python=selection['python']\n"
        + " env['PYTHONPATH']=os.pathsep.join(selection['compiler_roots'][1:]+selection['dependency_roots'])\n"
        + " args=['-S'," + repr(str(bootstrap)) + "," + repr(str(configuration)) + ",*args]\n"
        + "env['PYTHONDONTWRITEBYTECODE']='1'\n"
        + "os.execve(python,[python,'-B',*args],env)\n"
    )


def verify_reference_policy(document, settings, policy):
    observed = document.get("numeric_contract", {}).get("host_math", {}).get("policy", "native")
    if observed != settings.host_math_policy or document.get("comparison_policy") != policy:
        raise ValueError("selected reference and linked build use different numerical policies")


def source_route(document):
    """Decode an explicit caller selection; effect premises are not inferred."""
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.head_source_route import HeadBatchSourceRoute
    from merlin.llvmlower.sibling_source_route import SiblingSourceRoute

    if type(document) is not dict:
        raise ValueError("source route must be a JSON object")
    kinds = {
        "independent_heads": (HeadBatchSourceRoute, {"shared_input_siblings", "whole_batch", "shared_rhs_row_views"}),
        "shared_input": (SiblingSourceRoute, {"partial"}),
    }
    if type(document.get("kind")) is not str or document["kind"] not in kinds:
        raise ValueError("unknown source route kind")
    constructor, flags = kinds[document["kind"]]
    required = {"kind", "source_ir_sha256", "effects"}
    if not required <= document.keys() or document.keys() - required - flags:
        raise ValueError("source route has missing or unknown fields")
    effects = document["effects"]
    if type(effects) is not dict or set(effects) != {
        "qualification_identity", "no_retained_aliases", "no_releases", "complete_before_return"
    }:
        raise ValueError("source route needs all explicit borrowed-effect premises")
    selected = constructor(source_ir_sha256=document["source_ir_sha256"], effects=BorrowedBufferEffects(**effects),
                           **{key: document[key] for key in flags if key in document})
    selected.record()
    return selected


def unique_object(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate JSON field: " + key)
        document[key] = value
    return document


def build_selected(builder, stage, *, source_route=None, **kwargs):
    """Leave the ordinary baseline invocation unchanged when no route is selected."""
    if source_route is not None:
        kwargs["source_route"] = source_route
    return builder(stage, **kwargs)


def retain_producers(output, sources, prepare):
    retained = output / "producer"
    retained.mkdir()
    rows = []
    for source in sources:
        before = prepare.pin(source)
        destination = retained / Path(source).name
        destination.write_bytes(Path(source).read_bytes())
        if prepare.pin(source) != before or prepare.sha(destination) != before["sha256"]:
            raise ValueError("producer changed while retaining its bytes")
        rows.append({"original": before, "retained": prepare.pin(destination)})
    return rows


def run(args):
    from merlin.common import owned_process
    from merlin.model_check import admission, full_build as FB, full_quantized as FQ
    from merlin.model_check.cli import tree_sha256
    from merlin.model_check.models import load, selected_float_policy

    prepare = sibling("prepare")
    compiler, library, stage = (p.resolve(strict=True) for p in (args.compiler, args.compiler_library, args.stage))
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(root) for root in (compiler, library, stage)):
        raise ValueError("select fresh output outside frozen inputs")
    if not 1 <= args.run_timeout <= 14400:
        raise ValueError("run timeout must be within 1..14400 seconds")
    models_bytes, models_pin = selected_bytes(args.models, prepare)
    models = load(args.models, source_bytes=models_bytes)
    model = models.model(args.model)
    model.program(args.program)
    settings = FB.BuildSettings.of(models, model, args.host_package, op_profile=args.op_profile)
    if args.host_elementwise_fusion:
        settings = replace(settings, host_elementwise_fusion=True)
    policy = selected_float_policy(models)
    reference = args.reference.resolve(strict=True)
    reference_document = args.reference_document.resolve(strict=True)
    if output.is_relative_to(reference.parent):
        raise ValueError("output overlaps the selected reference")
    reference_rules = sibling("reference_admission")
    document_bytes, document_pin = selected_bytes(reference_document, prepare)
    reference_record = json.loads(document_bytes, object_pairs_hook=unique_object)
    reference_selection = dict(reference=reference, reference_sha256=args.reference_sha256,
                               models=args.models, model=args.model, program=args.program, stage=stage,
                               host_math_policy=settings.host_math_policy, comparison_policy=policy)
    reference_admission = reference_rules.admit(reference_record, **reference_selection)
    pins = [models_pin, document_pin, *[prepare.pin(p) for p in (reference, settings.host_dts,
                                    settings.board_catalog, args.host_python, args.candidate_python)]]
    producer_sources = [Path(__file__), Path(prepare.__file__), Path(__file__).with_name("candidate_python.py"),
                        Path(reference_rules.__file__)]
    pins.extend(prepare.pin(p) for p in producer_sources)
    route_path = args.source_route.resolve(strict=True) if args.source_route is not None else None
    selected_route = None
    if route_path is not None:
        route_pin = prepare.pin(route_path)
        selected_route = source_route(json.loads(route_path.read_bytes(), object_pairs_hook=unique_object))
        if prepare.pin(route_path) != route_pin:
            raise ValueError("selected source route changed while reading")
        pins.append(route_pin)

    def verify():
        prepare.verify_compiler(compiler, args.compiler_inventory, args.compiler_sha256)
        prepare.tree_pin(library, args.compiler_library_sha256)
        if tree_sha256(compiler) != args.submission_tree_sha256:
            raise ValueError("model_check submission tree differs from selected identity")
        if reference_rules.admit(reference_record, **reference_selection) != reference_admission:
            raise ValueError("reference admission changed")
        if any(prepare.pin(p["path"]) != p for p in pins):
            raise ValueError("selected model/reference/runtime input changed")

    verify()
    output.mkdir(parents=True)
    producer_retention = retain_producers(output, producer_sources, prepare)
    retained_pins = [row["retained"] for row in producer_retention]
    pins.extend(retained_pins)
    temporary = output / "tmp"
    temporary.mkdir()
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(temporary),
                      MERLIN_OUT_ROOT=str(output / "out"), MERLIN_COMPILER_PYTHON=str(args.host_python.absolute()))
    probe_environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(compiler / "compiler_library"), str(library)]))
    runtime = prepare.runtime_probe(args.candidate_python, probe_environment, output, 60)
    if runtime["returncode"]:
        raise ValueError("selected frozen compiler runtime probe failed")
    host_code = (
        "import json,sys; from pathlib import Path; import torch_mlir.ir,torch_mlir.passmanager; "
        "print(json.dumps({'executable':sys.executable,'files':[str(Path(torch_mlir.ir.__file__).resolve()),"
        "str(Path(torch_mlir.passmanager.__file__).resolve()),*map(str,sorted((Path(torch_mlir.ir.__file__).parent/"
        "'_mlir_libs').glob('*.so')))]}))"
    )
    host = owned_process.run([str(args.host_python), "-B", "-c", host_code], env=dict(os.environ),
                             capture_output=True, timeout=60)
    (output / "host-runtime.stdout").write_bytes(host.stdout)
    (output / "host-runtime.stderr").write_bytes(host.stderr)
    if host.returncode:
        raise ValueError("selected host Python cannot import the upstream MLIR lowering runtime")
    host_observation = json.loads(host.stdout)
    runtime_pins = [prepare.pin(p) for p in host_observation["files"]]
    for row in [*runtime["observed"]["modules"].values(), *runtime["observed"]["solver_libraries"]]:
        runtime_pins.append(prepare.pin(row["path"]))
    origins = output / "compiler-imports"
    origins.mkdir()
    bootstrap = output / "candidate-python.py"
    bootstrap.write_bytes(Path(__file__).with_name("candidate_python.py").read_bytes())
    configuration = output / "candidate-environment.json"
    candidate = {
        "python": str(args.candidate_python.absolute()), "outer_python": sys.executable,
        "entrypoint": str(compiler / "compiler.py"),
        "compiler_roots": [str(compiler), str(compiler / "compiler_library"), str(library)],
        "dependency_roots": sorted({str(Path(row["path"]).parent.parent)
                                    for row in runtime["observed"]["modules"].values()}),
        "origins": str(origins),
    }
    configuration.write_text(json.dumps(candidate, indent=2) + "\n")
    wrapper = output / "compiler-python" / "python3"
    wrapper.parent.mkdir()
    wrapper.write_text(candidate_wrapper(candidate, bootstrap, configuration))
    wrapper.chmod(0o755)
    os.environ["PATH"] = str(wrapper.parent) + os.pathsep + os.environ.get("PATH", "")
    pins.extend(prepare.pin(p) for p in (bootstrap, configuration, wrapper))
    record = {
        "schema": "merlin.model_study.full_forward.v1", "status": "building", "settings": settings.record(),
        "compiler_source_sha256": args.compiler_sha256, "submission_tree_sha256": args.submission_tree_sha256,
        "candidate_environment": candidate, "host_runtime": host_observation, "runtime_pins": runtime_pins,
        "inputs": pins, "candidate_runtime": runtime, "producer": prepare.pin(Path(__file__)),
        "reference_admission": reference_admission,
        "producer_retention": producer_retention,
        **({"source_route": {"input": route_pin, "selection": selected_route.record(),
                              "effects_scope": "explicit caller premises; not an independent implementation qualification"}}
           if selected_route is not None else {}),
        "scope": "one captured forward; no full recurrent-model, physical or timing qualification",
        "admission": "ordinary structural linked-ELF service; not a no-FSM instruction certificate",
        "hardware_execution": "not_run", "numerical_status": "not_run",
    }
    destination = output / "result.json"

    def save():
        destination.write_text(json.dumps(record, indent=2) + "\n")

    started = time.monotonic()
    save()
    try:
        print("Building linked captured forward", flush=True)
        build = build_selected(FB.build, stage, source_route=selected_route,
                         target=models.target, label=args.model + "/" + args.program, submission=compiler,
                         submission_sha256=args.submission_tree_sha256, settings=settings,
                         directory=output / "out" / "image-build", admission=admission.service(models.target), offload=True)
        (output / "linked-build.json").write_text(json.dumps(build, indent=2) + "\n")
        record.update(status="running_spike", elf=prepare.pin(build["elf"]), build_hash=build["build_hash"])
        save()
        print("Running linked forward on Spike", flush=True)
        console, elapsed = FB.run(models.target, Path(build["elf"]), timeout_s=args.run_timeout)
        (output / "console.bin").write_bytes(console)
        actual, diagnostic = FB.read(console, stage, build["build_hash"])
        import numpy as np

        np.savez(output / "actual.npz", **{"out" + str(i): value for i, value in enumerate(actual)})
        with np.load(reference, allow_pickle=False) as archive:
            expected = [np.asarray(archive["out" + str(i)]) for i in range(len(archive.files))]
        bindings = [{"program": args.program, "step": 0, "output": i} for i in range(len(expected))]
        comparisons = FQ.compare_outputs(actual, expected, bindings, policy)
        passed = bool(comparisons) and all(row["pass"] for row in comparisons)
        record.update(status="completed" if passed else "numerical_failure", numerical_status="pass" if passed else "fail",
                      comparisons=comparisons, comparison_policy=policy, diagnostics=diagnostic, spike_wall_seconds=elapsed,
                      console=prepare.pin(output / "console.bin"), actual=prepare.pin(output / "actual.npz"))
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        try:
            verify()
            if any(prepare.pin(row["path"]) != row for row in runtime_pins):
                raise ValueError("selected runtime bytes changed")
            record["inputs_unchanged"] = True
        except Exception as error:
            record.update(status="inputs_changed", inputs_unchanged=False, input_error=str(error))
        record["wall_seconds"] = time.monotonic() - started
        save()
    print(json.dumps({"status": record["status"], "result": str(destination)}), flush=True)
    return 0 if record["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler", "compiler-inventory", "compiler-library", "candidate-python", "host-python", "stage",
                 "models", "host-package", "reference", "reference-document", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("compiler-sha256", "submission-tree-sha256", "compiler-library-sha256", "reference-sha256",
                 "model", "program"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--run-timeout", type=int, default=600)
    parser.add_argument("--op-profile", action="store_true")
    parser.add_argument("--host-elementwise-fusion", action="store_true",
                        help="explicitly enable the existing shared host fusion pass as a separate condition")
    parser.add_argument("--source-route", type=Path,
                        help="explicit typed source-selection JSON, including inspected synchronous-effect premises")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
