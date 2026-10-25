"""Recompute actual call effects in an explicitly pinned producer subprocess.

The selected example supplies declarations and the adapter, never candidate
effect annotations. This is trusted projection evidence, not a Lean or target
qualification. Run directly with only the selected source on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

if __package__:
    from .lean_graph_loops_capture import _producer_sources, _reference
else:
    from lean_graph_loops_capture import _producer_sources, _reference


def project(*, source, revision, example, iterations, before, after, output, index_bits, max_dynamic_calls):
    sys.dont_write_bytecode = True
    import merlin.compile.scheduling.periodic_effects as periodic
    from merlin.compile.scheduling.temporary_storage import StorageEffects
    from merlin.sched.ir import Call, Kernel, Loop, check_structure, parse_kernel

    source = source.resolve()
    if (
        subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip()
        != revision
    ):
        raise ValueError("actual producer revision differs from the selected revision")
    example_path = (source / example).resolve()
    if not example_path.is_relative_to(source / "examples"):
        raise ValueError("selected declaration source must be an actual producer example")
    spec = importlib.util.spec_from_file_location("graph_loop_selected_effects", example_path)
    if spec is None or spec.loader is None:
        raise ValueError("selected adapter cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    declared, objects, banks, adapter, admission, options, _iset = module.fixture(iterations)
    source_kernel = parse_kernel(before.read_bytes().decode())
    target_kernel = parse_kernel(after.read_bytes().decode())
    if any(type(kernel) is not Kernel or check_structure(kernel) for kernel in (source_kernel, target_kernel)):
        raise ValueError("effect projection requires valid actual Kernels")
    if (source_kernel.name, source_kernel.args, source_kernel.attrs) != (declared.name, declared.args, declared.attrs):
        raise ValueError("actual source boundary differs from selected declaration context")
    if len(source_kernel.body) != 1 or type(source_kernel.body[0]) is not Loop:
        raise ValueError("one actual counted loop is required")
    loop = source_kernel.body[0]
    if loop.extent != iterations or not loop.body or any(type(call) is not Call for call in loop.body):
        raise ValueError("actual loop differs from selected finite domain")
    if len(loop.body) * iterations > max_dynamic_calls:
        raise ValueError("complete actual effect domain exceeds the selected budget")
    blocks = [tuple(periodic._closed(call, loop.var, index) for call in loop.body) for index in range(iterations)]
    expected = tuple(call for block in blocks for call in block)
    if (target_kernel.name, target_kernel.args, target_kernel.attrs) != (
        source_kernel.name,
        source_kernel.args,
        source_kernel.attrs,
    ) or target_kernel.body != expected:
        raise ValueError("effects do not bind to the complete actual materialized call sequence")
    effects = []
    for block in blocks:
        rows = []
        for call in block:
            if any((call.unit, call.produces, call.awaits, call.stages, call.sets, call.assumes)):
                raise ValueError("execution metadata needs a separately qualified effect projection")
            effect = adapter.project(call)
            if type(effect) is not StorageEffects or effect.completion == "unknown":
                raise ValueError("unknown instruction or completion effect refuses projection")
            rows.append(asdict(effect))
        effects.append(rows)
    closure = _producer_sources(source, revision, (example_path,))
    try:
        checked = periodic.check_drained_period(
            source_kernel,
            objects,
            banks,
            adapter,
            admission,
            **dict(options, index_bits=index_bits, max_dynamic_calls=max_dynamic_calls),
        )
        checked.revalidate()
        admission_result = {"accepted": True, "report": checked.report()}
    except ValueError as error:
        admission_result = {"accepted": False, "error": str(error), "type": type(error).__name__}
    # Stateful callbacks and changing code are not silently assigned old evidence.
    if effects != [[asdict(adapter.project(call)) for call in block] for block in blocks]:
        raise ValueError("selected adapter changed its actual effect interpretation")
    if closure != _producer_sources(source, revision, (example_path,)):
        raise ValueError("selected producer dependencies changed during projection")
    payload = {
        "schema": "merlin.graph.loop_effect_projection.v1",
        "before": _reference(before),
        "after": _reference(after),
        "producer_revision": revision,
        "producer_source": closure,
        "blocks": effects,
        "objects": [asdict(obj) for obj in objects],
        "banks": [asdict(bank) for bank in banks],
        "external_windows": [asdict(window) for window in options["external_windows"]],
        "private_ownership": options["private_ownership"],
        "adapter_identities": {
            "configuration": adapter.configuration_identity,
            "numerical_policy": adapter.numerical_policy_identity,
            "qualification": adapter.qualification_identity,
        },
        "source_admission": admission_result,
        "python_projection_trusted": True,
        "adapter_independently_qualified": False,
        "pipeline_receipt_issued": False,
    }
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "before", "after", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("revision", "example"):
        parser.add_argument("--" + name, required=True)
    for name in ("iterations", "index-bits", "max-dynamic-calls"):
        parser.add_argument("--" + name, type=int, required=True)
    result = project(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "producer_revision": result["producer_revision"],
                "source_admission": result["source_admission"]["accepted"],
            }
        )
    )


if __name__ == "__main__":
    main()
