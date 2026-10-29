"""Add selected admission data to the unchanged v1 actual effect projection.

This isolated producer process reopens the pinned declaration and recomputes its
actual effects. It does not change the delivered v1 projection implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__:
    from .lean_graph_loops_capture import _producer_sources
    from .lean_graph_loops_effects import project as project_v1
else:
    from lean_graph_loops_capture import _producer_sources
    from lean_graph_loops_effects import project as project_v1


def project(**arguments):
    sys.dont_write_bytecode = True
    from merlin.sched.ir import parse_kernel

    output = arguments["output"]
    base = project_v1(**dict(arguments, output=output.with_suffix(".v1.json")))
    source = arguments["source"].resolve()
    example = (source / arguments["example"]).resolve()
    spec = importlib.util.spec_from_file_location("graph_capacity_admission", example)
    if spec is None or spec.loader is None:
        raise ValueError("selected capacity declaration cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _declared, objects, banks, adapter, admission, options, _iset = module.fixture(arguments["iterations"])
    actual = parse_kernel(arguments["after"].read_bytes().decode())
    static_calls = len(base["blocks"][0])
    expected = [
        [asdict(adapter.project(call)) for call in actual.body[index : index + static_calls]]
        for index in range(0, len(actual.body), static_calls)
    ]
    identities = {
        "configuration": adapter.configuration_identity,
        "numerical_policy": adapter.numerical_policy_identity,
        "qualification": adapter.qualification_identity,
    }
    if (
        expected != base["blocks"]
        or identities != base["adapter_identities"]
        or [asdict(obj) for obj in objects] != base["objects"]
        or [asdict(bank) for bank in banks] != base["banks"]
        or [asdict(window) for window in options["external_windows"]] != base["external_windows"]
        or options["private_ownership"] is not base["private_ownership"]
    ):
        raise ValueError("reopened admission changed its actual effect/declaration context")
    selected = json.loads(json.dumps(asdict(admission)))
    if base["source_admission"]["accepted"] and selected != json.loads(
        json.dumps(base["source_admission"]["report"]["admission"])
    ):
        raise ValueError("reopened admission differs from actual production checking")
    closure = _producer_sources(source, arguments["revision"], (example,))
    if closure != base["producer_source"]:
        raise ValueError("admission/source dependency context changed during reprojection")
    result = dict(base, schema="merlin.graph.loop_capacity_projection.v1", admission=selected)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


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
                "pending_capacity": result["admission"]["max_pending"],
                "source_admission": result["source_admission"]["accepted"],
            }
        )
    )


if __name__ == "__main__":
    main()
