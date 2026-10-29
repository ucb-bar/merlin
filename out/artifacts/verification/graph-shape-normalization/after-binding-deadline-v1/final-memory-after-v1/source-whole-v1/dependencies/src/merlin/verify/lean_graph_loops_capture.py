"""Capture actual counted-loop materialization from an explicitly pinned producer.

Run this file in an isolated subprocess whose PYTHONPATH selects the reference
production checkout. It emits evidence, never an accepted pipeline receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path


def _reference(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _producer_sources(source: Path, revision: str, extra: tuple[Path, ...]) -> list[dict]:
    """Bind every loaded producer module to both its bytes and selected Git blob."""
    paths = set(extra)
    for name, module in tuple(sys.modules.items()):
        if name != "merlin" and not name.startswith("merlin."):
            continue
        location = getattr(module, "__file__", None)
        if location is None:
            continue
        path = Path(location).resolve()
        if not path.is_relative_to(source / "src") or path.suffix != ".py":
            raise ValueError("loaded producer module escaped the explicitly selected source")
        paths.add(path)
    relative = [str(path.relative_to(source)) for path in sorted(paths)]
    tree = subprocess.run(
        ["git", "-C", str(source), "ls-tree", "-z", revision, "--", *relative],
        capture_output=True,
        check=True,
    ).stdout
    blobs = {}
    for entry in tree.split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            mode, kind, identity = metadata.split(b" ")
            if mode not in (b"100644", b"100755") or kind != b"blob":
                raise ValueError("producer dependency is not a regular selected source blob")
            blobs[name.decode()] = identity.decode()
    records = []
    for path, name in zip(sorted(paths), relative, strict=True):
        data = path.read_bytes()
        identity = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blobs.get(name) != identity:
            raise ValueError("actual producer dependency differs from the pinned revision")
        records.append(dict(_reference(path), git_blob=identity))
    return records


def _expression(value, forms) -> dict:
    const, variable, summation, scaled, select = forms
    if type(value) is const:
        if type(value.value) is not int:
            raise ValueError("index constants must be actual integers")
        return {"kind": "constant", "value": value.value}
    if type(value) is variable:
        return {"kind": "variable", "name": value.name}
    if type(value) is summation:
        return {"kind": "sum", "terms": [_expression(term, forms) for term in value.terms]}
    if type(value) is scaled:
        return {"kind": "scaled", "value": _expression(value.expr, forms), "factor": value.factor}
    if type(value) is select:
        return {
            "kind": "select",
            "name": value.var,
            "equals": value.equals,
            "yes": _expression(value.then, forms),
            "no": _expression(value.other, forms),
        }
    raise ValueError("unsupported counted-loop index expression")


def _call(command, forms, pointer, null, expr) -> dict:
    if any((command.unit, command.produces, command.awaits, command.stages, command.sets, command.assumes)):
        raise ValueError("this loop family requires separately projected storage/completion commands")
    arguments = []
    for name, value in command.args:
        if isinstance(value, expr):
            item = {"kind": "integer", "expression": _expression(value, forms)}
        elif type(value) is pointer:
            item = {"kind": "pointer", "space": value.tensor, "offset": _expression(value.offset, forms)}
        elif value is null:
            item = {"kind": "null"}
        else:
            raise ValueError("floating or opaque call arguments need a separate exact projection")
        arguments.append([name, item])
    return {"instruction": command.instr, "arguments": arguments}


def capture(*, source: Path, revision: str, example: str, output: Path, iterations: int, index_bits: int) -> dict:
    sys.dont_write_bytecode = True
    # Imports are intentionally delayed until the caller selected the producer.
    import merlin.compile.scheduling.periodic_effects as periodic
    import merlin.compile.scheduling.temporary_storage as storage
    import merlin.sched.ir.expr as expr_module
    import merlin.sched.ir.kernel as kernel_module
    from merlin.sched.ir import NULL, Call, Kernel, Loop, Ptr, concretize, instances
    from merlin.sched.ir.expr import Const as EConst
    from merlin.sched.ir.expr import Expr, Scaled, Select, Sum, Var

    source = source.resolve()
    observed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    if observed != revision:
        raise ValueError("producer revision differs from the selected immutable source")
    modules = [kernel_module, expr_module, periodic, storage]
    paths = [Path(module.__file__).resolve() for module in modules]
    if any(not path.is_relative_to(source / "src") for path in paths):
        raise ValueError("producer imports escaped the explicitly selected source checkout")
    example_path = (source / example).resolve()
    if not example_path.is_relative_to(source / "examples"):
        raise ValueError("capture requires an explicitly selected producer example")
    paths.append(example_path)
    before_sources = [_reference(path) for path in paths]
    spec = importlib.util.spec_from_file_location("graph_loop_actual_producer", example_path)
    if spec is None or spec.loader is None:
        raise ValueError("producer example cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel, objects, banks, adapter, admission, options, instruction_set = module.fixture(iterations)
    if type(kernel) is not Kernel or kernel.attrs or len(kernel.body) != 1 or type(kernel.body[0]) is not Loop:
        raise ValueError("capture supports one actual counted loop with no unknown kernel attributes")
    loop = kernel.body[0]
    if any(type(command) is not Call for command in loop.body):
        raise ValueError("nested loops and opaque statements are outside this first family")
    options = dict(options, index_bits=index_bits)
    checked = periodic.check_drained_period(kernel, objects, banks, adapter, admission, **options)
    # These are the EXACT production instantiations used by check_drained_period.
    blocks = [
        tuple(periodic._closed(command, loop.var, iteration) for command in loop.body)
        for iteration in range(loop.extent)
    ]
    flattened = tuple(command for block in blocks for command in block)
    realized = replace(kernel, body=flattened)
    independent = [(command.instr, concretize(command, environment)) for command, environment in instances(kernel)]
    actual = [(command.instr, concretize(command, {})) for command in flattened]
    if independent != actual:
        raise ValueError("production materialization differs from complete Kernel.instances order or operands")
    forms = EConst, Var, Sum, Scaled, Select
    before = {
        "schema": "merlin.graph.counted_kernel.v1",
        "name": kernel.name,
        "arguments": [asdict(argument) for argument in kernel.args],
        "attrs": [],
        "loop": {
            "variable": loop.var,
            "extent": loop.extent,
            "bits": index_bits,
            "body": [_call(command, forms, Ptr, NULL, Expr) for command in loop.body],
        },
    }
    after = {
        "schema": "merlin.graph.closed_kernel.v1",
        "name": realized.name,
        "arguments": [asdict(argument) for argument in realized.args],
        "attrs": [],
        "body": [_call(command, forms, Ptr, NULL, Expr) for command in realized.body],
    }
    effects = [[asdict(adapter.project(command)) for command in block] for block in blocks]
    emitted = checked.emit_c(instruction_set, symbol="graph_loop_generated")
    checked.revalidate()
    if [_reference(path) for path in paths] != before_sources:
        raise ValueError("producer source changed during actual capture")
    closure = _producer_sources(source, revision, (example_path,))
    output.mkdir(parents=True, exist_ok=False)
    products = {
        "before.json": json.dumps(before, sort_keys=True, indent=2) + "\n",
        "after.json": json.dumps(after, sort_keys=True, indent=2) + "\n",
        "before.kernel": kernel.text(),
        "after.kernel": realized.text(),
        "counted-emitted.c": emitted,
        "effects.json": json.dumps(
            {
                "blocks": effects,
                "objects": [asdict(obj) for obj in objects],
                "banks": [asdict(bank) for bank in banks],
                "admission": asdict(admission),
                "external_windows": [asdict(window) for window in options["external_windows"]],
                "private_ownership": options["private_ownership"],
                "adapter_identities": {
                    "configuration": adapter.configuration_identity,
                    "numerical_policy": adapter.numerical_policy_identity,
                    "qualification": adapter.qualification_identity,
                },
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        "production-check.json": json.dumps(checked.report(), sort_keys=True, indent=2) + "\n",
    }
    for name, text in products.items():
        (output / name).write_text(text)
    implementation = output / "capture-implementation.py"
    implementation.write_bytes(Path(__file__).read_bytes())
    result = {
        "schema": "merlin.graph.loop_capture.v1",
        "producer_revision": revision,
        "producer_source": closure,
        "capture_implementation": _reference(implementation),
        "products": {name: _reference(output / name) for name in products},
        "kernel_before_sha256": kernel.digest(),
        "kernel_after_sha256": realized.digest(),
        "iterations": loop.extent,
        "static_calls": len(loop.body),
        "dynamic_calls": len(flattened),
        "production_closed_and_instances_agree": True,
        "python_projection_trusted": True,
        "lean_checked": False,
        "external_output_coverage_proved": False,
        "renderer_semantics_proved": False,
        "pipeline_verified": False,
        "target_execution_qualified": False,
    }
    (output / "CAPTURE.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--example", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--index-bits", type=int, choices=(32, 64), default=64)
    args = parser.parse_args()
    result = capture(
        source=args.source,
        revision=args.revision,
        example=args.example,
        output=args.output,
        iterations=args.iterations,
        index_bits=args.index_bits,
    )
    print(
        json.dumps({key: result[key] for key in ("producer_revision", "iterations", "dynamic_calls", "lean_checked")})
    )


if __name__ == "__main__":
    main()
