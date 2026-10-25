"""Retain independent actual source/provider declarations and original Kernel.

This intake executes no machine code or numerical oracle. Production source
readers/provider mappings run in a separate process at the pinned reference.
Physical compilation, when requested, retains the actual relocated Kernel and
declared effects. It does not execute machine code or qualify effect contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


def ref(path):
    path = path.resolve(strict=True)
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def save(path, value):
    def data(item):
        if type(item) is bytes:
            return dict(encoding="hex", hex=item.hex(), bytes=len(item))
        if type(item) is dict:
            return {key: data(value) for key, value in item.items()}
        if type(item) in (list, tuple):
            return [data(value) for value in item]
        return item

    path.write_text(json.dumps(data(value), indent=2, sort_keys=True) + "\n")


def loaded_files(reference):
    code = {}
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(reference):
            relative = path.relative_to(reference).as_posix()
            expected = subprocess.check_output(["git", "show", f"{REVISION}:{relative}"], cwd=reference)
            if path.read_bytes() != expected:
                raise ValueError("loaded reference source differs from pinned commit: " + relative)
        code[name] = ref(path)
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        row = line.split(maxsplit=5)
        if len(row) == 6 and row[5].startswith("/"):
            path = Path(row[5])
            if not path.is_file():
                raise ValueError("loaded file has no stable filesystem binding: " + row[5])
            libraries[str(path.resolve())] = ref(path)
    return dict(
        loaded_modules=code,
        mapped_host_files=libraries,
        python=ref(Path(sys.executable)),
        python_version=sys.version,
        configuration=ref(Path(sys.executable).parent.parent / "pyvenv.cfg"),
    )


def worker(args):
    root = args.reference.resolve(strict=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != REVISION:
        raise ValueError("different reference source revision")
    from test_multi_output_provider import MultiOutputProvider, original_model
    from test_multi_output_source import Case, source

    from merlin.compile.scheduling.multioutput_fusion import read_source_multioutput
    from merlin.transforms.certificates import ir_text

    output = args.destination.resolve()
    output.mkdir(parents=True, exist_ok=False)
    case = Case(tuple(args.shape), args.width, args.shared)
    module = source(case)
    generics = tuple(op for op in module.walk() if op.name == "linalg.generic")[:3]
    original = ir_text(module)
    ordinals = {op: ordinal for ordinal, op in enumerate(module.walk())}
    from xdsl.ir import BlockArgument

    def locator(value):
        if isinstance(value, BlockArgument):
            parent = value.owner.parent_op()
            return dict(kind="argument", owner_ordinal=ordinals[parent], index=value.index, type=str(value.type))
        return dict(kind="result", owner_ordinal=ordinals[value.owner], index=value.index, type=str(value.type))

    selected_model = original_model(output / "selected", case)
    provider = MultiOutputProvider(output / "provider", selected_model, (), case, grouped=True)
    logical = read_source_multioutput(module, generics)
    mapping = provider.lower_multioutput(logical)
    if mapping is None:
        raise ValueError("actual original provider refused this source")
    selected, adapter, _, _ = selected_model
    specification = json.loads(selected.fusion.declarations_json)
    before_calls = tuple(provider.lower(stage) for stage in (logical.producer, *logical.epilogues))
    if any(call is None for call in before_calls):
        raise ValueError("original per-stage provider declined")
    (output / "source.mlir").write_text(original)
    (output / "retained-source.mlir").write_text(ir_text(module))
    save(output / "original-provider-kernel.json", asdict(mapping.kernel))
    save(output / "original-provider-bindings.json", asdict(mapping.bindings))
    save(output / "original-provider-objects.json", [asdict(o) for o in mapping.objects])
    save(output / "selected-fusion-declarations.json", specification)
    save(output / "selected-storage-declarations.json", json.loads(selected.storage.declarations_json))
    save(output / "selected-record.json", selected.record())
    if args.physical:
        from merlin.compile.scheduling.multioutput_fusion import compile_source_multioutput

        configuration = provider.selection()
        law = configuration.multi_output.composition_law
        product = compile_source_multioutput(
            module, generics, mapping.kernel, mapping.objects, adapter, selected_model[2], selected_model[3],
            selected_contracts=selected, bindings=mapping.bindings, composition_law=law,
            budgets=configuration.budgets, private_ownership=True,
        )
        product.revalidate()
        save(output / "actual-output-kernel.json", asdict(product.compiled.kernel))
        save(output / "actual-output-conformance.json", product.conformance)
        save(output / "declared-composition-law.json", asdict(law))
        save(output / "actual-output-effects.json", [asdict(adapter.project(call)) for call in product.compiled.kernel.body])
        for index, data in enumerate(product.images()):
            (output / f"actual-image-{index}.bin").write_bytes(data)
    stage_records = []
    for index, (stage, call, baseline) in enumerate(
        zip((logical.producer, *logical.epilogues), mapping.kernel.body, before_calls, strict=True)
    ):
        effect = adapter.project(call)
        if effect is None or effect.elementwise is None:
            raise ValueError("actual original provider omitted scalar semantics")
        declaration = specification["elementwise"][call.instr]
        for name, data in (
            ("source", stage.scalar_ir),
            ("adapter", effect.elementwise.function_ir),
            ("selected", declaration["scalar_ir"].encode()),
        ):
            (output / f"stage-{index}-{name}.mlir").write_bytes(data)
        save(output / f"stage-{index}-effect.json", asdict(effect))
        save(output / f"stage-{index}-baseline-kernel.json", asdict(baseline.kernel))
        save(output / f"stage-{index}-baseline-bindings.json", asdict(baseline.bindings))
        stage_records.append(
            dict(
                index=index,
                instruction=call.instr,
                semantic_role=effect.semantic_key,
                input_indices=list(logical.input_indices[index]),
                source_operation_ordinal=ordinals[generics[index]],
                source_operands=[locator(value) for value in generics[index].operands],
                source_results=[locator(value) for value in generics[index].results],
                scalar_source=ref(output / f"stage-{index}-source.mlir"),
                adapter_declaration=ref(output / f"stage-{index}-adapter.mlir"),
                selected_declaration=ref(output / f"stage-{index}-selected.mlir"),
            )
        )
    if ir_text(module) != original:
        raise ValueError("source/provider intake changed the original module")
    save(output / "HOST-INTAKE.json", loaded_files(root))
    capture = dict(
        schema="merlin.graph-regions.declared-source-intake.v1",
        reference_revision=revision,
        reference_root=str(root),
        generator=ref(Path(__file__)),
        case=asdict(case),
        source=ref(output / "source.mlir"),
        retained_source=ref(output / "retained-source.mlir"),
        stages=stage_records,
        source_inputs=[str(value.type) for value in logical.inputs],
        source_outputs=[str(value.type) for value in logical.outputs],
        input_value_locators=[locator(value) for value in logical.inputs],
        output_value_locators=[locator(value) for value in logical.outputs],
        original_provider_kernel=ref(output / "original-provider-kernel.json"),
        host_intake=ref(output / "HOST-INTAKE.json"),
        physical_compilation="actual production compile_source_multioutput" if args.physical else "not run: source-only intake",
        machine_execution_run=False,
        numerical_execution_run=False,
        graph_hashes_used_as_proof=False,
        execution_contracts_proved=False,
        approved_pass_verification=False,
    )
    save(output / "CAPTURE.json", capture)


def capture(reference: Path, destination: Path, *, shape=(3, 47), width=1, shared=False, physical=False):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, (reference / "src", reference / "merlin/tests/infra"))),
        PYTHONDONTWRITEBYTECODE="1",
        MERLIN_VERIFY_LOG=str(destination.resolve() / "verify.log"),
    )
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--reference",
        str(reference.resolve()),
        "--destination",
        str(destination.resolve()),
        "--width",
        str(width),
        "--shape",
        *map(str, shape),
    ]
    if shared:
        command.append("--shared")
    if physical:
        command.append("--physical")
    result = subprocess.run(command, cwd=reference, env=env, capture_output=True, text=True, timeout=90)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save(
        destination.parent / (destination.name + "-process.json"),
        dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
    )
    result.check_returncode()
    return json.loads((destination / "CAPTURE.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(1, 2), default=1)
    parser.add_argument("--shape", type=int, nargs=2, default=(3, 47))
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--physical", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        print(
            json.dumps(
                capture(args.reference, args.destination, shape=tuple(args.shape), width=args.width,
                        shared=args.shared, physical=args.physical)
            )
        )


if __name__ == "__main__":
    main()
