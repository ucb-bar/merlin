"""Execute the real plural wrapper with its checked native lifetime boundary.

The reused public fixture declares neutral C implementation premises. This is
ordinary host execution and ownership evidence, not target qualification.
"""

import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_plural_fresh_tensor_writer import contract
from test_plural_fresh_tensor_writer import fixture as tensor_fixture
from test_plural_fresh_tensor_writer import test_plural_results_native_against_separate_calls_and_reference as run_pair
from xdsl.dialects import llvm

from merlin.llvmlower import (
    abi,
    borrowed_buffer_calls,
    dealloc_lifetime,
    fresh_tensor_writer,
    llvm_dialect_product,
    pipeline,
)
from merlin.llvmlower.borrowed_buffer_calls import CHECKPOINT, REPORT, RESULTS_SCHEMA, from_writers, verify_native
from merlin.llvmlower.fresh_tensor_writer import prove_lowered_writer_results
from merlin.llvmlower.llvm_dialect_product import verify_llvm_dialect_product
from merlin.verify import host_call_alignment
from merlin.verify.host_call_alignment import import_host_entry
from merlin.xdsl_dialects._common import text


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


@pytest.mark.parametrize("mutation", ["report", "checkpoint", "effects"])
def test_real_native_observer_cannot_refresh_changed_products_or_original_effects(
    tmp_path, upstream_host_tools, mutation
):
    from merlin.llvmlower.fresh_tensor_writer import rewrite_fresh_tensor_writers

    writer, module = contract(), tensor_fixture()
    rewrite_fresh_tensor_writers(module, (writer,))
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text(text(module, generic=True))
    reached = []

    def observer(paths, initial):
        assert initial
        reached.append(True)
        if mutation == "effects":
            object.__setattr__(writer.borrowed_effects, "qualification_identity", "different-valid-original-law")
        else:
            name = REPORT if mutation == "report" else CHECKPOINT
            path = next(path for path in paths if path.name == name)
            path.write_text(path.read_text() + "\n ")

    selected = from_writers(prepared, (writer,), (), lambda: None, observer)
    with pytest.raises(ValueError, match="changed"):
        pipeline.lower_to_llvm_ir(
            prepared.read_text(), workdir=tmp_path / "lower", vectorize=False, borrowed_buffer_calls=selected
        )
    assert reached == [True]
    recipe = json.loads((tmp_path / "lower" / "lowering_recipe.json").read_text())
    assert recipe["status"] == "invoked" and "returned_llvm_ir" not in recipe


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_real_plural_model_checked_boundary_and_every_returned_state(
    tmp_path, monkeypatch, upstream_host_tools, optimization
):
    original_lower, original_load = pipeline.lower_to_llvm_ir, abi.HostModel.load
    history, selections, observations = [], [], {}
    owner_paths = (
        Path(__file__),
        Path(run_pair.__code__.co_filename),
        *(
            Path(module.__file__)
            for module in (
                abi,
                pipeline,
                borrowed_buffer_calls,
                dealloc_lifetime,
                fresh_tensor_writer,
                llvm_dialect_product,
                host_call_alignment,
            )
        ),
    )
    original_owners = [pin(path) for path in owner_paths]
    adapter_pin = None

    def original():
        for item in (*original_owners, adapter_pin):
            assert pin(item["path"]) == item

    def checked_lower(source, **kwargs):
        nonlocal adapter_pin
        if adapter_pin is None:
            adapter_pin = pin(tmp_path / "adapter.c")
        original()
        if Path(kwargs["workdir"]).name != "plural-lower":
            result = original_lower(source, **kwargs)
            original()
            return result
        selected_path = tmp_path / "plural.mlir"
        assert selected_path.read_text() == source
        originals = tuple((Path(item["path"]), item["sha256"]) for item in (*original_owners, adapter_pin))

        def observe(paths, initial):
            for path in paths:
                current = pin(path)
                if initial:
                    assert str(path) not in observations
                    observations[str(path)] = current
                else:
                    assert observations[str(path)] == current

        selected = from_writers(selected_path, (contract(),), originals, original, observe)
        assert selected.record()["schema"] == RESULTS_SCHEMA
        actual = {}
        result = original_lower(
            source, borrowed_buffer_calls=selected, retain_llvm_dialect=True, lowering_selection=actual, **kwargs
        )
        product = verify_llvm_dialect_product(actual["llvm_dialect_product"]["path"], llvm_ir=result)
        assert Path(product["llvm_dialect"]["path"]).is_file()
        selections.append((selected, source, Path(kwargs["workdir"])))
        return result

    class RecordingModel:
        def __init__(self, image, path):
            self.image, self.lib, self.path = image, image.lib, Path(path)
            assert image.image_sha256 == pin(path)["sha256"]
            self.iteration = 0

        def __call__(self, descriptors):
            ctypes_types = (
                ctypes.c_int8,
                ctypes.c_int16,
                ctypes.c_int8,
                ctypes.c_int32,
                ctypes.c_int16,
                ctypes.c_int32,
            )
            views, owners = [], []
            for (address, shape), dtype in zip(descriptors, ctypes_types, strict=True):
                count = int(np.prod(shape))
                owners.append(
                    np.ctypeslib.as_array((dtype * (count + 4)).from_address(address - 2 * ctypes.sizeof(dtype)))
                )
                views.append(owners[-1][2:-2].reshape(shape))
            inputs = [value.copy() for value in views[:2]]
            self.image(descriptors)
            self.lib.actual_calls.restype = ctypes.c_int
            archive = tmp_path / f"{self.path.stem}-call-{self.iteration}.npz"
            np.savez(
                archive,
                **{f"input{i}": value for i, value in enumerate(inputs)},
                **{f"after{i}": value.copy() for i, value in enumerate(views)},
                **{f"guard{i}": np.concatenate((owner[:2], owner[-2:])) for i, owner in enumerate(owners)},
            )
            history.append(
                dict(
                    arm=self.path.stem,
                    iteration=self.iteration,
                    arrays=pin(archive),
                    image=pin(self.path),
                    actual_calls=self.lib.actual_calls(),
                )
            )
            self.iteration += 1

    def recording_load(path, *args, **kwargs):
        return RecordingModel(
            original_load(path, *args, image_policy=abi.PrivateHostImagePolicy(tmp_path), **kwargs), path
        )

    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", checked_lower)
    monkeypatch.setattr(abi.HostModel, "load", staticmethod(recording_load))
    run_pair(tmp_path, "ranked_c", optimization, upstream_host_tools)
    assert len(history) == 6 and len(selections) == 1
    selected, source, work = selections[0]
    report = verify_native(selected, source, work)
    assert report["schema"] == "merlin.borrowed_buffer_lifetime.v2" and report["status"] == "checked"
    assert report["accepted_symbols"] == ["borrowed", "writer"]
    assert (work / CHECKPOINT).is_file() and (work / REPORT).is_file()
    artifacts = {
        role: dict(path=item["path"], sha256=item["sha256"], size_bytes=item["bytes"])
        for role, item in (("llvm_ir", pin(tmp_path / "plural.ll")), ("object", pin(tmp_path / "plural.o")))
    }
    module, imported = import_host_entry(
        artifacts,
        entry="forward",
        include_functions=("writer",),
        diagnostics=tmp_path / "actual-LLVM",
        max_ir_bytes=1 << 20,
    )
    module.verify()
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, llvm.FuncOp)}
    call = next(
        op
        for op in functions["forward"].walk()
        if isinstance(op, llvm.CallOp) and op.callee is not None and op.callee.root_reference.data == "writer"
    )
    proof = prove_lowered_writer_results(
        functions["writer"],
        call,
        contract(),
        argument_shapes=((2, 3), (3,), (2, 3), (3,)),
        element_bytes=(1, 2, 4, 2),
        allocator_alignment=16,
    )
    assert proof["actual_output_parameter_starts"] == [24, 29]
    (tmp_path / "actual-output-owners.json").write_text(json.dumps(proof, indent=2))
    files = [
        *tmp_path.glob("*.mlir"),
        *tmp_path.glob("*.ll"),
        *tmp_path.glob("*.o"),
        *tmp_path.glob("*.so"),
        tmp_path / "adapter.c",
        tmp_path / "actual-output-owners.json",
    ]
    products = [pin(path) for path in files]
    # Final callback-free reopen after all original validation callbacks.
    for item in [*original_owners, adapter_pin, *products, *observations.values(), *(row["arrays"] for row in history)]:
        assert pin(item["path"]) == item
    (tmp_path / "pair.json").write_text(
        json.dumps(
            dict(
                schema="neutral_plural_borrowed_native_pair_v1",
                optimization=optimization,
                source_owners=original_owners,
                products=products,
                native_observations=list(observations.values()),
                history=history,
                actual_llvm_import=imported,
                scope="neutral host outputs, current native lifetime and conditional output-owner proof only",
            ),
            indent=2,
        )
    )
