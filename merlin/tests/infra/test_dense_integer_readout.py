"""Exact integer readouts retain i32 reductions and verify actual store modes."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_compact_dense import encode, project
from test_dense_source import compile_source, contraction
from test_selected_dense_contracts import inputs, schedule

from merlin.compile.scheduling.block_schedule import Store
from merlin.compile.scheduling.dense_source import DenseSourceUnavailable
from merlin.frontends.linalg_mlir import parse_mlir_text

QUALIFICATION = "neutral-independent-i32-clamp-i8-readout-v1"
SCALAR_IR = """module {func.func @readout(%x:i32)->i8 {
 %zero=arith.constant 0:i32
 %high=arith.constant 127:i32
 %positive=arith.maxsi %x,%zero:i32
 %bounded=arith.minsi %positive,%high:i32
 %out=arith.trunci %bounded:i32 to i8
 func.return %out:i8
}}"""


def readout_inputs(tmp_path, *, mutate=None, **kwargs):
    def declare(facts, software, dense):
        facts["output_bits"] = 8
        facts["codes"].append(5)
        narrow = deepcopy(software["isa"]["instructions"][-1])
        narrow["name"] = "store_i8"
        narrow["encoding"][0]["fixed"]["fact"][-1] = 5
        software["isa"]["instructions"].append(narrow)
        dense["output_readout"] = dict(
            schema="merlin.dense_integer_readout.v1",
            scalar_ir=SCALAR_IR,
            output=dict(dtype="i8", bits={"fact": ["facts", "output_bits"]}),
            instruction="store_i8",
        )
        if mutate:
            mutate(facts, software, dense)

    selected, problem, storage, adapter = inputs(tmp_path, mutate=declare, **kwargs)
    readout = selected.dense.output_readout

    def encode_readout(op):
        if type(op) is Store and op.readout is not None:
            assert op.readout == readout.identity
            return replace(encode(replace(op, readout=None)), instr=readout.instruction_name)
        return encode(op)

    def project_readout(call):
        if call.instr == readout.instruction_name:
            return replace(project(replace(call, instr="store")), readout=readout.identity)
        return project(call)

    return selected, problem, storage, replace(adapter, encode=encode_readout, project=project_readout)


def readout_source(shape, dtypes, *, extra_use=False, altered=False):
    m, k, n = shape
    x, y, z = dtypes
    text = f"""module {{func.func @forward(%a:tensor<{m}x{k}x{x}>,%b:tensor<{k}x{n}x{y}>)
    -> tensor<{m}x{n}xi8> {{
    {contraction(shape, dtypes)}
    %e=tensor.empty():tensor<{m}x{n}xi8>
    %y=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
    iterator_types=["parallel","parallel"]}} ins(%r:tensor<{m}x{n}x{z}>)
    outs(%e:tensor<{m}x{n}xi8>) {{^bb0(%x:i32,%unused:i8):
      %zero=arith.constant 0:i32
      %high=arith.constant {126 if altered else 127}:i32
      %positive=arith.maxsi %x,%zero:i32
      %bounded=arith.minsi %positive,%high:i32
      %out=arith.trunci %bounded:i32 to i8
      linalg.yield %out:i8
    }}->tensor<{m}x{n}xi8>
    {"%second=tensor.cast %r:tensor<" + str(m) + "x" + str(n) + "xi32> to tensor<?x?xi32>" if extra_use else ""}
    return %y:tensor<{m}x{n}xi8>
    }}}}"""
    module = parse_mlir_text(text)
    ops = tuple(op for op in module.walk() if op.name == "linalg.generic")
    return module, ops[0], ops[1]


@pytest.mark.parametrize(
    "tile,widths,shape,shared",
    [
        ((2, 3, 4), (1, 1, 4), (5, 14, 13), False),
        ((3, 2, 5), (2, 2, 4), (7, 9, 16), True),
    ],
)
def test_output_width_is_separate_from_complete_accumulator(tmp_path, tile, widths, shape, shared):
    model = readout_inputs(tmp_path, tile=tile, widths=widths, shape=shape, shared=shared)
    readout = model[0].dense.output_readout
    baseline = schedule(model)
    result = schedule(model, output_readout=readout, readout_qualification_identity=QUALIFICATION)
    assert baseline.kernel.args[-1].dtype == "i32"
    assert result.kernel.args[-1].dtype == "i8"
    assert model[1].accumulator_bytes == 4
    assert result.realization["padded_high_water_bytes"] == baseline.realization["padded_high_water_bytes"]
    assert result.realization["requested_store_bytes"] * 4 == baseline.realization["requested_store_bytes"]
    assert result.realization["requested_load_bytes"] == baseline.realization["requested_load_bytes"]
    result.revalidate()


def test_actual_store_must_reconstruct_readout_identity(tmp_path):
    model = readout_inputs(tmp_path)
    selected, _, _, adapter = model
    actual_project = adapter.project
    corrupted = replace(
        adapter,
        project=lambda c: replace(actual_project(c), readout=None) if c.instr == "store_i8" else actual_project(c),
    )
    with pytest.raises(ValueError, match="readout"):
        schedule(
            model,
            adapter=corrupted,
            output_readout=selected.dense.output_readout,
            readout_qualification_identity=QUALIFICATION,
        )


@pytest.mark.parametrize("qualification", [None, "", " "])
def test_readout_declaration_alone_never_qualifies_execution(tmp_path, qualification):
    model = readout_inputs(tmp_path)
    with pytest.raises(ValueError, match="qualification"):
        schedule(model, output_readout=model[0].dense.output_readout, readout_qualification_identity=qualification)


@pytest.mark.parametrize("mutation", ["output_width", "float", "effects", "instruction"])
def test_readout_declaration_requires_exact_integer_formats_and_effects(tmp_path, mutation):
    def change(facts, software, dense):
        if mutation == "output_width":
            facts["output_bits"] = 32
        elif mutation == "float":
            dense["output_readout"]["scalar_ir"] = (
                "module {func.func @r(%x:f32)->i8 { %r=arith.fptosi %x:f32 to i8 func.return %r:i8 }}"
            )
        elif mutation == "effects":
            software["isa"]["instructions"][-1]["effects"]["reads"].append("input_lhs")
        else:
            dense["output_readout"]["instruction"] = "store"

    with pytest.raises(ValueError, match="readout|integer"):
        readout_inputs(tmp_path, mutate=change)


@pytest.mark.parametrize("mutation", ["remove", "replace", "nested"])
def test_compiled_readout_cannot_be_removed_or_replaced_after_checking(tmp_path, mutation):
    model = readout_inputs(tmp_path)
    readout = model[0].dense.output_readout
    result = schedule(model, output_readout=readout, readout_qualification_identity=QUALIFICATION)
    if mutation == "remove":
        result = replace(result, output_readout=None, checked_readout=None)
    elif mutation == "replace":
        changed = replace(readout, scalar_ir=SCALAR_IR.replace("127", "126"))
        result = replace(result, output_readout=changed, checked_readout=changed.identity)
    else:
        object.__setattr__(readout, "output_dtype", "i32")
    with pytest.raises(ValueError, match="readout|checked"):
        result.revalidate()


def test_arbitrary_qualification_text_cannot_impersonate_a_different_readout(tmp_path):
    model = readout_inputs(tmp_path)
    original = model[0].dense.output_readout
    changed = replace(original, scalar_ir=SCALAR_IR.replace("127", "126"))
    result = schedule(
        model, output_readout=original, readout_qualification_identity=f"original:readout={changed.identity}:reviewed"
    )
    with pytest.raises(ValueError, match="checked"):
        replace(result, output_readout=changed, checked_readout=changed.identity).revalidate()


def test_original_source_readout_publishes_final_type_and_keeps_contraction(tmp_path):
    model = readout_inputs(tmp_path)
    module, dense, output = readout_source((5, 14, 13), model[0].dense.dtypes)
    result = compile_source(model, module, dense, output_operation=output, readout_qualification_identity=QUALIFICATION)
    assert result.operation is dense and result.operations == (dense, output)
    assert result.output_dtype == "i8" and result.output_readout is model[0].dense.output_readout
    assert result.compiled.kernel.args[-1].dtype == "i8"
    result.revalidate()


@pytest.mark.parametrize("extra_use,altered", [(True, False), (False, True)])
def test_unsupported_original_composition_preserves_dense_only_route(tmp_path, extra_use, altered):
    model = readout_inputs(tmp_path)
    module, dense, output = readout_source((5, 14, 13), model[0].dense.dtypes, extra_use=extra_use, altered=altered)
    with pytest.raises(DenseSourceUnavailable, match="consumer|differs"):
        compile_source(model, module, dense, output_operation=output, readout_qualification_identity=QUALIFICATION)
    baseline = compile_source(model, module, dense)
    assert baseline.output_dtype == "i32" and baseline.operations == (dense,)
