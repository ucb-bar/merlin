"""Independently constructed source slices retain runtime coordinate guards."""

import json
import signal
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
from test_execution_model_correspondence import printed
from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.ir import Block, Region

from merlin.common.paths import repo_root
from merlin.frontends.execution_call_views import validate
from merlin.frontends.execution_source_coordinates import SourceCoordinates
from merlin.llvmlower import toolchain
from merlin.llvmlower.abi import ScalarArg
from merlin.llvmlower.owned_result_program import save
from merlin.llvmlower.owned_results import compile_owned_results

DYNAMIC = builtin.DYNAMIC_INDEX
NAMES = ("rows", "cols", "s0", "s1", "origin")


@pytest.mark.parametrize(
    "profile,policy",
    (
        (None, "runtime-logical-coordinates-v1"),
        ("t2m.execution-checked-call-views.v1", "runtime-logical-coordinates-v1"),
        ("t2m.execution-checked-call-views.v2", "runtime-logical-coordinates-v1"),
        ("t2m.execution-checked-call-views.v3", None),
        ("t2m.execution-checked-call-views.v3", "physical-alias"),
    ),
)
def test_runtime_coordinates_require_the_known_explicit_policy_and_profile(profile, policy):
    bindings = {"calls": [{"source_view_policy": policy, "source_tensor_views": {"view": {}}}]}
    if profile is not None:
        bindings["execution_view_binding_profile"] = profile
    with pytest.raises(ValueError):
        validate(bindings)


def source_program(*, damage=None):
    typ = builtin.TensorType(builtin.i64, [DYNAMIC, DYNAMIC])
    types = [typ, *([builtin.i64] * len(NAMES))]
    block = Block(arg_types=types)

    def emit(operation, role="source"):
        block.add_op(operation)
        return operation.results[0] if operation.results else None

    def constant(number):
        return emit(arith.ConstantOp.from_int_and_width(number, builtin.IndexType()))

    zero, one, two, three = [constant(n) for n in range(4)]
    rows = emit(tensor.DimOp(block.args[0], zero))
    cols = emit(tensor.DimOp(block.args[0], one))

    def ceil_div(number, divisor):
        quotient = emit(arith.DivUIOp(number, divisor))
        remainder = emit(arith.RemUIOp(number, divisor))
        nonzero = emit(arith.CmpiOp(remainder, zero, "ne"))
        increment = emit(arith.SelectOp(nonzero, one, zero))
        return emit(arith.AddiOp(quotient, increment))

    clipped = emit(arith.SelectOp(emit(arith.CmpiOp(cols, two, "ult")), cols, two))
    sizes = [ceil_div(rows, two), ceil_div(emit(arith.SubiOp(cols, clipped)), three)]
    predicate = emit(arith.ConstantOp.from_int_and_width(1, builtin.i1))
    for size in sizes:
        positive = emit(arith.CmpiOp(size, zero, "sgt"))
        predicate = emit(arith.AndIOp(predicate, positive))
    if damage == "condition":
        predicate = emit(arith.XOrIOp(predicate, emit(arith.ConstantOp.from_int_and_width(1, builtin.i1))))
    fields = {"static_offsets": [0, DYNAMIC], "static_sizes": [DYNAMIC, DYNAMIC], "static_strides": [2, 3]}
    extraction = tensor.ExtractSliceOp.build(
        operands=[block.args[0], [clipped], sizes, []],
        result_types=[typ],
        properties={name: builtin.DenseArrayBase.from_list(builtin.i64, value) for name, value in fields.items()},
    )
    empty = tensor.EmptyOp(sizes, typ)
    positive_block = Block([extraction, scf.YieldOp(extraction.result)])
    empty_block = Block([empty, scf.YieldOp(empty.tensor)])
    conditional = scf.IfOp(predicate, [typ], Region(positive_block), Region(empty_block))
    view = emit(conditional)
    config = {"root": "state", "source_arg": 1, "shape": ["rows", "cols"], "strides": ["s0", "s1"], "offset": "origin"}
    record = {
        "view": {
            "binding": config,
            "root_schema_operand": "owner",
            "root_type": str(typ),
            "view_type": str(typ),
            "layout": {
                "schema": "t2m.runtime-source-coordinates.v1",
                "root_order": "logical-dense-row-major",
                "steps": [
                    {
                        "op": "scf.if",
                        "result_type": str(typ),
                        "offsets": fields["static_offsets"],
                        "sizes": fields["static_sizes"],
                        "strides": fields["static_strides"],
                    }
                ],
            },
        }
    }
    provenance = {
        "source_argument_binding": {"bindings": [None, {"kind": "pointer", "allocation": "state", "offset": "origin"}]}
    }
    if damage == "record":
        record = deepcopy(record)
        record["view"]["layout"]["steps"][0]["strides"] = [1, 3]
    if damage == "branch":
        empty_block.insert_op_before(arith.ConstantOp.from_int_and_width(7, builtin.i64), empty)
    SourceCoordinates(emit).check_views(
        record,
        {"state": "owner"},
        {"owner": block.args[0], "view": view},
        dict(zip(NAMES, block.args[1:], strict=True)),
        provenance,
    )
    block.add_op(func.ReturnOp(view))
    module = builtin.ModuleOp([func.FuncOp("forward", (types, [typ]), Region(block))])
    module.verify()
    return printed(module)


@pytest.fixture(scope="module")
def programs(tmp_path_factory):
    if not toolchain.available():
        pytest.skip("existing native compiler required")
    root = tmp_path_factory.mktemp("source-coordinates")
    made = {}
    for name, damage in (("valid", None), ("condition", "condition")):
        program = compile_owned_results(source_program(damage=damage), workdir=root / name, timeout=30)
        save(program)
        made[name] = program
    return made


def test_runtime_logical_slices_keep_exact_words_and_empty_boundary_shapes(programs):
    for shape in ((2, 2), (9, 17), (1, 1), (0, 0), (0, 5), (5, 0)):
        original = (np.arange(np.prod(shape), dtype=np.int64) + (1 << 56) + 1).reshape(shape)
        wanted = original[::2, 2::3]
        coordinates = (*wanted.shape, 2 * shape[1], 3, min(2, shape[1]))
        (actual,), _ = programs["valid"].execute((original, *(ScalarArg(n, "i64") for n in coordinates)))
        assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())


@pytest.mark.parametrize("damage", ("record", "branch"))
def test_records_and_branch_effects_cannot_supply_coordinate_semantics(damage):
    with pytest.raises(ValueError):
        source_program(damage=damage)


@pytest.mark.parametrize(
    "kind,coordinates,message",
    (
        ("valid", (2, 2, 8, 3, 2), "kernel argument disagrees"),
        ("valid", (-1, 1, 8, 3, 2), "kernel argument disagrees"),
        ("valid", (2, 1, 7, 3, 2), "kernel argument disagrees"),
        ("valid", (2, 1, 8, 3, 0), "kernel argument disagrees"),
        ("condition", (2, 1, 8, 3, 2), "empty-slice branch differs"),
    ),
)
def test_false_runtime_coordinates_abort_before_returning_values(programs, tmp_path, kind, coordinates, message):
    inputs = tmp_path / "inputs.npz"
    np.savez(
        inputs,
        in0=np.arange(12, dtype=np.int64).reshape(3, 4),
        **{f"in{i + 1}": np.asarray(n, np.int64) for i, n in enumerate(coordinates)},
    )
    script = tmp_path / "invoke.py"
    script.write_text("""import sys,ctypes,resource
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1]+'/src')
from merlin.llvmlower.owned_result_program import load
from merlin.llvmlower.abi import ScalarArg
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
libc=ctypes.CDLL(None)
libc.setbuf.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
libc.setbuf(ctypes.c_void_p.in_dll(libc,'stdout'),None)
program=load(Path(sys.argv[2])/'owned-result-program.json')
with np.load(sys.argv[3],allow_pickle=False) as data:
 values=[data[f'in{i}'].copy() for i in range(len(data.files))]
Path(sys.argv[4]).write_text('invoking')
outputs,_=program.execute((values[0],*(ScalarArg(int(v),'i64') for v in values[1:])))
np.savez(sys.argv[5],**{f'out{i}':v for i,v in enumerate(outputs)})
""")
    ready, output = tmp_path / "ready", tmp_path / "outputs.npz"
    done = subprocess.run(
        [
            sys.executable,
            str(script),
            str(repo_root()),
            str(programs[kind].source.parent),
            str(inputs),
            str(ready),
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert ready.read_text() == "invoking"
    assert done.returncode == -signal.SIGABRT and message in done.stdout + done.stderr
    assert not output.exists()
    (tmp_path / "outcome.json").write_text(
        json.dumps(
            {"returncode": done.returncode, "source_guard": message, "stdout": done.stdout, "stderr": done.stderr},
            indent=2,
        )
        + "\n"
    )
