"""Runtime source dimensions drive real reshapes and checked index products."""

import json
import signal
import subprocess
import sys

import numpy as np
import pytest
from test_execution_model_correspondence import printed
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.ir import Block, Region

from merlin.common.paths import repo_root
from merlin.frontends.execution_call_contracts import CallContracts
from merlin.frontends.execution_runtime_views import POLICY, RuntimeViewContracts
from merlin.llvmlower import toolchain
from merlin.llvmlower.abi import ScalarArg
from merlin.llvmlower.owned_result_program import save
from merlin.llvmlower.owned_results import compile_owned_results

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="existing native compiler required")
DYNAMIC = builtin.DYNAMIC_INDEX


@pytest.fixture(scope="module")
def programs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime-view-contracts")
    made = {}
    for kind in ("matrix", "singleton", "factor", "product"):
        scalar = kind == "product"
        source = builtin.TensorType(builtin.i64, [DYNAMIC, DYNAMIC] if kind == "matrix" else [DYNAMIC])
        types = [builtin.i64] * 3 if scalar else [source, builtin.i64, builtin.i64] if kind == "matrix" else [source]
        block = Block(arg_types=types)

        def append(op, role):
            block.add_op(op)
            return op.results[0] if op.results else None

        contracts = RuntimeViewContracts(append)
        if scalar:
            values = [append(arith.IndexCastOp(value, builtin.IndexType()), "source-index") for value in block.args]
            product = contracts.elements(values)
            number = append(arith.IndexCastOp(product, builtin.i64), "source-word")
            packed = append(tensor.FromElementsOp(number), "source-result")
            result = append(
                tensor.CollapseShapeOp.build(
                    operands=[packed],
                    result_types=[builtin.TensorType(builtin.i64, [])],
                    properties={"reassociation": builtin.ArrayAttr([])},
                ),
                "source-scalar-result",
            )
        else:
            shape = [DYNAMIC, DYNAMIC] if kind == "matrix" else [1 if kind == "singleton" else 2, DYNAMIC]
            target = builtin.TensorType(builtin.i64, shape)
            dims = {
                name: append(arith.IndexCastOp(value, builtin.IndexType()), "source-dimension")
                for name, value in zip(("rows", "cols"), block.args[1:])
            }
            result = CallContracts(append).view(
                block.args[0],
                target,
                {"shape": ["rows", "cols"]},
                {"tensor_views": POLICY},
                builtin.DictionaryAttr({}),
                dimensions=dims,
                role="caller-to-kernel-view" if kind == "matrix" else "kernel-to-caller-view",
            )
        block.add_op(func.ReturnOp(result))
        module = builtin.ModuleOp([func.FuncOp("forward", (types, [result.type]), Region(block))])
        module.verify()
        program = compile_owned_results(printed(module), workdir=root / kind, timeout=30)
        save(program)
        made[kind] = program
    return root, made


def test_dynamic_same_rank_geometry_and_empty_shapes_are_runtime_values(programs):
    _, programs = programs
    for source, target in (((2, 3), (3, 2)), ((3, 4), (2, 6)), ((0, 7), (0, 13)), ((2, 0), (5, 0))):
        x = (np.arange(np.prod(source), dtype=np.int64) + 2**53 + 1).reshape(source)
        before = x.tobytes()
        (actual,), observation = programs["matrix"].execute((x, *(ScalarArg(n, "i64") for n in target)))
        assert (actual.shape, actual.dtype, actual.tobytes()) == (target, x.dtype, before)
        assert x.tobytes() == before and observation["source_calls"] == 1


@pytest.mark.parametrize("kind,sizes", [("singleton", (0, 1, 7, 19)), ("factor", (0, 2, 6, 18))])
def test_result_unknown_axis_follows_checked_element_count(programs, kind, sizes):
    _, programs = programs
    for size in sizes:
        x = np.arange(size, dtype=np.int64) - (1 << 62)
        (actual,), _ = programs[kind].execute((x,))
        width = 1 if kind == "singleton" else 2
        assert actual.shape == (width, size // width) and actual.tobytes() == x.tobytes()


@pytest.mark.parametrize(
    "sizes", [(0, 1 << 62, 1 << 62), (1 << 62, 0, 1 << 62), (1 << 62, 1 << 62, 0), (3, 7, 11), ((1 << 62) - 1, 1, 1)]
)
def test_products_handle_empty_domains_before_unsigned_and_signed_overflow(programs, sizes):
    _, programs = programs
    (value,), _ = programs["product"].execute(tuple(ScalarArg(n, "i64") for n in sizes))
    assert value.shape == () and int(value) == int(sizes[0]) * sizes[1] * sizes[2]


@pytest.mark.parametrize(
    "kind,values,message",
    [
        ("matrix", ((2, 3), 2, 4), "element count mismatch"),
        ("matrix", ((2, 3), -2, -3), "negative extent"),
        ("factor", ((3,),), "element count mismatch"),
        ("product", (1 << 32, 1 << 32, 1), "element product overflow"),
        ("product", (1 << 62, 2, 1), "element product overflow"),
        ("product", (-1, 0, 7), "negative extent"),
    ],
)
def test_bad_runtime_contracts_abort_in_actual_source_before_results(programs, tmp_path, kind, values, message):
    _, programs = programs
    args = (
        [np.asarray(n, np.int64) for n in values]
        if kind == "product"
        else [
            np.arange(np.prod(values[0]), dtype=np.int64).reshape(values[0]),
            *(np.asarray(n, np.int64) for n in values[1:]),
        ]
    )
    np.savez(tmp_path / "inputs.npz", **{f"in{i}": value for i, value in enumerate(args)})
    script = tmp_path / "invoke.py"
    script.write_text("""import sys,resource,ctypes
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
arguments=tuple(ScalarArg(int(value),'i64') if value.shape==() else value for value in values)
Path(sys.argv[4]).write_text('invoking')
outputs,_=program.execute(arguments)
np.savez(sys.argv[5],**{f'out{i}':value for i,value in enumerate(outputs)})
""")
    ready, output = tmp_path / "ready", tmp_path / "outputs.npz"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            str(repo_root()),
            str(programs[kind].source.parent),
            str(tmp_path / "inputs.npz"),
            str(ready),
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert ready.read_text() == "invoking"
    assert process.returncode == -signal.SIGABRT and message in process.stdout + process.stderr
    assert not output.exists()
    (tmp_path / "outcome.json").write_text(
        json.dumps(
            {
                "returncode": process.returncode,
                "source_guard": message,
                "stdout": process.stdout,
                "stderr": process.stderr,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("shape", [(DYNAMIC, DYNAMIC), (0, DYNAMIC)])
def test_ambiguous_result_axes_cannot_be_invented_from_a_flat_count(shape):
    value = Block(arg_types=[builtin.TensorType(builtin.i64, [DYNAMIC])]).args[0]
    with pytest.raises(ValueError, match="ambiguous"):
        RuntimeViewContracts(lambda *args: pytest.fail("emitted before refusing ambiguous geometry")).adapt(
            value, builtin.TensorType(builtin.i64, shape), [], {}, input_view=False
        )
