"""Provisional TorchAO materialization checks and independent CPU/native comparisons.

These tests do not establish captured-call execution refinement or approve a numerical contract.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_torchao_affine import _module_text

from merlin.common.paths import merlin_dir, repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.torchao_affine import lower_torchao_affine_quant
from merlin.verify.lean import differential as diff
from merlin.verify.lean import local_passes
from merlin.verify.lean.export import ExportError, export_file
from merlin.xdsl_dialects._common import text as to_text


def actual(source: str) -> tuple[str, str, int]:
    module = parse_mlir_text(source)
    before = to_text(module, generic=True)
    count = lower_torchao_affine_quant(module)
    module.verify()
    return before, to_text(module, generic=True), count


def exported(source: str, work: Path, tools) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    mlir, path = work / "subject.mlir", work / "subject.json"
    mlir.write_text(source)
    export_file(mlir, path, backend="mlir", m2m_python=tools.compiler_python)
    return path


def checked(before: str, after: str, work: Path, tools) -> dict:
    paths = [exported(s, work / side, tools) for s, side in ((before, "before"), (after, "after"))]
    result = subprocess.run(
        [
            str(repo_root() / "out/build/lean/build/bin/merlin-lean-local-passes"),
            "torchao-affine",
            *(str(p) for p in paths),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=360,
    )
    answer = json.loads(result.stdout)
    (work / "checker.json").write_text(json.dumps(answer, indent=2) + "\n")
    assert answer["status"] == "materialized_contract_only"
    assert not answer["source_call_refinement_proved"]
    return answer


@pytest.mark.parametrize("shape", [(2, 4), (1, 2, 4)])
def test_actual_pass_complete_materialization(tmp_path, fidelity_tools, shape):
    before, after, count = actual(_module_text(shape))
    assert count == 2
    answer = checked(before, after, tmp_path, fidelity_tools)
    assert answer["accepted"] and answer["scalar_bodies"] == [True]


def test_complete_program_calls_and_unrelated_functions(tmp_path, fidelity_tools):
    source = (
        _module_text((2, 4))
        .replace("func.func @forward", "func.func private @body")
        .replace(
            "\n}",
            "\n func.func @forward(%x: tensor<2x4xf32>) -> (tensor<2xf32>, tensor<2x4xi8>) {"
            "\n %s, %q = func.call @body(%x) : (tensor<2x4xf32>) -> (tensor<2xf32>, tensor<2x4xi8>)"
            "\n return %s, %q : tensor<2xf32>, tensor<2x4xi8> }"
            "\n func.func @unrelated() -> i32 { %c = arith.constant 7 : i32"
            " %d = arith.constant 7 : i32 %sum = arith.addi %c, %d : i32 return %sum : i32 }\n}",
        )
    )
    before, after, count = actual(source)
    assert count == 2 and checked(before, after, tmp_path / "baseline", fidelity_tools)["accepted"]
    wrong = after.replace("value = 7 : i32", "value = 8 : i32")
    assert wrong != after
    assert not checked(before, wrong, tmp_path / "unrelated", fidelity_tools)["accepted"]


def test_compiler_valid_mutations_are_rejected_by_lean(tmp_path, fidelity_tools):
    before, after, _ = actual(_module_text((2, 4)))
    assert checked(before, after, tmp_path / "baseline", fidelity_tools)["accepted"]
    source_decl = before.splitlines()[1:3]
    mutations = {
        "retained_opaque_calls": before,
        "round_away": after.replace('"math.roundeven"', '"math.round"'),
        "wrong_reciprocal": after.replace('"arith.divf"(%21, %19)', '"arith.divf"(%19, %21)'),
        "wrong_source": after.replace('"arith.mulf"(%18, %24)', '"arith.mulf"(%19, %24)'),
        "wrong_qmin": after.replace("-1.270000e+02", "-1.280000e+02"),
        "wrong_denominator": after.replace("1.270000e+02", "1.275000e+02"),
        "wrong_epsilon": after.replace("1.000000e-05", "1.000000e-04"),
        "fastmath": after.replace("#arith.fastmath<none>", "#arith.fastmath<fast>"),
        "wrong_yield_use": after.replace('"linalg.yield"(%29)', '"linalg.yield"(%20)'),
        "wrong_scale_access": after.replace(
            "affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>",
            "affine_map<(d0, d1) -> (0)>, affine_map<(d0, d1) -> (d0, d1)>",
        ),
        "retained_declarations": after.replace(
            '"builtin.module"() ({', '"builtin.module"() ({\n' + "\n".join(source_decl)
        ),
        "missing_provenance": after.replace('prov.transforms = "torchao_quantize_affine"', 'prov.transforms = "wrong"'),
        "changed_module_recipe": after.replace(
            'prov.quantization = "int8_dyn_act_int8_weight"', 'prov.quantization = "other"'
        ),
    }
    for name, mutant in mutations.items():
        assert mutant != after, name
        assert not checked(before, mutant, tmp_path / name, fidelity_tools)["accepted"], name


def test_invalid_ssa_is_compiler_refusal(tmp_path, fidelity_tools):
    _, after, _ = actual(_module_text((2, 4)))
    with pytest.raises(ExportError):
        exported(after.replace('"linalg.yield"(%29)', '"linalg.yield"(%undefined)'), tmp_path, fidelity_tools)


def test_unsupported_keepdim_layout_is_explicit(tmp_path, fidelity_tools):
    before, after, count = actual(_module_text((2, 4), scale_shape=(2, 1)))
    assert count == 2 and not checked(before, after, tmp_path, fidelity_tools)["accepted"]


ORACLE = """import hashlib,importlib.metadata,json,sys,torch
from pathlib import Path
import torchao.quantization.quant_primitives as qp
import torchao.quantization.quant_api as qa
assert torch.__version__ == '2.10.0+cpu', torch.__version__
version=importlib.metadata.version('torchao'); assert version==sys.argv[2],version
j=json.load(open(sys.argv[1])); x=torch.tensor(j['bits'],dtype=torch.uint32).view(torch.float32).reshape(j['shape'])
block=tuple([1]*(x.dim()-1)+[x.shape[-1]])
s,z=qp.choose_qparams_affine(x,qp.MappingType.SYMMETRIC,block,torch.int8,quant_min=-127,quant_max=127,eps=1e-5,scale_dtype=None)
q=qp.quantize_affine(x,block,s,z,torch.int8,quant_min=-127,quant_max=127)
print(json.dumps({'torch':torch.__version__,'torchao':version,'python':sys.version,
 'sources':{str(Path(p).name):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [qp.__file__,qa.__file__]},
 'zero_point':z.flatten().tolist(), 'scale_bits':[int(v)&0xffffffff for v in s.view(torch.int32).flatten().tolist()],
 'q_bits':[int(v)&255 for v in q.flatten().tolist()]}))
"""


@pytest.mark.parametrize("case", ["ties", "ulp", "tiny", "zeros", "r3", "signed-zero-fractions"])
def test_native_against_both_versioned_torchao_oracles(tmp_path, fidelity_tools, case):
    data = np.load(merlin_dir() / "tests/data/torchao_affine/reference.npz")
    if case == "signed-zero-fractions":
        x = np.array([[0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 1.0, -1.0]], dtype=np.float32)
    else:
        x = np.ascontiguousarray(data[f"{case}::x"])
    bits = x.view(np.uint32).reshape(-1).tolist()
    before, after, _ = actual(_module_text(x.shape))
    assert checked(before, after, tmp_path / "check", fidelity_tools)["accepted"]
    path = exported(after, tmp_path / "export", fidelity_tools)
    inputs, outputs = diff.signature(json.loads(path.read_text()), "forward")
    mlir = tmp_path / "after.mlir"
    mlir.write_text(after)
    native = diff.execute_mlir(
        mlir, [[bits]], tmp_path / "native", fidelity_tools, inputs=inputs, outputs=outputs, function="forward"
    )
    request = tmp_path / "oracle-request.json"
    request.write_text(json.dumps({"bits": bits, "shape": list(x.shape)}))
    for short, version in (("015", "0.15.0"), ("016", "0.16.0")):
        python = Path(os.environ[f"MERLIN_TORCHAO_{short}_PYTHON"])
        result = subprocess.run(
            [str(python), "-B", "-c", ORACLE, str(request), version], check=True, capture_output=True, text=True
        )
        (tmp_path / f"torchao-{short}.json").write_text(result.stdout)
        (tmp_path / f"torchao-{short}.stderr").write_text(result.stderr)
        oracle = json.loads(result.stdout)
        assert all(z == 0 for z in oracle["zero_point"])
        assert native["results"] == [[str(v) for v in oracle["scale_bits"] + oracle["q_bits"]]]


def test_provisional_materialization_receipt(tmp_path, fidelity_tools):
    before, after, _ = actual(_module_text((2, 4)))
    paths = [tmp_path / f"{side}.mlir" for side in ("before", "after")]
    for p, text in zip(paths, (before, after), strict=True):
        p.write_text(text)
    receipt = local_passes.verify(
        "torchao-affine",
        *paths,
        tmp_path / "receipts",
        compiler_python=fidelity_tools.compiler_python,
        suffix=(paths[1],),
    )
    assert receipt["status"] == "proved_materialization_provisional", receipt.get("reason")
    local_passes.replay(receipt, compiler_python=fidelity_tools.compiler_python)
    forged = copy.deepcopy(receipt)
    forged["source_call_refinement_proved"] = True
    with pytest.raises(ValueError):
        local_passes.replay(forged, compiler_python=fidelity_tools.compiler_python)


def test_nonfinite_reference_behavior_is_not_a_native_quantization_proof(tmp_path, fidelity_tools):
    """Record CPU observations without treating invalid casts or MLIR poison as defined data."""
    request = tmp_path / "oracle-request.json"
    request.write_text(
        json.dumps(
            {
                "shape": [2, 4],
                "bits": [
                    0x7FC00000,
                    0,
                    0x80000000,
                    0x3F000000,
                    0x7F800000,
                    0xFF800000,
                    0x3F800000,
                    0xBF800000,
                ],
            }
        )
    )
    for short, version in (("015", "0.15.0"), ("016", "0.16.0")):
        result = subprocess.run(
            [os.environ[f"MERLIN_TORCHAO_{short}_PYTHON"], "-B", "-c", ORACLE, str(request), version],
            check=True,
            capture_output=True,
            text=True,
        )
        (tmp_path / f"nonfinite-reference-{short}.json").write_text(result.stdout)
        oracle = json.loads(result.stdout)
        assert oracle["q_bits"] == [0] * 8
        assert oracle["scale_bits"][0] & 0x7FFFFFFF > 0x7F800000
        assert oracle["scale_bits"][1] == 0x7F800000


def test_empty_affine_reference_refusal(tmp_path, fidelity_tools):
    before, after, count = actual(_module_text((0, 4)))
    assert count == 2 and not checked(before, after, tmp_path / "check", fidelity_tools)["accepted"]
    request = tmp_path / "oracle-request.json"
    request.write_text(json.dumps({"shape": [0, 4], "bits": []}))
    for short, version in (("015", "0.15.0"), ("016", "0.16.0")):
        result = subprocess.run(
            [os.environ[f"MERLIN_TORCHAO_{short}_PYTHON"], "-B", "-c", ORACLE, str(request), version],
            capture_output=True,
            text=True,
        )
        (tmp_path / f"empty-reference-{short}.stdout").write_text(result.stdout)
        (tmp_path / f"empty-reference-{short}.stderr").write_text(result.stderr)
        assert result.returncode != 0 and "TypeError" in result.stderr and "NoneType" in result.stderr
