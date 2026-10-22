"""Actual pass boundaries, compiler-valid mutations, native bits and provisional replay."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import fix_bool_sitofp
from merlin.verify.lean import differential as diff
from merlin.verify.lean import local_passes
from merlin.verify.lean.export import ExportError, export_file
from merlin.xdsl_dialects._common import text as to_text

SCALAR = """module {
  func.func @subject(%b: i1, %other: i1) -> f32 {
    %r = arith.sitofp %b : i1 to f32
    return %r : f32
  }
}"""

NESTED = """module {
  func.func private @cast(%b: i1) -> f32 {
    %r = arith.sitofp %b : i1 to f32
    return %r : f32
  }
  func.func @subject(%mask: tensor<4xi1>) -> tensor<4xf32> {
    %e = tensor.empty() : tensor<4xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]} ins(%mask: tensor<4xi1>) outs(%e: tensor<4xf32>) {
      ^bb0(%b: i1, %old: f32):
      %f = func.call @cast(%b) : (i1) -> f32
      linalg.yield %f : f32
    } -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}"""


def actual(source: str) -> tuple[str, str, int]:
    module = parse_mlir_text(source)
    before = to_text(module, generic=True)
    count = fix_bool_sitofp(module)
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
    executable = repo_root() / "out/build/lean/build/bin/merlin-lean-local-passes"
    assert executable.is_file(), "build LocalPasses.lakefile.toml first"
    result = subprocess.run(
        [str(executable), "bool-to-float", *(str(p) for p in paths)],
        check=True,
        capture_output=True,
        text=True,
        timeout=360,
    )
    answer = json.loads(result.stdout)
    (work / "checker.json").write_text(json.dumps(answer, indent=2) + "\n")
    assert answer["status"] == "provisional_contract"
    return answer


@pytest.mark.parametrize("source", [SCALAR, NESTED, NESTED.replace("tensor<4x", "tensor<0x")])
def test_actual_pass_complete_program(tmp_path, fidelity_tools, source):
    before, after, count = actual(source)
    assert count == 1
    assert checked(before, after, tmp_path, fidelity_tools)["accepted"]


def test_scf_nested_sites_and_both_calls(tmp_path, fidelity_tools):
    source = """module {func.func @subject(%b: i1) -> f32 {
      %r = scf.if %b -> (f32) {
        %t = arith.sitofp %b : i1 to f32
        scf.yield %t : f32
      } else {
        %f = arith.sitofp %b : i1 to f32
        scf.yield %f : f32
      }
      return %r : f32 }}"""
    before, after, count = actual(source)
    assert count == 2
    assert checked(before, after, tmp_path, fidelity_tools)["accepted"]
    path = exported(after, tmp_path / "native-export", fidelity_tools)
    inputs, outputs = diff.signature(json.loads(path.read_text()), "subject")
    mlir = tmp_path / "after.mlir"
    mlir.write_text(after)
    native = diff.execute_mlir(
        mlir, [[[0]], [[1]]], tmp_path / "native", fidelity_tools, inputs=inputs, outputs=outputs
    )
    assert native["results"] == [["0"], ["1065353216"]]


@pytest.mark.parametrize("format", ["f16", "bf16", "f64"])
def test_unsupported_formats_are_refused(tmp_path, fidelity_tools, format):
    before, after, count = actual(SCALAR.replace("f32", format))
    assert count == 1  # Production coverage is wider than this proved f32 contract.
    answer = checked(before, after, tmp_path, fidelity_tools)
    assert not answer["accepted"] and answer["inventory"]["sites"] == 1


def test_valid_mutations_reach_actual_lean(tmp_path, fidelity_tools):
    before, after, _ = actual(SCALAR)
    assert checked(before, after, tmp_path / "baseline", fidelity_tools)["accepted"]
    cast = next(line for line in after.splitlines() if '"arith.uitofp"' in line)
    mutations = {
        "retained_signed_cast": before,
        "wrong_source": after.replace('"arith.uitofp"(%b)', '"arith.uitofp"(%other)'),
        "nonneg": after.replace("{prov.transforms", "{nonNeg, prov.transforms"),
        "fastmath": after.replace("{prov.transforms", "{fastmath = #arith.fastmath<nnan>, prov.transforms"),
        "rounding": after.replace("{prov.transforms", '{roundingmode = "towardzero", prov.transforms'),
        "wrong_use": after.replace(
            cast, cast + '\n    %wrong = "arith.constant"() <{value = -0.0 : f32}> : () -> f32'
        ).replace('"func.return"(%r)', '"func.return"(%wrong)'),
        "bypassed_cast": after.replace(cast, cast + '\n    %wrong = "arith.sitofp"(%b) : (i1) -> f32').replace(
            '"func.return"(%r)', '"func.return"(%wrong)'
        ),
        "wrong_format": after.replace("f32", "f64"),
        "missing_provenance": after.replace(' {prov.transforms = "bool_sitofp_uitofp"}', ""),
        "extra_dead_operation": after.replace(
            cast, cast + '\n    %dead = "arith.constant"() <{value = 1.0 : f32}> : () -> f32'
        ),
    }
    for name, mutant in mutations.items():
        assert mutant != after, name
        # Export runs the authoritative compiler verifier. Only then is rejection a Lean test.
        assert not checked(before, mutant, tmp_path / name, fidelity_tools)["accepted"], name


def test_complete_boundary_rejects_partial_repair_and_wrong_nested_use(tmp_path, fidelity_tools):
    source = NESTED.replace(
        "%f = func.call @cast(%b) : (i1) -> f32",
        "%f = arith.sitofp %b : i1 to f32",
    )
    before, after, count = actual(source)
    assert count == 2 and checked(before, after, tmp_path / "baseline", fidelity_tools)["accepted"]
    line = next(line for line in after.splitlines() if '"arith.uitofp"(%b)' in line)
    partial = after.replace(line, line.replace('"arith.uitofp"', '"arith.sitofp"'), 1)
    wrong = after.replace('"linalg.yield"(%f)', '"linalg.yield"(%old)')
    assert not checked(before, partial, tmp_path / "partial", fidelity_tools)["accepted"]
    assert not checked(before, wrong, tmp_path / "use", fidelity_tools)["accepted"]


def test_changed_internal_callee_is_refused(tmp_path, fidelity_tools):
    alternate = """func.func private @alternate(%b: i1) -> f32 {
      %z = arith.constant 0.0 : f32
      return %z : f32
    }"""
    before, after, _ = actual(NESTED.replace("module {", "module {" + alternate, 1))
    assert checked(before, after, tmp_path / "baseline", fidelity_tools)["accepted"]
    wrong = after.replace("callee = @cast", "callee = @alternate")
    assert wrong != after
    assert not checked(before, wrong, tmp_path / "callee", fidelity_tools)["accepted"]


def test_ssa_errors_are_compiler_refusals(tmp_path, fidelity_tools):
    _, after, _ = actual(SCALAR)
    invalid = after.replace('"func.return"(%r)', '"func.return"(%undefined)')
    with pytest.raises(ExportError):
        exported(invalid, tmp_path, fidelity_tools)


@pytest.mark.parametrize("source,values", [(SCALAR, [[[0], [1]], [[1], [0]]]), (NESTED, [[[0, 1, 1, 0]]])])
def test_native_and_pinned_torch_oracle(tmp_path, fidelity_tools, source, values):
    before, after, _ = actual(source)
    checked(before, after, tmp_path / "check", fidelity_tools)
    mlir = tmp_path / "after.mlir"
    mlir.write_text(after)
    path = exported(after, tmp_path / "export", fidelity_tools)
    inputs, outputs = diff.signature(json.loads(path.read_text()), "subject")
    native = diff.execute_mlir(mlir, values, tmp_path / "native", fidelity_tools, inputs=inputs, outputs=outputs)
    python = Path(os.environ["MERLIN_TORCH_PYTHON"])
    request = tmp_path / "torch-request.json"
    request.write_text(json.dumps(values))
    code = """import json,sys,torch
assert torch.__version__ == '2.10.0+cpu', torch.__version__
samples=json.load(open(sys.argv[1]));result=[]
for sample in samples:
 x=torch.tensor(sample[0],dtype=torch.bool).to(torch.float32)
 result.append([str(int(b)&0xffffffff) for b in x.view(torch.int32).tolist()])
print(json.dumps({'torch':torch.__version__,'results':result}))
"""
    reference = subprocess.run(
        [str(python), "-B", "-c", code, str(request)], check=True, capture_output=True, text=True
    )
    answer = json.loads(reference.stdout)
    (tmp_path / "torch.json").write_text(reference.stdout)
    assert native["results"] == answer["results"]


def test_provisional_receipt_and_replay(tmp_path, fidelity_tools):
    before, after, _ = actual(SCALAR)
    paths = [tmp_path / f"{side}.mlir" for side in ("before", "after")]
    for path, source in zip(paths, (before, after), strict=True):
        path.write_text(source)
    receipt = local_passes.verify(
        "bool-to-float",
        *paths,
        tmp_path / "receipts",
        compiler_python=fidelity_tools.compiler_python,
        suffix=(paths[1],),
    )
    assert receipt["status"] == "proved_policy_provisional", receipt.get("reason")
    assert receipt["answer"]["suffix_count"] == 1
    local_passes.replay(receipt, compiler_python=fidelity_tools.compiler_python)
    for field, value in (
        ("approved_full_verification", True),
        ("contract_review", "approved"),
        ("whole_function_defined", True),
    ):
        forged = copy.deepcopy(receipt)
        forged[field] = value
        with pytest.raises(ValueError):
            local_passes.replay(forged, compiler_python=fidelity_tools.compiler_python)
    forged = copy.deepcopy(receipt)
    forged["artifacts"].reverse()
    with pytest.raises(ValueError):
        local_passes.replay(forged, compiler_python=fidelity_tools.compiler_python)


def test_unrelated_suffix_mutation_is_refused(tmp_path, fidelity_tools):
    before, after, _ = actual(SCALAR)
    paths = [tmp_path / name for name in ("before.mlir", "after.mlir", "suffix.mlir")]
    wrong = after.replace('"arith.uitofp"(%b)', '"arith.uitofp"(%other)')
    for path, source in zip(paths, (before, after, wrong), strict=True):
        path.write_text(source)
    # This valid suffix reaches Lean's complete-program correspondence checker.
    exported(wrong, tmp_path / "export-suffix", fidelity_tools)
    receipt = local_passes.verify(
        "bool-to-float",
        *paths[:2],
        tmp_path / "receipts",
        compiler_python=fidelity_tools.compiler_python,
        suffix=(paths[2],),
    )
    assert receipt["status"] == "refused" and receipt["answer"]["suffix_count"] == 1


def test_actual_smolvla_has_no_bool_sitofp_sites(tmp_path, fidelity_tools):
    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    before, after, count = actual(capture.read_text())
    assert count == 0
    answer = checked(before, after, tmp_path, fidelity_tools)
    assert answer["inventory"]["sites"] == 0 and not answer["accepted"]


def test_actual_smolvla_denoise_two_sites(tmp_path, fidelity_tools):
    capture = merlin_dir() / "contract/capsules/model/M4_smolvla_denoise_gemmini/capsule.interface.mlir"
    before, after, count = actual(capture.read_text())
    assert count == 2
    answer = checked(before, after, tmp_path, fidelity_tools)
    assert answer["inventory"]["sites"] == 2 and answer["accepted"]


def test_native_float_mask_inputs_include_ieee_edges(tmp_path, fidelity_tools):
    bits = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x3F000000,
        0xBF000000,
        0x3FC00000,
        0xBFC00000,
        0x3F800000,
        0xBF800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0xFFC00000,
        0x7F800001,
        0xFF800001,
    ]
    source = """module {func.func @subject(%x: tensor<Nxf32>) -> tensor<Nxf32> {
      %e = tensor.empty() : tensor<Nxf32>
      %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]} ins(%x: tensor<Nxf32>) outs(%e: tensor<Nxf32>) {
        ^bb0(%v: f32, %old: f32):
        %zero = arith.constant 0.0 : f32
        %b = arith.cmpf une, %v, %zero : f32
        %out = arith.sitofp %b : i1 to f32
        linalg.yield %out : f32
      } -> tensor<Nxf32>
      return %r : tensor<Nxf32> }}""".replace("N", str(len(bits)))
    before, after, count = actual(source)
    assert count == 1 and checked(before, after, tmp_path / "check", fidelity_tools)["accepted"]
    path = exported(after, tmp_path / "export", fidelity_tools)
    inputs, outputs = diff.signature(json.loads(path.read_text()), "subject")
    mlir = tmp_path / "after.mlir"
    mlir.write_text(after)
    native = diff.execute_mlir(mlir, [[bits]], tmp_path / "native", fidelity_tools, inputs=inputs, outputs=outputs)
    code = """import json,sys,torch
bits=json.loads(sys.argv[1]); x=torch.tensor(bits,dtype=torch.uint32).view(torch.float32)
y=x.to(torch.bool).to(torch.float32).view(torch.int32)
print(json.dumps({'torch':torch.__version__,'results':[[str(int(v)) for v in y.tolist()]]}))
"""
    for label, python in (
        ("pinned", Path(os.environ["MERLIN_TORCH_PYTHON"])),
        ("supplemental", fidelity_tools.compiler_python),
    ):
        result = subprocess.run(
            [str(python), "-B", "-c", code, json.dumps(bits)], check=True, capture_output=True, text=True
        )
        oracle = json.loads(result.stdout)
        if label == "pinned":
            assert oracle["torch"] == "2.10.0+cpu"
        (tmp_path / f"torch-{label}.json").write_text(result.stdout)
        assert native["results"] == oracle["results"]
