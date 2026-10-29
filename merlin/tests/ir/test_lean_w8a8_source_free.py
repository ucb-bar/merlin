"""Reuse frozen-semantics proofs only after proving equality with the draft interpretation.

The bridge includes every function body and all call fuel; equality can also mean both sides
are stuck. It supplies neither numerical-domain facts nor support for source operations.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_defined_policy import qualify
from test_lean_w8a8_module import source as contraction
from test_lean_w8a8_nested_support import source as nested
from test_lean_w8a8_source_program import execute, source
from test_lean_w8a8_source_quantize import word

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8

pytestmark = pytest.mark.usefixtures("lean_oracle")


def inspect(document, work):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_source_free.lean"),
            str(document),
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "provisional_frozen_interpretation_agreement"
    assert report["theorem"] == "MerlinVerify.Rules.W8A8SourceFree.denote_eq"
    assert report["whole_pass_verified"] is False and report["initialized_execution_verified"] is False
    assert (proc.returncode == 0) == report["interpretations_agree"]
    (work / "source-free-report.json").write_text(json.dumps(report, indent=2))
    return report


def frozen_execute(document, samples, work):
    request = work / "frozen-program-request.json"
    request.write_text(
        json.dumps(
            {
                "contract": "w8a8_byte_model",
                "export": document.read_text(),
                "function": "forward",
                "index_bits": 64,
                "samples": samples,
            }
        )
    )
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", "Differential.lean", str(request)],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_nested_ordinary_programs_reuse_universal_definedness(tmp_path, lean_tools, width, depth):
    document = exported(nested(width, depth=depth, runtime_dequant=True, reference=True), tmp_path, lean_tools)
    assert inspect(document, tmp_path)["interpretations_agree"]
    assert qualify(document, document, tmp_path)["initialized_policy_refinement"]
    arguments = [[["1", "255", "127"], [str(word(1))], ["0"]]]
    assert execute(document, arguments, tmp_path) == frozen_execute(document, arguments, tmp_path)


def test_initialized_float_baseline_is_separate_from_policy_quantizer_domain(tmp_path, lean_tools):
    before = contraction()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    paths = []
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        work.mkdir()
        paths.append(exported(program, work, lean_tools))
        assert inspect(paths[-1], work)["interpretations_agree"]
    assert qualify(paths[0], paths[0], tmp_path)["initialized_policy_refinement"]
    report = qualify(*paths, tmp_path, scope="all")
    assert report["functions"] == [{"function": "forward", "source_initialized": True}]
    assert not report["initialized_policy_refinement"]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("depth,index", [(1, 0), (3, 0), (3, 1)])
def test_source_dequantization_feeds_initialized_ordinary_nested_regions(tmp_path, lean_tools, width, depth, index):
    body = "linalg.yield %value : f32"
    for level in range(depth - 1):
        body = body.replace("%value", f"%nested_value{level}")
        body = f"""%input{level} = tensor.splat %value : tensor<1xf32>
          %empty{level} = tensor.empty() : tensor<1xf32>
          %r{level} = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
            iterator_types = ["parallel"]}} ins(%input{level} : tensor<1xf32>) outs(%empty{level} : tensor<1xf32>) {{
              ^bb0(%nested_value{level}: f32, %unused{level}: f32):
                {body}
          }} -> tensor<1xf32>
          %index{level} = arith.constant {index} : index
          %result{level} = tensor.extract %r{level}[%index{level}] : tensor<1xf32>
          linalg.yield %result{level} : f32"""
    program = parse_mlir_text(f"""func.func @forward(%q: tensor<3xi8>, %s: tensor<f32>,
        %z: tensor<i{width}>) -> tensor<3xf32> {{
      %x = "quant_ext.dequantize_per_tensor"(%q, %s, %z)
        : (tensor<3xi8>, tensor<f32>, tensor<i{width}>) -> tensor<3xf32>
      %empty = tensor.empty() : tensor<3xf32>
      %r = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}} ins(%x : tensor<3xf32>) outs(%empty : tensor<3xf32>) {{
          ^bb0(%value: f32, %unused: f32):
            {body}
      }} -> tensor<3xf32>
      return %r : tensor<3xf32>
    }}""")
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not inspect(document, tmp_path)["interpretations_agree"]
    assert qualify(document, document, tmp_path)["initialized_policy_refinement"] == (index == 0)
    if index != 0:
        return
    actual = execute(document, [[["1", "255", "127"], [str(word(1))], ["0"]]], tmp_path)
    assert actual[0]["values"][0]["bits"] == [str(word(v)) for v in [1.0, -1.0, 127.0]]


@pytest.mark.parametrize("kind", ["internal", "external", "recursive"])
def test_interpretation_equality_includes_calls_and_stuck_results(tmp_path, lean_tools, kind):
    target = "forward" if kind == "recursive" else "callee"
    declaration = "" if kind == "recursive" else "func.func private @callee(tensor<1xf32>) -> tensor<1xf32>"
    if kind == "internal":
        declaration = """func.func private @callee(%x: tensor<1xf32>) -> tensor<1xf32> {
          return %x : tensor<1xf32>
        }"""
    program = parse_mlir_text(f"""module {{
      {declaration}
      func.func @forward(%x: tensor<1xf32>) -> tensor<1xf32> {{
        %r = func.call @{target}(%x) : (tensor<1xf32>) -> tensor<1xf32>
        return %r : tensor<1xf32>
      }}
    }}""")
    document = exported(program, tmp_path, lean_tools, generic=True)
    assert inspect(document, tmp_path)["interpretations_agree"]
    arguments = [[[str(word(1))]]]
    actual = execute(document, arguments, tmp_path)
    assert actual == frozen_execute(document, arguments, tmp_path)
    assert actual[0]["status"] == ("ok" if kind == "internal" else "stuck")
    # Existing definedness does not prove calls; semantic equality cannot supply that proof.
    assert not qualify(document, document, tmp_path)["initialized_policy_refinement"]


@pytest.mark.parametrize("kind", ["nested", "callee", "unused"])
def test_source_operations_cannot_be_hidden_from_the_bridge(tmp_path, lean_tools, kind):
    program = nested(depth=3) if kind == "nested" else source(64, (1,), wrapper=True)
    if kind == "unused":
        # Keep the quantizing kernel but replace only the wrapper with an identity.
        text = str(program)
        start = text.index("func.func @forward")
        program = parse_mlir_text(
            text[:start]
            + """func.func @forward(%x: tensor<1xf32>) -> tensor<1xf32> {
          return %x : tensor<1xf32>
        }
        }"""
        )
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    report = inspect(document, tmp_path)
    assert not report["interpretations_agree"]
    if kind in {"callee", "unused"}:
        assert {fn["function"]: fn["source_free"] for fn in report["functions"]} == {"kernel": False, "forward": True}


def test_equality_does_not_supply_definedness_for_poison(tmp_path, lean_tools):
    program = parse_mlir_text("""func.func @forward() -> tensor<2xf32> {
      %e = tensor.empty() : tensor<2xf32>
      return %e : tensor<2xf32>
    }""")
    document = exported(program, tmp_path, lean_tools)
    assert inspect(document, tmp_path)["interpretations_agree"]
    actual = execute(document, [[]], tmp_path)
    assert actual == frozen_execute(document, [[]], tmp_path)
    assert actual[0]["values"][0]["bits"] == [None, None]
    assert not qualify(document, document, tmp_path)["initialized_policy_refinement"]
