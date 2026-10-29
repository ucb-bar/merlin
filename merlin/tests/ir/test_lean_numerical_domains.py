"""Kernel checks for explicit numerical domains; no policy approval is issued."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def domain_tools():
    sdk = Path(os.environ.get("MERLIN_LEAN_SDK_BIN", "/UNSELECTED_LEAN_SDK"))
    if not all((sdk / name).is_file() for name in ("lake", "lean")):
        reason = "numerical-domain kernel checks require MERLIN_LEAN_SDK_BIN"
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)
    return SimpleNamespace(sdk_bin=sdk)


def check_query(tools, work, query):
    source = work / "Domain.lean"
    source.write_text(query)
    sdk = tools.sdk_bin
    result = subprocess.run(
        [str(sdk / "lake"), "env", str(sdk / "lean"), str(source)],
        cwd=repo_root() / "merlin/lean",
        env={**os.environ, "PATH": str(sdk) + os.pathsep + os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    (work / "stdout.log").write_text(result.stdout)
    (work / "stderr.log").write_text(result.stderr)
    return result


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", ["[]", "[0, 3]", "[2, 3]"])
def test_quantizer_domain_gives_defined_execution(domain_tools, tmp_path, width, shape):
    query = f"""import MerlinVerify.Rules.QuantizeDefinedness
open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Rules.QuantizeDefinedness
def model : FloatModel := {{ failingConversionModel with toInt := fun _ _ _ _ => some 0 }}
def input := Tensor.full {shape} (.float .f32) (.some (.flt .f32 0#32))
def scale := Tensor.full [] (.float .f32) (.some (.flt .f32 0x3f800000#32))
def zero := Tensor.full [] (.int {width}) (.some (.bits {width} 0))
theorem domain : Domain {width} model {shape} input scale zero := by
  refine ⟨0x3f800000#32, 0#{width}, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · intro p hp
    exact ⟨0#32, 0#8, by simp [input, Tensor.full, Tensor.canon, hp, Dequant.float32], rfl⟩
theorem successful : ∃ results,
    Quantize.run ⟨{shape}⟩ model [.tensor input, .tensor scale, .tensor zero] = .ok results ∧
    ResultsDefined results :=
  run_defined (by decide) (by cbv) domain
#print axioms successful
"""
    result = check_query(domain_tools, tmp_path, query)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "sorryAx" not in result.stdout


def test_initialized_inputs_cannot_discharge_failed_conversion(domain_tools, tmp_path):
    query = """import MerlinVerify.Rules.QuantizeDefinedness
open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Rules.QuantizeDefinedness
def input := Tensor.full [] (.float .f32) (.some (.flt .f32 0#32))
def scale := Tensor.full [] (.float .f32) (.some (.flt .f32 0x3f800000#32))
def zero := Tensor.full [] (.int 32) (.some (.bits 32 0#32))
example : Domain 32 failingConversionModel [] input scale zero := by
  refine ⟨0x3f800000#32, 0#32, rfl, rfl, ?_⟩
  intro p hp
  refine ⟨0#32, 0#8, by simp [input, Tensor.full, Tensor.canon, hp, Dequant.float32], ?_⟩
  change (none : Option (BitVec 8)) = some 0#8
  rfl
"""
    result = check_query(domain_tools, tmp_path, query)
    assert result.returncode != 0
    assert "Tactic `rfl` failed" in result.stdout
    assert "none" in result.stdout and "some" in result.stdout


def test_mixed_initialized_trace_retains_initial_value_and_order(domain_tools, tmp_path):
    query = """import MerlinVerify.Rules.Bf16F32AccInitialized
open MerlinVerify MerlinVerify.Sem.Ops.Linalg MerlinVerify.Rules
open Bf16F32AccContract Bf16F32AccInitialized Bf16F32AccTensor
def model : FloatModel where
  op := fun key _k args =>
    if key.op == "arith.addf" then 2 * args[0]! + args[1]!
    else if key.op == "arith.mulf" then args[0]! * args[1]! else 0
  cmp := fun _ _ _ _ => false
  convert := fun _ _ dst x => x.setWidth dst.width
  toInt := fun _ _ _ _ => none
  ofInt := fun _ _ k _ => 0#k.width
def g : Geometry := ⟨[2], [2], [], ⟨1, 0, [.dim 0]⟩, ⟨1, 0, [.dim 0]⟩, ⟨1, 0, []⟩, [2]⟩
def a : Idx → Option (FBits .bf16) := fun p => some (if p == [0] then 1 else 2)
def b : Idx → Option (FBits .f16) := fun _ => some 1
def initial : Idx → Option (FBits .f32) := fun _ => some 3
theorem geometry : Check.StructuredOk [g.leftMap, g.rightMap, g.outMap] 1
    [.tensor [2] (.float .bf16), .tensor [2] (.float .f16), .tensor [] (.float .f32)] [2] :=
  Check.structuredRanges?_spec (by cbv)
theorem actual_generic (ctx : Ctx) (call : CallSem) (env : Env) :
    genericSem ctx [g.leftMap, g.rightMap, g.outMap] [.reduction]
      [.tensor (floats .bf16 g.left a), .tensor (floats .f16 g.right b)]
      [.tensor (floats .f32 g.out initial)]
      (fun c => evalRegion Sem.Ops.opSem c call env
        (Bf16F32AccBody.accRegion .bf16 .f16 ⟨.none, .none⟩)) =
      .ok [.tensor (Tensor.canon g.out (.float .f32)
        (fun cell => value .f32 (accumulateFrom ctx.M ⟨.none, .none⟩ g a b initial cell)))] :=
  accumulateFrom_sound (ib := 64) ctx call env [.reduction] ⟨.none, .none⟩ g geometry (by cbv) a b initial
theorem nonzero_ordered :
    (resultFrom model ⟨.none, .none⟩ .f16 g a b initial).get [] = .some (.flt .f16 16) := by cbv
theorem no_reset :
    (resultFrom model ⟨.none, .none⟩ .f16 g a b (fun _ => some zero)).get [] ≠
    (resultFrom model ⟨.none, .none⟩ .f16 g a b initial).get [] := by cbv; decide
theorem poisoned_initial :
    (resultFrom model ⟨.none, .none⟩ .f16 g a b (fun _ => none)).get [] = .poison := by cbv
#print axioms nonzero_ordered
#print axioms actual_generic
#print axioms no_reset
#print axioms poisoned_initial
"""
    result = check_query(domain_tools, tmp_path, query)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "sorryAx" not in result.stdout
