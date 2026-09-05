"""The integer ``requant`` epilogue stage must arrive with its shift DECLARED, end to end.

``requant`` is the one stage in the command-buffer ABI's epilogue vocabulary that no capsule could
demand: it is in :data:`merlin.runtime.commandbuffer.EPILOGUE_STAGES`, in the JSON schema, in the
interface dialect's verifier and in all three numeric engines, and it was in neither of the two tuples
that decide which stages a capsule builder emits. ``check_defect_reach.py`` reported it unreachable on
the one target whose manifest demands it -- and an external hard-kernel campaign had already found the
defect it names ("requant stage dropped": a standalone integer requantization with no lowering, omitted
from the bundle rather than counted as failing).

Three things had to be true at once for a requant capsule to test anything, and each of them failed
SILENTLY rather than loudly:

1. the SHIFT reaches the capsule. The golden engine, the reference and the simulator each carried their
   own fallback of 4, so a capsule declaring the stage without a shift produced a golden and a reference
   that agreed with each other, by coincidence, at a number the backend under test was never told;
2. the commit NARROWS. ``_resolve_output_dtype`` consulted only ``acc_scale``, so the stage whose whole
   job is to narrow i32 -> i8 declared the wide accumulator dtype -- the property under test absent from
   the capsule meant to test it;
3. the requirement DEMANDS it. ``conformance._epilogue_axis`` walked a tuple that omitted the stage.

Every test here is written so that undoing one of those three fixes turns it red.
"""
from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import repo_root
from merlin.runtime.commandbuffer import EPILOGUE_STAGES
from merlin.runtime.reference import reference_outputs
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import conformance as CF
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import corpus_synth as CSY
from merlin.targetgen import model_slice_export as MSE
from merlin.targetgen.contract import interface_emit as IE

#: The target this repo's requant obligation is derived for. A test may name a target -- it is the
#: subject here -- but nothing about it is assumed: the shift and the narrow dtype are read back out of
#: the target's own profile rather than spelled in this file.
_TARGET = "gemmini"

#: A shift that is NOT the value the three engines fall back to (4). Every threading assertion below
#: uses one of these, so a broken thread shows up as a wrong number rather than as a coincidence.
_DECLARED_SHIFT = 3
_OVERRIDE_SHIFT = 5


def _binding(**over) -> CS.CorpusBinding:
    kw = dict(target=_TARGET, tile_dim=16, operand_dtype="int8", accum_dtype="i32",
              integer=True, tiers=["L0", "L1", "L2", "L3"], compare="exact_int",
              requant_output_dtype="i8", requant_shift=_DECLARED_SHIFT,
              classes_for=lambda **_: [])
    kw.update(over)
    return CS.CorpusBinding(**kw)


def _entry(**over) -> dict:
    entry = {
        "name": "RQ", "kind": "layer", "op": "matmul",
        "M": 16, "K": 32, "N": 16, "lhs": "A0", "weight": "W", "out": "Y0",
        "epilogue": ["requant"],
        "source_role": "handauthored_compiler_test",
        "source_reference": "requant epilogue declaration test",
    }
    entry.update(over)
    return entry


# ---------------------------------------------------------------------------------------------
# 1. the shift is declared, and it is the DECLARED one that both engines apply
# ---------------------------------------------------------------------------------------------

def test_capsule_and_interface_both_carry_the_declared_shift():
    capsule, mlir = CS.build_matmul(_entry(), _binding())

    assert capsule["operation"]["attributes"]["requant_shift"] == _DECLARED_SHIFT
    commit = next(c for c in IE.parse_interface_mlir(mlir)["commands"]
                  if c["opcode"] == "COMMIT")
    assert commit["attributes"]["requant_shift"] == _DECLARED_SHIFT
    assert commit["attributes"]["epilogue"] == ["requant"]


def test_an_entry_declaration_overrides_the_targets_default():
    """A capsule may requant by its own shift; the number the capsule states is the number emitted."""
    capsule, mlir = CS.build_matmul(_entry(requant_shift=_OVERRIDE_SHIFT), _binding())
    assert capsule["operation"]["attributes"]["requant_shift"] == _OVERRIDE_SHIFT
    commit = next(c for c in IE.parse_interface_mlir(mlir)["commands"] if c["opcode"] == "COMMIT")
    assert commit["attributes"]["requant_shift"] == _OVERRIDE_SHIFT


def test_golden_and_reference_agree_because_they_read_the_same_declared_shift():
    """The capsule's golden and the interface's reference must MATCH -- and match for the stated
    reason. The second half of this test is the one that matters: strip the declaration out of the
    emitted module and the two diverge, which is what proves the number is being read rather than
    independently defaulted to the same value."""
    capsule, mlir = CS.build_matmul(_entry(), _binding())

    golden = CG._recompute_golden(capsule)["Y0"]
    got = reference_outputs(IE.parse_interface_mlir(mlir))["Y0"]
    assert got == golden

    stripped = mlir.replace(f", requant_shift = {_DECLARED_SHIFT} : i64", "")
    assert stripped != mlir
    undeclared = reference_outputs(IE.parse_interface_mlir(stripped))["Y0"]
    assert undeclared != golden, (
        "with the declaration removed the reference falls back to its own shift and still matched the "
        "golden -- so this capsule cannot tell a backend that read the declared shift from one that "
        "invented its own")


def test_a_requant_stage_without_any_declared_shift_fails_closed():
    """Neither the entry nor the target declares one: refuse, rather than emit a capsule three engines
    will each grade with a private default."""
    with pytest.raises(ValueError, match="requant_shift"):
        CS.build_matmul(_entry(), _binding(requant_shift=None))


def test_a_declared_shift_without_the_stage_fails_closed():
    """A parameter nothing reads is the same silent wrong answer pointing the other way."""
    with pytest.raises(ValueError, match="requant"):
        CS.build_matmul(_entry(epilogue=[], requant_shift=_DECLARED_SHIFT), _binding())


def test_the_emitter_refuses_each_half_of_the_pair():
    with pytest.raises(ValueError, match="requant_shift"):
        MSE.emit_interface_mlir(lhs="A0", weight="W", out="Y0", M=16, K=16, N=16,
                                epilogue=["requant"], output_dtype="i8")
    with pytest.raises(ValueError, match="requant"):
        MSE.emit_interface_mlir(lhs="A0", weight="W", out="Y0", M=16, K=16, N=16,
                                epilogue=[], output_dtype="i32", requant_shift=_DECLARED_SHIFT)


def test_the_golden_engine_refuses_an_undeclared_shift():
    """The engine that computes the answer must not supply the one parameter nobody stated."""
    capsule, _ = CS.build_matmul(_entry(), _binding())
    capsule["operation"]["attributes"].pop("requant_shift")
    with pytest.raises(ValueError, match="requant_shift"):
        CG._recompute_golden(capsule)


# ---------------------------------------------------------------------------------------------
# 2. the commit narrows
# ---------------------------------------------------------------------------------------------

def test_a_requant_commit_narrows_to_the_targets_declared_output_dtype():
    """The stage exists to commit the wide accumulator back to the operand width. A capsule declaring
    the accumulator dtype would be satisfied by a backend that never narrowed."""
    binding = _binding()
    capsule, mlir = CS.build_matmul(_entry(), binding)

    assert binding.requant_output_dtype != binding.accum_dtype
    assert capsule["operation"]["attributes"]["output_dtype"] == binding.requant_output_dtype
    assert capsule["numeric_policy"]["dtype"] == binding.requant_output_dtype
    assert f"x{binding.requant_output_dtype}>" in mlir

    values = [v for row in CG._recompute_golden(capsule)["Y0"] for v in row]
    assert values and all(-128 <= v <= 127 for v in values)


def test_an_entry_declaration_still_wins_over_the_narrowing():
    """The entry's own output_dtype is the one declaration this resolver may not second-guess."""
    capsule, _ = CS.build_matmul(_entry(output_dtype="i32"), _binding())
    assert capsule["operation"]["attributes"]["output_dtype"] == "i32"


def test_the_convolution_builder_threads_the_shift_too():
    """The other builder that reads an entry's `epilogue:`. It emits its own interface text rather than
    going through the matmul emitter, so the threading is a second code path and needs its own check --
    an untested branch here would carry the stage into the capsule and drop it from the module."""
    binding = _binding()
    entry = {
        "name": "RQC", "kind": "layer", "op": "conv2d", "ifm": "IFM", "weight": "W", "out": "Y0",
        "ci": 4, "N": 16, "Himg": 6, "Wimg": 6, "kh": 3, "kw": 3,
        "epilogue": ["requant"],
        "source_role": "handauthored_compiler_test",
        "source_reference": "requant epilogue declaration test (conv)",
    }
    capsule, mlir = CS.build_conv2d(entry, binding)
    assert capsule["operation"]["attributes"]["requant_shift"] == binding.requant_shift
    assert capsule["operation"]["attributes"]["output_dtype"] == binding.requant_output_dtype
    conv = next(c for c in IE.parse_interface_mlir(mlir)["commands"] if c["opcode"] == "CONV2D")
    assert conv["attributes"]["requant_shift"] == binding.requant_shift
    assert reference_outputs(IE.parse_interface_mlir(mlir))["Y0"] == CG._recompute_golden(capsule)["Y0"]


# ---------------------------------------------------------------------------------------------
# 3. the requirement demands it
# ---------------------------------------------------------------------------------------------

def test_requant_is_in_the_builder_vocabulary_and_there_is_only_one_of_it():
    """Two hand-kept copies of this tuple, each annotated as mirroring the other, are what let the
    stage go missing. Both consumers must read the SAME object."""
    assert "requant" in CS.BUILDER_EPILOGUE_STAGES
    assert set(CS.BUILDER_EPILOGUE_STAGES) <= set(EPILOGUE_STAGES)
    assert CF._builder_epilogue_stages() is CS.BUILDER_EPILOGUE_STAGES


def test_the_derived_epilogue_requirement_demands_the_requant_stage():
    axis = CF._epilogue_axis(_TARGET)
    required = {str(r.get("stage")): r for r in (axis.get("required") or ())}
    assert "requant" in required, (
        f"{_TARGET} declares the stage's family fused-only, so a corpus that never asks for it lets a "
        f"backend that cannot emit it fail nothing: {sorted(required)}")
    assert required["requant"]["evidenced_by"], "a required stage with no evidence is a guess"


def _synthesized_requant_entry() -> dict:
    """The profile entry the generator's epilogue axis writes for the DERIVED requant obligation.

    Driven off a freshly derived epilogue axis rather than the tracked spec file, so this asserts the
    derivation -> synthesis chain and not the freshness of a checked-in artifact.
    """
    import yaml

    spec = yaml.safe_load(
        (repo_root() / "merlin" / "contract" / "capsules" / "conformance"
         / f"{_TARGET}.yaml").read_text(encoding="utf-8")) or {}
    spec["epilogue"] = CF._epilogue_axis(_TARGET)
    entries = [e for e in (CSY.synthesize(spec).get("capsules") or ())
               if "requant" in [str(x) for x in (e.get("epilogue") or ())]]
    assert entries, "the epilogue axis wrote no member for the required 'requant' stage"
    return entries[0]


def test_the_synthesized_obligation_annotates_the_generalization_axis():
    """The axis label goes under `generalization`. `semantic` is a free-form op-semantics label and
    anything written there is silently discarded, so a member annotated that way would be emitted,
    counted, and credited to no axis at all."""
    entry = _synthesized_requant_entry()
    assert entry.get("generalization", {}).get("generalization_axis") == "epilogue"
    assert "generalization_axis" not in (entry.get("semantic") or {})


def test_the_synthesized_obligation_builds_into_a_narrowing_capsule_with_a_declared_shift():
    """End to end: the derived requirement's own entry, through the real builder, on the target's own
    derived binding -- the capsule an agent would actually be graded on."""
    binding = _binding()
    entry = dict(_synthesized_requant_entry())
    for axis in ("M", "K", "N"):
        if axis in entry:
            entry[axis] = CSY._tile_int(entry[axis], binding.tile_dim)

    capsule, mlir = CS.build_matmul(entry, binding)
    attrs = capsule["operation"]["attributes"]
    assert attrs["epilogue"] == ["requant"]
    assert attrs["requant_shift"] == binding.requant_shift
    assert attrs["output_dtype"] == binding.requant_output_dtype
    assert reference_outputs(IE.parse_interface_mlir(mlir))["Y0"] == CG._recompute_golden(capsule)["Y0"]


# ---------------------------------------------------------------------------------------------
# 4. the defect class is reachable
# ---------------------------------------------------------------------------------------------

def _defect_reach():
    path = repo_root() / "build_tools" / "scripts" / "check_defect_reach.py"
    spec = importlib.util.spec_from_file_location("_check_defect_reach", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_such_a_capsule_makes_the_requant_defect_class_reachable():
    """The reach checker asks whether SOME capsule demands the stage. A capsule built the way the
    epilogue axis builds one must answer yes -- otherwise a green corpus run says nothing about a
    backend that drops the stage."""
    dr = _defect_reach()
    capsule, _ = CS.build_matmul(_entry(), _binding())
    assert not dr.epilogue_reach([])["requant"]
    assert dr.epilogue_reach([capsule])["requant"] == [capsule["name"]]
