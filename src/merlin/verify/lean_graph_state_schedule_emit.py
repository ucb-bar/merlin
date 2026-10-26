# ruff: noqa: E501
# Exact literal Lean statement equations remain byte-bound and readable.
"""Whole actual-module order certificates on existing Scalar/Pipeline semantics."""

from __future__ import annotations

from .lean_graph_state_scalar import budget, program, types
from .lean_graph_state_schedule_errors import ScheduleDomainRefusal, ScheduleUnavailable, require


def prefix_author(prefix, mode):
    from .lean_graph_state_integer import author

    try:
        return author(prefix, mode)
    except ValueError as error:
        # These are specific arithmetic/domain refusals in the READONLY author.
        # Identity/runtime and unexpected author errors do not share this class.
        known = (
            "actual ordered expression lacks a supported exact local rewrite proof",
            "complete retained call/loop/observation skeleton differs", "retained effect descriptor/result order differs",
            "ordered loop carried types differ", "complete observation/operand vector arity differs",
            "runtime representation proof has wrong static expression width/type",
            "wrong complete typed expression vector arity", "source index arithmetic outside generated control bridges is unsupported",
            "inline/empty/tail explicitly outside supported no-tail bulk mode", "two-input changed function submode required",
            "exactly one top-level changed loop required", "one to four carries supported",
            "actual source/bulk/segmented control joins differ", "supported factor2..4/tile2..16/signed32-or64 profile required",
        )
        if str(error) in known:
            raise ScheduleDomainRefusal(str(error)) from error
        raise ScheduleUnavailable("unexpected prefix author error: " + str(error)) from error


def emit(capture, *, mode="graph"):
    require(mode in ("graph", "initialized-symbolic-provider-v1", "initialized-scalar-provider-v1"), "unknown supported scheduling model mode")
    pairs = capture["pairs"]
    require(type(pairs) is list and pairs, "nonempty complete actual order trace required")
    registry = pairs[0]["registry"]
    require(all(pair["registry"] == registry and pair["types"] == pairs[0]["types"] for pair in pairs),
            "one common schedule registry/type interpretation required")
    prefix, old_roots, model = "", [], None
    if capture["prefix"] is not None:
        require(mode != "initialized-scalar-provider-v1", "standalone scalar witness mode is separate from the seven-buffer model")
        prefix, old_roots, model = prefix_author(capture["prefix"], mode)
        prefix = prefix.replace("Merlin.GraphStateIntegerInstance", "Merlin.GraphStateScheduleInstance.Prefix")
        old_roots = [name.replace("Merlin.GraphStateIntegerInstance", "Merlin.GraphStateScheduleInstance.Prefix") for name in old_roots]
    else:
        require(mode != "initialized-symbolic-provider-v1", "initialized combined model requires actual seven-buffer prefix")
    value = "[" + ",".join("{inputs := " + types(r["inputs"]) + ",outputs := " + types(r["outputs"]) + "}" for r in registry) + "]"
    text = "import Merlin.GraphStateSchedule\nimport Merlin.GraphStateScheduleCountermodels\n" + prefix
    text += "set_option maxRecDepth 4096\nset_option maxHeartbeats 2000000\nnamespace Merlin.GraphStateScheduleInstance\n"
    text += "open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline Merlin.GraphState.Schedule\n"
    if prefix:
        text += "abbrev registry := Prefix.registry\ntheorem registryBound : registry = " + value + " := by decide\n"
    else:
        text += "def registry : List Signature := " + value + "\n"
    defined = [i for i, row in enumerate(pairs[0]["functions"]) if not row["declaration"]]
    require(bool(defined), "defined complete function required")
    for index in defined:
        first = pairs[0]["functions"][index]
        records = [first["before"], *[pair["functions"][index]["after"] for pair in pairs]]
        text += f"namespace F{index}\ndef inputs : List Ty := " + types(first["inputs"]) + "\ndef outputs : List Ty := " + types(first["outputs"]) + "\n"
        for stage, raw in enumerate(records):
            budget(raw, len(first["inputs"]), [0])
            text += f"def S{stage} : Program := " + program(raw) + "\n"
        for stage in range(len(pairs)):
            text += f"theorem accepted{stage} : check registry inputs outputs S{stage} S{stage+1} = true := by decide\n"
            text += f"theorem actual{stage} : Preserves registry inputs.length S{stage} S{stage+1} := check_sound accepted{stage}\n"
        text += f"end F{index}\n"
    for stage in range(len(pairs) + 1):
        text += f"def E{stage} : List Entry := [" + ",".join(
            f"{{inputs := F{i}.inputs, outputs := F{i}.outputs, body := F{i}.S{stage}}}" for i in defined) + "]\n"
    middles = "[" + ",".join(f"E{i}" for i in range(1, len(pairs))) + "]"
    text += f"theorem accepted : checkSteps registry E0 {middles} E{len(pairs)} = true := by decide\n"
    text += f"theorem schedule_complete {{Value World Error : Type}} (model : Model Value World Error registry) :\n  PreservesEntries model E0 E{len(pairs)} := checkSteps_sound {middles} accepted model\n"
    if prefix:
        text += "theorem exactMiddle : Prefix.Suffix.finalEntries = E0 := by decide\n"
        # The prefix's root and the generic scheduler root operate on identical
        # actual Programs/registry. An IR hash is not a semantic bridge premise.
        text += f"theorem complete {{World Error : Type}} (external : External World Error registry)\n"
        text += "    (typed : ∀ identity arguments world values final,\n      Merlin.GraphState.Integer.Values Merlin.GraphState.Integer.Canonical (signature registry identity).inputs arguments →\n      external.run identity arguments world = .ok (values,final) →\n      Merlin.GraphState.Integer.Values Merlin.GraphState.Integer.Canonical (signature registry identity).outputs values) :\n"
        text += f"  Merlin.GraphState.Integer.PreservesEntriesCanonical (Prefix.Suffix.M Merlin.GraphState.Integer.primitive external) Prefix.originalEntries E{len(pairs)} := by\n"
        text += "  apply Merlin.GraphState.Integer.compose_suffix _ (Prefix.complete external typed)\n  rw [exactMiddle]\n  exact schedule_complete _\n"
    roots = ["Merlin.GraphState.Schedule.checkPair_sound", "Merlin.GraphState.Schedule.checkSteps_sound",
             "Merlin.GraphState.Schedule.checked_reached", "Merlin.GraphStateScheduleInstance.accepted",
             "Merlin.GraphStateScheduleInstance.schedule_complete", *old_roots]
    roots += ["Merlin.GraphState.Schedule.Countermodels." + name for name in (
        "both_statically_typed", "first_failure", "reordered_failure", "partial_actions_cannot_be_reordered",
        "regions_statically_typed", "anchored_region_failure", "moved_region_failure", "region_totality_cannot_be_inferred_from_empty_effects",
    )]
    if prefix:
        roots += ["Merlin.GraphStateScheduleInstance.exactMiddle", "Merlin.GraphStateScheduleInstance.complete"]
    if model is not None:
        functions = capture["prefix"]["prefix"]["functions"]
        index = next(i for i, f in enumerate(functions) if not f["declaration"] and f["before"] != f["after"])
        text += f"namespace FinalWitness\nopen Prefix.IntegerProvider\n"
        last = len(pairs)
        for fail, result in (("false", f".ok ([17],{model['success']['world']})"), ("true", f".error (.external {model['failure']['error']},{model['failure']['world']})")):
            label = "success" if fail == "false" else "failure"
            text += f"theorem final_{label} : evalProgram (chosen {fail}) F{index}.S{last} arguments 7 = {result} := by\n"
            for step in reversed(range(last)):
                text += f"  rw [← F{index}.actual{step} Int Nat Int (chosen {fail}) arguments 7 rfl]\n"
            text += f"  change evalProgram (chosen {fail}) Prefix.Suffix.F{index}.S4 arguments 7 = _\n  exact Prefix.IntegerProvider.actual_final_{label}\n"
            roots.append("Merlin.GraphStateScheduleInstance.FinalWitness.final_" + label)
        text += "end FinalWitness\n"
    text += "end Merlin.GraphStateScheduleInstance\n"
    if mode == "initialized-scalar-provider-v1":
        from .lean_graph_state_schedule_models import standalone

        return standalone(capture, text, roots)
    return text, roots, model
