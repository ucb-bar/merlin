"""The correctness gate a whole-model bundle is graded by, DECLARED before the run.

WHY A DECLARED GATE. A tolerance chosen after seeing the result is not a gate, and this tree has the
receipts for both halves of that. So a gate names its reference, its comparison, its tolerances and
what the harness prints, up front — and refuses a combination that cannot mean what it would claim.

THE REFUSAL THAT MATTERS MOST. In an ``*_int8_*`` recapture, ``golden.npy`` is a **weight-only-int8**
reference: the weights are quantized and the activations stay fp32. Merlin's path is W8A8. Grading
W8A8 against it measures activation-quantization error, and that cost real time once — a K1 RVV run
scored **cos 0.484** against ``golden.npy`` and **cos 1.0, rel 0.0** against ``golden_w8a8.npy``, and
the 0.484 was chased as a codegen defect. So :func:`gate_for` REFUSES ``golden.npy`` for a W8A8
bundle by name and says which generator produces the right reference. Right now
``tiny_llama_int8_w8a8_consistent`` ships *only* ``golden.npy``, so that refusal fires on the real
tree and names the missing artifact rather than grading against the wrong one.

THE SECOND REFUSAL. ``golden_w8a8.npy`` is an **execution** reference — merlin's own int8 datapath.
A host run scores ``cos 1.0 / rel 0.0`` against it by construction, because the two sides are the
same program; ``make_w8a8_independent_golden.py``'s own docstring records a case where it reproduced
a shipped golden bit-for-bit while the post-fix code differed by 0.0755. It answers "did the device
reproduce the host compiler", which is a real question, and it is NOT evidence about the arithmetic.
A gate may use it only when it declares that scope.

WHAT IS NOT A GATE. Dumping every logit. A TinyLlama step is ``(1, 8, 32000)`` = 256,000 floats, and
the console is the binding constraint on what is gradeable at all (``dump_cap = 4096`` exists for
exactly this). So a large output declares a digest plus per-row ``ARGMAX``, and the gate says so.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["ReferenceKind", "Comparison", "CorrectnessGate", "gate_for", "GateError",
           "REFERENCE_KINDS", "COMPARISONS", "CONSOLE_DUMP_CAP"]


class GateError(ValueError):
    """The declared gate cannot mean what it claims, and the message says what to fix."""


#: Every reference this repo knows how to produce, with what it is evidence ABOUT. The scope text is
#: not documentation: :func:`gate_for` quotes it into the refusal when a gate misuses one.
REFERENCE_KINDS: Mapping[str, Mapping[str, Any]] = {
    "fp32": {
        "file": "golden.npy",
        "quantization": "none",
        "scope": "the unquantized model; a W8A8 datapath differs from it by design",
        "generator": "the capture itself",
    },
    "weight_only_int8": {
        "file": "golden.npy",
        "quantization": "weights int8, activations fp32",
        "scope": ("weight-only int8; grading a W8A8 run against it measures ACTIVATION "
                  "quantization error, which is how a correct run once scored cos 0.484 and was "
                  "chased as a codegen defect"),
        "generator": "the capture itself",
    },
    "w8a8_execution": {
        "file": "golden_w8a8.npy",
        "quantization": "W8A8, by merlin's own int8 datapath",
        "scope": ("whether the DEVICE reproduced what the host compiler computes -- a host run "
                  "scores cos 1.0 / rel 0.0 by construction, so it is not evidence about the "
                  "arithmetic"),
        "generator": "build_tools/scripts/make_w8a8_golden.py",
    },
    "w8a8_independent": {
        "file": "golden_w8a8.independent.npy",
        "quantization": "W8A8, by torchao int8_dyn_act_int8_weight in torch eager",
        "scope": ("the arithmetic itself; fail-closed on the bundle's quantized weights not "
                  "matching bit-for-bit, which is what makes the number citable"),
        "generator": "build_tools/scripts/make_w8a8_independent_golden.py",
    },
    "pt2e_integer": {
        "file": "golden_integer.npy",
        "quantization": "W8A8, by a PT2E integer reference over quantized_decomposed Q/DQ pairs",
        "scope": ("an exact integer reference; available ONLY for a TorchAO PT2E Q/DQ graph, so it "
                  "does not transfer to a torchao int8_dyn_act_int8_weight capture"),
        "generator": "model2MLIR m2m.capture.pt2e_integer_reference",
    },
    "eager_same_precision": {
        "file": "session_goldens.npz",
        "quantization": "same precision as the device path, in torch eager",
        "scope": "a multi-step trajectory, keyed by the session contract's declared output",
        "generator": "the capture's session contract",
    },
}

#: How a gate compares. Each carries what it can and cannot conclude.
COMPARISONS: Mapping[str, str] = {
    "exact_elementwise": ("every element within an absolute+relative tolerance, with a count of "
                          "violations; only meaningful against an exact integer reference"),
    "tolerance_and_topk": ("a stated tolerance plus the argmax per row; the right shape for a "
                           "language model, where the ranking is the result and the logits are not"),
    "trajectory": ("a declared key over a multi-step session, compared step by step; a single step "
                   "is not the trajectory and must not be reported as it"),
}

#: Above this many output elements the harness prints a digest and per-row argmax instead of values.
#: The console is the binding constraint on what is gradeable at all.
CONSOLE_DUMP_CAP = 4096

#: Reference kinds whose activations are NOT quantized. A W8A8 bundle graded against one of these
#: measures activation-quantization error and nothing it claims to.
_ACTIVATION_UNQUANTIZED = frozenset({"fp32", "weight_only_int8"})

#: Comparisons that require an exact reference, i.e. one with no rounding disagreement to absorb.
_NEEDS_EXACT_REFERENCE = frozenset({"exact_elementwise"})
_EXACT_REFERENCES = frozenset({"pt2e_integer"})


@dataclass(frozen=True)
class CorrectnessGate:
    """One model's declared gate. Every field is decided before the run, not after."""

    model: str
    datapath: str                    # "w8a8" | "fp32"
    reference_kind: str
    comparison: str
    atol: float
    rtol: float
    #: Elements in the graded output. Decides whether the harness may print values at all.
    output_elements: int
    #: The expected argmax, when the model has a single stable one (ResNet-50's is 258). None means
    #: the gate checks argmax AGREEMENT with the reference instead of a literal.
    expected_argmax: int | None = None
    steps: int = 1
    session_key: str = ""
    scope_note: str = ""

    @property
    def reference_file(self) -> str:
        return str(REFERENCE_KINDS[self.reference_kind]["file"])

    @property
    def prints_values(self) -> bool:
        """Whether the harness may dump the output, or must fall back to digest + argmax."""
        return self.output_elements <= CONSOLE_DUMP_CAP

    def to_dict(self) -> dict[str, Any]:
        kind = REFERENCE_KINDS[self.reference_kind]
        return {"schema": "merlin_bundle_correctness_gate_v1", "model": self.model,
                "datapath": self.datapath, "reference_kind": self.reference_kind,
                "reference_file": self.reference_file,
                "reference_scope": kind["scope"], "reference_generator": kind["generator"],
                "comparison": self.comparison,
                "comparison_licence": COMPARISONS[self.comparison],
                "atol": self.atol, "rtol": self.rtol,
                "output_elements": self.output_elements, "prints_values": self.prints_values,
                "console_dump_cap": CONSOLE_DUMP_CAP,
                "expected_argmax": self.expected_argmax, "steps": self.steps,
                "session_key": self.session_key, "scope_note": self.scope_note,
                "declared": "before the run; a tolerance chosen after seeing the result is not a gate"}


def gate_for(*, model: str, datapath: str, reference_kind: str, comparison: str,
             atol: float, rtol: float, output_elements: int,
             expected_argmax: int | None = None, steps: int = 1,
             session_key: str = "", scope_note: str = "",
             available_references: Mapping[str, bool] | None = None) -> CorrectnessGate:
    """Build a gate, or refuse and name what would have made it meaningless.

    ``available_references`` maps a reference kind to whether its file exists in the bundle. Supplied,
    it turns "this gate needs a reference nobody has generated" into a refusal that names the
    generator to run -- which is the state ``tiny_llama_int8_w8a8_consistent`` is in today.
    """
    if reference_kind not in REFERENCE_KINDS:
        raise GateError(f"reference kind {reference_kind!r} is not one of "
                        f"{sorted(REFERENCE_KINDS)}; a reference nobody can produce is not a gate")
    if comparison not in COMPARISONS:
        raise GateError(f"comparison {comparison!r} is not one of {sorted(COMPARISONS)}")
    if datapath not in ("w8a8", "fp32"):
        raise GateError(f"datapath {datapath!r} must be 'w8a8' or 'fp32'; it decides which "
                        f"references can mean anything")
    if atol < 0 or rtol < 0:
        raise GateError("tolerances must be non-negative")
    if output_elements <= 0:
        raise GateError("a gate over no output elements checks nothing")
    if steps < 1:
        raise GateError("a gate must grade at least one step")

    kind = REFERENCE_KINDS[reference_kind]

    # THE 0.484 REFUSAL.
    if datapath == "w8a8" and reference_kind in _ACTIVATION_UNQUANTIZED:
        better = "w8a8_independent"
        raise GateError(
            f"{model}: a w8a8 datapath cannot be graded against the {reference_kind!r} reference "
            f"({kind['file']}) -- {kind['scope']}. Use {better!r} "
            f"({REFERENCE_KINDS[better]['file']}), produced by "
            f"{REFERENCE_KINDS[better]['generator']}")

    if datapath == "fp32" and reference_kind not in ("fp32",):
        raise GateError(
            f"{model}: an fp32 datapath graded against the {reference_kind!r} reference would be "
            f"measuring the reference's quantization, not the datapath")

    # An exact elementwise comparison needs a reference with no rounding to absorb.
    if comparison in _NEEDS_EXACT_REFERENCE and reference_kind not in _EXACT_REFERENCES:
        raise GateError(
            f"{model}: {comparison!r} needs an exact reference ({sorted(_EXACT_REFERENCES)}); "
            f"{reference_kind!r} is {kind['quantization']}, so an elementwise equality would be "
            f"asserting that two different roundings agree")

    if comparison == "trajectory":
        if steps < 2:
            raise GateError(
                f"{model}: a trajectory gate over {steps} step(s) is a single step, and a single "
                f"step must not be reported as the trajectory")
        if not session_key:
            raise GateError(f"{model}: a trajectory gate must name the session output it grades")
    elif steps != 1:
        raise GateError(
            f"{model}: {comparison!r} grades one step, so declaring {steps} is a scope claim the "
            f"comparison does not support -- use 'trajectory'")

    if reference_kind == "w8a8_execution" and not scope_note:
        raise GateError(
            f"{model}: the {reference_kind!r} reference answers only whether the device reproduced "
            f"the host compiler ({kind['scope']}); a gate using it must declare that scope in "
            f"scope_note so the number is not read as evidence about the arithmetic")

    if available_references is not None and not available_references.get(reference_kind):
        raise GateError(
            f"{model}: the {reference_kind!r} reference ({kind['file']}) is not present in this "
            f"bundle. Produce it with {kind['generator']} rather than falling back to a reference "
            f"that measures something else")

    return CorrectnessGate(
        model=model, datapath=datapath, reference_kind=reference_kind, comparison=comparison,
        atol=atol, rtol=rtol, output_elements=output_elements, expected_argmax=expected_argmax,
        steps=steps, session_key=session_key, scope_note=scope_note)
