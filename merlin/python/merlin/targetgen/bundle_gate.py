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
           "REFERENCE_KINDS", "COMPARISONS", "CONSOLE_DUMP_CAP", "CHANNELS",
           "reference_digest", "gate_from_session_contract",
           "grades_unquantized_activations", "reference_spread", "ReferenceSpread"]


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
        "weights": "full", "activations": "full",
    },
    "weight_only_int8": {
        "file": "golden.npy",
        "quantization": "weights int8, activations fp32",
        # THE 0.484 CASE. Quantized weights, UNQUANTIZED activations -- which is why one field
        # cannot express it: it is quantized enough to refuse an fp32 datapath and unquantized
        # enough to refuse a W8A8 correctness gate.
        "weights": "quantized", "activations": "full",
        "scope": ("weight-only int8; grading a W8A8 run against it measures ACTIVATION "
                  "quantization error, which is how a correct run once scored cos 0.484 and was "
                  "chased as a codegen defect"),
        "generator": "the capture itself",
    },
    "w8a8_execution": {
        "file": "golden_w8a8.npy",
        "quantization": "W8A8, by merlin's own int8 datapath",
        "weights": "quantized", "activations": "quantized",
        "scope": ("whether the DEVICE reproduced what the host compiler computes -- a host run "
                  "scores cos 1.0 / rel 0.0 by construction, so it is not evidence about the "
                  "arithmetic"),
        "generator": "build_tools/scripts/make_w8a8_golden.py",
    },
    "w8a8_independent": {
        "file": "golden_w8a8.independent.npy",
        "quantization": "W8A8, by torchao int8_dyn_act_int8_weight in torch eager",
        "weights": "quantized", "activations": "quantized",
        "scope": ("the arithmetic itself; fail-closed on the bundle's quantized weights not "
                  "matching bit-for-bit, which is what makes the number citable"),
        "generator": "build_tools/scripts/make_w8a8_independent_golden.py",
    },
    "pt2e_integer": {
        "file": "golden_integer.npy",
        "quantization": "W8A8, by a PT2E integer reference over quantized_decomposed Q/DQ pairs",
        "weights": "quantized", "activations": "quantized",
        "scope": ("an exact integer reference; available ONLY for a TorchAO PT2E Q/DQ graph, so it "
                  "does not transfer to a torchao int8_dyn_act_int8_weight capture"),
        "generator": "model2MLIR m2m.capture.pt2e_integer_reference",
    },
    "eager_same_precision": {
        "file": "session_goldens.npz",
        "quantization": "same precision as the device path, in torch eager",
        "scope": "a multi-step trajectory, keyed by the session contract's declared output",
        "generator": "the capture's session contract",
        # Its precision is DEFINED as the datapath's, so it is admissible on either datapath --
        # unlike a reference whose precision is fixed independently of what the device runs.
        "weights": "follows_datapath", "activations": "follows_datapath",
    },
    "eager_fp32": {
        "file": "session_quality_fp32.npz",
        "quantization": "none -- torch eager at full precision",
        "weights": "full", "activations": "full",
        "scope": ("how much accuracy the QUANTIZATION cost, which is a model-quality claim. It is "
                  "the reference the 0.484 incident was graded against by mistake: a correct W8A8 "
                  "datapath differs from it by design, so it can never be a correctness gate"),
        "generator": "the capture's session contract",
    },
}

#: The two claims a session contract declares references for. They are different questions, and
#: reporting one as the other is the 0.484 incident: an unquantized reference used as a correctness
#: gate reads a by-design quantization difference as a codegen defect.
CHANNELS: Mapping[str, Mapping[str, Any]] = {
    "correctness": {
        "field": "correctness",
        "asks": "did the device compute what the compiler's precision says it should",
        "admits_unquantized_reference": False,
    },
    "quality": {
        "field": "quality",
        "asks": "how much accuracy the quantization cost, versus full precision",
        "admits_unquantized_reference": True,
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

#: Every precision a reference may declare, on each of its two axes. ``follows_datapath`` means the
#: precision is DEFINED as whatever the device runs, so it is admissible on either datapath; the
#: other two are fixed independently of the device and so constrain which datapath they can grade.
#:
#: TWO AXES, NOT ONE, and the reason is ``weight_only_int8``: its weights are int8 while its
#: activations stay fp32. A single "is it quantized" field puts it on the wrong side of one rule
#: whichever value it takes -- and the value that reads as "quantized" is the one that switches OFF
#: the 0.484 refusal, i.e. the failure is silent and in the permissive direction.
_PRECISIONS: frozenset[str] = frozenset({"full", "quantized", "follows_datapath"})

def grades_unquantized_activations(reference_kind: str) -> bool:
    """Whether this reference's ACTIVATIONS are full-precision, which is the 0.484 axis.

    THE ONLY place this rule is expressed. It was briefly two -- a module-level frozenset and an
    inline field read -- and a mutation flipping the set's axis passed the whole suite, because the
    two could disagree silently and only one was on the path the tests exercised. A rule stated
    twice is a rule that can be half-fixed.
    """
    kind = REFERENCE_KINDS.get(reference_kind) or {}
    return kind.get("activations") == "full"


#: The reference kinds that predicate currently selects. Derived, never listed: a reference added
#: without an ``activations`` axis is placed by :func:`grades_unquantized_activations` as False and
#: refused earlier, by the axis check, rather than silently admitted here.
_ACTIVATION_UNQUANTIZED = frozenset(
    name for name in REFERENCE_KINDS if grades_unquantized_activations(name))

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
    #: Independent ranking rows in the graded output. A language model's result IS the ranking, and
    #: it is per TOKEN: one global argmax over TinyLlama's (1, 8, 32000) = 256,000 logits agrees
    #: with the reference whenever the single largest logit anywhere happens to land in the same
    #: place, and says nothing about the other seven positions. So the rows are declared and each
    #: is ranked separately.
    rows: int = 1
    steps: int = 1
    session_key: str = ""
    scope_note: str = ""
    #: Which claim this gate makes. ``correctness`` grades the datapath against a reference at its
    #: own precision; ``quality`` measures what the quantization cost against full precision. They
    #: are different numbers, and a recorded block that did not say which is the 0.484 shape.
    channel: str = "correctness"

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
                "output_elements": self.output_elements, "rows": self.rows,
                "elements_per_row": self.output_elements // self.rows,
                "prints_values": self.prints_values,
                "console_dump_cap": CONSOLE_DUMP_CAP,
                "expected_argmax": self.expected_argmax, "steps": self.steps,
                "session_key": self.session_key, "scope_note": self.scope_note,
                "channel": self.channel, "channel_asks": CHANNELS[self.channel]["asks"],
                "declared": "before the run; a tolerance chosen after seeing the result is not a gate"}


def gate_for(*, model: str, datapath: str, reference_kind: str, comparison: str,
             atol: float, rtol: float, output_elements: int,
             expected_argmax: int | None = None, rows: int = 1, steps: int = 1,
             session_key: str = "", scope_note: str = "",
             channel: str = "correctness",
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
    if rows < 1:
        raise GateError("a gate must grade at least one ranking row")
    if output_elements % rows:
        raise GateError(
            f"{model}: {output_elements} element(s) do not divide into {rows} ranking row(s); a "
            f"row-wise argmax over a ragged split would rank across a row boundary")
    if rows > 1 and expected_argmax is not None:
        raise GateError(
            f"{model}: a literal expected_argmax describes ONE ranking, but {rows} rows were "
            f"declared. Per-row agreement with the reference is the check that means something "
            f"for a multi-row output")
    if channel not in CHANNELS:
        raise GateError(f"channel {channel!r} is not one of {sorted(CHANNELS)}; a gate that does "
                        f"not say which claim it makes invites its number being read as the other")

    kind = REFERENCE_KINDS[reference_kind]

    weights_precision = str(kind.get("weights") or "")
    activations_precision = str(kind.get("activations") or "")  # noqa: F841 -- checked below
    if weights_precision not in _PRECISIONS or activations_precision not in _PRECISIONS:
        raise GateError(
            f"{model}: reference {reference_kind!r} does not declare both a weight and an "
            f"activation precision from {sorted(_PRECISIONS)}; whether it can grade this datapath "
            f"is UNKNOWN and is refused rather than assumed")

    # THE 0.484 REFUSAL -- a quantized datapath graded against a full-precision reference measures
    # activation-quantization error. Scoped to channels that do not ADMIT an unquantized reference:
    # the `quality` channel's whole purpose is exactly that comparison, so refusing it there would
    # refuse the measurement rather than the misattribution.
    if (datapath == "w8a8" and grades_unquantized_activations(reference_kind)
            and not CHANNELS[channel]["admits_unquantized_reference"]):
        better = "w8a8_independent"
        raise GateError(
            f"{model}: a w8a8 datapath cannot be graded against the {reference_kind!r} reference "
            f"({kind['file']}) on the {channel!r} channel -- {kind['scope']}. Use {better!r} "
            f"({REFERENCE_KINDS[better]['file']}), produced by "
            f"{REFERENCE_KINDS[better]['generator']}")

    # The mirror image: an unquantized datapath graded against a quantized reference would be
    # measuring the REFERENCE's quantization. A datapath-following reference is fine on either.
    if datapath == "fp32" and "quantized" in (weights_precision, activations_precision):
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
        rows=rows, steps=steps, session_key=session_key, scope_note=scope_note, channel=channel)


# ---------------------------------------------------------------------------------------------
# Deriving the gate from the capture's own declaration
# ---------------------------------------------------------------------------------------------
#
# A session contract already declares everything a trajectory gate needs -- the reference kind, the
# file, the keyed array, the output index, the step count, and a `reference_sha256`. Choosing those
# again here would be choosing a gate; reading them is deriving one. So :func:`gate_from_session_contract`
# is the only way a session-scoped gate should be built, and the caller supplies only the two things
# the contract does NOT declare: the tolerances, and which datapath is being graded.
#
# `reference_sha256` was declared by the schema and verified by nothing. Its subject is not the file:
# an ``.npz`` is a zip whose bytes carry timestamps and are not reproducible, so hashing the container
# would fail for every capture. Measured against all four captures on disk (8 of 8 channels), the
# digest is over the KEYED ARRAY's contiguous float32 bytes -- which is why :func:`reference_digest`
# hashes that and nothing else. Getting this subject wrong is not a harmless miss: it reports every
# capture as corrupt, which is indistinguishable from the check being broken and is how a verification
# gets switched off.


@dataclass(frozen=True)
class ReferenceSpread:
    """How far apart a capture's OWN two references are, on the array a gate grades.

    A capture that ships both an ``eager_same_precision`` and an ``eager_fp32`` reference has
    already measured its own quantization spread. That number is the floor on any meaningful
    tolerance: asserting the DEVICE is closer to the same-precision reference than that reference is
    to its own full-precision sibling is asserting the device is more faithful than the arithmetic
    it implements.
    """

    max_absolute: float
    mean_absolute: float
    max_relative: float
    cosine: float
    elements: int
    def to_dict(self) -> dict[str, Any]:
        return {"max_absolute": self.max_absolute, "mean_absolute": self.mean_absolute,
                "max_relative": self.max_relative, "cosine": self.cosine,
                "elements": self.elements}


def reference_spread(a_path: Any, b_path: Any, key: str) -> ReferenceSpread:
    """Measure the gap between two references on ``key``. Both must hold the same shape."""
    import numpy as np

    arrays = []
    for path in (a_path, b_path):
        with np.load(path) as data:
            if key not in data.files:
                raise GateError(f"{path}: key {key!r} is absent, so no spread can be measured")
            arrays.append(np.ascontiguousarray(data[key], dtype=np.float64))
    first, second = arrays
    if first.shape != second.shape:
        raise GateError(
            f"the two references have shapes {first.shape} and {second.shape}; a spread between "
            f"different shapes is not a number about this model")
    difference = np.abs(first - second)
    magnitude = np.maximum(np.abs(second), np.finfo(np.float64).tiny)
    norms = float(np.linalg.norm(first) * np.linalg.norm(second))
    return ReferenceSpread(
        max_absolute=float(difference.max()),
        mean_absolute=float(difference.mean()),
        max_relative=float((difference / magnitude).max()),
        cosine=(float(first.ravel() @ second.ravel() / norms) if norms > 0 else 0.0),
        elements=int(first.size))


def reference_digest(golden_path: Any, key: str) -> str:
    """The digest a session contract's ``reference_sha256`` declares, over the bytes it declares it of.

    The subject is the keyed array's contiguous float32 content -- NOT the ``.npz`` container, whose
    zip framing is not byte-reproducible. Established by agreement with every declared digest in the
    tree rather than assumed.
    """
    import hashlib

    import numpy as np

    with np.load(golden_path) as data:
        if key not in data.files:
            raise GateError(f"{golden_path}: the contract's key {key!r} is absent from the golden "
                            f"(it holds {sorted(data.files)}), so there is nothing to digest")
        values = np.ascontiguousarray(data[key], dtype=np.float32)
    return hashlib.sha256(values.tobytes()).hexdigest()


def gate_from_session_contract(contract: Mapping[str, Any], *, model: str, datapath: str,
                               bundle_dir: Any, atol: float, rtol: float,
                               channel: str = "correctness",
                               verify_digest: bool = True,
                               require_derivable_tolerance: bool = True) -> CorrectnessGate:
    """Build the gate the capture DECLARES, verifying the reference is the one it declared.

    Everything but ``atol``/``rtol`` and ``datapath`` comes off the contract. A contract that
    declares a reference this module cannot place, omits its digest, or whose golden's bytes disagree
    with the declared digest is REFUSED -- a golden regenerated by different code keeps its filename,
    and grading against it silently is the failure this exists to stop.

    THE TOLERANCE IS CHECKED AGAINST THE CAPTURE, not merely required up front. Declaring a
    tolerance before the run is the right discipline and is not sufficient: the NUMBER can still be
    arbitrary. Measured on SmolVLA, an atol/rtol of 1e-4 declared in good faith turns out to sit
    inside the model's own quantization spread -- comparing the capture's two references to each
    other, 98.4% of elements fail it and the worst disagreement is 5.36e-2. A gate at that tolerance
    reports a failure for every conforming datapath, which is how a correct run gets chased as a
    codegen defect (this repo's cos-0.484 incident, in a subtler form).

    So when the contract ships both a correctness and a quality reference, the spread between THEM
    is measured and a tolerance tighter than it is refused with the numbers. ``eager_fp32`` is the
    reference the quality channel grades against, so the spread is exactly "how far this model's
    quantized arithmetic is from full precision" -- a floor no device can be held below.
    """
    from pathlib import Path

    import numpy as np

    if channel not in CHANNELS:
        raise GateError(f"channel {channel!r} is not one of {sorted(CHANNELS)}")
    field = str(CHANNELS[channel]["field"])
    spec = contract.get(field)
    if not isinstance(spec, Mapping) or not spec:
        raise GateError(
            f"{model}: the session contract declares no {field!r} block, so it declares no "
            f"{channel} reference; a gate invented here would be a gate chosen by the grader")

    scope = str(spec.get("scope") or "")
    if scope != "trajectory":
        raise GateError(
            f"{model}: the contract declares {field}.scope={scope!r}; this builder grades the "
            f"declared multi-step trajectory and must not silently regrade a different scope")

    reference_kind = str(spec.get("reference") or "")
    if reference_kind not in REFERENCE_KINDS:
        raise GateError(
            f"{model}: the contract declares {field}.reference={reference_kind!r}, which this "
            f"module cannot place among {sorted(REFERENCE_KINDS)}. A reference whose meaning is "
            f"unknown is refused rather than graded against")

    if (not CHANNELS[channel]["admits_unquantized_reference"]
            and grades_unquantized_activations(reference_kind)):
        raise GateError(
            f"{model}: {reference_kind!r} is the {channel!r} channel's reference in this contract, "
            f"but {REFERENCE_KINDS[reference_kind]['scope']}")

    golden_name = str(spec.get("golden") or "")
    if not golden_name:
        raise GateError(f"{model}: the contract's {field} block names no golden file")
    golden = Path(bundle_dir) / golden_name
    if not golden.is_file():
        raise GateError(
            f"{model}: the contract's {field} golden is absent: {golden}. Produce it with "
            f"{REFERENCE_KINDS[reference_kind]['generator']} rather than grading against a "
            f"reference that measures something else")

    key = str(spec.get("key") or "")
    if not key:
        raise GateError(f"{model}: the contract's {field} block names no keyed array to grade")

    declared_digest = str(spec.get("reference_sha256") or "")
    if verify_digest:
        if not declared_digest:
            raise GateError(
                f"{model}: the contract's {field} block declares no reference_sha256, so the golden "
                f"on disk cannot be shown to be the one the capture produced. Fail closed: a "
                f"regenerated golden keeps its filename")
        actual = reference_digest(golden, key)
        if actual != declared_digest:
            raise GateError(
                f"{model}: {golden_name}[{key!r}] digests to {actual} but the contract declares "
                f"{declared_digest}. The golden on disk is not the one this contract was written "
                f"against, and grading a datapath against it would attribute someone else's "
                f"reference to this run")

    with np.load(golden) as data:
        values = np.asarray(data[key])
    if values.ndim < 2:
        raise GateError(
            f"{model}: {golden_name}[{key!r}] has shape {tuple(values.shape)}, which carries no "
            f"per-step axis, so it cannot be the trajectory the contract declares")
    observed_steps = int(values.shape[0])
    per_step_elements = int(np.prod(values.shape[1:]))

    declared_steps = int(contract.get("steps") or 0)
    if declared_steps and declared_steps != observed_steps:
        raise GateError(
            f"{model}: the contract declares {declared_steps} step(s) but {golden_name}[{key!r}] "
            f"holds {observed_steps}; grading the shorter one and reporting the declared count is "
            f"how a partial trajectory gets published as the whole one")
    steps = declared_steps or observed_steps

    # THE TOLERANCE FLOOR, measured from the capture's own pair of references.
    spread = None
    other = "quality" if channel == "correctness" else "correctness"
    other_spec = contract.get(other)
    if isinstance(other_spec, Mapping) and other_spec.get("golden"):
        other_golden = Path(bundle_dir) / str(other_spec["golden"])
        if other_golden.is_file() and str(other_spec.get("key") or "") == key:
            spread = reference_spread(golden, other_golden, key)
            floor = spread.max_absolute
            if require_derivable_tolerance and (atol + rtol) < floor:
                raise GateError(
                    f"{model}: the declared tolerance (atol {atol:g} + rtol {rtol:g}) is tighter "
                    f"than this capture's OWN reference spread. Its {reference_kind!r} and "
                    f"{str(other_spec.get('reference'))!r} references differ by up to {floor:.6g} "
                    f"(mean {spread.mean_absolute:.6g}, cosine {spread.cosine:.9f}) over "
                    f"{spread.elements} elements. A gate this tight fails for every conforming "
                    f"datapath, because it asserts the device is closer to one reference than that "
                    f"reference is to its own full-precision sibling. Declare a tolerance at or "
                    f"above {floor:.6g}, or pass require_derivable_tolerance=False and say why")

    scope_note = (f"{CHANNELS[channel]['asks']}; over the contract-declared {steps}-step "
                  f"trajectory of output {int(spec.get('output_index', 0))}, key {key!r}"
                  + (f"; the capture's own two references differ by up to "
                     f"{spread.max_absolute:.6g} (cosine {spread.cosine:.9f}), which is the floor "
                     f"this tolerance was checked against" if spread is not None else ""))
    return gate_for(model=model, datapath=datapath, reference_kind=reference_kind,
                    comparison="trajectory", atol=atol, rtol=rtol,
                    output_elements=per_step_elements, steps=steps, session_key=key,
                    scope_note=scope_note, channel=channel,
                    available_references={reference_kind: True})
