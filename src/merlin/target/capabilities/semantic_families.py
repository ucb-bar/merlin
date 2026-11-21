"""Canonical semantic-computation families — the target-agnostic vocabulary the capability contract
and the eligibility oracle both speak.

Coverage is argued over *semantic families*, not framework op names: ``aten.linear``, ``onnx.Gemm``,
``stablehlo.dot_general``, ``linalg.matmul`` and an ``einsum`` all arrive at the same **contraction**
family, so a compiler that supports the family supports all of them. This module is the single source
of truth for that vocabulary; :mod:`merlin.target.capabilities.eligibility` and the coverage certificate read it.

Two layers:

- **Primitives** — the closed set of irreducible computation kinds a compute unit can be said to run:
  ``contraction`` (reduce-over-k of a product: matmul/conv/attention scores),
  ``reduction`` (reduce-over-an-axis: sum/max/argmax),
  ``elementwise_map`` (per-element map: add/mul/gelu/silu/cast/bias),
  ``movement`` (data motion without arithmetic: transpose/reshape/pack/copy/dma), and
  ``synchronization`` (ordering/visibility: barrier/fence — a SIMT/spatial concern).

- **Composites** — named patterns declared as a tuple of primitives (``attention`` = contraction +
  reduction + elementwise_map, ``normalization`` / ``softmax`` = reduction + elementwise_map). A target
  may declare a composite directly (it fuses the pattern) or be judged on the primitives it covers.

The mapping FROM the capture's ``prov.family`` / ``prov.op`` tags (see
:mod:`merlin.analysis.dse.guidance.attribution` ``OPC_*``) and FROM routing ``OpDemand.op`` names lives here so
callers never re-derive it. No target literals, no regex — a structural dict lookup.
"""

from __future__ import annotations

# ---- semantic_families -------------------------------------------------------------------------

#: The irreducible computation kinds. Closed set — a new one is a real capability, not a spelling.
PRIMITIVES: tuple[str, ...] = (
    "contraction",
    "reduction",
    "elementwise_map",
    "movement",
    "synchronization",
)

#: Named fused patterns, each declared as the primitives it is built from. A target that lowers the
#: whole pattern as one kernel declares the composite; otherwise it is scored on the primitives.
COMPOSITES: dict[str, tuple[str, ...]] = {
    "attention": ("contraction", "reduction", "elementwise_map"),
    "normalization": ("reduction", "elementwise_map"),
    "softmax": ("reduction", "elementwise_map"),
}

#: Every family name callers may use.
FAMILIES: frozenset[str] = frozenset(PRIMITIVES) | frozenset(COMPOSITES)


def operation_form(op: str | None, *, carrier_op: str | None = None) -> str | None:
    """Classify a movement operation's *work*, not its storage layout.

    A transfer/copy and an axis permutation share the ``movement`` family but
    require different hardware mechanisms. Prefer the actual IR carrier when
    available; a provenance label may describe a larger enclosing region.
    Unknown forms stay unknown so a form-constrained capability cannot claim
    them by family membership alone.
    """
    if carrier_op in ("linalg.transpose", "merlin_iface.transpose", "merlin_iface.permute"):
        return "permutation"
    if carrier_op in ("linalg.copy", "memref.copy", "merlin_iface.movement"):
        return "copy"
    # These typed tensor operations carry distinct transfer geometries. A
    # declaration for a plain copy does not establish indexed extraction,
    # insertion into a larger tensor, or concatenation of many inputs.
    tensor_forms = {
        "tensor.extract_slice": "slice_extract",
        "tensor.insert_slice": "slice_insert",
        "tensor.concat": "concatenation",
    }
    if carrier_op in tensor_forms:
        return tensor_forms[carrier_op]
    if carrier_op is not None:
        return None
    if op in ("transpose", "permute"):
        return "permutation"
    if op in ("movement", "copy"):
        return "copy"
    return None


# --- capture prov.family -> canonical family ---------------------------------------------------
# The capture tags each op with a coarse prov.family; this pins each to a canonical family. Keys are
# the strings emitted by model2MLIR (mirrored by merlin.analysis.dse.guidance.attribution).
_PROV_FAMILY: dict[str, str] = {
    "contraction": "contraction",
    "conv": "contraction",
    "attention": "attention",
    "reduction": "reduction",
    "reduce": "reduction",
    "normalization": "normalization",
    "elementwise": "elementwise_map",
    "activation": "elementwise_map",
    "layout": "movement",
    "movement": "movement",
    "copy": "movement",
    "synchronization": "synchronization",
    # THE REST OF THE TAGS THE STUDY MODELS' CAPTURES ACTUALLY EMIT. This table was grown one capture at
    # a time and drifted behind `llvmlower.op_profile.FAMILY_CATEGORY`, whose docstring says in as many
    # words that its keys are "the families the study models' own captures emit" -- it knew `quantize`,
    # `minmax` and `pool`; this one did not. A tag only that table recognises resolves to None here, and
    # `from_prov` fails closed, so the region became UNKNOWN: not "the hardware cannot do this" but
    # "nobody named it", reported identically. Measured on a captured ResNet-50: 100 of 116 host regions
    # the compiler refused were refused under a family word neither table could turn into a capability.
    #
    # Derived rather than re-typed: `tests/targetgen/test_residual_seam.py` holds this table
    # against `FAMILY_CATEGORY`, so the next tag a capture emits cannot resolve in one and not the other.
    "quantize": "elementwise_map",
    "cast": "elementwise_map",
    "bitwise": "elementwise_map",
    "compare": "elementwise_map",
    "minmax": "elementwise_map",
    "pool": "reduction",
    "arg_reduce": "reduction",
    "scan": "reduction",
    "concat": "movement",
    "resize": "movement",
    "gather_scatter": "movement",
    "fill": "movement",
    "iota": "movement",
}

#: Capture tags that deliberately resolve to NO family, each with the reason. Declared rather than
#: merely absent, for the same reason ``CONFIG``/``FLUSH`` are declared absent from
#: :data:`_ISA_CLASS_FAMILY`: "we looked at this and it has no canonical primitive" and "nobody has
#: added it yet" are different states, and only the second is a bug. A region carrying one of these is
#: reported UNKNOWN, which is the correct answer -- forcing it into contraction or elementwise_map
#: would manufacture a capability out of a computation no declared primitive describes.
PROV_FAMILY_UNMAPPED: dict[str, str] = {
    "spectral": (
        "an FFT/DCT-style transform is a butterfly over a twiddle schedule: not a reduce-over-k, not a "
        "reduce-over-an-axis, not a per-element map and not data motion. It needs a primitive of its "
        "own before any target can be said to declare or lack it"
    ),
}

# --- shared ISA semantic-class -> canonical family ---------------------------------------------
# The closed class vocabulary a target declares in ``encoding.semantic_class`` (and that the RoCC trace
# decoder speaks). Closed and target-agnostic, exactly like _PROV_FAMILY above. Classes that are pure
# plumbing (configuration, cache maintenance) map to NOTHING rather than to a family: configuring the
# datapath is not a computation, and letting CONFIG license ``elementwise_map`` would manufacture a
# capability out of a setup instruction.
_ISA_CLASS_FAMILY: dict[str, str] = {
    # reduce-over-k products, and the preload/loop scaffolding that only exists to feed one
    "COMPUTE_PRELOADED": "contraction",
    "COMPUTE_ACCUMULATE": "contraction",
    "PRELOAD": "contraction",
    "LOOP_WS": "contraction",
    "LOOP_CONV": "contraction",
    # data motion without arithmetic
    "MVIN": "movement",
    "MVIN2": "movement",
    "MVIN3": "movement",
    "MVOUT": "movement",
    "LOAD": "movement",
    "STORE": "movement",
    # ordering / visibility
    "FENCE": "synchronization",
    # CONFIG / CONFIG_EX / CONFIG_LD / CONFIG_ST / FLUSH: plumbing, deliberately absent (see below)
}

#: The shared classes that are PLUMBING -- configuration and cache maintenance. Named here, rather than
#: only left out of :data:`_ISA_CLASS_FAMILY`, so a consumer that must know "does this class issue a
#: memory request of its own?" asks this module instead of re-listing the vocabulary.
_ISA_PLUMBING_CLASSES = frozenset({"CONFIG", "CONFIG_EX", "CONFIG_LD", "CONFIG_ST", "FLUSH"})

#: Contraction classes whose operands are already on chip, so they issue no memory request. The loop
#: classes are NOT here: a hardware loop unroller issues loads the decoded stream cannot see.
_ISA_ON_CHIP_CLASSES = frozenset({"PRELOAD", "COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"})

#: Which way a shared class moves data, or which movement a configuration class configures: ``in`` onto
#: the chip, ``out`` off it. Part of the same closed vocabulary; a target's mnemonics never appear here.
_ISA_CLASS_DIRECTION: dict[str, str] = {
    "MVIN": "in",
    "MVIN2": "in",
    "MVIN3": "in",
    "LOAD": "in",
    "MVOUT": "out",
    "STORE": "out",
    "CONFIG_LD": "in",
    "CONFIG_ST": "out",
}

# --- structural ISA ROLE -> canonical family ----------------------------------------------------
# A self-hosted-ISA target declares no ``encoding.semantic_class`` at all: its vocabulary is its own
# mnemonics (``MXUMatMul``, ``TensorComputeBinary``, ...), which _ISA_CLASS_FAMILY must never learn --
# that is the name-matching this repo exists to avoid. What such a target DOES yield is a structural
# ROLE census (:func:`merlin.target.isa._taxonomy_worker._role_for_pattern`), which reads
# each instruction's OWN typed operands and answers "what datapath does this describe?" -- accumulator
# from tensor+weight sources is a matmul, tensor->tensor is a vector epilogue, and so on. Those roles
# are a closed, universal systolic vocabulary, so they pin to families exactly as the declared classes
# above do, and every target gains a family vocabulary whether or not it declares one.
#
# ⚠️ Deliberately NOT derivable here: ``reduction``. A reduce-over-an-axis is expressed with the same
# tensor->tensor instructions as a per-element map, so the operand census cannot separate them -- the
# distinction lives in the loop structure, not the instruction. Callers get ``elementwise_map`` for the
# tensor-compute roles and must treat a missing ``reduction`` as UNKNOWN, never as "cannot reduce".
#: ``{role: (family, requires)}``. ``requires`` names families the licence is only valid ALONGSIDE — an
#: epilogue that exists solely on another unit's output path is not a standalone capability.
ISA_ROLE_FAMILY: dict[str, tuple[str, tuple[str, ...]]] = {
    "matmul": ("contraction", ()),
    # tensor base+offset load/store: data motion, the role-census twin of MVIN / MVOUT
    "memory": ("movement", ()),
    # the vector unary/binary epilogue -- a per-element map over tensor registers
    "tensor_compute_unary": ("elementwise_map", ()),
    "tensor_compute_binary": ("elementwise_map", ()),
    # a SCALED accumulator pop is a requant on the readout path: an elementwise map available only fused
    # with the contraction that filled the accumulator, never standalone.
    "acc_readout_scaled": ("elementwise_map", ("contraction",)),
    # weight_load / acc_seed / acc_readout are contraction PLUMBING: they feed or drain the mesh and
    # license nothing on their own -- reading them as "contraction" would let a target that merely pushes
    # weights claim it can multiply. scalar is host-side control. All absent on purpose, exactly as
    # CONFIG is absent above: setting the datapath up is not a computation.
}

# --- routing OpDemand.op / prov.op -> canonical family -----------------------------------------
# When only an op name is available (no family tag), pin it structurally. Softmax/normalization ops
# resolve to their composite so eligibility can ask for the fused capability or the primitives.
_OP_FAMILY: dict[str, str] = {
    # contraction: any reduce-over-k product (matmul/conv/attention-scores/batched GEMV)
    "matmul": "contraction",
    "batch_matmul": "contraction",
    "matmul_batched": "contraction",
    "addmm": "contraction",
    # A contraction and a residual add presented as two regions. Its PRIMARY family is the contraction
    # it contains; the add is credited through `composed_families`, the same way a fused epilogue is --
    # the capsule exists to test whether the two are joined, so claiming the add as its own family here
    # would count the seam as evidence for a standalone elementwise capability nobody claimed.
    "residual_seam": "contraction",
    "sibling_feed": "contraction",
    "linear": "contraction",
    "fused_matmul_bias": "contraction",
    "gemv_batched": "contraction",
    "k_chain": "contraction",
    # a matmul that REUSES the resident weight across calls -- the reuse is a
    # scheduling property, the payload is still a reduce-over-k product.
    "resident_reuse": "contraction",
    "patch_embed": "contraction",
    "conv2d": "contraction",
    "conv1d": "contraction",
    "conv3d": "contraction",
    "depthwise_conv2d": "contraction",
    "convolution": "contraction",
    # attention: the FUSED scaled-dot-product pattern is the composite (contraction + reduction +
    # elementwise_map). Its QK and PV PIECES are not: each is a single reduce-over-k product --
    # `Q @ K^T` and `P @ V` -- with no reduction and no per-element map, so each is a plain
    # contraction, differing from a matmul only in that one operand arrives untransposed.
    #
    # Calling a piece by the composite's name made it unrunnable on paper. Measured on gemmini:
    # `C7_attention_qk_i8` (declares `op: attention_qk`) resolved to the `attention` composite, whose
    # primitives include `reduction`, which the target does not declare -- verdict "target declares no
    # capability for family 'attention'". The identical computation with K pre-transposed
    # (`C5_attention_qk_matmul`, `op: matmul`) resolved to `contraction` and passed at the RTL tier.
    # The hardware was never the constraint: gemmini's contraction capability already declares
    # `transpose: True`. The taxonomy was.
    "sdpa": "attention",
    # The ATen spelling of the same fused pattern, as a captured frontend graph names it.
    "scaled_dot_product_attention": "attention",
    "attention": "attention",
    "attention_full": "attention",
    # the same fused pattern with the score scale supplied as a full elementwise tensor (a mask/scale
    # tile) instead of a scalar: still contraction + reduction + elementwise_map.
    "attention_scaled": "attention",
    "attention_qk": "contraction",
    "attention_pv": "contraction",
    # both operands are runtime activations (no stored weight); still a single reduce-over-k product
    "activation_matmul": "contraction",
    # the MX-format piece keeps the composite: it carries the scale-block handling, not just a product
    "attention_mx": "attention",
    # softmax / normalization: reduction + elementwise composites
    "softmax": "softmax",
    "layer_norm": "normalization",
    "layernorm": "normalization",
    "rms_norm": "normalization",
    "rmsnorm": "normalization",
    "rmsnorm_qkv": "normalization",
    "gemma_4norm": "normalization",
    # reduction: reduce over an axis
    "reduce": "reduction",
    "reduce_sum": "reduction",
    "reduce_mean": "reduction",
    "sum": "reduction",
    "max": "reduction",
    "argmax": "reduction",
    # Pooling is a reduction over a sliding window -- structurally the same reduce this table already
    # names for the bare `max`/`sum` it is built from. Missing these spellings sent every pooling capsule
    # to `family = None` ("unrecognized semantic family, fail-closed"), where it was indistinguishable in
    # the aggregate from a region the HARDWARE cannot do. Measured: 3 of one target's 5 unclassified
    # capsules were maxpool2d / avgpool2d / global_average, all carrying must_accelerate: true -- graded
    # work the eligibility oracle could not name.
    "maxpool2d": "reduction",
    "avgpool2d": "reduction",
    "maxpool": "reduction",
    "avgpool": "reduction",
    # ...and the SNAKE-CASE spellings a torch capture emits for the same two windows. The entries above
    # were added from one capture's vocabulary and the underscore forms from another's were never added,
    # so `max_pool2d` -- the spelling in `prov.op` of the ResNet-50 stem -- still resolved to None while
    # `maxpool2d` resolved. The same window under two spellings is one capability, and a table that
    # recognises only one of them reports the other as work the hardware cannot name.
    "max_pool2d": "reduction",
    "avg_pool2d": "reduction",
    "adaptive_max_pool2d": "reduction",
    "global_average": "reduction",
    "global_avg_pool": "reduction",
    "adaptive_avg_pool2d": "reduction",
    # elementwise_map: per-element maps (activations, bias, rotary, scaling, requant)
    "add": "elementwise_map",
    "mul": "elementwise_map",
    "sub": "elementwise_map",
    "div": "elementwise_map",
    "pow": "elementwise_map",
    "bias": "elementwise_map",
    "bias_add": "elementwise_map",
    # Two quantized tensors summed elementwise into one output domain (a residual connection).
    "residual_add": "elementwise_map",
    "gelu": "elementwise_map",
    "relu": "elementwise_map",
    "acc_scale": "elementwise_map",
    "silu": "elementwise_map",
    "geglu": "elementwise_map",
    # normalize-then-scale with the variance supplied as an INPUT: rsqrt then multiply, both per
    # element. Deliberately NOT the `normalization` composite -- that decomposes to (reduction,
    # elementwise_map), and this program contains no reduction at all, so naming it normalization
    # would demand a reduction capability of a target for work this capsule never asks it to do (the
    # `attention_qk` mislabel documented above, in the other direction).
    "fused_norm_scale": "elementwise_map",
    "erf": "elementwise_map",
    "tanh": "elementwise_map",
    "sigmoid": "elementwise_map",
    "exp": "elementwise_map",
    "cast": "elementwise_map",
    "rope": "elementwise_map",
    "rope_qkv": "elementwise_map",
    "logit_softcap": "elementwise_map",
    "embed_scale": "elementwise_map",
    "requant": "elementwise_map",
    # `relu` and `acc_scale` sit above beside the other activations because they are the spellings the
    # COMMAND-BUFFER ABI itself uses for its epilogue stages ("ordered subset of [bias_add, requant,
    # acc_scale, relu]"), and two of the four resolved while two did not. That asymmetry is not
    # cosmetic: a fused capsule is credited for the family its epilogue exercises, so an unresolved
    # stage name silently withheld coverage a capsule had genuinely earned.
    # ...and the per-element spellings a CAPTURED model actually contains beyond the hand corpus. Each is
    # a spelling of the same primitive, not a new capability: an integer bitwise op, a predicate, and a
    # dtype conversion are all per-element maps, and an index generator is a map over the index
    # (out[i] = f(i)). Measured across six captured models: without these, 1222 of 12013 regions were
    # UNCLASSIFIED, so a coverage number could not say whether they were opportunities or scalar work.
    # A contract names its unit's ops with the family word itself ("elementwise", beside "matmul").
    # `movement` and `reduce` are already keys here for the same reason; `elementwise` was missing,
    # so a unit declaring [matmul, elementwise] and no explicit semantic_capabilities block derived
    # ONLY {contraction} -- dropping elementwise from its own capability denominator.
    "elementwise": "elementwise_map",
    "bitwise": "elementwise_map",
    "compare": "elementwise_map",
    "dtype_cast": "elementwise_map",
    "arange": "elementwise_map",
    "iota": "elementwise_map",
    # THE TWO LARGEST HOLES A DEPLOYABLE INTEGER MODEL LEAVES, and both are spellings of maps this
    # table already names under another word. A per-tensor quantize is `real -> round(real/s) + zp`
    # clamped to the output's range: the same per-element affine map as `requant`/`acc_scale`, which
    # resolve. A `minmax` is a bounded clamp: the same map as `relu`, which resolves, with two declared
    # bounds instead of one implied zero. Measured on a captured ResNet-50's command buffer: 50 regions
    # tagged `quantize_per_tensor` and 49 tagged `minmax` resolved to None, where a region the hardware
    # genuinely cannot do and a region nobody named are indistinguishable -- so neither could ever
    # appear in a conformance cell, and no capsule could be synthesized to demand either.
    "quantize": "elementwise_map",
    "quantize_per_tensor": "elementwise_map",
    "quantize_per_channel": "elementwise_map",
    "dequantize": "elementwise_map",
    "dequantize_per_tensor": "elementwise_map",
    "dequantize_per_channel": "elementwise_map",
    "requantize": "elementwise_map",
    "minmax": "elementwise_map",
    "clamp": "elementwise_map",
    "hardtanh": "elementwise_map",
    "relu6": "elementwise_map",
    # THE OP-LEVEL HALF OF THE CAPTURE VOCABULARY, and it was never held against this table. The
    # cross-hold that exists compares `op_profile.FAMILY_CATEGORY` (prov.FAMILY tags) with
    # `_PROV_FAMILY`; `op_profile.OP_CATEGORY` and `ROLE_CATEGORY` are the prov.OP and prov.ROLE
    # spellings of the same captures and were held against nothing, so nine of them classified for COST
    # and resolved to no CAPABILITY. Each below is a spelling of a primitive this table already names:
    # the activation-quantize chain is a scale and an affine map (per element) around an absolute-max
    # search (a reduce over the tensor); an im2col convolution is the contraction it lowers to; and a
    # metadata-only reshape moves nothing but its own shape.
    "act_scale": "elementwise_map",
    "act_quantize": "elementwise_map",
    "act_amax": "reduction",
    "contraction": "contraction",
    "convolution_im2col_matmul": "contraction",  # read alias (legacy_labels.CONV_OP_ALIASES)
    "convolution_gathered_matmul": "contraction",
    "gather": "movement",
    "view": "movement",
    "squeeze": "movement",
    "unsqueeze": "movement",
    # movement: data motion without arithmetic
    "movement": "movement",
    "transpose": "movement",
    "permute": "movement",
    "reshape": "movement",
    "expand": "movement",
    # Repeats each slice along one axis: a broadcast copy, like ``expand`` materialized.
    "repeat_interleave": "movement",
    "pack": "movement",
    "copy": "movement",
    # ...captured-model movement spellings. A constant fill writes a value per element with no arithmetic
    # on an input, and a gather/embedding lookup moves rows by index -- motion, not compute. `fill` alone
    # was 1097 of those 1222 unclassified regions, i.e. the single largest hole in the vocabulary, and it
    # is init code no accelerator needs to claim.
    "fill": "movement",
    "index_gather": "movement",
    "embedding": "movement",
    # synchronization: ordering / visibility
    "barrier": "synchronization",
    "fence": "synchronization",
}


# --- spelling folding: ONE capability, however a capture punctuates it -------------------------
# THE SAME BUG TWICE, UNDER ITS OWN WARNING. `maxpool2d` was added to `_OP_FAMILY` beneath a comment
# explaining that a missing pooling spelling had sent every pooling capsule to `family = None`; the
# snake_case `max_pool2d` from a second capture's vocabulary was not added, and the ResNet-50 stem's
# pooling region resolved to None anyway. Both spellings are now keys, and listing keys is exactly the
# repair that failed the first time: the NEXT capture punctuates something else and the table is a
# spelling behind again.
#
# So the lookup folds instead. A separator character carries no meaning in an op name -- `max_pool2d`,
# `maxpool2d` and `max-pool-2d` are three punctuations of one op, never three capabilities -- so the
# fold removes separators and nothing else. It removes no WORDS, which is why `add` and `bias_add` stay
# distinct: they differ by a word, not by punctuation.
#
# Structural, not a pattern match: `str.replace` over a closed set of separator characters, and the
# folded index is built from the declared tables themselves, so a table edit needs no second edit here.
# A fold that made two DECLARED keys collide on DIFFERENT families would silently remap one of them, so
# :func:`check` refuses that rather than letting the index decide which won.
_SEPARATORS: tuple[str, ...] = ("_", "-", ".", " ")


def fold_spelling(name: str) -> str:
    """``name`` with separators removed and case normalized — the key two spellings of one op share."""
    text = name.strip().lower()
    for sep in _SEPARATORS:
        text = text.replace(sep, "")
    return text


def _folded_index(table: dict[str, str]) -> dict[str, str]:
    """``{folded key: family}`` for a declared table. A key that folds onto another declared key with a
    DIFFERENT family is left out rather than allowed to win arbitrarily; :func:`check` reports it."""
    index: dict[str, str] = {}
    conflicted: set[str] = set()
    for key, fam in table.items():
        folded = fold_spelling(key)
        if folded in index and index[folded] != fam:
            conflicted.add(folded)
        index[folded] = fam
    for folded in conflicted:
        index.pop(folded, None)
    return index


def _folded_conflicts(table: dict[str, str]) -> list[str]:
    """Declared keys that fold together while naming different families."""
    seen: dict[str, tuple[str, str]] = {}
    problems: list[str] = []
    for key, fam in table.items():
        folded = fold_spelling(key)
        prior = seen.get(folded)
        if prior is not None and prior[1] != fam:
            problems.append(f"{prior[0]!r} -> {prior[1]!r} and {key!r} -> {fam!r} are one spelling folded")
        else:
            seen[folded] = (key, fam)
    return problems


_PROV_FAMILY_FOLDED: dict[str, str] = _folded_index(_PROV_FAMILY)
_OP_FAMILY_FOLDED: dict[str, str] = _folded_index(_OP_FAMILY)
#: Folded spellings that must NOT resolve, because the declared tag they fold to is deliberately
#: unmapped. Without this a fold would quietly give ``spectral`` a family the declaration refuses it.
_UNMAPPED_FOLDED: frozenset[str] = frozenset(fold_spelling(tag) for tag in PROV_FAMILY_UNMAPPED)


def _lookup(table: dict[str, str], folded: dict[str, str], raw: str) -> str | None:
    """Exact declared spelling first, then the folded one. ``None`` when neither resolves."""
    hit = table.get(raw.strip().lower())
    if hit is not None:
        return hit
    key = fold_spelling(raw)
    if key in _UNMAPPED_FOLDED:
        return None
    return folded.get(key)


def from_isa_class(isa_class: str | None) -> str | None:
    """Canonical family for a SHARED ISA semantic-class name; ``None`` if unrecognized.

    ⚠️ Scope: this table covers only the **shared, closed** class vocabulary a target declares in its
    contract's ``encoding.semantic_class`` — the same human-owned vocabulary the compiler and the trace
    decoder both speak. It is emphatically NOT a place to map a target's own instruction mnemonics: a
    target that names its reduction ``VREDSUM_BF`` must be classified from the STRUCTURE of that
    instruction (its typed operands and its behaviour — see the ISA role census), never from the letters
    in its name, or we are back to the string-matching this repo exists to avoid. Unrecognized returns
    ``None`` so the caller records UNKNOWN rather than guessing.
    """
    if not isa_class:
        return None
    return _ISA_CLASS_FAMILY.get(isa_class.strip().upper())


def movement_classes(direction: str) -> frozenset[str]:
    """The shared ``movement``-family classes that move data ``direction`` (``in`` / ``out``)."""
    return frozenset(
        name
        for name, family in _ISA_CLASS_FAMILY.items()
        if family == "movement" and _ISA_CLASS_DIRECTION.get(name) == direction
    )


def configuration_classes(direction: str) -> frozenset[str]:
    """The shared plumbing classes that configure movement in ``direction`` (``in`` / ``out``)."""
    return frozenset(name for name in _ISA_PLUMBING_CLASSES if _ISA_CLASS_DIRECTION.get(name) == direction)


def local_only_classes() -> frozenset[str]:
    """The shared classes that issue NO memory request of their own: plumbing, and contraction steps
    whose operands are already on chip. Any other class may touch memory."""
    return _ISA_PLUMBING_CLASSES | _ISA_ON_CHIP_CLASSES


def from_isa_role(isa_role: str | None) -> str | None:
    """Canonical family for a STRUCTURAL ISA role from the role census; ``None`` if it maps to nothing.

    The companion to :func:`from_isa_class` for a target that declares no shared class vocabulary. The
    role is derived from an instruction's own typed operands, so this stays a structural lookup and never
    reads a target's mnemonics. ``None`` for control/scalar plumbing and for any unknown role — callers
    fail closed (record UNKNOWN) rather than guessing a family.
    """
    if not isa_role:
        return None
    hit = ISA_ROLE_FAMILY.get(isa_role.strip().lower())
    return hit[0] if hit else None


def isa_role_requires(isa_role: str | None) -> tuple[str, ...]:
    """Families the role's licence is only valid alongside; empty when it stands alone or is unknown."""
    if not isa_role:
        return ()
    hit = ISA_ROLE_FAMILY.get(isa_role.strip().lower())
    return hit[1] if hit else ()


def families_from_roles(roles) -> frozenset[str]:
    """The canonical families a role census evidences — the family vocabulary of a self-hosted-ISA
    target. Roles that pin to nothing (contraction plumbing, scalar/control, unknown) drop out silently.

    A role whose licence ``requires`` another family only counts when that family is ALSO evidenced: a
    scaled accumulator pop proves a fused requant exists, not that the target can map elementwise on its
    own. An empty result means the census evidenced no computation family — a real answer, not an error.
    """
    seen = {r.strip().lower() for r in (roles or ()) if r}
    standalone = {ISA_ROLE_FAMILY[r][0] for r in seen if r in ISA_ROLE_FAMILY and not ISA_ROLE_FAMILY[r][1]}
    conditional = {
        ISA_ROLE_FAMILY[r][0]
        for r in seen
        if r in ISA_ROLE_FAMILY and ISA_ROLE_FAMILY[r][1] and all(req in standalone for req in ISA_ROLE_FAMILY[r][1])
    }
    return frozenset(standalone | conditional)


def carrier_provenance_family(prov_family: str, *, carrier_op: str, reduces: bool | None) -> str:
    """Correct an enclosing contraction tag using the actual carrier's work.

    A parallel generic is deliberately unknown: its region's op name cannot
    prove a family. A named transpose independently specifies permutation.
    Other tags and carriers retain their original capture vocabulary.
    """
    if prov_family == "contraction":
        if carrier_op == "linalg.generic" and reduces is False:
            return ""
        if carrier_op == "linalg.transpose":
            return "layout"
    return prov_family


def from_prov(prov_family: str | None, prov_op: str | None = None) -> str | None:
    """Canonical family for a captured op's ``prov.family`` (with ``prov.op`` as a tiebreaker).

    Returns ``None`` when the tags carry no recognizable family — callers must fail closed (treat the
    region as UNKNOWN / ineligible-by-default), never guess a family.
    """
    if prov_family:
        fam = _lookup(_PROV_FAMILY, _PROV_FAMILY_FOLDED, prov_family)
        if fam is not None:
            return fam
    if prov_op:
        return from_op(prov_op)
    return None


def from_op(op: str | None) -> str | None:
    """Canonical family for a routing ``OpDemand.op`` / bare op name; ``None`` if unrecognized."""
    if not op:
        return None
    return _lookup(_OP_FAMILY, _OP_FAMILY_FOLDED, op)


def primitives_of(family: str) -> tuple[str, ...]:
    """The primitive(s) a family decomposes to — itself for a primitive, its parts for a composite."""
    if family in COMPOSITES:
        return COMPOSITES[family]
    return (family,) if family in PRIMITIVES else ()


def is_family(name: str) -> bool:
    return name in FAMILIES


def check() -> list[str]:
    """Invariant (empty list == OK): every composite decomposes to declared primitives, and every
    mapping target is a declared family. Wire into a structure test."""
    problems: list[str] = []
    for comp, parts in COMPOSITES.items():
        for p in parts:
            if p not in PRIMITIVES:
                problems.append(f"composite {comp!r} references non-primitive {p!r}")
    for src, fam in {**_PROV_FAMILY, **_OP_FAMILY, **_ISA_CLASS_FAMILY}.items():
        if fam not in FAMILIES:
            problems.append(f"mapping {src!r} -> {fam!r} is not a declared family")
    for role, (fam, requires) in ISA_ROLE_FAMILY.items():
        if fam not in FAMILIES:
            problems.append(f"role {role!r} -> {fam!r} is not a declared family")
        for req in requires:
            if req not in FAMILIES:
                problems.append(f"role {role!r} requires {req!r}, not a declared family")
    # A fold that merges two declared keys onto DIFFERENT families would make one of them resolve to
    # the other's capability, which is worse than not resolving: a region would be judged against a
    # capability nobody claimed for it. Reported here rather than resolved by an ordering rule.
    for label, table in (("prov.family", _PROV_FAMILY), ("prov.op", _OP_FAMILY)):
        for clash in _folded_conflicts(table):
            problems.append(f"{label} spellings collide when folded: {clash}")
    # ...and an unmapped tag must stay unmapped through the fold too, or the declaration that it has no
    # primitive would be silently overridden by a neighbouring spelling.
    for tag in PROV_FAMILY_UNMAPPED:
        if from_prov(tag) is not None:
            problems.append(f"prov.family {tag!r} is declared unmapped yet resolves to {from_prov(tag)!r}")
    return problems


PRIMITIVE_FORM_AXES: dict[str, tuple[str, ...]] = {
    "contraction": ("occupancy", "aspect", "reduction_depth", "operand_capacity", "accumulator_capacity", "transfer"),
    "reduction": ("occupancy", "reduction_depth", "operand_capacity", "transfer"),
    "elementwise_map": ("occupancy", "operand_capacity", "transfer"),
    "movement": ("occupancy", "operand_capacity", "transfer"),
    "synchronization": (),
}

OP_STRUCTURAL: dict[str, str] = {
    "model": (
        "a whole captured program, not one computation. Its families are whatever its regions contain, "
        "which is the model's property and not the op's; classifying it as any single family would "
        "credit or demand a capability on the strength of a container"
    ),
    "host_island_seam": (
        "a PLACEMENT structure — the accelerator -> host -> accelerator handoff. What it exercises is "
        "the seam itself (who owns which region, and whether the operands survive the crossing), so it "
        "belongs to no computation family and no datapath declares it"
    ),
}

_OP_STRUCTURAL_FOLDED: dict[str, str] = {fold_spelling(op): why for op, why in OP_STRUCTURAL.items()}


def form_axes(family: str) -> tuple[str, ...]:
    """The form axes ``family``'s shape space has — the union over its primitives for a composite.

    Union, not intersection: a fused attention contains a contraction, so its form space contains the
    contraction's aspect and accumulator axes whether or not its reduction half has them. Scoring it on
    the intersection would drop exactly the axes the fused kernel is hardest on.
    """
    prims = primitives_of(family)
    if not prims:
        return ()
    seen: list[str] = []
    for prim in prims:
        for axis in PRIMITIVE_FORM_AXES.get(prim, ()):
            if axis not in seen:
                seen.append(axis)
    return tuple(seen)


def structural_reason(op: str | None) -> str | None:
    """Why ``op`` names a capsule STRUCTURE rather than a computation; ``None`` if it does not.

    Spelling-folded like :func:`from_op`, so a declaration is not defeated by punctuation.
    """
    if not op:
        return None
    hit = OP_STRUCTURAL.get(op.strip().lower())
    if hit is not None:
        return hit
    return _OP_STRUCTURAL_FOLDED.get(fold_spelling(op))


# ---- operation_numerics ------------------------------------------------------------------------
# Separate reviewed arithmetic declarations from observed storage/compute widths.


def integer_split_k_limit(numeric_policy: dict | None, *, tile_dim: int, full_k: int) -> dict | None:
    """Derive the worst-case signed-integer primitive K from selected arithmetic.

    This is a target-neutral numerical obligation shared by capture selection and
    actual mesh dispatch. A full contraction may need several bounded primitive
    tiles, followed by exact i32 aggregation before its epilogue. The caller
    must separately establish that the declaration was reviewed and that a
    compiler performs the split; this calculation proves neither.
    """
    semantics = (numeric_policy or {}).get("numerical_semantics") or {}
    internal = semantics.get("internal_arithmetic") or {}
    if not internal:
        return None
    if internal.get("full_operation_overflow_policy") != "bounded_exact_requires_each_partial_sum":
        raise ValueError("integer mesh has no supported selected partial-sum policy")
    operand_bits = internal.get("signed_operand_bits")
    mac_bits = internal.get("mac_result_bits")
    if (
        type(operand_bits) is not int
        or type(mac_bits) is not int
        or not 2 <= operand_bits <= 32
        or not 2 <= mac_bits <= 64
        or semantics.get("accumulator_dtype") not in {"i32", "int32"}
        or internal.get("mac_result_overflow") != f"wrap_to_{mac_bits}_bits"
        or type(tile_dim) is not int
        or tile_dim < 1
        or type(full_k) is not int
        or full_k < 1
    ):
        raise ValueError("integer mesh has unsupported selected MAC/accumulator arithmetic")
    maximum_operand = 1 << (operand_bits - 1)
    maximum_product = maximum_operand * maximum_operand
    full_bound = full_k * maximum_product
    if full_bound > (1 << 31) - 1:
        raise ValueError("integer mesh full contraction may exceed exact i32 aggregation")
    limit = ((1 << (mac_bits - 1)) - 1) // maximum_product
    aligned = (limit // tile_dim) * tile_dim
    return {
        "max_true_k": limit,
        "max_padded_k": limit,
        "tile_k": aligned,
        "full_result_bound": full_bound,
        "mac_result_bits": mac_bits,
        "signed_operand_bits": operand_bits,
        "maximum_product": maximum_product,
        "aggregation": "exact_integer_sum_before_epilogue",
    }


def integer_partial_sum_bound(
    semantics: dict, *, reduction_extent: int, lhs_values, rhs_values, initial_values=()
) -> dict:
    """Sufficient absolute bound for every partial sum of a concrete contraction.

    This is not a tensor execution model. The caller must select the actual
    reduction extent and include all initial accumulator/bias values. The bound
    deliberately ignores cancellation and conservatively uses the positive
    signed limit, so it cannot approve an overflowing internal MAC by relying on
    wider storage or a small final output.
    """
    if type(reduction_extent) is not int or reduction_extent < 1:
        raise ValueError("reduction_extent must be a positive integer")
    internal = semantics.get("internal_arithmetic") or {}
    bits, operand_bits = internal.get("mac_result_bits"), internal.get("signed_operand_bits")
    if type(bits) is not int or bits < 2 or type(operand_bits) is not int or operand_bits < 2:
        return {"status": "unknown", "reason": "selected internal signed operand/result widths are absent"}
    values = [list(lhs_values), list(rhs_values), list(initial_values)]
    if not values[0] or not values[1] or any(type(value) is not int for vector in values for value in vector):
        raise ValueError("partial-sum bounds require nonempty integral lhs/rhs values")
    lo, hi = -(1 << (operand_bits - 1)), (1 << (operand_bits - 1)) - 1
    if any(not lo <= value <= hi for vector in values[:2] for value in vector):
        raise ValueError("observed operands exceed the selected signed operand width")
    lhs, rhs, initial = (max((abs(value) for value in vector), default=0) for vector in values)
    bound = initial + reduction_extent * lhs * rhs
    limit = (1 << (bits - 1)) - 1
    return {
        "status": "proven_safe" if bound <= limit else "may_overflow",
        "scope": "all partial sums of the supplied contraction and initial addends; not full-kernel execution",
        "bound": bound,
        "signed_positive_limit": limit,
        "mac_result_bits": bits,
        "reduction_extent": reduction_extent,
        "maximum_absolute_operands": [lhs, rhs],
        "maximum_absolute_initial_addend": initial,
        "formula": "max_abs(initial) + reduction_extent * max_abs(lhs) * max_abs(rhs)",
    }


def integer_matrix_partial_sum_bound(
    semantics: dict, *, lhs_shape, rhs_shape, lhs_values, rhs_values, initial_values=(), max_products=1 << 28
) -> dict:
    """Bound every concrete matrix partial sum without cancellation or order.

    Global operand maxima can combine values that never multiply each other.
    This computes the tighter nonnegative envelope ``abs(A) @ abs(B)`` and
    adds the largest absolute initial addend. Every subset/order of a dot
    product is bounded by that envelope. No signed result is evaluated here.
    The work cap is a host proof budget, not a target capacity or wider MAC law.
    """
    shapes = (lhs_shape, rhs_shape)
    if (
        any(
            not isinstance(shape, (tuple, list))
            or len(shape) != 2
            or any(type(value) is not int or value < 1 for value in shape)
            for shape in shapes
        )
        or lhs_shape[1] != rhs_shape[0]
    ):
        raise ValueError("concrete matrix bound requires matching positive rank-2 extents")
    if type(max_products) is not int or max_products < 1:
        raise ValueError("concrete matrix bound requires a positive host proof budget")
    lhs, rhs, initial = list(lhs_values), list(rhs_values), list(initial_values)
    if len(lhs) != lhs_shape[0] * lhs_shape[1] or len(rhs) != rhs_shape[0] * rhs_shape[1]:
        raise ValueError("concrete matrix bound operands differ from their complete shapes")
    coarse = integer_partial_sum_bound(
        semantics, reduction_extent=lhs_shape[1], lhs_values=lhs, rhs_values=rhs, initial_values=initial
    )
    if coarse["status"] == "unknown":
        return coarse
    products = lhs_shape[0] * lhs_shape[1] * rhs_shape[1]
    if products > max_products or coarse["bound"] >= 1 << 63:
        return {
            "status": "unknown",
            "reason": "concrete matrix proof exceeds its host work or exact signed-i64 arithmetic budget",
            "products": products,
            "max_products": max_products,
        }
    import numpy as np

    a = np.abs(np.asarray(lhs, dtype=np.int64).reshape(lhs_shape))
    b = np.abs(np.asarray(rhs, dtype=np.int64).reshape(rhs_shape))
    bound = int((a @ b).max()) + coarse["maximum_absolute_initial_addend"]
    return {
        **coarse,
        "status": "proven_safe" if bound <= coarse["signed_positive_limit"] else "may_overflow",
        "bound": bound,
        "global_maximum_bound": coarse["bound"],
        "matrix_shapes": [list(lhs_shape), list(rhs_shape)],
        "method": "concrete_absolute_product_envelope",
        "formula": "max(abs(lhs) @ abs(rhs)) + max_abs(initial)",
        "scope": "every partial sum in every reduction order for these exact matrix operands and initial addends",
    }


def _unknowns(value, path="semantics") -> list[str]:
    if isinstance(value, dict):
        return [unknown for key, member in value.items() for unknown in _unknowns(member, f"{path}.{key}")]
    if isinstance(value, list):
        return [unknown for index, member in enumerate(value) for unknown in _unknowns(member, f"{path}.{index}")]
    return (
        [path]
        if value is None or (isinstance(value, str) and (not value or value.lower().startswith("unknown")))
        else []
    )


def _explicit_contract(contract: dict | None) -> dict:
    if not isinstance(contract, dict):
        return {"status": "unknown", "reason": "no operation-scoped reviewed numerical contract"}
    semantics, evidence = contract.get("semantics"), contract.get("evidence")
    unknowns = _unknowns(semantics) if isinstance(semantics, dict) and semantics else ["semantics"]
    if not isinstance(evidence, dict) or not evidence:
        unknowns.append("evidence")
    if contract.get("status") != "reviewed":
        unknowns.append("review_status")
    return {
        "status": "unknown" if unknowns else "resolved",
        "semantics": semantics,
        "unknowns": sorted(unknowns),
        "qualification": "reviewed declaration, not measured numerical conformance",
    }


def operation_numerical_contracts(entry: dict, software_spec: dict | None, host_capabilities: dict | None) -> dict:
    row = entry["observed_signature"]
    spec = software_spec or {}
    declarations = {declaration["id"]: declaration for declaration in spec.get("operations") or []}
    accelerator = []
    for admission in entry["software_admissions"]:
        if admission["placement"] not in {"accelerator", "fused_accelerator"} or admission["status"] != "admitted":
            continue
        declaration = declarations[admission["declaration"]]
        if "numerical_contract" in declaration:
            decision = _explicit_contract(declaration["numerical_contract"])
        elif row.get("semantic_family") == "contraction":
            semantics = spec.get("numerical_semantics") or {}
            unknowns = _unknowns(semantics)
            engine = (semantics.get("model") or {}).get("engine")
            required = {"operand_dtype", "accumulator_dtype", "readout_dtype", "subnormal_operand_flush"}
            if engine == "integer_reference":
                required.add("overflow")
            elif engine == "specir_fp_reduce":
                required.update({"rounding", "reduction_order", "reduction_cadence", "product_rounding"})
            else:
                unknowns.append("model.engine")
            unknowns.extend(sorted(required - set(semantics)))
            if spec.get("status") != "reviewed":
                unknowns.append("software_spec_review")
            decision = {
                "status": "unknown" if unknowns else "resolved",
                "unknowns": sorted(set(unknowns)),
                "semantics": semantics,
                "basis": "selected contraction numerical semantics",
                "qualification": "reviewed declaration, not measured numerical conformance",
            }
        else:
            decision = _explicit_contract(None)
        accelerator.append({"declaration": declaration["id"], **decision})
    host = []
    for profile in entry["host_admission"].get("profiles") or []:
        if profile["status"] != "admitted" or not profile.get("reviewed"):
            continue
        selected = (host_capabilities or {}).get(profile["profile"]) or {}
        document = selected.get("capability_spec") or {}
        declarations = {declaration["id"]: declaration for declaration in document.get("operations") or []}
        for admission in profile.get("decisions") or []:
            if admission["status"] != "admitted":
                continue
            declaration = declarations[admission["declaration"]]
            host.append(
                {
                    "profile": profile["profile"],
                    "declaration": declaration["id"],
                    **_explicit_contract(declaration.get("numerical_contract")),
                }
            )

    def summarize(decisions):
        return {
            "status": "resolved" if any(decision["status"] == "resolved" for decision in decisions) else "unknown",
            "declarations": decisions,
            "scope": "operation-specific; widths and precision profile alone are insufficient",
        }

    return {"accelerator": summarize(accelerator), "host": summarize(host)}
