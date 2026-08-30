"""Canonical semantic-computation families — the target-agnostic vocabulary the capability contract
and the eligibility oracle both speak.

Coverage is argued over *semantic families*, not framework op names: ``aten.linear``, ``onnx.Gemm``,
``stablehlo.dot_general``, ``linalg.matmul`` and an ``einsum`` all arrive at the same **contraction**
family, so a compiler that supports the family supports all of them. This module is the single source
of truth for that vocabulary; :mod:`merlin.targetgen.eligibility` and the coverage certificate read it.

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
:mod:`merlin.dse_guidance.attribution` ``OPC_*``) and FROM routing ``OpDemand.op`` names lives here so
callers never re-derive it. No target literals, no regex — a structural dict lookup.
"""
from __future__ import annotations

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

# --- capture prov.family -> canonical family ---------------------------------------------------
# The capture tags each op with a coarse prov.family; this pins each to a canonical family. Keys are
# the strings emitted by model2MLIR (mirrored by merlin.dse_guidance.attribution).
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
    "MVOUT": "movement",
    "LOAD": "movement",
    "STORE": "movement",
    # ordering / visibility
    "FENCE": "synchronization",
    # CONFIG / CONFIG_EX / CONFIG_LD / CONFIG_ST / FLUSH: plumbing, deliberately absent
}

# --- structural ISA ROLE -> canonical family ----------------------------------------------------
# A self-hosted-ISA target declares no ``encoding.semantic_class`` at all: its vocabulary is its own
# mnemonics (``MXUMatMul``, ``TensorComputeBinary``, ...), which _ISA_CLASS_FAMILY must never learn --
# that is the name-matching this repo exists to avoid. What such a target DOES yield is a structural
# ROLE census (:func:`merlin.targetgen.oracle_helpers.isa_introspect._role_for_pattern`), which reads
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
    "addmm": "contraction",
    "linear": "contraction",
    "fused_matmul_bias": "contraction",
    "gemv_batched": "contraction",
    "k_chain": "contraction",
    # a weight-stationary matmul that REUSES the resident weight across calls -- the reuse is a
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
    "attention": "attention",
    "attention_full": "attention",
    "attention_qk": "contraction",
    "attention_pv": "contraction",
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
    "gelu": "elementwise_map",
    "silu": "elementwise_map",
    "geglu": "elementwise_map",
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
    # movement: data motion without arithmetic
    "movement": "movement",
    "transpose": "movement",
    "reshape": "movement",
    "expand": "movement",
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
    standalone = {ISA_ROLE_FAMILY[r][0] for r in seen if r in ISA_ROLE_FAMILY
                  and not ISA_ROLE_FAMILY[r][1]}
    conditional = {ISA_ROLE_FAMILY[r][0] for r in seen if r in ISA_ROLE_FAMILY
                   and ISA_ROLE_FAMILY[r][1]
                   and all(req in standalone for req in ISA_ROLE_FAMILY[r][1])}
    return frozenset(standalone | conditional)


def from_prov(prov_family: str | None, prov_op: str | None = None) -> str | None:
    """Canonical family for a captured op's ``prov.family`` (with ``prov.op`` as a tiebreaker).

    Returns ``None`` when the tags carry no recognizable family — callers must fail closed (treat the
    region as UNKNOWN / ineligible-by-default), never guess a family.
    """
    if prov_family:
        fam = _PROV_FAMILY.get(prov_family.strip().lower())
        if fam is not None:
            return fam
    if prov_op:
        return _OP_FAMILY.get(prov_op.strip().lower())
    return None


def from_op(op: str | None) -> str | None:
    """Canonical family for a routing ``OpDemand.op`` / bare op name; ``None`` if unrecognized."""
    if not op:
        return None
    return _OP_FAMILY.get(op.strip().lower())


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
    return problems
