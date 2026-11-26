"""Merlin's host CPU target: the host C emitter families a placement may select.

These are implementation claims of the checked host C emitters in this package
and the pinned host math supplier, in the same records an OOT target declares.
They neither admit a source program nor authorize a precision change; each
emitter's own source, domain and publication checks remain mandatory. A device
target's adapter that runs these routes names its own route for each family.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.compiler.backend.interface import (
    CompilerDescription, ImplementationFamily, SourceDomain, TargetPipeline,
)
from merlin.compiler.backend.protocol import LoweringRecipe, SourcePattern
from merlin.compiler.quantization.policy_generation import NumericalFamily

PROVIDER = "host"
PIPELINE = "host.c"
ABI = "merlin.host-c-region.v1"
ARTIFACTS = (("c-source", "wrapper"),)
_STORAGE_POLICIES = ("raw-bit-pattern", "signed-i8", "i32-wrap-k-ascending")
# A family names a host emitter route, never a model, node identity, or shape whitelist.
_FAMILIES = (
    ("layout", ("reshape", "copy", "gather", "transpose", "concat", "insert_slice", "identity"),
     "one closed dense byte-preserving layout admitted by the existing host emitter"),
    ("bias-add", ("bf16_add",),
     "explicitly selected host BF16 bias addition; original BF16 RNE and disjoint dense buffers"),
    ("ordered-select", ("relu",),
     "BF16 ordered x < +0 select; all BF16 bit patterns; preserve signed zero and NaN payloads"),
    ("window-max", ("bf16_window_max",),
     "ordered BF16 NCHW window maximum; non-NaN inputs, first-equal tie, checked strided/dilated/padded maps"),
    ("batch-norm", ("bf16_batch_norm_inference",),
     "checked BF16 inference normalization with exact declared binary32 channel steps and fused element update"),
    ("sigmoid", ("bf16_sigmoid",),
     "checked source expf/add/divide/BF16-RNE order; pinned host supplier and exact receipt qualification obligation"),
    ("gelu", ("bf16_gelu",),
     "checked source binary32 tanh or erf GELU order; pinned correctly rounded host supplier; finite BF16 inputs"),
    ("layer-norm", ("bf16_layer_norm",),
     "checked pinned torch 2.11.0+cpu AVX2/AVX512 layer norm kernel (chunked Welford moments); finite inputs and results"),
    ("f32-silu", ("f32_silu",),
     "checked source f32 SiLU in the pinned torch 2.11.0+cpu vector arithmetic, x/(1+expf(-x)); finite inputs"),
    ("f32-rsqrt", ("source_f32_rsqrt_unqualified",),
     "checked source f32 rsqrt operation as two binary32 roundings, 1/sqrt(x); finite nonnegative inputs"),
    ("f32-sum", ("source_f32_reduce_add_unqualified",),
     "checked +0-seeded sum of a source f32 mean over the last axis in the declared order torch-2.11-cpu-f32-sum-v1; "
     "not torch eager for a noncontiguous operand; finite inputs and results; one row of 32,768 or more elements and "
     "reductions lowered from other source operations stay refused"),
    ("f32-softmax", ("source_f32_softmax_unqualified",),
     "checked source f32 softmax over the last axis in the declared order host-softmax-f32-left-fold-div-rne-v1 "
     "(left-fold maximum and sum, pinned correctly rounded expf, one division); not torch eager's vector order"),
    ("bf16-contraction", ("bf16_affine_contraction",),
     "checked BF16 contraction over one affine loop domain (unfolded convolution); binary32 products (exact unless underflow), ascending binary32 sums"),
    ("sign-flip", ("bf16_neg",),
     "exact BF16 sign-bit flip for every bit pattern, including signed zero and NaN payloads"),
    ("f32-add", ("f32_add",),
     "source f32 add on the checked host emitter; binary32 RNE in source operand order"),
    ("f32-sub", ("f32_sub",),
     "source f32 sub on the checked host emitter; binary32 RNE in source operand order"),
    ("f32-mul", ("f32_mul",),
     "source f32 mul on the checked host emitter; binary32 RNE in source operand order"),
    ("f32-div", ("f32_div",),
     "source f32 div on the checked host emitter; binary32 RNE in source operand order"),
)


def family_id(name):
    return f"{PROVIDER}.{name}"


def family_operations():
    """Each family identity with the source operations its pattern roots."""
    return {family_id(name): operations for name, operations, _ in _FAMILIES}


def describe_compiler():
    return CompilerDescription(
        PROVIDER,
        SourceDomain("merlin.semantic", tuple(sorted({op for _, ops, _ in _FAMILIES for op in ops})),
                     ("bf16", "f32"),
                     (("scope", "checked host C emitter families; other kinds refuse"),
                      ("admission", "each emitter's closed-region source, domain and publication checks"),
                      ("implementation-pin", "host emitter and host math supplier sources; C toolchain unpinned"))),
        tuple(SourcePattern(family_id(name), operations, PIPELINE, 1) for name, operations, _ in _FAMILIES),
        tuple(ImplementationFamily(family_id(name), (family_id(name),), PIPELINE, (("admission", premise),))
              for name, _, premise in _FAMILIES),
        (TargetPipeline(PIPELINE, "merlin.semantic", "c", ARTIFACTS, ABI,
                        ("source-admission", "host-c-emission", "checked-c-export")),),
    )


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _constraints(*entries):
    return tuple((path, _json(value)) for path, value in entries)


def _family(name, implementation_policy, *, constraints=(), source_policies=_STORAGE_POLICIES, **contract):
    return NumericalFamily(
        family_id(name), family_id(name), source_policies, implementation_policy, constraints=constraints,
        numerical_contract=tuple((key, _json(value)) for key, value in sorted(dict(
            schema="merlin.numerical-family.v1",
            evidence_status="implementation_claim_pending_admission",
            source_equivalence="conditional_on_declared_source_and_runtime_domains",
            admission="existing closed-region recognizer, resource/ABI and publication checks",
            **contract,
        ).items())),
    )


def describe_numerical_families() -> tuple[NumericalFamily, ...]:
    """Typed numerical contracts of this description's families, one per family."""
    return (
        _family(
            "layout", "raw-byte-host-layout.v1",
            transformation="exact raw-byte permutation or copy; no numerical conversion",
            runtime={"checks": ["closed dense affine bounds", "host emitter overlap and storage checks"]},
            layout="preserve logical element bytes and explicit index-map operand order",
        ),
        _family(
            "bias-add", "ieee754-bf16-add-rne-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 2),
                (("result", "dtype"), "bf16"),
                (("operands", "0", "dtype"), "bf16"),
                (("operands", "1", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "ieee754-bf16-add-rne-v1"),
                (("operands", "0", "producer", "attributes", "semantic_kind"), "prequantized_contraction"),
                (("operands", "1", "producer", "operation"), "gather"),
            ),
            arithmetic={"operation": "add", "rounding": "BF16 RNE", "operands": "original source order",
                            "bias_position": "after contraction readout restoration and output cast",
                            "forbidden_substitutions": ["MXU seed bias", "primitive VPU FP32 RNE then BF16 chop"]},
            runtime={"domain": "NaNs and invalid infinity operations refuse before writes",
                         "checks": ["original checked host arithmetic emitter", "dense disjoint buffers"],
                         "publication": "existing source host arithmetic checks remain mandatory"},
        ),
        _family(
            "ordered-select", "bf16-ordered-lt-positive-zero-select-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 1),
                (("result", "dtype"), "bf16"), (("operands", "0", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "bf16-ordered-lt-positive-zero-select-v1"),
                (("attributes", "input_domain"), "all-bf16-bit-patterns"),
            ),
            arithmetic={"operation": "ordered x < +0 select +0 else original bits",
                        "negative_zero": "preserve negative zero", "nan": "preserve sign and payload bits",
                        "negative": "negative finite values and negative infinity become positive zero",
                        "positive": "unchanged bits"},
            runtime={"domain": "all 65536 BF16 bit patterns",
                     "checks": ["identical operand/result type and pointwise maps", "dense disjoint buffers"],
                     "publication": "original checked host ordered-select emitter"},
        ),
        _family(
            "window-max", "ordered-window-max-first-equal-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 1), (("result", "dtype"), "bf16"),
                (("operands", "0", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "ordered-window-max-first-equal-v1"),
                (("attributes", "input_domain"), "non-nan-bf16"),
            ),
            arithmetic={"scan": "authored NCHW window loop order; replace only when strictly greater",
                        "ties": "retain first equal, including signed-zero ties", "padding": "negative infinity"},
            runtime={"domain": "non-NaN BF16", "checks": ["strided/dilated/padded window maps",
                         "declared positive window extents", "bounded indices and disjoint buffers"],
                     "publication": "NaN/domain scan before output stores"},
        ),
        _family(
            "batch-norm", "torch-cpu-bf16-batch-norm-inference-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("result", "dtype"), "bf16"), (("operands", "0", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "torch-cpu-bf16-batch-norm-inference-v1"),
                (("attributes", "input_domain"), "finite-bf16"),
            ),
            arithmetic={"epsilon": "authored finite nonnegative epsilon_f32_bits; unchanged binary32 bits",
                        "channel_order": ["RN32(var+epsilon)", "RN32(sqrt)", "RN32(reciprocal)",
                                          "RN32(inverse*weight)", "RN32_fused(bias-mean*alpha)"],
                        "element_order": ["RN32_fused(x*alpha+beta)", "BF16 RNE"],
                        "carriers": "exact integer arithmetic implements each declared binary32 rounding"},
            runtime={"domain": "finite BF16 activation; finite BF16/f32 channel terms; variance+epsilon positive",
                     "checks": ["activation identity and channel-axis maps", "authored mean/var/weight/bias roles",
                                "channel shapes and immutable constant encodings", "finite channel intermediates",
                                "dense disjoint buffers"],
                     "publication": "all term and activation domain checks before stores"},
        ),
        _family(
            "sigmoid", "torch-cpu-bf16-sigmoid-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 1), (("result", "dtype"), "bf16"),
                (("operands", "0", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "torch-cpu-bf16-sigmoid-v1"),
                (("attributes", "input_domain"), "non-nan-bf16"),
            ),
            arithmetic={"order": ["exact BF16 widening", "pinned expf(-x)", "RN32(1+e)",
                                  "RN32(1/d)", "BF16 RNE"],
                        "supplier": "policy-authenticated embedded CORE-MATH expf source and support header",
                        "effects": "errno and FP flags unobserved"},
            runtime={"domain": "non-NaN BF16", "environment": "binary32 RNE and gradual underflow; no fast-math",
                     "checks": ["supplier policy/source identity", "pointwise identity maps and disjoint buffers",
                                "NaN and FP environment preflight"],
                     "qualification": "requires independently linked exact receipt; declaration is not qualification",
                     "publication": "NaN and unsupported environment refuse before stores"},
        ),
        _family(
            "gelu", "bf16-gelu-f32-opmath-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 1), (("result", "dtype"), "bf16"),
                (("operands", "0", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "bf16-gelu-f32-opmath-v1"),
                (("attributes", "input_domain"), "finite-bf16"),
            ),
            arithmetic={"tanh": ["exact BF16 widening", "RN32(x*x)", "RN32(*x)", "RN32(0.044715*cube)",
                                 "RN32(x+that)", "RN32(sqrt(2/pi)*sum)", "pinned tanhf", "RN32(1+t)",
                                 "RN32(x*0.5)", "RN32(product)", "BF16 RNE"],
                        "none": ["exact BF16 widening", "RN32(x*(1/sqrt(2)))", "pinned erff", "RN32(1+e)",
                                 "RN32(x*0.5)", "RN32(product)", "BF16 RNE"],
                        "coefficients": "source binary32 literal bits",
                        "supplier": "policy-authenticated embedded correctly rounded CORE-MATH tanhf or erff",
                        "effects": "errno and FP flags unobserved"},
            runtime={"domain": "finite BF16", "environment": "binary32 RNE and gradual underflow; no fast-math",
                     "checks": ["supplier policy/source identity", "pointwise identity maps and disjoint buffers",
                                "non-finite input and FP environment preflight"],
                     "qualification": "requires independently linked exact receipt; declaration is not qualification",
                     "publication": "non-finite input and unsupported environment refuse before stores"},
        ),
        _family(
            "layer-norm", "torch-2.11-cpu-fma-bf16-layer-norm-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("result", "dtype"), "bf16"), (("operands", "0", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "torch-2.11-cpu-fma-bf16-layer-norm-v1"),
                (("attributes", "input_domain"), "finite-bf16"),
            ),
            arithmetic={"kernel": "torch eager CPU BF16 layer_norm as the pinned 2.11.0+cpu wheel dispatches it on AVX2 "
                                  "and AVX512 hosts: 16-lane BF16 vectors as two 8-lane binary32 halves, 7 fused multiply-adds",
                        "order": ["exact BF16 widening", "RowwiseMoments chunked Welford (chunk 16), "
                                  "cascade and lane combination", "RN32(m2/n)", "RN32(var+eps)", "IEEE sqrt", "RN32(1/root)",
                                  "RN32(-mean*rstd)", "fma(x, rstd, offset)", "fma(that, weight or 1, bias or 0)", "BF16 RNE"],
                        "fused": "one IEEE fused multiply-add at 7 sites: inner Welford m1 and m2, AddMomentsVec m2, scalar AddMoments "
                                 "m1 and sum, and both output steps; the AddMomentsVec m1 and tail m2 updates round "
                                 "product and sum separately",
                        "not_matched": "the wheel's DEFAULT capability and other torch builds",
                        "epsilon": "authored binary32 bits", "affine": "optional BF16 weight and bias of the normalized shape"},
            runtime={"domain": "finite BF16 input and affine terms; finite binary32 statistics and results",
                     "environment": "binary32 RNE and gradual underflow; no fast-math",
                     "checks": ["row and trailing affine maps", "affine shapes and constant encodings",
                                "dense disjoint buffers", "FP environment preflight"],
                     "qualification": "requires independently linked exact receipt; declaration is not qualification",
                     "publication": "a first pass evaluates every row without stores; any refusal writes nothing"},
        ),
        _family(
            "f32-silu", "torch-2.11-cpu-f32-silu-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 1), (("result", "dtype"), "f32"), (("operands", "0", "dtype"), "f32"),
                (("attributes", "numeric_semantics"), "torch-2.11-cpu-f32-silu-v1"),
                (("attributes", "input_domain"), "finite-f32"),
            ),
            arithmetic={"kernel": "torch eager CPU f32 and BF16 SiLU as the pinned 2.11.0+cpu wheel runs it",
                        "order": ["SLEEF expf(-x), 1.0 ULP, fused steps", "RN32(1+e)", "RN32(x/d)"],
                        "source_form": "the capture's x*sigmoid(x) of one original SiLU node",
                        "not_matched": "the wheel's DEFAULT capability and other torch builds"},
            runtime={"domain": "finite binary32", "environment": "binary32 RNE and gradual underflow; no fast-math",
                     "checks": ["pointwise identity maps and disjoint buffers", "non-finite input and FP environment preflight"],
                     "qualification": "requires independently linked exact receipt; declaration is not qualification",
                     "publication": "non-finite input and unsupported environment refuse before stores"},
        ),
        _family(
            "f32-rsqrt", "torch-cpu-f32-rsqrt-two-rounding-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 1), (("result", "dtype"), "f32"), (("operands", "0", "dtype"), "f32"),
                (("attributes", "numeric_semantics"), "source-f32-rsqrt-unqualified-v1"),
                (("attributes", "source_operation"), "aten.rsqrt.default"),
            ),
            arithmetic={"order": ["IEEE sqrt", "RN32(1/root)"]},
            runtime={"domain": "finite nonnegative binary32", "environment": "binary32 RNE and gradual underflow; no fast-math",
                     "checks": ["pointwise identity maps and disjoint buffers", "domain and FP environment preflight"],
                     "qualification": "requires independently linked exact receipt; declaration is not qualification",
                     "publication": "out-of-domain input and unsupported environment refuse before stores"},
        ),
        _family(
            "f32-sum", "torch-2.11-cpu-f32-sum-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 2), (("result", "dtype"), "f32"), (("operands", "0", "dtype"), "f32"),
                (("attributes", "numeric_semantics"), "source-f32-add-unqualified-v1"),
                (("attributes", "reduction_order"), "unresolved"),
                (("attributes", "source_operation"), "aten.mean.dim"),
            ),
            arithmetic={"kernel": "declared order for the sum inside a source f32 mean over the last axis; EXACT only "
                                  "against this order. Torch eager equals it under AVX512, AVX2, and DEFAULT when its "
                                  "operand is contiguous along the reduced axis",
                        "order": ["eight lanes, each four partials of a four-level cascade over the row's vectors",
                                  "scalar tail fold added to lane 0", "lanes folded in ascending order",
                                  "rows under 8 elements: one four-partial cascade"],
                        "excluded": "a reduction lowered from any other source operation, such as softmax or layer "
                                    "norm, and one row of 32,768 or more elements",
                        "not_matched": "an eager operand that is not contiguous along the reduced axis"},
            runtime={"domain": "finite binary32 inputs and sums", "environment": "binary32 RNE and gradual underflow; no fast-math",
                     "checks": ["+0 seed, last axis, dense disjoint buffers", "domain and FP environment preflight"],
                     "qualification": "requires independently linked exact receipt; declaration is not qualification",
                     "publication": "a first pass evaluates every row without stores; any refusal writes nothing"},
        ),
        _family(
            "f32-softmax", "host-softmax-f32-left-fold-div-rne-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 1), (("result", "dtype"), "f32"), (("operands", "0", "dtype"), "f32"),
                (("attributes", "numeric_semantics"), "source-f32-softmax-unqualified-v1"),
                (("attributes", "reduction_order"), "unresolved"),
            ),
            arithmetic={"kernel": "declared order at source f32 precision, the same on every host; not torch eager's "
                                  "vector order, which differs between CPU capabilities",
                        "order": ["left-fold maximum from -inf", "RN32(x - maximum)", "correctly rounded expf",
                                  "left-fold RN32 sum from +0", "RN32(e / sum)"],
                        "source_form": "a complete last-axis softmax lowered from one source softmax; an additive "
                                       "mask before it stays its own f32 add"},
            runtime={"domain": "binary32 scores; NaN results are canonical 0x7fc00000",
                     "environment": "binary32 RNE and gradual underflow; no fast-math or contraction",
                     "checks": ["dense disjoint buffers", "FP environment preflight"],
                     "qualification": "requires independently linked exact receipt; declaration is not qualification",
                     "publication": "exponentials are stored and normalized in place per row"},
        ),
        _family(
            "bf16-contraction", "bf16-exact-product-f32-ltr-rne-bf16-rne-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("result", "dtype"), "bf16"), (("operands", "0", "dtype"), "bf16"),
                (("operands", "1", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "source-bf16-f32-affine-contraction-v1"),
                (("attributes", "input_domain"), "finite-bf16"),
                (("attributes", "reduction_order"), "unresolved"),
            ),
            arithmetic={"order": ["exact BF16 widening", "RN32(x*w)", "RN32 sum from +0 in ascending lexicographic "
                                  "reduction-loop order", "RN32(+bias) when present", "BF16 RNE"],
                        "reduction_order": "source unresolved; ascending order is the declared host choice",
                        "source_form": "unfolded (im2col) gather, BF16 widening, f32 matmul, channel bias, one BF16 cast",
                        "weights": "original BF16 bits; no FP8 recoding"},
            runtime={"domain": "finite BF16 operands; K*max|x|*max|w|+max|bias| below 2^127",
                     "environment": "binary32 RNE and gradual underflow; no fast-math",
                     "checks": ["gathered map within source bounds", "unit output, weight and bias maps",
                                "dense disjoint buffers", "FP environment preflight"],
                     "qualification": "requires independently linked exact receipt; declaration is not qualification",
                     "publication": "domain and bound checks before stores"},
        ),
        _family(
            "sign-flip", "bf16-sign-bit-flip-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 1), (("result", "dtype"), "bf16"), (("operands", "0", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "bf16-sign-bit-flip-v1"),
            ),
            arithmetic={"operation": "flip only the sign bit", "domain": "all 65536 BF16 bit patterns",
                        "nan": "payload bits preserved", "zero": "+0 and -0 exchange",
                        "torch": "agrees for every non-NaN pattern; torch's scalar path returns canonical NaN"},
            runtime={"checks": ["identical operand/result type and pointwise maps", "dense disjoint buffers"],
                     "publication": "original checked host sign-flip emitter"},
        ),
        _family(
            "f32-add", "ieee754-f32-add-rne-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 2), (("result", "dtype"), "f32"),
                (("operands", "0", "dtype"), "f32"), (("operands", "1", "dtype"), "f32"),
                (("attributes", "numeric_semantics"), "ieee754-f32-add-rne-v1"),
            ),
            arithmetic={"operation": "add", "rounding": "binary32 RNE", "operands": "original source order"},
            runtime={"checks": ["checked host emitter source contract", "dense disjoint buffers"],
                     "publication": "existing checked host emitter"},
        ),
        _family(
            "f32-sub", "ieee754-f32-sub-rne-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 2), (("result", "dtype"), "f32"),
                (("operands", "0", "dtype"), "f32"), (("operands", "1", "dtype"), "f32"),
                (("attributes", "numeric_semantics"), "ieee754-f32-sub-rne-v1"),
            ),
            arithmetic={"operation": "sub", "rounding": "binary32 RNE", "operands": "original source order"},
            runtime={"checks": ["checked host emitter source contract", "dense disjoint buffers"],
                     "publication": "existing checked host emitter"},
        ),
        _family(
            "f32-mul", "ieee754-f32-mul-rne-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 2), (("result", "dtype"), "f32"),
                (("operands", "0", "dtype"), "f32"), (("operands", "1", "dtype"), "f32"),
                (("attributes", "numeric_semantics"), "ieee754-f32-mul-rne-v1"),
            ),
            arithmetic={"operation": "mul", "rounding": "binary32 RNE", "operands": "original source order"},
            runtime={"checks": ["checked host emitter source contract", "dense disjoint buffers"],
                     "publication": "existing checked host emitter"},
        ),
        _family(
            "f32-div", "ieee754-f32-div-rne-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 2), (("result", "dtype"), "f32"),
                (("operands", "0", "dtype"), "f32"), (("operands", "1", "dtype"), "f32"),
                (("attributes", "numeric_semantics"), "ieee754-f32-div-rne-v1"),
            ),
            arithmetic={"operation": "div", "rounding": "binary32 RNE", "operands": "original source order"},
            runtime={"checks": ["checked host emitter source contract", "dense disjoint buffers"],
                     "publication": "existing checked host emitter"},
        ),
    )


def _source_roots():
    """Explicit shallow roots: the host emitters and the host math supplier, no recursion."""
    from merlin.runtime import host_math_supplier

    here = Path(__file__).parent
    return (("host-c", here, (".py",)),
            ("host-ops", here / "host_ops", (".py",)),
            ("host-math", host_math_supplier.ROOT, (".py", ".c", ".h", ".json")),
            ("host-math-core", host_math_supplier.ROOT / "third_party/core-math", (".c", ".h")),
            ("host-math-pow", host_math_supplier.ROOT / "third_party/core-math/pow", (".c", ".h")))


def implementation_sha256():
    """Hash sorted, named direct sources of the host emitters, without executing them.

    This is a source pin, not a claim to identify the C toolchain that builds them.
    """
    digest = hashlib.sha256()
    for label, root, suffixes in sorted(_source_roots()):
        for path in sorted(root.iterdir()):
            if not path.is_symlink() and path.is_file() and path.suffix in suffixes:
                digest.update(f"{label}/{path.name}".encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def lowering_recipe(guards):
    """The recipe of one host family route; ``guards`` must name a declared family."""
    if dict(guards).get("family") not in family_operations():
        raise ValueError("host lowering recipe names no declared host family")
    recipe = LoweringRecipe(implementation_sha256(), PIPELINE, guards, ARTIFACTS, ABI)
    describe_compiler().validate_recipe(recipe)
    return recipe


def validate_recipe(recipe):
    """Reject stale source pins and undeclared families in a host recipe."""
    describe_compiler().validate_recipe(recipe)
    if dict(recipe.guards).get("family") not in family_operations() or recipe.artifacts != ARTIFACTS:
        raise ValueError("host lowering recipe differs from the declared host families")
    if recipe.implementation_sha256 != implementation_sha256():
        raise ValueError("host lowering implementation sources changed; rediscover proposals")
