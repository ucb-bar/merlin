"""Inventory exact spec declarations and separately classify TVM implementation evidence.

This is an audit tool, not a compiler pass or an execution-admission substitute.
It refuses a different specification and unclassified baseline operators.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import yaml

SPEC_SHA256 = "769b77782ea3d7b9bc35c0f21d2797ce09697a0f2aed52c7b8a1dde10a7de9f4"
SPEC_COMMIT = "8ec99308cb1182b9b896a6075300312266fadf6f"

# Exact declaration IDs. These descriptions identify existing implementation
# candidates; none expands the source declaration or supplies missing evidence.
IMPLEMENTATIONS = {
    "contraction": ("candidate", "i8*i8 -> i32; split/cache/blockize/tensorize; OS/WS primitive compute", "gemmini_sketch.py", "Every retained PE partial <=524287 and accumulator partial <=2147483647, including seed; all logical ranks/layout/tails and raw-exit audit required."),
    "elementwise_map": ("candidate", "Producer readout: explicit f32 product, RNE, saturation, optional ReLU", "gemmini_epilogue.py", "Selected scheme and multiplier bits, scalar command partition, nonnegative scale/order proof, pinned RNE oracle; threshold proof alone does not discharge all emission audits."),
    "elementwise_map_standalone": ("candidate_subset", "Same-shape one/two widened byte terms with signed-byte coefficients and constant i32 bias", "gemmini_affine.py", "Bound all prefixes. Full-domain wrapping i32 add/sub/scalar-add is outside bounded-exact admission; standalone full-i32 readout needs its own bound/identity proof."),
    "elementwise_map_after_operand_sum": ("candidate_subset", "Exact byte sum followed by a single proved narrow readout", "gemmini_affine.py", "Use this exact rounding contract; no bias epilogue; per-scale exhaustive contract and emission evidence remain required."),
    "elementwise_map_after_per_operand_sum": ("candidate_subset", "Two independently rounded/saturated byte operands, exact i32 sum, proved narrow readout", "gemmini_affine.py", "Rank 2 only, no broadcast/bias/pool/raw exit. Screen all 65536 pairs for EVERY scale triple; current two fixtures do not admit arbitrary triples or ranks."),
    "pooling_on_readout": ("composition_only", "Existing standalone matcher gathers a window, widens to accumulator, and emits pooled mvout", "gemmini_pool.py", "Declaration is fused readout, not blanket standalone maxpool. Require selected composition/producer proof and complete region audit. Gathered padding is host work, not an implicit padded-pool capability."),
    "reduction_standalone": ("candidate_subset", "Bounded widened-i8 sum; already-present selected reciprocal/readout for window mean", "gemmini_reduction.py", "Resolve standalone hardware facts. Mesh-against-ones must separately satisfy contraction rank/bounds; arbitrary f32/i64 reduction is not admitted by similarity."),
    "movement": ("candidate_subset", "Materialized static copies, broadcasts, uniform fills and selectors; legal views can be elided", "gemmini_elementwise.py", "Resolve actual standalone movement forms/facts and each selector/alias proof. i32/f32 accumulator-carrier copies are not the F1 int8-transfer implementation."),
    "movement_bitwise": ("candidate_subset", "Opaque byte-copy TensorIntrin for bool/i16/f16/bf16/i64/f64; existing i8 DMA", "gemmini_bytes.py", "Explicit declared dtype/rank, closed selectors, LE byte storage, disjointness, unit scale, no pooling, nonzero address, bounded rows/bytes. complex<f32> is declared but not implemented."),
    "component_linear": ("not_implemented", "No general signed-radix-256 component lowering in this TVM branch", None, "Do not relabel byte affine as full integer component support. Requires WS-only component proof and exact host recombination."),
    "component_bilinear": ("not_implemented", "No general multi-limb bilinear lowering in this TVM branch", None, "Requires WS-only component proof, raw full-i32 exits, bounded partials and exact host recombination; no float/int64 arithmetic inference."),
}

BASELINE_GROUPS = [
    (["mm", "bmm", "addmm", "convolution"], ["contraction"], "device_subregion_candidate", "Only admitted integer contraction and seed; preserve any source output conversion separately."),
    (["sum.dim_IntList"], ["reduction_standalone", "contraction"], "device_subregion_candidate", "Bounded byte-source sum; i64 ABI result conversion is retained host work."),
    (["add.Tensor", "add.Scalar", "sub.Scalar"], ["elementwise_map_standalone"], "outside_declared_exact_domain", "Historical full-i32 boundary fixtures wrap. The simplified backend removes this accumulator-add mapping and retains host semantics; bounded-exact affine does not authorize this domain."),
    (["sub.Tensor", "mul.Scalar", "div.Scalar", "neg"], [], "host_only_for_baseline", "Historical full-i32 forms remain host. New byte-affine/readout expressions are different typed programs, not support for these overloads."),
    (["relu"], ["elementwise_map_standalone"], "device_subregion_candidate", "Historical i8 max(x,0): unit load and identity-scale narrow ReLU; no f32 ReLU claim."),
    (["mean.dim", "mean.default", "adaptive_avg_pool1d", "_adaptive_avg_pool2d", "_adaptive_avg_pool3d", "avg_pool2d_backward", "_adaptive_avg_pool2d_backward", "col2im"], [], "host_only_for_baseline", "Historical floating arithmetic is not the declared integer sum/window-mean program."),
    (["convolution_backward", "avg_pool1d", "avg_pool2d", "avg_pool3d"], ["movement"], "host_arithmetic_with_movement_candidate", "Only separately proved movement stages can be device work; whole floating operator is not admitted."),
    (["repeat", "cat", "constant_pad_nd", "reflection_pad1d", "reflection_pad2d", "reflection_pad3d", "replication_pad2d", "replication_pad3d", "diagonal", "permute", "slice.Tensor", "unsqueeze", "view"], ["movement", "movement_bitwise"], "movement_or_view_candidate", "Historical i8 static selector/copy; prove physical commands and aliases. Uniform fill and host-created regions are separately accounted."),
    (["slice_scatter", "select_scatter", "as_strided", "select.int", "split_with_sizes", "squeeze.dim", "squeeze.dims"], ["movement"], "movement_facts_required", "Historical i32/f32 carrier transfer uses accumulator path; do not claim F1 byte-transfer conformance without changing/proving the actual path."),
    (["_to_copy"], ["elementwise_map_standalone"], "widening_candidate", "Historical signed-i8 to i32 unit-scale accumulator load; numeric conversion is not bitwise movement."),
    (["clone", "fill.Scalar", "full", "full_like", "scalar_tensor"], ["movement"], "elided_or_host_abi_copy", "Current fixture elides or retains host ABI copy; no executed accelerator arithmetic claim."),
]


def audit(spec_path, tvm_root):
    data = spec_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SPEC_SHA256:
        raise ValueError("Specification differs from the exact user-selected 8ec99308 bytes")
    spec = yaml.safe_load(data)
    if set(spec["operations"]) != set(IMPLEMENTATIONS):
        raise ValueError("Declaration inventory changed; review every added/removed operation")
    contracts = tvm_root / "apps/gemmini/operator_contracts.py"
    requested = None
    for node in ast.parse(contracts.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REQUESTED_OPERATORS" for t in node.targets):
            requested = ast.literal_eval(node.value)
    classified = {}
    for names, declarations, status, reason in BASELINE_GROUPS:
        for name in names:
            if name in classified:
                raise ValueError("Duplicate baseline classification: " + name)
            classified[name] = dict(operator=name, declaration_paths=["operations." + x for x in declarations], status=status, reason=reason, qualified=False)
    if requested is None or set(requested) != set(classified) or len(requested) != 51:
        raise ValueError("Baseline inventory differs; do not infer mappings for new operators")
    rows = []
    for name, declaration in spec["operations"].items():
        status, mapping, module, obligations = IMPLEMENTATIONS[name]
        source = tvm_root / "python/tvm/relax/backend/contrib" / module if module else None
        rows.append(dict(declaration_path="operations." + name, declaration=declaration,
                         implementation_status=status, mapping=mapping,
                         source=str(source) if source else None,
                         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
                         outstanding_obligations=obligations, qualified=False))
    return dict(schema="merlin.tvm.spec_derivation_audit.v1", spec_commit=SPEC_COMMIT,
                spec_sha256=SPEC_SHA256, execution_provider_revision=spec["execution_provider_revision"],
                execution_admission=False,
                reason="Traceability audit only; source/provider, resolved facts, per-program compiler proofs and instruction/execution gates must be bound separately.",
                declarations=rows, baseline_ops=[classified[name] for name in requested],
                transfer_contracts=spec["transfer_contracts"],
                restrictions=spec["restrictions"],
                baseline_source_sha256=hashlib.sha256(contracts.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--tvm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.spec, args.tvm)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(declarations=len(result["declarations"]), baseline_ops=len(result["baseline_ops"]), execution_admission=False)))


if __name__ == "__main__":
    main()
