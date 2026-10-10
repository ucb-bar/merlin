"""Host-owned Phase 0 writer implementation."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

from merlin.targetgen import capsule_golden as CG  # noqa: E402
from merlin.targetgen import corpus_spec as CS  # noqa: E402
from merlin.targetgen import golden_store as GS  # noqa: E402
from merlin.targetgen import numeric_falsifiability as NF  # noqa: E402

from .golden_cache import _golden_cached
from .numerics import (
    _float_golden,
    _mx_attention_golden,
    _mx_gemv_batched_golden,
    _mx_golden,
    _simt_golden,
    float_semantics,
    specir_oracle_source_identity,
)
from .sealed_generation import capture_source

INTEGER_CONTRACTION_BOUND_OPS = frozenset(
    {
        "matmul",
        "linear",
        "matmul_bias",
        "fused_matmul_bias",
        "resident_reuse",
        "host_island_seam",
        "residual_seam",
        "conv2d",
        "scope_chain",
        "attention_qk",
    }
)


def _m2m_unavailable_reason() -> str:
    if "MERLIN_PHASE0_FROZEN_SOURCE_MAP" in os.environ:
        if os.environ.get("MERLIN_PHASE0_M2M_REQUIRED") == "1":
            return "selected Model2MLIR capture runtime is unavailable"
        return "frozen Phase 0 has no selected Model2MLIR capture runtime"
    return "model2MLIR capture runtime unavailable (set MERLIN_M2M_PYTHON)"


def _with_attestations(written, source, *, start: int):
    """Record on the capsule the sealed-runner attestations of the captures it was built from."""
    attestations = list(getattr(source, "attestations", ()) or ())[start:]
    if written is None or not attestations:
        return written
    path = Path(written) / "capsule.yaml"
    capsule = yaml.safe_load(path.read_bytes())
    capsule["capture_execution_attestations"] = attestations
    path.write_text(yaml.safe_dump(capsule, sort_keys=False))
    return written


def _skip_or_require_m2m(entry: dict) -> None:
    reason = _m2m_unavailable_reason()
    required = os.environ.get("MERLIN_PHASE0_M2M_REQUIRED") == "1"
    verified = os.environ.get("MERLIN_PHASE0_EVIDENCE_MODE") == "verified"
    if required or verified:
        raise ValueError(f"{entry['name']}: {reason}; requested frontend capsule cannot be omitted")
    print(f"  [skip] {entry['name']}: {reason}")


def _integer_reference_bound(entry: dict, cap: dict) -> dict:
    """Screen concrete integer contraction stimuli against selected internal width."""
    from merlin.targetgen.operation_numerics import integer_partial_sum_bound

    semantics = entry.get("numerical_semantics") or {}
    operation = cap["operation"]
    op, attrs = operation["op"], operation.get("attributes") or {}
    if op == "component_program":
        from .component_execution_budget import source_for_capsule
        from .component_integer_bounds import derive

        return derive(source_for_capsule(cap), semantics)
    policy = (semantics.get("internal_arithmetic") or {}).get("full_operation_overflow_policy")
    if policy != "bounded_exact_requires_each_partial_sum":
        if op == "host_island_seam":
            raise ValueError("host-island integer contractions require a selected internal-width bound policy")
        return {"status": "unknown", "reason": "no selected full-operation internal-width bound policy"}
    if op not in INTEGER_CONTRACTION_BOUND_OPS:
        return {"status": "not_applicable", "reason": "this writer path has no modeled integer contraction"}
    if op == "scope_chain":
        families = attrs.get("scope_families")
        if (
            not isinstance(families, list)
            or len(families) < 3
            or families[:2] != ["movement", "contraction"]
            or any(family != "elementwise_map" for family in families[2:])
        ):
            raise ValueError("integer scope chain needs one selected contraction and only post-contraction maps")
    leaves = CG.materialize_capsule_leaves(cap)

    def bounded_matmul(lhs_name: str, rhs_name: str, *, initial_values=()) -> dict:
        if lhs_name not in leaves or rhs_name not in leaves:
            raise ValueError("integer contraction bound requires concrete lhs and weight operands")
        lhs, rhs = leaves[lhs_name], leaves[rhs_name]
        if len(lhs.shape) != 2 or len(rhs.shape) != 2 or lhs.shape[1] != rhs.shape[0]:
            raise ValueError("integer matmul bound requires matching rank-2 reduction extents")
        result = integer_partial_sum_bound(
            semantics,
            reduction_extent=lhs.shape[1],
            lhs_values=[int(v) for v in lhs.data],
            rhs_values=[int(v) for v in rhs.data],
            initial_values=initial_values,
        )
        if result["status"] != "proven_safe":
            raise ValueError(f"integer mathematical golden cannot qualify selected internal MAC width: {result}")
        return result

    if op == "resident_reuse":
        weight_name, matmuls = attrs.get("weight"), attrs.get("matmuls")
        if not isinstance(weight_name, str) or not isinstance(matmuls, list) or not matmuls:
            raise ValueError("resident reuse bound requires a weight and nonempty matmul roster")
        members = []
        for member in matmuls:
            if (
                not isinstance(member, dict)
                or not isinstance(member.get("lhs"), str)
                or not isinstance(member.get("out"), str)
            ):
                raise ValueError("resident reuse bound has a malformed matmul member")
            members.append(
                {
                    "lhs": member["lhs"],
                    "out": member["out"],
                    "partial_sum_bound": bounded_matmul(member["lhs"], weight_name),
                }
            )
        bounds = [member["partial_sum_bound"] for member in members]
        return {
            "status": "proven_safe",
            "scope": "each listed resident-weight contraction; not full-kernel execution",
            "bound": max(bound["bound"] for bound in bounds),
            "signed_positive_limit": bounds[0]["signed_positive_limit"],
            "mac_result_bits": bounds[0]["mac_result_bits"],
            "members": members,
        }

    if op == "host_island_seam":
        from merlin.runtime.tensor import Tensor

        role, transform = attrs.get("comparison_role"), attrs.get("host_transform")
        if (
            role not in {"island", "no_island"}
            or {"island": "xor_low_bit", "no_island": "none"}[role] != transform
            or attrs.get("accelerator_contractions") != 2
            or attrs.get("shared_accelerator_epilogue") != "saturating_i32_to_i8"
            or (semantics.get("internal_arithmetic") or {}).get("signed_operand_bits") != 8
        ):
            raise ValueError("host-island bound requires the declared two-contraction i8 seam")
        lhs_name, first_weight_name, second_weight_name = (attrs.get("lhs"), attrs.get("weight0"), attrs.get("weight1"))
        if (
            not all(isinstance(name, str) for name in (lhs_name, first_weight_name, second_weight_name))
            or set(leaves) != {lhs_name, first_weight_name, second_weight_name}
            or len(leaves) != 3
        ):
            raise ValueError("host-island bound requires exactly its three concrete input tensors")
        lhs, first_weight, second_weight = (leaves[lhs_name], leaves[first_weight_name], leaves[second_weight_name])
        dimensions = tuple(attrs.get(key) for key in ("M", "K", "H", "N"))
        if (
            any(type(value) is not int or value < 1 for value in dimensions)
            or lhs.shape != dimensions[:2]
            or first_weight.shape != dimensions[1:3]
            or second_weight.shape != dimensions[2:]
            or any(tensor.dtype != "i8" for tensor in (lhs, first_weight, second_weight))
        ):
            raise ValueError("host-island bound operand ABI differs from its declared contractions")
        first = bounded_matmul(lhs_name, first_weight_name)
        # The first sum is proven inside the selected MAC width above. Its
        # saturating narrow and optional bitwise map therefore give the exact
        # signed i8 operand bytes consumed by the second contraction.
        middle = lhs.matmul(first_weight).to_i8()
        if transform == "xor_low_bit":
            mask = attrs.get("xor_mask")
            if type(mask) is not int or not 0 < mask < 128:
                raise ValueError("host-island bound has an invalid low-bit XOR mask")
            middle = Tensor(middle.shape, [value ^ mask for value in middle.data], "i8")
        second = integer_partial_sum_bound(
            semantics,
            reduction_extent=dimensions[2],
            lhs_values=middle.data,
            rhs_values=second_weight.data,
        )
        if second["status"] != "proven_safe":
            raise ValueError(f"integer mathematical golden cannot qualify selected internal MAC width: {second}")
        return {
            "status": "proven_safe",
            "scope": (
                "both host-island contractions on the concrete input and derived middle tensor; "
                "not full-kernel execution"
            ),
            "bound": max(first["bound"], second["bound"]),
            "signed_positive_limit": first["signed_positive_limit"],
            "mac_result_bits": first["mac_result_bits"],
            "members": [
                {"region": "contraction_0", "partial_sum_bound": first},
                {"region": "contraction_1", "partial_sum_bound": second},
            ],
        }

    def name(role, declared):
        return attrs.get(declared) or next((row["name"] for row in cap["inputs"] if row.get("role") == role), None)

    if op == "attention_qk":
        lhs_name, rhs_name = attrs.get("q"), attrs.get("k")
    else:
        lhs_name, rhs_name = name("input", "ifm" if op == "conv2d" else "lhs"), name("weight", "weight")
    if lhs_name not in leaves or rhs_name not in leaves:
        raise ValueError("integer contraction bound requires concrete lhs and weight operands")
    lhs, rhs = leaves[lhs_name], leaves[rhs_name]
    if op in {"matmul", "linear", "matmul_bias", "fused_matmul_bias"}:
        initial = [
            int(value) for key, tensor in leaves.items() if key not in {lhs_name, rhs_name} for value in tensor.data
        ]
        return bounded_matmul(lhs_name, rhs_name, initial_values=initial)
    if op == "attention_qk" and rhs.shape[-1] != lhs.shape[-1]:
        raise ValueError("attention score reduction extents differ between query and key")
    if op == "scope_chain" and rhs.shape[-1] != lhs.shape[-1]:
        raise ValueError("integer scope chain reduction extents differ between lhs and transposed weight")
    reduction_extent = rhs.shape[0] if op == "conv2d" else lhs.shape[-1]
    initial = [int(value) for key, tensor in leaves.items() if key not in {lhs_name, rhs_name} for value in tensor.data]
    result = integer_partial_sum_bound(
        semantics,
        reduction_extent=reduction_extent,
        lhs_values=[int(v) for v in lhs.data],
        rhs_values=[int(v) for v in rhs.data],
        initial_values=initial,
    )
    if result["status"] != "proven_safe":
        raise ValueError(f"integer mathematical golden cannot qualify selected internal MAC width: {result}")
    return result


def _source_integer_reference_bound(entry: dict, cap: dict, directory: Path) -> dict:
    """Bound exact source i8 matmul inputs, including the zero initial sum.

    A source-backed model or quantized PyTorch graph can contain contractions
    whose internal operands are not the capsule's external inputs. Isolated
    PyTorch integer matmul has captured bytes; spec matmul has program operands
    that must be checked alongside its separately materialized capsule inputs.
    """
    semantics = entry.get("numerical_semantics") or {}
    internal = semantics.get("internal_arithmetic") or {}
    if internal.get("full_operation_overflow_policy") != "bounded_exact_requires_each_partial_sum":
        raise ValueError("source-backed integer contraction lacks a selected internal-width bound policy")
    if entry.get("source") == "spec" or entry.get("spec_ref"):
        return _spec_integer_reference_bound(entry, cap, directory)
    operation = cap.get("operation") or {}
    attrs = operation.get("attributes") or {}
    if (
        entry.get("kind") == "model"
        or entry.get("op") == "model"
        or not (entry.get("source") == "pytorch" or entry.get("pytorch_ref"))
        or entry.get("capture_op") != "int_matmul"
        or operation.get("op") != "matmul"
        or cap.get("kind") == "model"
        or (cap.get("numeric_policy") or {}) != {"compare": "exact_int", "dtype": "i32"}
        or semantics.get("operand_dtype") not in {"int8", "i8"}
        or semantics.get("accumulator_dtype") != "i32"
        or attrs.get("epilogue") not in ([], ())
        or len(cap.get("inputs") or []) != 2
        or not CG.is_exact_pytorch_integer_source(cap)
    ):
        raise ValueError("source-backed integer contraction lacks a verified isolated i8 matmul and exact inputs")
    scoped = {**cap, "__dir__": str(directory)}
    # This checks the saved host output against an independent recomputation
    # from the captured bytes. It also refuses missing/mismatched input bytes.
    CG.golden(scoped, directory)
    bound = _integer_reference_bound(entry, scoped)
    if bound["status"] != "proven_safe":
        raise ValueError("source-backed integer contraction has no proven internal partial-sum bound")
    return bound


def _spec_integer_reference_bound(entry: dict, cap: dict, directory: Path) -> dict:
    """Bound both the spec program and the separately materialized capsule operands."""
    from merlin.targetgen.operation_numerics import integer_partial_sum_bound

    operation = cap.get("operation") or {}
    attrs = operation.get("attributes") or {}
    lhs_name, rhs_name, out_name = (attrs.get(key) for key in ("lhs", "weight", "out"))
    inputs = cap.get("inputs") or []
    by_name = {row.get("name"): row for row in inputs if isinstance(row, dict)}
    if (
        entry.get("kind") == "model"
        or entry.get("op") != "matmul"
        or operation.get("op") != "matmul"
        or cap.get("kind") == "model"
        or not isinstance(entry.get("spec_ref"), str)
        or cap.get("spec_ref") != entry["spec_ref"]
        or (cap.get("numeric_policy") or {}) != {"compare": "exact_int", "dtype": "i32"}
        or attrs.get("epilogue") not in ([], ())
        or not all(isinstance(name, str) for name in (lhs_name, rhs_name, out_name))
        or len(inputs) != 2
        or set(by_name) != {lhs_name, rhs_name}
        or (entry.get("numerical_semantics") or {}).get("operand_dtype") not in {"int8", "i8"}
        or (entry.get("numerical_semantics") or {}).get("accumulator_dtype") != "i32"
    ):
        raise ValueError("spec-backed integer contraction lacks an isolated exact i8 matmul")
    golden = GS.load_golden(directory) or {}
    if not isinstance(golden, dict):
        raise ValueError("spec-backed integer contraction lacks an independent golden document")
    provenance = golden.get("oracle_provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("spec-backed integer contraction lacks exact program operand provenance")
    saved_inputs = provenance.get("inputs") or {}
    if (
        golden.get("golden_source") != f"specir_program_{entry['spec_ref'].partition(':')[0]}"
        or provenance.get("spec_ref") != entry["spec_ref"]
        or not isinstance(saved_inputs, dict)
        or set(saved_inputs) != {lhs_name, rhs_name}
    ):
        raise ValueError("spec-backed integer contraction lacks exact program operand provenance")

    def matrix(name: str, role: str) -> list[list[int]]:
        spec, saved = by_name[name], saved_inputs[name]
        shape = spec.get("shape")
        rows = saved.get("decoded") if isinstance(saved, dict) else None
        if (
            spec.get("role") != role
            or spec.get("dtype") != "i8"
            or not isinstance(saved, dict)
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(dim) is not int or dim < 1 for dim in shape)
            or saved.get("shape") != shape
            or not isinstance(rows, list)
            or len(rows) != shape[0]
            or any(
                not isinstance(row, list)
                or len(row) != shape[1]
                or any(type(value) is not int or not -128 <= value <= 127 for value in row)
                for row in rows
            )
        ):
            raise ValueError("spec-backed integer contraction has incomplete exact i8 operand values")
        return rows

    lhs, rhs = matrix(lhs_name, "input"), matrix(rhs_name, "weight")
    if len(lhs[0]) != len(rhs):
        raise ValueError("spec-backed integer matmul has mismatched reduction extents")
    recomputed = [
        [sum(lhs[m][k] * rhs[k][n] for k in range(len(rhs))) for n in range(len(rhs[0]))] for m in range(len(lhs))
    ]
    outputs = golden.get("outputs")
    observed = outputs.get(out_name) if isinstance(outputs, dict) else None
    if (
        not isinstance(observed, list)
        or any(not isinstance(row, list) or any(type(value) is not int for value in row) for row in observed)
        or observed != recomputed
    ):
        raise ValueError("spec-backed integer golden differs from exact operand recomputation")
    program_bound = integer_partial_sum_bound(
        entry["numerical_semantics"],
        reduction_extent=len(rhs),
        lhs_values=[value for row in lhs for value in row],
        rhs_values=[value for row in rhs for value in row],
    )
    if program_bound["status"] != "proven_safe":
        raise ValueError(f"spec-backed integer golden cannot qualify selected internal MAC width: {program_bound}")
    capsule_bound = _integer_reference_bound(entry, cap)
    if capsule_bound["status"] != "proven_safe":
        raise ValueError("spec-backed integer capsule inputs have no proven internal partial-sum bound")
    dominant = max((program_bound, capsule_bound), key=lambda bound: bound["bound"])
    return {
        **dominant,
        "scope": "spec-program and separately materialized integer-capsule matmul operands",
        "members": [
            {"operand_stream": "spec_program", "partial_sum_bound": program_bound},
            {"operand_stream": "capsule_materialized", "partial_sum_bound": capsule_bound},
        ],
    }


def _is_source_backed(entry: dict) -> bool:
    return bool(
        entry.get("kind") == "model"
        or entry.get("op") == "model"
        or entry.get("source") in {"pytorch", "spec"}
        or entry.get("pytorch_ref")
        or entry.get("spec_ref")
    )


# ------------------------------------------------------------------------------------------------
def _write_capsule(entry, binding, out_root, facts_sha: str = "", *, capture=None, component_only: bool = False):
    """Write one capsule, then GUARANTEE it carries its generalization-intent block.

    The stamp is a post-step rather than something each writer does, because there are four writers
    (direct-MLIR, pytorch-sourced, spec-sourced, whole-model) and three of them build their capsule dict
    themselves and return early. Stamping inside ``corpus_spec.build`` alone left 14 of atlas's 33
    capsules unannotated -- exactly the silent-gap failure mode this block exists to close -- so it is
    applied here, at the one point every path must pass through.
    """
    if type(component_only) is not bool:
        raise TypeError("component_only must be an explicit Boolean")
    written = _write_capsule_inner(
        entry, binding, out_root, facts_sha, **({"capture": capture} if capture is not None else {})
    )
    if not written:
        return written
    d = Path(written) if not isinstance(written, Path) else written
    capf = d / "capsule.yaml" if d.is_dir() else None
    if capf is None or not capf.exists():
        return written
    cap = yaml.safe_load(capf.read_text()) or {}
    dirty = False
    regime, _ = CS.entry_binding(entry, binding)
    model_capsule = entry.get("kind") == "model" or entry.get("op") == "model"
    if regime == "int" and _is_source_backed(entry):
        if model_capsule:
            # A whole model's contractions read internal tensors, so it is qualified by the isolated
            # verification of every device group it forms; without that evidence it is refused
            # exactly as before (see model_qualification).
            from .model_qualification import qualify

            bound = qualify(entry, cap, d, binding, Path(out_root))
            cap["model_qualification"] = bound
        else:
            bound = _source_integer_reference_bound(entry, cap, d)
        golden = GS.load_golden(d)
        source = golden.get("golden_source") if isinstance(golden, dict) else None
        if source != "host_torch_eager" and not (isinstance(source, str) and source.startswith("specir_program_")):
            raise ValueError("source-backed integer bound requires its independently captured golden")
        cap["integer_partial_sum_bound"] = bound
        golden["integer_partial_sum_bound"] = bound
        golden["qualification"] = (
            "source-backed integer model qualified by isolated per-group verification; "
            "target execution and full-model numerics unverified"
            if model_capsule
            else "source-backed integer contraction with concrete operand-stream internal-width bounds; "
            "target execution and full-mesh ordering unverified"
        )
        GS.write_golden(d, golden)
        dirty = True
    if not (cap.get("semantic") or {}).get("generalization_axis"):
        _, eb = CS.entry_binding(entry, binding)
        cap["semantic"] = CS._semantic_block(entry, eb)
        dirty = True
    dirty = _backfill_required_classes(cap, binding) or dirty
    _validate_lane_declaration(entry, binding)
    dirty = _carry_declared_blocks(entry, cap) or dirty
    # AFTER the declared blocks are carried, because `lanes` reaches the capsule THERE. Checking before
    # it read an empty lanes block and passed everything -- a verification that cannot see what it
    # verifies is worse than none, because it reports the assertion as checked.
    _verify_a_forbidden_lane_is_provable(d, cap, getattr(binding, "target", None))
    dirty = _cap_oracle_tiers(entry, cap) or dirty
    if not component_only:
        dirty = _stamp_member_geometry(cap, binding) or dirty
    # THE TOLERANCE MUST BE FALSIFIABLE AT THIS GOLDEN'S SCALE, and here is the first point at which
    # both the capsule and its golden exist for EVERY writer -- the same reason the generalization stamp
    # lives here. A profile declares ONE absolute tolerance for a whole target, which is the right shape
    # for a datapath error budget and the wrong shape for a small-magnitude output: a softmax capsule
    # whose golden spans 0.0139..0.1523 was graded at `atol: 0.25`, so zeros, the mean and the midrange
    # all passed it. It reported a numeric pass and proved nothing.
    if (d / GS.DOCUMENT).is_file() and (cap.get("numeric_policy") or {}).get("atol") is not None:
        _gdoc = GS.load_golden(d) or {}
        _pol, _prov = NF.falsifiable_policy(
            cap["numeric_policy"], _gdoc.get("outputs") or {}, name=str(entry.get("name") or d.name)
        )
        if cap.get("numeric_policy") != _pol or cap.get("numeric_falsifiability") != _prov:
            cap["numeric_policy"] = _pol
            cap["numeric_falsifiability"] = _prov
            dirty = True
    if dirty:
        capf.write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    _write_capsule_readme(entry, cap, d)
    return written


#: Profile-entry keys that describe what a capsule is FOR rather than what it computes, and which every
#: writer must carry through untouched. They are stamped in the same post-step as the generalization
#: block, and for the same reason: three of the four writers build their capsule dict themselves, so a
#: key handled in only one of them is silently absent from two thirds of the corpus.
#:
#: ``performance``      which optimization level the capsule exercises and which schedule lever its cycle
#:                      count can see. A capsule is otherwise mute about this, so a perf corpus and a
#:                      functional corpus are indistinguishable once generated.
#: ``comparison_group`` the capsule's place in a set whose cycle counts are comparable to one another --
#:                      a fused implementation against the parts it replaces. The field has been declared
#:                      on four capsules since they were written and consumed by nothing, which is the
#:                      same thing as not existing.
#: ``pass_requirements`` the compiler-obligation classes a capsule demands, which is the ONLY link
#:                      between a catalogued pass and a concrete capsule that requires it
#:                      (``check_pass_obligations.py`` rejects a pass no capsule obliges). It was
#:                      hand-written onto two capsules and unknown to this generator, so every
#:                      regeneration silently deleted the corpus's only pass obligations.
#: ``lanes``               the interop/negative-lane contract: which execution lanes must have carried
#:                      work, and (``forbid``) which must have carried none. Only the whole-model writer
#:                      emitted it, so a model_slice capsule declaring lanes silently lost them -- which
#:                      is how the first host-only capsule generated with `lanes: None` and asserted
#:                      nothing at all.
_DECLARED_BLOCKS = (
    "component_coverage",
    "input_palette",
    "performance",
    "comparison_group",
    "pass_requirements",
    "lanes",
    # The oracle-tier ceiling and the sibling a capped member rests on. Declared
    # once in a profile and carried onto every member derived from it, so the
    # link between a screened member and the capsule that certifies it is
    # machine-readable rather than prose. See merlin.targetgen.tier_policy.
    "max_oracle_tier",
    "max_timing_tier",
    "extends",
)


def _carry_declared_blocks(entry: dict, cap: dict) -> bool:
    """Copy the profile entry's declared intent blocks onto the capsule. Never overwrites one already
    there (a hand-authored capsule is the source of record), and never invents one."""
    dirty = False
    for key in _DECLARED_BLOCKS:
        value = entry.get(key)
        if value is None or cap.get(key) is not None:
            continue
        cap[key] = dict(value) if isinstance(value, dict) else value
        dirty = True
    return dirty


def _cap_oracle_tiers(entry: dict, cap: dict) -> bool:
    """Trim a capsule's required tiers to the deepest one its SIZE can afford, and say what it rests on.

    ``corpus_spec.build`` gives every capsule the target's full tier list, which is right for a
    capsule sized to the tile edge and wrong for one sized to an application: a shape too large to
    simulate cycle-accurately cannot demand the cycle-accurate tier, and demanding it anyway makes
    the whole corpus unrunnable rather than making the capsule affordable.

    ``extends`` is carried onto the capsule for the same reason it exists at all -- an L2-only
    capsule is admissible only as an extension of a sibling that WAS certified, so the thing it rests
    on has to be readable from the capsule itself rather than inferred from a naming convention.
    """
    # The cap may come from the profile entry or already sit on a carried capsule (a hand-authored or
    # materialized member). Either way a written capsule must never demand a tier above its own cap:
    # `required_oracle_tiers: [..., L3]` beside `max_oracle_tier: L2` is unrunnable as a corpus.
    cap_to = str(entry.get("max_oracle_tier") or cap.get("max_oracle_tier") or "")
    if not cap_to:
        return False
    tiers = [str(t) for t in (cap.get("required_oracle_tiers") or ())]
    if cap_to not in tiers:
        raise ValueError(
            f"{cap.get('name')!r} caps its oracle tier at {cap_to!r}, which is not among the tiers "
            f"this target declares ({tiers}); a cap onto a tier that does not exist would silently "
            f"leave the capsule demanding everything"
        )
    trimmed = tiers[: tiers.index(cap_to) + 1]
    changed = trimmed != tiers or cap.get("max_oracle_tier") != cap_to
    cap["required_oracle_tiers"] = trimmed
    cap["max_oracle_tier"] = cap_to
    if entry.get("extends"):
        cap["extends"] = str(entry["extends"])
        changed = True
    return changed


#: ``source_role`` the corpus synthesizer stamps on every entry it derives. Mirrors
#: ``corpus_synth.SOURCE_ROLE``; compared as data so a hand-authored capsule and a derived
#: one can be told apart where the two need different handling.
SYNTH_ROLE = "derived_sweep"


class UnprovableForbid(ValueError):
    """A capsule forbids the mesh on a program the target would legitimately accelerate.

    A distinct type because the right response depends on who wrote the capsule. A HAND-AUTHORED one
    is a contradiction its author must resolve, and aborting is how they find out. A SYNTHESIZED one
    is not: synthesis is pure -- it derives entries from the requirement without building or
    classifying anything -- so the axis genuinely cannot know that `normalization` decomposes into
    regions this target admits. The generator is the first place that fact exists, and the honest
    response there is to drop the capsule and REPORT the family as uncovered, which is the same
    fail-closed shape as `host_only_unsynthesizable`: a requirement that produced no capsule stays
    visible, and nothing can pass in its place.
    """


def _verify_a_forbidden_lane_is_provable(d: Path, cap: dict, target: str | None) -> None:
    """A capsule may only forbid the mesh if its own program has nothing the mesh may legitimately take.

    CLASSIFIED, not predicted. Whether a capsule is host-only is a property of the regions its written
    interface contains, and the only honest way to know is to ask the classifier the coverage gate asks
    (`boundary.profile_capsule`). Deriving it from the family instead is nearly right and not right
    enough: `normalization` decomposes into a reduction and an elementwise map, so the family-level rule
    catches a target that admits either -- and still passed a target admitting NEITHER whose rmsnorm
    program turned out to contain an eligible region anyway.

    Why it must raise rather than quietly drop the assertion. `forbid: [on_mesh]` says the submission
    must NOT accelerate this; on a program containing admitted work that is a demand to leave
    performance on the table, and a compiler doing the right thing is recorded as violating a lane. The
    capsule is wrong, not the compiler, and the generator is where that is still cheap to fix.
    """
    if not target:
        return
    forbid = {str(x) for x in ((cap.get("lanes") or {}).get("forbid") or ())}
    if "on_mesh" not in forbid:
        return
    from merlin.targetgen import boundary as BD

    prof = BD.profile_capsule(d, str(target))
    if prof.kind == BD.HOST_ONLY:
        return
    raise UnprovableForbid(
        f"{cap.get('name')!r} forbids `on_mesh`, but {str(target)!r} classifies its program as "
        f"{prof.kind!r} rather than host-only: it contains region(s) the manifest admits, so the "
        f"assertion demands the compiler decline work it is entitled to do. Choose a family whose "
        f"decomposition this target admits nothing of, or drop the forbid"
    )


def _validate_lane_declaration(entry: dict, binding) -> None:
    """Refuse an unreachable or self-contradictory lane declaration AT GENERATION TIME.

    The whole-model writer already ran ``_checked_lanes``; the other writers did not, because they never
    carried lanes at all. Now that every writer does, the check has to move with it -- a bar the target's
    declared units make unreachable is not a capability test, it is a wall, and the place to catch it is
    where an author can still fix it.
    """
    lanes = entry.get("lanes") or {}
    if not lanes:
        return
    from merlin.targetgen.capsule_source import _checked_lanes

    _checked_lanes(entry, binding)  # raises on an unreachable `require`
    forbid = [str(x) for x in (lanes.get("forbid") or ())]
    both = sorted(set(str(x) for x in (lanes.get("require") or ())) & set(forbid))
    if both:
        raise ValueError(
            f"{entry.get('name')!r}: lane(s) {both} are both required and forbidden; one "
            f"of the two assertions can never hold"
        )
    target = getattr(binding, "target", None)
    if forbid and target:
        from merlin.targetgen.routing import reachable_lanes

        unreachable = sorted(set(forbid) - reachable_lanes(target))
        if unreachable:
            raise ValueError(
                f"{entry.get('name')!r}: forbids lane(s) {unreachable} that {target!r} cannot populate "
                f"anyway, so the assertion is vacuously true and tests nothing"
            )


def _stamp_member_geometry(cap: dict, binding) -> bool:
    """Record which shape class an OBJECTIVE member occupies, and whether real models present it.

    A perf member exists to make generated code faster on shapes that matter, and nothing in a
    generated capsule said which shapes those were. MEASURED on this repo's corpus: 27 of the 29
    classifiable OBJECTIVE members sit in a geometric class the target's own census -- derived from
    real captures -- does not contain, and every reachable class in that census has no members. That
    was invisible from every artifact and answerable only by running a script.

    Stamped here for the same reason the generalization block is: four writers build their capsule
    dict themselves, so a key handled in one of them is absent from three quarters of the corpus.

    DELIBERATELY NOT A GATE. The census's mass-carrying class is recorded as unbuildable on this
    target, so refusing off-census members would emit an empty corpus. Recording the placement makes
    the hole readable from the tracked capsule; deciding what to do about it is the corpus's job, not
    the writer's.
    """
    perf = cap.get("performance")
    if not isinstance(perf, dict) or perf.get("member_class") != "OBJECTIVE":
        return False
    target = str(getattr(binding, "target", "") or "")
    if not target:
        return False
    from merlin.perf.member_geometry import stamp_for

    block = stamp_for(cap, target=target)
    # None and a block saying `in_census: false` are different answers: the first is "this member's
    # geometry is unreadable here", the second is "it was read and no capture presents it". Writing
    # the first as the second would turn an unpriced op into a coverage claim.
    if block is None:
        return False
    # THE GRANULARITY THE EXTENTS WERE MINTED AGAINST, recorded beside them. Sweep axes are written as
    # multiples of the target's own tile, so an extent alone does not say how many array row blocks it
    # spans -- and that count decides, for the array-side stream families, whether a member can
    # exercise their lever at all or is their control. A record, never a bound: nothing fails on it.
    tile = getattr(binding, "tile_dim", None)
    if isinstance(tile, int) and not isinstance(tile, bool) and tile >= 1:
        block = {**block, "row_block": int(tile)}
    if perf.get("shape_geometry") == block:
        return False
    perf["shape_geometry"] = block
    return True


def _write_capsule_readme(entry: dict, cap: dict, d: Path) -> None:
    """Write the capsule's ``README.md`` -- the 5th of the five files a capsule is DEFINED to have.

    The generator only ever emitted four of them, so every generated capsule was incomplete by the
    corpus's own definition and the materialized public view failed its own completeness check the moment
    a capsule arrived without a hand-written README. Derived from the profile entry and the capsule, so it
    cannot go stale: the prose is the entry's ``comment`` when it has one, otherwise a sentence built from
    the op, the source it was authored from, and the operand shapes/dtypes. Never overwrites a README that
    is already there -- the hand-written ones are the frozen source-of-record."""
    rd = d / "README.md"
    if rd.exists():
        return
    name = cap.get("name") or entry.get("name", "")
    prose = (entry.get("comment") or "").strip()
    if not prose:
        op = (cap.get("operation") or {}).get("op") or entry.get("op") or "unknown"
        ops = ", ".join(
            f"{i.get('name')}{list(i.get('shape') or [])}:{i.get('dtype')}"
            for i in (cap.get("inputs") or [])
            if i.get("name")
        )
        src = cap.get("source_reference") or entry.get("source_reference") or ""
        prose = f"{name}: {op}" + (f" over {ops}" if ops else "")
        prose += f", authored from {src}." if src else "."
    line = " ".join(
        f"{k}={v}"
        for k, v in (
            ("kind", cap.get("kind") or entry.get("kind")),
            ("label", cap.get("label") or entry.get("label")),
            ("op", (cap.get("operation") or {}).get("op") or entry.get("op")),
            ("modes", (cap.get("expected") or {}).get("modes", {})),
        )
        if v is not None
    )
    rd.write_text(f"# {name}\n\n{prose}\n\n{line}\n", encoding="utf-8")


def _backfill_required_classes(cap: dict, binding) -> bool:
    """Fill an EMPTY ``expected.instruction_classes`` from the target's own derived taxonomy.

    The source-backed writers (pytorch / spec / model) build their capsule dict themselves and leave this
    empty, so a contraction authored in PyTorch shipped with no coverage requirement at all while the
    direct-MLIR twin next to it carried the full systolic sequence -- the L1 coverage assertion silently
    did not apply to exactly the frontend-faithful capsules the generalization corpus is made of.

    Derived, never hardcoded: the slots come from the op's family in the closed vocabulary and are mapped
    to class names through THIS target's role census. Fail-closed at every step -- an op that owes no
    contraction, an undecidable taxonomy, or a role the target does not have all leave the list empty
    rather than inventing a demand. Only ever fills an empty list; never edits an authored one."""
    exp = cap.get("expected")
    if not isinstance(exp, dict) or exp.get("instruction_classes"):
        return False
    op = (cap.get("operation") or {}).get("op")
    if not op:
        return False
    attrs = (cap.get("operation") or {}).get("attributes", {}) or {}
    modes = exp.get("modes", {}) or {}
    from merlin.targetgen import isa_taxonomy as IT

    tax = IT.taxonomy_for_target(binding.target)  # {} when the target ships no ISA definition
    if not tax or not tax.get("by_class"):
        return False
    want = IT.required_classes_for_op(
        tax,
        op=op,
        output_dtype=attrs.get("output_dtype") or (cap.get("numeric_policy") or {}).get("dtype"),
        epilogue=tuple(attrs.get("epilogue", []) or []),
        movement=op in ("movement", "copy") or bool(modes.get("movement")),
    )
    if not want:
        return False
    exp["instruction_classes"] = list(want)
    return True


def _roster_captures() -> dict:
    """Captured bundles available as derivation evidence, keyed by model name.

    Same store and key normalisation as `check_conformance_coverage._captures`, so a micro model is
    derived from exactly the captures the requirement was derived from.
    """
    from merlin.common.paths import artifacts_dir

    root = artifacts_dir() / "recaptures"
    if not root.is_dir():
        return {}
    out = {}
    for d in sorted(root.iterdir()):
        m = d / "model.mlir"
        if m.is_file():
            out[d.name.replace("_fp32_consistent", "").replace("_consistent", "")] = m
    return out


def _emit_micro_model_loader(entry: dict, target: str, out_root, *, capture_dtype: str | None = None) -> bool:
    """Write the derived micro model's loader into its capsule directory, or say why not.

    `micro_model.spec` selects operations whose standalone placement is supported by the selected
    hardware and software contracts, and puts the others on the host. The run passes its frozen capture
    and software-spec snapshots, so the loader cannot drift with an ambient target file.
    """
    from merlin.targetgen import micro_model as MM

    # Derived-only execution passes exact run-owned snapshots. Pop the internal
    # selector before the entry becomes a public capsule or provenance record.
    captures = entry.pop("_frozen_application_captures", None)
    software_spec = entry.pop("_frozen_software_spec", None)
    if captures is None:
        captures = _roster_captures()
    if not captures:
        print(f"  [skip] {entry['name']}: no captured model is available to derive the inventory from")
        return False
    try:
        spec = MM.spec(target, captures, software_spec=software_spec, capture_dtype=capture_dtype)
        src = MM.emit_pytorch(spec)
    except MM.UnwritableLayer as exc:
        print(f"  [skip] {entry['name']}: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 -- an underivable spec is not a crash
        print(f"  [skip] {entry['name']}: micro-model spec unavailable: {type(exc).__name__}: {exc}")
        return False
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    loader = d / "capsule.pytorch.py"
    loader.write_text(src, encoding="utf-8")
    entry["loader"] = str(loader)
    entry.setdefault("model", entry["name"])
    print(f"  [micro] {entry['name']}: {spec.composition()} over {len(spec.layers)} derived layer(s)")
    return True


def _write_capsule_inner(entry, binding, out_root, facts_sha: str = "", *, capture=None):
    regime, eb = CS.entry_binding(entry, binding)
    if entry.get("op") == "component_program":
        from .component_integer_bounds import preflight_entry

        preflight_entry(entry, binding=binding)
    if entry.get("input_palette") is not None and regime == "mx":
        raise ValueError("component input palettes require separately declared typed block-scale inputs on MX")
    # Whole-model capsule: a small representative network lowered end-to-end via model2MLIR, graded vs its
    # host torch-eager output, GATED so it runs only after the op suite proves itself. Additive: skipped
    # (loudly) when the m2m venv is absent.
    if entry.get("kind") == "model" or entry.get("op") == "model":
        from merlin.targetgen import capsule_source as CSRC

        if entry.get("materialized_capture"):
            artifact = CSRC.materialized_model_artifacts(entry["materialized_capture"])
            return CSRC.write_model_capsule(entry, eb, out_root, artifact=artifact)
        src = capture if capture is not None else capture_source()
        if not src.available():
            _skip_or_require_m2m(entry)
            return None
        # A DERIVED micro model writes its own loader first. Without this the entry names a loader that
        # does not exist, and the capsule that the composition axis exists to produce cannot be built.
        if entry.get("micro_model") and not _emit_micro_model_loader(
            entry, eb.target, out_root, capture_dtype=entry.get("capture_dtype") or eb.operand_dtype
        ):
            return None
        start = len(getattr(src, "attestations", ()) or ())
        return _with_attestations(CSRC.write_model_capsule(entry, eb, out_root, source=src), src, start=start)
    # PREFERRED source: a capsule defined in PyTorch (frontend-faithful), lowered to linalg via model2MLIR
    # with a host torch-eager golden. Opt in per entry (``source: pytorch``). Restricted to the float
    # regime: a host-eager float reference is graded with tolerance, matching the merlin_iface float
    # interface; int/MX datapaths keep the direct-MLIR engines below (the endorsed fallback for the
    # dtypes torch/torchAO does not faithfully model, e.g. int8xint8 systolic or block-scaled MX).
    if entry.get("source") == "pytorch" or entry.get("pytorch_ref"):
        from merlin.targetgen.application_inventory import int_mm_source_is_qualified

        # AN ENTRY THAT NAMES A QUANTIZATION SCHEME has said which arithmetic its program must contain,
        # so the float-regime restriction below does not apply to it. The restriction exists because a
        # host-eager float reference cannot grade an int/MX datapath -- but that is a statement about
        # the DEFAULT weight-only capture, which emits a float matmul over dequantized weights. A W8A8
        # scheme emits `aten._int_mm` accumulating in i32, which IS the mesh's arithmetic, and torch
        # eager then computes the same quantized math, so the golden is right by construction.
        # For operation-derived slices, the source's exact digest-bound integer arithmetic is checked
        # independently. Static and dynamic whole-model quantization remain distinct source schemes.
        if entry.get("quant_scheme") or (
            entry.get("capture_op") == "int_matmul"
            and int_mm_source_is_qualified(entry.get("application_signature_match") or {})
        ):
            pass
        elif regime != "simt":
            raise ValueError(
                f"pytorch source for capsule {entry['name']!r} needs a float dtype "
                f"(got regime {regime!r} for {eb.operand_dtype!r}); author int/MX capsules "
                f"via the direct-MLIR engine"
            )
        from merlin.targetgen import capsule_source as CSRC

        src = capture if capture is not None else capture_source()
        if not src.available():
            # Diagnostic derivation without a selected runtime may skip a
            # frontend capsule. Selected or verified runs must fail closed.
            _skip_or_require_m2m(entry)
            return None
        start = len(getattr(src, "attestations", ()) or ())
        return _with_attestations(CSRC.write_pytorch_capsule(entry, eb, out_root, source=src), src, start=start)
    # Spec source: a capsule whose PROGRAM + bit-exact golden come from the specir verification spec itself
    # (``spec_ref: '<gen>:op.<name>'``). Additive: a gen without a specir program emitter (or no specir) is
    # skipped loudly rather than sinking the target.
    if entry.get("source") == "spec" or entry.get("spec_ref"):
        from merlin.targetgen import capsule_source as CSRC

        src = CSRC.SpecRefSource()
        if not src.available():
            print(f"  [skip] {entry['name']}: spec source needs specir (set SPECIR_ROOT)")
            return None
        try:
            return CSRC.write_spec_capsule(entry, eb, out_root, source=src)
        except CSRC.SpecProgramUnavailable as e:
            print(f"  [skip] {entry['name']}: {e}")
            return None
    cap, mlir = CS.build(entry, eb)
    if cap["operation"]["op"] == "component_program":
        cap["numerical_semantics"] = copy.deepcopy(entry.get("numerical_semantics") or {})
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8"
    )
    if regime == "int":
        bound = _integer_reference_bound(entry, cap)
        cap["integer_partial_sum_bound"] = bound
        (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
        if cap["operation"]["op"] == "component_program":
            from .component_numerics import evaluate

            outputs = evaluate(cap)
            source = "merlin_tensor_component_program"
        else:
            outputs = CG.golden({**cap, "__dir__": ""})
            source = "merlin_tensor_int"
        GS.write_golden(
            d,
            {
                "golden_source": source,
                "integer_partial_sum_bound": bound,
                "qualification": "mathematical reference; target execution and full-mesh ordering unverified",
                "outputs": outputs,
            },
        )
    elif regime == "specir":
        selected_semantics = float_semantics(entry, eb)
        oracle_source = specir_oracle_source_identity(selected_semantics)
        outputs, prov = _golden_cached(_float_golden, entry, eb, facts_sha, oracle_source=oracle_source)
        if specir_oracle_source_identity(selected_semantics) != oracle_source:
            raise OSError("selected SpecIR oracle source changed while computing the golden")
        GS.write_golden(
            d,
            {
                "golden_source": (
                    "specir_refmodel_float"
                    if selected_semantics["selection_status"] == "explicit"
                    else "specir_refmodel_fp8_e4m3_bf16"
                ),
                "oracle_provenance": {
                    "engine": "specir.oracle.dtypes + specir.oracle.refmodel.fp_reduce",
                    "source_identity": oracle_source,
                    "datapath": selected_semantics,
                    # How the datapath decodes an operand code. A unit that admits only normal operands
                    # reads a zero exponent field as zero; the golden decodes it the same way, so the two
                    # references implement ONE datapath (see the target's profile ``datapath`` block).
                    "operand_decode": ("subnormal_flush_to_zero" if eb.subnormal_operand_flush else "exact"),
                    "operand_dtype": eb.cap_dtype(eb.operand_dtype),
                    "accum_dtype": eb.cap_dtype(eb.accum_dtype),
                    "output_dtype": selected_semantics["readout_dtype"],
                    "note": "INDEPENDENT of the target RTL (not self-oracle); specir refmodel is the reference.",
                    "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                    "inputs": prov,
                },
                "outputs": outputs,
            },
        )
    elif regime == "mx":
        # matmul/linear -> the single MX GEMM golden; attention_mx -> the fused flash-attention composition
        # (two MX GEMMs + a bf16 softmax), both over the SAME validated mx_ref engine.
        if entry.get("op") == "attention_mx":
            outputs, prov = _golden_cached(_mx_attention_golden, entry, eb, facts_sha)
            engine = (
                "mlc.validate.mx_ref.mx_matmul x2 (QK & PV, the engine's transcription of the MX "
                "host golden lib/golden/mx_golden.cpp) + numpy bf16 row-softmax; P requantized to mxfp8 per-row"
            )
            datapath = (
                "O = mx_matmul(softmax(mx_matmul(Q,K^T)/sqrt(H) [+softcap]), V); E8M0 per 32-elt "
                "K group; bf16 accumulate + bf16 softmax"
            )
        elif entry.get("op") == "gemv_batched":
            outputs, prov = _golden_cached(_mx_gemv_batched_golden, entry, eb, facts_sha)
            engine = "mlc.validate.mx_ref.mx_matmul x B (independent batched MX GEMMs stacked row-major)"
            datapath = "B x [M,H]@[H,N] on the mx_pe; one E8M0 scale per 32-elt K group; bf16 accumulate"
        else:
            outputs, prov = _golden_cached(_mx_golden, entry, eb, facts_sha)
            engine = (
                "mlc.validate.mx_ref.mx_matmul (the engine's transcription of the MX host golden "
                "lib/golden/{mx_fp_math.h,mx_golden.cpp}; mirrors the RTL, bit-exact vs spike)"
            )
            datapath = (
                "16-deep systolic per-column acc schedule (ACC_E/ACC_M); one E8M0 scale per "
                "32-elt K group; bf16 accumulate"
            )
        GS.write_golden(
            d,
            {
                "golden_source": "mlc_mx_ref_hardware_semantics",
                "oracle_provenance": {
                    "engine": engine,
                    "datapath": datapath,
                    "operand_dtype": eb.cap_dtype(eb.operand_dtype),
                    "block_scale": "e8m0",
                    "output_dtype": "bf16",
                    "note": (
                        "NOT specir (the SpecIR float refmodel is a different datapath); "
                        "MX is a distinct block-scaled datapath."
                    ),
                    "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                    "inputs": prov,
                },
                "outputs": outputs,
            },
        )
    else:  # simt (IEEE fp16/bf16/f32)
        outputs, prov = _golden_cached(_simt_golden, entry, eb, facts_sha)
        GS.write_golden(
            d,
            {
                "golden_source": "ieee_simt_f32_accumulate",
                "oracle_provenance": {
                    "engine": "numpy IEEE float (CVFPU fp32 accumulate; format-rounded operands)",
                    "operand_dtype": eb.cap_dtype(eb.operand_dtype),
                    "accum_dtype": "f32",
                    "output_dtype": "f32",
                    "note": "SIMT cores do ordinary IEEE math; reference is independent of any accelerator model.",
                    "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                    "inputs": prov,
                },
                "outputs": outputs,
            },
        )
    return d
