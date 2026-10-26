"""Finite composition × scheme sources from explicit hardware facts.

This is the composition axis of corpus_synth, not a second corpus generator.
Source constants are synthetic; missing target premises never acquire defaults.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from functools import lru_cache

from merlin.targetgen.required_regimes import regime_identity
from merlin.targetgen.required_regimes import represents_required_regime as _represents_required_regime
from merlin.targetgen.required_regimes import required_regimes as _required_regimes
from merlin.targetgen.required_regimes import selected_schemes as _selected_schemes


def synthesize_compositions_enabled(workload_spec: Mapping | None) -> bool:
    """Validate the descriptor's optional composition synthesis selection."""
    enabled = (workload_spec or {}).get("synthesize_compositions", True)
    if not isinstance(enabled, bool):
        raise ValueError("workload_spec.synthesize_compositions must be a bool")
    return enabled


def coverage(entries, *, excluded_spec=None, validation=None):
    """Count declared composition pairs, retaining explicit seal exclusions."""
    if validation is not None:
        return validated_coverage(entries, validation, out_of_scope=excluded_spec is not None)
    if excluded_spec is not None:
        schemes = ((excluded_spec.get("quantization") or {}).get("schemes") or {}).get("schemes") or [{"id": None}]
        return [
            dict(
                composition=chain["id"],
                scheme=scheme["id"],
                count=0,
                status="out_of_scope",
                reason="workload_spec.synthesize_compositions is false",
            )
            for chain in sorted((excluded_spec.get("compositions") or {}).get("chains", []), key=lambda c: c["id"])
            for scheme in sorted(schemes, key=lambda s: s["id"])
        ]
    chains, counts = {}, Counter()
    for entry in entries:
        if entry.get("op") != "composition_program":
            continue
        semantic = entry["generalization"]
        chain = semantic["composition_chain"]
        chains[chain["id"]] = chain
        scheme = semantic["quantization_scheme"]
        axes = (("scale_source", scheme["scale_source"]),) if scheme.get("scale_source") == "runtime_input" else ()
        counts[(chain["id"], scheme["id"], axes)] += 1
    rows = []
    for (chain_id, scheme, axes), count in sorted(counts.items()):
        # A cross-check row carries the justification its chain declares beside the selection.
        selection = chains[chain_id].get("quantization_schemes") or {}
        justification = next(
            (
                row["justification"]
                for row in selection.get("cross_checks", [])
                if row["scheme"] == scheme and row.get("axes", {}) == dict(axes)
            ),
            None,
        )
        row = dict(composition=chain_id, scheme=scheme, count=count)
        if axes:
            row["axes"] = dict(axes)
        if scheme in selection.get("primary", ()) and not axes:
            row["role"] = "primary"
        elif justification is not None:
            row.update(role="cross_check", justification=justification)
        else:
            row["role"] = "UNKNOWN"
        rows.append(row)
    return rows


def _positive(facts, key):
    value = facts.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"composition corpus requires derived positive {key}")
    return value


@lru_cache(maxsize=1)
def _searched_scale_grid():
    """Search rational candidates for f32 disagreement and pred(2**e) witnesses."""
    import numpy as np

    from merlin.runtime.exact_readout import form_multiplier, pred_pow2_vectors

    disagreements, predecessors = [], []
    # A public source-search budget, unrelated to any target or model constants.
    accumulators = np.arange(-2048, 2049, dtype=np.int64)
    for denominator in range(3, 257):
        for numerator in (1, 3):
            rational = Fraction(numerator, denominator)
            if rational >= 1 or rational.denominator & (rational.denominator - 1) == 0:
                continue
            multiplier = float(form_multiplier(rational))
            exact = np.array([round(int(a) * Fraction(multiplier)) for a in accumulators])
            rounded = np.rint(accumulators.astype(np.float32) * np.float32(multiplier)).astype(np.int64)
            differing = np.flatnonzero((exact != rounded) & (abs(exact) <= 127))
            if differing.size and len(disagreements) < 2 and multiplier not in [r["multiplier"] for r in disagreements]:
                disagreements.append(dict(multiplier=multiplier, witness=int(accumulators[differing[0]])))
            if len(disagreements) < 2:
                continue
            vectors = pred_pow2_vectors(multiplier, acc_lo=-32768, acc_hi=32768, e_range=range(7))
            products = vectors.astype(np.float32) * np.float32(multiplier)
            targets = [np.nextafter(np.float32(2**e), np.float32(0)) for e in range(7)]
            if any(np.any(products == target) for target in targets):
                predecessors.append(dict(multiplier=multiplier, vectors=[int(v) for v in vectors]))
                if multiplier not in [r["multiplier"] for r in disagreements]:
                    return tuple(disagreements), tuple(predecessors)
    raise ValueError("composition scale property search exhausted its finite budget")


def arbitrary_scale_grid():
    """Return private copies so callers cannot mutate the cached search result."""
    return copy.deepcopy(_searched_scale_grid())


def scheme_scales(scheme):
    if scheme["scale_value"] == "power_of_two":
        return [0.0625, 0.125, 0.25], []
    if scheme["scale_value"] != "arbitrary_f32":
        raise ValueError("composition scale domain is unresolved")
    disagreements, predecessors = arbitrary_scale_grid()
    grid = [r["multiplier"] for r in disagreements] + [predecessors[0]["multiplier"]]
    return grid, [*disagreements, *predecessors]


def held_out_shapes(manifest=None):
    """Read shapes only inside trusted generation; never publish the exclusion set."""
    import yaml

    from merlin.common.paths import merlin_dir

    manifest = manifest or merlin_dir() / "contract/capsules/MANIFEST.yaml"
    document = yaml.safe_load(manifest.read_text())
    shapes = set()
    for member in document.get("generated", []):
        if not member.startswith("_model_layers/"):
            continue
        path = manifest.parent / member / "capsule.yaml"
        cap = yaml.safe_load(path.read_text())
        attrs = cap["operation"].get("attributes", {})
        inputs = {row["name"]: row for row in cap["inputs"]}
        if cap["operation"]["op"] in {"matmul", "linear"}:
            a, w = inputs[attrs["lhs"]]["shape"], inputs[attrs["weight"]]["shape"]
            shapes.add((a[0], a[1], w[1]))
        elif cap["operation"]["op"] == "conv2d":
            shapes.add(tuple(v for row in cap["inputs"] for v in row["shape"]))
            image = inputs[attrs["ifm"]]["shape"]
            weights = inputs[attrs["weight"]]["shape"]
            from merlin.runtime.commandbuffer import conv_out_dims

            # The held-out convolution's flattened contraction is excluded too.
            height, width = conv_out_dims(
                image[1], image[2], attrs["kh"], attrs["kw"], attrs["stride"], attrs["padding"], attrs["dilation"]
            )
            shapes.add((image[0] * height * width, weights[0], weights[1]))
        else:
            raise ValueError("held-out manifest has an unsupported shape carrier")
    if not shapes:
        raise ValueError("held-out manifest provides no model-layer shapes")
    return frozenset(shapes)


def assert_generic_shape(entry, excluded):
    if (entry["M"], entry["K"], entry["N"]) in excluded:
        raise ValueError("composition shape equals a held-out model-layer instance")


def _producer(chain):
    producers = chain["producer"]
    return ("movement" if "movement" in producers else producers[0]) if isinstance(producers, list) else producers


# Witnesses of the arbitrary_f32 scale search; they have no meaning under another scale value.
_SCALE_REGRESSIONS = ("regression_vectors", "regression_second", "pred_pow2")
# Rows the campaign never executed: neither admitted nor evaluated refusals.
_UNEXECUTED_STATES = frozenset({"deferred_by_stratification", "not_selected_for_execution"})
# Shape classes that place one output edge off the tile; every selected scheme keeps one.
_TAILS = ("partial", "m_tail", "n_tail", "k_tail")


@lru_cache(maxsize=64)
def operand_sum_rounding_witnesses(scales, operand_bits):
    """Operand pairs whose source f32 order differs from one rational rounding.

    The source rounds each scaled operand to f32, rounds their sum to f32 and
    then rounds to even; a reference that rounds the exact rational sum once
    differs on exactly these pairs of the signed operand domain.
    """
    from merlin.llvmlower.requantization import f32

    left_scale, right_scale = (f32(float(scale)) for scale in scales)
    bound = 1 << (operand_bits - 1)
    return tuple(
        (left, right)
        for left in range(-bound, bound)
        for right in range(-bound, bound)
        if round(f32(f32(f32(left) * left_scale) + f32(f32(right) * right_scale)))
        != round(left * Fraction(left_scale) + right * Fraction(right_scale))
    )


def _variant_groups(chain, facts):
    """Shape classes as rotation units; an expected-refuse twin shares its positive's scheme."""
    groups = [("aligned",), *((tail,) for tail in _TAILS), ("sub_tile",), ("multi_block",)]
    producer = _producer(chain)
    if producer == "operand_sum":
        return groups + [("scaled_operands",)]
    if producer == "movement":
        return groups + [
            ("standalone_pool", "standalone_pool_unfenced"),
            ("padded_pool", "padded_pool_unfenced"),
            ("window_mean",),
            ("pool_below_tile",),
            ("pool_above_tile",),
        ]
    if chain.get("extension"):
        return groups + [("two_digits",), ("noninteger_coefficient",)]
    if chain.get("multiplicity"):
        return groups + [("different_exits",)]
    groups += [("k_beyond_pass",), ("bias_relu",), ("fused_pool",), *((label,) for label in _SCALE_REGRESSIONS)]
    if "accumulator" in chain.get("exits", []):
        groups += [("accumulator_exit",)]
    if "OS" in facts.get("dataflows", []):
        groups += [("os_constant", "os_bound_outside", "os_runtime")]
    return groups


def _pairs(groups, selected):
    """Every selected scheme gets the aligned case and one tail; other shape classes appear once.

    Shape classes rotate over the selected schemes in declaration order. Scale-search
    regressions skip to the next arbitrary_f32 scheme and are omitted when none is selected.
    """
    pairs = [(scheme, "aligned") for scheme in selected]
    turn = 0
    for group in groups[1:]:
        candidates = selected
        if any(label in _SCALE_REGRESSIONS for label in group):
            candidates = [s for s in selected if s["scale_value"] == "arbitrary_f32"]
            if not candidates:
                continue
        scheme = next(s for s in selected[turn:] + selected[:turn] if s in candidates)
        turn = (selected.index(scheme) + 1) % len(selected)
        pairs += [(scheme, label) for label in group]
    for scheme in selected:
        if not any(s is scheme and label in _TAILS for s, label in pairs):
            pairs.append((scheme, "partial"))
    return pairs


def _case(chain, scheme, label, facts):
    dim, width = _positive(facts, "tile_dim"), _positive(facts, "columns_per_scale")
    m, k, n = dim, dim, dim
    if label == "partial":
        m, k, n = dim + 1, dim + 1, dim + 1
    elif label == "sub_tile":
        m = k = n = max(1, dim - 1)
    elif label == "multi_block":
        n = 2 * width + 1
    elif label in {"m_tail", "n_tail", "k_tail"}:
        m += int(label == "m_tail")
        n += int(label == "n_tail")
        k += int(label == "k_tail")
    elif label == "k_beyond_pass":
        k = 2 * dim + 1
    grid, vectors = scheme_scales(scheme)
    producer = _producer(chain)
    path = facts["output_path"]
    stages = [s["kind"] for s in path["readout"] if s["kind"] != "pool"]
    exit_kind = "accumulator" if label == "accumulator_exit" else "requant"
    case = dict(
        producer=producer,
        readout_stages=stages,
        accumulator_stages=[],
        exit=exit_kind,
        dataflow=("WS" if "WS" in facts["dataflows"] else facts["dataflows"][0]) if "dataflows" in chain else None,
        shape_class=label,
        output_columns=n,
        accumulator_range=[-32768, 32767],
        members=["producer", "readout"],
    )
    if label == "bias_relu":
        case["accumulator_stages"] = ["preload"]
        case["members"].append("bias")
    if producer == "operand_sum":
        case.update(
            operand_multipliers=grid[:2]
            if scheme["operand_sum_rounding"] == "per_operand" or label == "scaled_operands"
            else [1.0, 1.0],
            operand_requantization="round_saturate_each",
            negates_load_minimum=False,
        )
    if (producer == "movement" and label != "window_mean") or label == "fused_pool":
        window = dim - 1 if label == "pool_below_tile" else dim + 1 if label == "pool_above_tile" else 2
        window = max(1, window)
        side = max(dim + 1, window + 1)
        m = side * side
        case.update(
            pool_in_dims=[side, side],
            pool_window=[window, window],
            pool_stride=[window, window],
            padding=[1, 1] if label.startswith("padded_pool") else [0, 0],
            indices_live=False,
            reduction_kind="max",
        )
        case["readout_stages"].append("pool")
        case["members"].append("pool")
        if producer == "movement":
            case["scratchpad_source_pooling_fenced"] = not label.endswith("_unfenced")
        if label.startswith("standalone_pool"):
            grid = [1.0]
            if "activation" in case["readout_stages"]:
                case["readout_stages"].remove("activation")
    if label == "window_mean":
        k = 2 * 2  # Synthetic square window, independent of target geometry.
        grid = [1 / k]
        case.update(producer="contraction", reduction_kind="sum", constant_weights=[[1] * n for _ in range(k)])
    if label.startswith("os_"):
        bits = _positive(facts, "mesh_signed_bits")
        operand_bits = _positive(facts, "operand_signed_bits")
        k = (1 << (bits - 1)) // (1 << (operand_bits - 1))
        if label == "os_bound_outside":
            k += 1
        if k < 1 or k * n > 1_000_000:
            raise ValueError("OS boundary exceeds the finite source budget")
        # Flattened spatial 1x1 convolution: [H*W, CI] @ [CI, CO].
        case.update(
            dataflow="OS",
            constant_weights=None if label == "os_runtime" else [[1] * n for _ in range(k)],
            seed_range=[0, 0],
            pointwise_kernel=[1, 1],
        )
    if chain.get("extension"):
        bits = _positive(facts, "operand_signed_bits")
        coefficient = (
            (1 << (bits - 1)) + 1 if label == "two_digits" else 0.5 if label == "noninteger_coefficient" else 1
        )
        case.update(
            coefficient=coefficient,
            source_k=k,
            residual_width=n,
            source_partial_bound=32768,
            residual_range=[-(1 << (bits - 1)), (1 << (bits - 1)) - 1],
        )
        case["members"].append("residual")
    if chain.get("multiplicity"):
        count = chain["multiplicity"]["min"]
        case.update(
            immutable_shared_input=True,
            resident_bytes=m * k,
            siblings=[dict(exit="accumulator" if i else "requant", readout_independent=True) for i in range(count)],
        )
        case["members"] += [f"sibling_{i}" for i in range(1, count)]
    if label in _SCALE_REGRESSIONS:
        m = max(1, len(vectors[-1].get("vectors", []))) if vectors else dim
        selected = 2 if label == "pred_pow2" else 1 if label == "regression_second" else 0
        grid = [grid[selected]]
        case["readout_stages"] = [s for s in case["readout_stages"] if s != "activation"]
        case["regression_vectors"] = [vectors[selected]] if vectors else []
        case["accumulator_stages"] = ["preload"]
    operand_bound = 1 << (_positive(facts, "operand_signed_bits") - 1)
    source_bound = k * operand_bound * operand_bound if case["producer"] == "contraction" else 2 * operand_bound
    if case.get("constant_weights") is not None:
        source_bound = operand_bound * max(
            sum(abs(row[column]) for row in case["constant_weights"]) for column in range(n)
        )
    if "source_partial_bound" in case:
        case["source_partial_bound"] = source_bound
        source_bound += abs(case["coefficient"]) * operand_bound
    if "preload" in case["accumulator_stages"]:
        rows = case.get("regression_vectors") or []
        values = sorted({v for row in rows for v in row.get("vectors", [row.get("witness", 0)])}) or [-7, 0, 7]
        case["bias_values"] = values
        source_bound += max(abs(v) for v in values)
    case["accumulator_range"] = [-int(source_bound), int(source_bound)]
    case["output_columns"] = n
    from .column_readout_capsules import scheme_column_scales

    gains = scheme_column_scales(scheme, n, grid)
    block_width = width if scheme["weight_scale_granularity"] == "column_block" else n
    case["multipliers"] = gains[::block_width]
    if scheme.get("scale_source") == "runtime_input" and exit_kind != "accumulator":
        case["runtime_scale_input"] = "ReadoutScale"
    if exit_kind == "accumulator":
        case.update(readout_stages=[], multipliers=[])
    if case.get("reduction_kind") == "sum":
        from merlin.compile.composition_fusion import require_exact_mean_window

        require_exact_mean_window(k)
    return dict(M=m, K=k, N=n, composition_instance=case)


def entries(software_spec, facts, *, excluded_shapes=None):
    """Each chain under its declared schemes: aligned and tail per scheme, other shapes once."""
    from merlin.targetgen.quantization_spec import resolve_quantization_schemes
    from merlin.targetgen.software_spec import validate_compositions

    chains = validate_compositions(software_spec["compositions"])["chains"]
    if any("dataflows" in c for c in chains) and not facts.get("dataflows"):
        raise ValueError("composition corpus requires derived dataflows")
    resolved, _ = resolve_quantization_schemes(software_spec, facts)
    schemes = sorted(resolved["schemes"], key=lambda s: s["id"])
    if not schemes or any(s["status"] != "resolved" for s in schemes):
        raise ValueError("composition corpus requires resolved declared schemes")
    excluded = held_out_shapes() if excluded_shapes is None else excluded_shapes
    result = []
    for chain in sorted(chains, key=lambda c: c["id"]):
        selected = _selected_schemes(chain, schemes, resolved["axes"])
        pairs = _pairs(_variant_groups(chain, facts), selected)
        seen = set()
        for scheme, label in pairs:
            identity = scheme["id"] + ("_runtime_input" if scheme.get("scale_source") == "runtime_input" else "")
            if (identity, label) in seen:
                continue
            seen.add((identity, label))
            shape = _case(chain, scheme, label, facts)
            entry = dict(
                name=f"SY_composition_{chain['id']}_{identity}_{label}",
                cat="layers",
                kind="layer",
                op="composition_program",
                operand_dtype=facts["dtype"],
                source_role="derived_sweep",
                label="public",
                source_reference="composition axis: declared chain × scheme × fact-derived shape class",
                lanes={"require": ["on_mesh"]},
                stimulus_range=[-8, 7],
                **shape,
            )
            if label in _SCALE_REGRESSIONS and shape["composition_instance"].get("regression_vectors"):
                regressions = shape["composition_instance"]["regression_vectors"]
                values = sorted({v for row in regressions for v in row.get("vectors", [row.get("witness", 0)])})
                entry["M"] = len(values)
                entry["input_palette"] = dict(
                    schema="merlin.component_input_palette.v1",
                    inputs=[
                        dict(name="W0", axis="linear", values=[0], offset=0),
                    ],
                )
            if "preload" in shape["composition_instance"]["accumulator_stages"]:
                palette = entry.setdefault("input_palette", dict(schema="merlin.component_input_palette.v1", inputs=[]))
                palette["inputs"].append(
                    dict(name="Bias", axis=0, values=shape["composition_instance"]["bias_values"], offset=0)
                )
            instance = shape["composition_instance"]
            if instance["producer"] == "operand_sum" and scheme["operand_sum_rounding"] == "single_rounding":
                # Spread pairs on which source f32 order and one rational rounding disagree.
                witnesses = operand_sum_rounding_witnesses(
                    tuple(instance["operand_multipliers"]), _positive(facts, "operand_signed_bits")
                )
                count = min(len(witnesses), entry["M"] * entry["N"], 8)
                if count:
                    chosen = [witnesses[index * len(witnesses) // count] for index in range(count)]
                    palette = entry.setdefault(
                        "input_palette", dict(schema="merlin.component_input_palette.v1", inputs=[])
                    )
                    palette["inputs"].extend(
                        dict(name=name, axis="linear", values=[pair[side] for pair in chosen], offset=0)
                        for side, name in enumerate(("Left", "Right"))
                    )
            if "runtime_scale_input" in shape["composition_instance"]:
                palette = entry.setdefault("input_palette", dict(schema="merlin.component_input_palette.v1", inputs=[]))
                palette["inputs"].append(
                    dict(
                        name=shape["composition_instance"]["runtime_scale_input"],
                        axis="linear",
                        values=shape["composition_instance"]["multipliers"][:1],
                        offset=0,
                    )
                )
            assert_generic_shape(entry, excluded)
            entry["generalization"] = dict(
                composition_chain_id=chain["id"],
                quantization_scheme_id=scheme["id"],
                shape_class=label,
                generalization_axis="composition",
                semantic_family="elementwise_map"
                if shape["composition_instance"]["producer"] == "operand_sum"
                else shape["composition_instance"]["producer"],
                composition_chain=copy.deepcopy(chain),
                quantization_scheme=copy.deepcopy(scheme),
                composition_instance=copy.deepcopy(shape["composition_instance"]),
                composition_facts=copy.deepcopy(facts),
                must_accelerate=True,
                eligible=True,
            )
            result.append(entry)
    return result


def validation_binding(software_spec, workload_spec=None):
    """A record is explicit contract data; external completeness lists are never inputs."""
    declared = ((software_spec or {}).get("compositions") or {}).get("validation")
    selected = (workload_spec or {}).get("fusion_validation")
    if declared is not None and selected is not None and declared != selected:
        raise ValueError("conflicting fusion validation selections")
    binding = declared if declared is not None else selected
    from merlin.common.digest import is_sha256

    if binding is not None and (
        not isinstance(binding, dict)
        or not is_sha256(binding.get("sha256"))
        or not isinstance(binding.get("path"), str)
        or not binding["path"]
    ):
        raise ValueError("fusion validation requires a declared path and sha256")
    return binding


def fusion_structure_class(semantic):
    """Classify practice-level operator shapes solely from graph semantics."""
    nodes = semantic.get("nodes", [])
    if semantic["structure"] == "shared_input":
        return "shared_input"
    if semantic["structure"] != "single" or len(nodes) != 1:
        return semantic["structure"]
    node = nodes[0]
    if node.get("residual"):
        return "residual_k_extension"
    if node["form"] == "operand_sum":
        return "scaled_loads"
    if node.get("realization") == "mean":
        return "mean"
    if node["form"] == "window_movement":
        return "standalone_pool"
    if any(stage["kind"] == "pool" for stage in node["readout"]["stages"]):
        return "fused_pool_" + node["readout"].get("pool_padding_class", "unpadded")
    return "contraction_epilogue"


def admitted_regime(semantic):
    """One cover obligation per scheme × semantic structure class × exit."""
    from merlin.common.jsonio import canonical_json

    # Keep heterogeneous sibling schemes/exits, but not their order or realization.
    return canonical_json(
        dict(
            structure=fusion_structure_class(semantic),
            schemes=sorted({node["scheme"] for node in semantic.get("nodes", [])}),
            exits=sorted({node["readout"]["exit"] for node in semantic.get("nodes", [])}),
        )
    ).decode()


def _operator_regimes(semantic):
    """A mixed sibling graph can witness several scheme/exit obligations."""
    from merlin.common.jsonio import canonical_json

    return sorted(
        {
            canonical_json(
                dict(
                    structure=fusion_structure_class(semantic),
                    schemes=[node["scheme"]],
                    exits=[node["readout"]["exit"]],
                )
            ).decode()
            for node in semantic.get("nodes", [])
        }
    )


def _case_obligations(row, admitted):
    from merlin.common.jsonio import canonical_json

    semantic = admitted[row["candidate_id"]]["semantic"]
    obligations = {(row["regime"], row["shape_class"]), ("dataflow", semantic.get("dataflow"))}
    for node in semantic["nodes"]:
        for key, value in (
            ("realization", node.get("realization")),
            ("stages", node["readout"]["stages"]),
            ("numeric_domain", node.get("regime", {})),
            ("seed_source", node.get("seed_source")),
            ("producer", node["form"]),
        ):
            obligations.add((row["regime"], key, canonical_json(value).decode()))
    obligations.add((row["regime"], "producer_forms", canonical_json(semantic.get("producer_forms", [])).decode()))
    return obligations


def _representative(options):
    tails = [row for row in options if row["shape_class"] in _TAILS]
    aligned = [row for row in options if row["shape_class"] == "aligned"]
    return (tails or aligned or options)[0]


def _regime_waived(chain, scheme, axes):
    return any(
        row.get("scheme") == scheme
        and row.get("axes", {}) == axes
        and isinstance(row.get("reason"), str)
        and row["reason"].strip()
        for row in (chain or {}).get("regime_waivers", [])
    )


def required_regime_blockers(software_spec, rows):
    """Retain every spec obligation, including regimes absent from enumeration."""
    blockers = []
    for chain, scheme, axes in _required_regimes(software_spec):
        matches = [row for row in rows if _represents_required_regime(row["semantic"], chain, scheme, axes)]
        if any(
            row.get("admitted") and row.get("state") == "candidate" and row.get("required_case_ids") for row in matches
        ):
            continue
        if _regime_waived(chain, scheme, axes):
            continue
        blockers.append(
            dict(
                regime=regime_identity(chain, scheme, axes),
                reason="no executed, admitted representative" if matches else "not enumerated",
                counts_by_state=dict(sorted(Counter(row.get("state", "unknown") for row in matches).items())),
            )
        )
    return blockers


def _require_spec_regimes(software_spec, semantics):
    from merlin.targetgen.corpus_synth import SynthesisError

    missing = [
        regime_identity(chain, scheme, axes)
        for chain, scheme, axes in _required_regimes(software_spec)
        if not any(_represents_required_regime(semantic, chain, scheme, axes) for semantic in semantics)
        and not _regime_waived(chain, scheme, axes)
    ]
    if missing:
        from merlin.common.jsonio import canonical_json

        raise SynthesisError(
            "required fusion regimes lack an admitted sampled representative: " + canonical_json(missing).decode()
        )


def sample_validated(
    validation, suite, *, software_spec=None, budget=160, workload_count=0, target_count=None, selection_path=None
):
    """Cover scheme/structure/exit obligations, then fill with diverse cross-checks."""
    from merlin.targetgen.corpus_synth import SynthesisError, require_budget

    require_budget(workload_count, budget)
    spec_target = ((software_spec or {}).get("compositions") or {}).get("target_count")
    if target_count is not None and spec_target is not None and target_count != spec_target:
        raise ValueError("workload composition_target_count conflicts with spec target_count")
    if target_count is None:
        target_count = spec_target if spec_target is not None else 50
    if type(target_count) is not int or target_count < 1:
        raise ValueError("composition target_count must be a positive integer")
    admitted = {
        row["candidate_id"]: row
        for row in validation["rows"]
        if row["admitted"] and row.get("state") not in _UNEXECUTED_STATES
    }
    if software_spec is None:
        raise SynthesisError("fusion composition sampling requires the selected software spec")
    blockers = required_regime_blockers(software_spec, validation["rows"])
    if blockers:
        from merlin.common.jsonio import canonical_json

        selection = dict(
            schema="merlin.fusion_corpus_selection.v1", selected=[], seal_blockers=blockers, status="blocked"
        )
        if selection_path is not None:
            from merlin.common.jsonio import write_canonical_json

            write_canonical_json(selection_path, selection)
        error = SynthesisError(
            "required fusion regimes lack an executed, admitted representative: " + canonical_json(blockers).decode()
        )
        error.selection = selection
        raise error
    if not admitted:
        raise SynthesisError("fusion composition synthesis requires at least one admitted candidate")
    _require_spec_regimes(software_spec, [row["semantic"] for row in admitted.values()])
    pool = []
    for case in suite["rows"]:
        candidate = admitted.get(case["candidate_id"])
        if candidate is not None:
            pool.append(
                dict(
                    candidate_id=case["candidate_id"],
                    case_id=case["case_id"],
                    shape_class=case["shape_class"],
                    case_sha256=case.get("sha256"),
                    regime=admitted_regime(candidate["semantic"]),
                    regimes=_operator_regimes(candidate["semantic"]),
                )
            )
    pool.sort(key=lambda row: (row["candidate_id"], row["case_id"]))
    regimes = sorted({regime for row in admitted.values() for regime in _operator_regimes(row["semantic"])})
    selected = {}
    for regime in regimes:
        options = [row for row in pool if regime in row["regimes"]]
        if not options:
            raise SynthesisError("admitted fusion regime has no validated case")
        row = _representative(options)
        selected[row["case_id"]] = {**row, "role": "sampled_regime"}
    minimum = len(selected)
    # Explicit axis obligations can distinguish cases inside one operator cover cell.
    # They remain required cross-checks even when the requested fill is smaller.
    for chain, scheme, axes in _required_regimes(software_spec):
        if _regime_waived(chain, scheme, axes):
            continue
        if any(
            _represents_required_regime(admitted[row["candidate_id"]]["semantic"], chain, scheme, axes)
            for row in selected.values()
        ):
            continue
        options = [
            row
            for row in pool
            if _represents_required_regime(admitted[row["candidate_id"]]["semantic"], chain, scheme, axes)
        ]
        if not options:
            raise SynthesisError("required fusion regime has no validated case")
        row = _representative(options)
        selected[row["case_id"]] = {**row, "role": "cross_check"}
    over_limit = len(selected) > 60 or len(selected) + workload_count > budget
    if over_limit:
        import warnings

        warnings.warn(
            f"mandatory admitted regime cover needs {minimum} composition capsules "
            f"({len(selected)} with required spec cross-checks) plus {workload_count} workloads; "
            f"exceeds composition limit 60 or budget {budget}; emitting full cover",
            stacklevel=2,
        )
    allocation = max(len(selected), min(target_count, 60, budget - workload_count, len(pool)))
    covered = set().union(*(_case_obligations(row, admitted) for row in selected.values()))
    while len(selected) < allocation:
        options = [row for row in pool if row["case_id"] not in selected]
        row = min(
            options, key=lambda r: (-len(_case_obligations(r, admitted) - covered), r["candidate_id"], r["case_id"])
        )
        selected[row["case_id"]] = {**row, "role": "cross_check"}
        covered.update(_case_obligations(row, admitted))
    from merlin.common.jsonio import canonical_sha256

    unsampled = sorted(set(regimes) - {regime for row in selected.values() for regime in row["regimes"]})
    if unsampled:
        raise SynthesisError(f"admitted fusion regimes were not sampled: {unsampled}")
    _require_spec_regimes(
        software_spec,
        [admitted[row["candidate_id"]]["semantic"] for row in selected.values()],
    )
    return dict(
        schema="merlin.fusion_corpus_selection.v1",
        pool_sha256=canonical_sha256(pool),
        selected=sorted(selected.values(), key=lambda row: (row["candidate_id"], row["case_id"])),
        minimum_cover=minimum,
        mandatory_count=minimum,
        over_limit=over_limit,
        limit_excess=dict(
            composition=max(0, len(selected) - 60), budget=max(0, len(selected) + workload_count - budget)
        ),
        seal_blockers=[],
        target_count=target_count,
        admitted_regimes=regimes,
        unsampled_regimes=unsampled,
        budget=budget,
        workload_count=workload_count,
        composition_count=len(selected),
        cross_checks=len(selected) - minimum,
        underfill=max(0, target_count - len(selected)),
        tie_break="candidate_id, case_id",
        not_admitted=[
            dict(candidate_id=row["candidate_id"], reasons=row["reasons"])
            for row in sorted(validation["rows"], key=lambda r: r["candidate_id"])
            if not row["admitted"] and row.get("state") not in _UNEXECUTED_STATES
        ],
        deferred=[
            dict(candidate_id=row["candidate_id"], state=row["state"])
            for row in sorted(validation["rows"], key=lambda r: r["candidate_id"])
            if not row["admitted"] and row.get("state") in _UNEXECUTED_STATES
        ],
    )


def validated_coverage(entries, validation, *, out_of_scope=False):
    counts = Counter()
    for entry in entries:
        selection = (entry.get("generalization") or {}).get("fusion_validation")
        if selection:
            counts[(selection["candidate_id"], selection["role"])] += 1
    rows = []
    sampled = set()
    for candidate in sorted(validation["rows"], key=lambda row: row["candidate_id"]):
        identity = candidate["candidate_id"]
        regime = admitted_regime(candidate["semantic"])
        roles = [role for cid, role in counts if cid == identity] or ["sampled_regime"]
        deferred = not candidate["admitted"] and candidate.get("state") in _UNEXECUTED_STATES
        for role in sorted(roles):
            count = counts[(identity, role)] if not out_of_scope else 0
            if count:
                sampled.update(_operator_regimes(candidate["semantic"]))
            rows.append(
                dict(
                    candidate_id=identity,
                    regime=regime,
                    role=role,
                    count=count,
                    state=candidate.get("state"),
                    admitted=int(candidate["admitted"]),
                    # Refusals were evaluated (statically or by execution); deferrals never ran.
                    not_admitted=int(not candidate["admitted"] and not deferred),
                    deferred=int(deferred),
                    status="out_of_scope" if out_of_scope else "deferred" if deferred else candidate["outcome"],
                    reasons=candidate["reasons"],
                    **({"reason": "workload_spec.synthesize_compositions is false"} if out_of_scope else {}),
                )
            )
    if not out_of_scope:
        for regime in sorted(
            {regime for row in validation["rows"] if row["admitted"] for regime in _operator_regimes(row["semantic"])}
            - sampled
        ):
            rows.append(dict(regime=regime, role="sampled_regime", count=0, status="not_sampled"))
    return rows
