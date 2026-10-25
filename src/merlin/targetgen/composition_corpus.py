"""Finite composition × scheme sources from explicit hardware facts.

This is the composition axis of corpus_synth, not a second corpus generator.
Source constants are synthetic; missing target premises never acquire defaults.
"""

from __future__ import annotations

import copy
from fractions import Fraction
from functools import lru_cache


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


def _variants(chain, facts):
    variants = ["aligned", "partial", "sub_tile", "multi_block", "m_tail", "n_tail", "k_tail"]
    producer = _producer(chain)
    if producer == "operand_sum":
        return variants + ["scaled_operands"]
    if producer == "movement":
        return variants + ["standalone_pool", "window_mean", "pool_below_tile", "pool_above_tile"]
    if chain.get("extension"):
        return variants + ["two_digits", "noninteger_coefficient"]
    if chain.get("multiplicity"):
        return variants + ["different_exits"]
    variants += ["k_beyond_pass", "bias_relu", "fused_pool", "regression_vectors", "regression_second", "pred_pow2"]
    if "accumulator" in chain.get("exits", []):
        variants += ["accumulator_exit"]
    if "OS" in facts.get("dataflows", []):
        variants += ["os_constant", "os_bound_outside", "os_runtime"]
    return variants


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
            padding=[1, 1] if label == "padded_pool" else [0, 0],
            indices_live=False,
            reduction_kind="max",
        )
        case["readout_stages"].append("pool")
        case["members"].append("pool")
        if label == "standalone_pool":
            grid = [1.0]
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
    if label in {"regression_vectors", "regression_second", "pred_pow2"}:
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
    if exit_kind == "accumulator":
        case.update(readout_stages=[], multipliers=[])
    return dict(M=m, K=k, N=n, composition_instance=case)


def entries(software_spec, facts, *, excluded_shapes=None):
    """Minimal all-scheme cross per chain, then rotating pairwise shape probes."""
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
        variants = _variants(chain, facts)
        # All schemes share the minimal aligned case and one independent tail
        # probe. This covers every pair without a quadratic shape cross.
        pairs = [(scheme, label) for scheme in schemes for label in ("aligned", "partial")]
        pairs += [(schemes[index % len(schemes)], label) for index, label in enumerate(variants[2:])]
        for scheme in schemes:
            for label in variants:
                if (
                    label in {"standalone_pool", "window_mean"}
                    or label.startswith("os_")
                    or (
                        scheme["scale_value"] == "arbitrary_f32"
                        and label in {"regression_vectors", "regression_second", "pred_pow2"}
                    )
                ):
                    pairs.append((scheme, label))
        seen = set()
        for scheme, label in pairs:
            if (scheme["id"], label) in seen:
                continue
            seen.add((scheme["id"], label))
            shape = _case(chain, scheme, label, facts)
            entry = dict(
                name=f"SY_composition_{chain['id']}_{scheme['id']}_{label}",
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
            if label in {"regression_vectors", "regression_second", "pred_pow2"} and shape["composition_instance"].get(
                "regression_vectors"
            ):
                regressions = shape["composition_instance"]["regression_vectors"]
                values = sorted({v for row in regressions for v in row.get("vectors", [row.get("witness", 0)])})
                entry["M"] = len(values)
                entry["input_palette"] = dict(
                    schema="merlin.component_input_palette.v1",
                    inputs=[
                        dict(name="W0", axis="linear", values=[0], offset=0),
                    ],
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
