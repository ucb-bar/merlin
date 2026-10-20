"""Hardware-path-derived column-readout demands, independent of private model shapes.

Gains and small workload extents are synthetic. The gain-block width and numeric
formats come only from the selected output path. These describe tests for a
compiler, never a target implementation or a successful execution verdict.
"""

from __future__ import annotations

import copy
import math
import struct

from merlin.common import quant_formats as QF

THREE_BLOCKS_V1 = "independent_three_blocks_v1"
DISTINCT_SCALES_V2 = "independent_distinct_scales_v2"
_DISTINCT_COLUMNS_LIMIT = 4096  # Workload/source budget, never a hardware capacity.


def _distinct_scales(columns):
    """Exact positive binary-f32 gains within this profile's finite source budget."""
    if type(columns) is not int or not 1 <= columns <= _DISTINCT_COLUMNS_LIMIT:
        raise ValueError("distinct-scale profile exceeds its finite column budget")
    denominator = 1 << (columns - 1).bit_length()
    gains = [(index + 1) / denominator for index in range(columns)]
    if len(set(gains)) != columns or any(
        not math.isfinite(value) or value <= 0 or struct.unpack("<f", struct.pack("<f", value))[0] != value
        for value in gains
    ):
        raise ValueError("distinct-scale profile requires unique exact positive f32 gains")
    return gains


def with_profile(spec, profile=None):
    """Opt in to one wider public demand without changing historical requirements.

    The extra source exercises three selected scale blocks with independent
    logical-column gains. Its geometry establishes no emitted store count,
    timing support, placement or numerical execution verdict.
    """
    if profile is None:
        return spec
    if profile not in (THREE_BLOCKS_V1, DISTINCT_SCALES_V2):
        raise ValueError("unknown column_readout_profile")
    case = "three_blocks_v1" if profile == THREE_BLOCKS_V1 else "distinct_scales_v2"
    required = (spec.get("groups") or {}).get("required") or []
    candidates = [
        row
        for row in required
        if row.get("op") == "matmul"
        and (row.get("column_readout") or {}).get("case") == "aligned"
        and row["column_readout"].get("logical_granularity") == "per_channel"
        and "bias_add" in row.get("epilogue", ())
    ]
    if not candidates:
        raise ValueError("column_readout_profile requires an admitted independent-column matmul path with bias preload")
    out = copy.deepcopy(spec)
    rows = out["groups"]["required"]
    for original in candidates:
        row = copy.deepcopy(original)
        width = row["column_readout"]["columns_per_scale"]
        if type(width) is not int or width < 1:
            raise ValueError("column_readout_profile requires a positive selected scale width")
        if profile == DISTINCT_SCALES_V2:
            _distinct_scales(3 * width)
        row["column_readout"].update(columns=3 * width, case=case, logical_granularity="per_channel")
        row["signature"] = f"matmul|independent_column_scale|{row['dtype']}|{width}|{case}"
        if not any(existing["signature"] == row["signature"] for existing in rows):
            rows.append(row)
    out["groups"]["column_readout_profile"] = profile
    return out


def _same_format(value, wanted):
    return bool(value and QF.has(value) and QF.get(value).name == QF.get(wanted).name)


def unavailable(path):
    """Why a declared column scale cannot yet be stated by these capsule builders."""
    if not path:
        return None
    readout = {row["kind"]: row for row in path.get("readout") or ()}
    scale = readout.get("scale") or {}
    if scale.get("granularity") != "per_command":
        return None
    if len(readout) != len(path.get("readout") or ()) or not (path.get("exits") or {}).get("requant"):
        return "the selected path has repeated readout stages or no declared narrowing exit"
    width = scale.get("columns_per_scale")
    if type(width) is not int or width < 1:
        return "the selected path does not derive a positive columns_per_scale"
    if not _same_format((path.get("accumulator") or {}).get("dtype"), "int32"):
        return "column-readout capsules require the declared integer accumulator format"
    for stage, field in ((readout.get("convert") or {}, "to"), (scale, "dtype")):
        if not _same_format(stage.get(field), "float32") or stage.get("rounding") != "half_even":
            return "column-readout capsules cannot state this conversion/scale format and rounding"
    requant = readout.get("requant") or {}
    if (not _same_format(requant.get("to"), "int8") or requant.get("rounding") != "half_even"
            or requant.get("clamp") != [-128, 127]):
        return "column-readout capsules cannot state this narrowing format, rounding or clamp"
    return None


def unstated(path):
    """Output-path features still missing a capsule statement."""
    if not path:
        return {}
    out = {}
    if any(row.get("kind") == "accumulate" for row in path.get("accumulator_stages") or ()):
        out["accumulate"] = "the command-buffer stage vocabulary has no accumulated residual operand"
    why = unavailable(path)
    if why:
        out["column_group_scale"] = why
    return out


def requirements(path, dtypes, carriers=("matmul",), *, tails=True):
    """Aligned and partial gain-block cases, from declared facts and public carriers."""
    if not path or unavailable(path):
        return []
    scale = next((row for row in path.get("readout") or () if row["kind"] == "scale"), {})
    if scale.get("granularity") != "per_command":
        return []
    width = scale["columns_per_scale"]
    epilogue = []
    if any(row["kind"] == "preload" for row in path.get("accumulator_stages") or ()):
        epilogue.append("bias_add")
    epilogue.append("acc_scale")
    if any(row["kind"] == "activation" and row.get("function") == "relu" for row in path.get("readout") or ()):
        epilogue.append("relu")
    rows = []
    for dtype in sorted(set(dtypes)):
        if not _same_format(dtype, "int8"):
            continue
        for op in sorted(set(carriers) & {"matmul", "conv2d"}):
            for tail in ((False, True) if tails else (False,)):
                case = "tail" if tail else "aligned"
                rows.append({
                    "signature": f"{op}|column_group_scale|{dtype}|{width}|{case}",
                    "op": op, "dtype": dtype, "epilogue": list(epilogue), "groups": 0, "observed_in": [],
                    "evidenced_by": ["derived_output_path"],
                    "column_readout": {"columns_per_scale": width, "columns": 2 * width + int(tail), "case": case},
                })
    # Preserve the existing block-uniform demands and add independent logical
    # columns. The candidate owns any physical partitioning implementation.
    if width > 1:
        for row in list(rows):
            per_column = copy.deepcopy(row)
            per_column["signature"] = row["signature"].replace("|column_group_scale|", "|independent_column_scale|")
            per_column["column_readout"]["logical_granularity"] = "per_channel"
            rows.append(per_column)
    return rows


def requirements_for_oracle(path, oracle, groups):
    """Use declared contraction formats and carriers observed in the public group inventory."""
    contraction = (getattr(oracle, "cap_map", {}) or {}).get("contraction")
    return requirements(path, getattr(contraction, "dtypes", ()), {"matmul", *(row["op"] for row in groups)},
                        tails=getattr(contraction, "arbitrary_mnk", False))


def entry(requirement):
    """A generic accelerator workload for this requirement, without captured values."""
    row = requirement["column_readout"]
    width, columns = row["columns_per_scale"], row["columns"]
    independent = row.get("logical_granularity") == "per_channel"
    # Deliberate workload constants, not hardware facts or model parameters.
    gains = (0.5, 0.125, 0.75)
    by_column = (
        _distinct_scales(columns) if row["case"] == "distinct_scales_v2"
        else [gains[(i if independent else i // width) % len(gains)] for i in range(columns)]
    )
    result = {
        "name": f"SY_{'independent_column' if independent else 'column'}_readout_"
                f"{requirement['op']}_{requirement['dtype']}_{row['case']}",
        "cat": "layers", "kind": "layer", "label": "public", "op": requirement["op"],
        "operand_dtype": requirement["dtype"], "epilogue": list(requirement["epilogue"]),
        "output_dtype": "i8",
        "M_tiles": 1, "K_tiles": 1, "N": columns,
        "ci": width, "Himg": 2, "Wimg": 3, "kh": 1, "kw": 1,
        **({"stride": [1, 1], "padding": [0, 0, 0, 0], "dilation": [1, 1]}
           if requirement["op"] == "conv2d" else {}),
        "acc_scale_by_column": by_column,
        "source_role": "derived_sweep",
        "source_reference": (
            f"selected physical readout shares one f32 gain across {width} lanes; "
            "independent logical-column source gains require a compiler implementation"
            if independent else f"selected output path permits one f32 gain per {width} columns; synthetic gain changes"
        ),
        "lanes": {"require": ["on_mesh"]},
        "generalization": {"generalization_axis": "column_readout", "must_accelerate": True, "eligible": True},
    }
    if row["case"] in ("three_blocks_v1", "distinct_scales_v2"):
        # Synthetic exact values, independent of model data and any measured
        # schedule: mixed signs, positive ties, ReLU and saturating narrowing.
        result["stimulus_range"] = [-128, 127]
        result["input_palette"] = {
            "schema": "merlin.component_input_palette.v1",
            "inputs": [
                {"name": "A0", "axis": 0, "values": [1, -1], "offset": 0},
                {"name": "W", "axis": 1, "values": [8, -8, 0, 1, -1, 2, -2, 127, -128][:columns], "offset": 0},
                {
                    "name": "B",
                    "axis": "linear",
                    "values": [0, 1, -1, 3, -3, 255, -255, 1023, -1023][:columns],
                    "offset": 0,
                },
            ],
        }
    return result


def matches(cap, requirement):
    """Whether the materialized capsule actually demands this column-readout case."""
    from merlin.runtime.commandbuffer import declared_acc_scale_by_column

    operation = cap.get("operation") or {}
    attrs = operation.get("attributes") or {}
    if operation.get("op") != requirement["op"] or attrs.get("epilogue") != requirement["epilogue"]:
        return False
    columns, width = (requirement["column_readout"][key] for key in ("columns", "columns_per_scale"))
    weight = next((row for row in cap.get("inputs") or () if row.get("role") == "weight"), {})
    if not _same_format(weight.get("dtype"), requirement["dtype"]) or (weight.get("shape") or [None])[-1] != columns:
        return False
    if "on_mesh" not in (cap.get("lanes") or {}).get("require", ()):
        return False
    try:
        gains = declared_acc_scale_by_column(attrs, columns, op="column-readout coverage")
    except (ValueError, TypeError, OverflowError):
        return False
    if not (all(math.isfinite(v) and v > 0 for v in gains) and len(set(gains)) > 1):
        return False
    uniform = all(len(set(gains[i:i + width])) == 1 for i in range(0, columns, width))
    return not uniform if requirement["column_readout"].get("logical_granularity") == "per_channel" else uniform
