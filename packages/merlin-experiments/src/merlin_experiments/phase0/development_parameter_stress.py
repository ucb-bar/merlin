"""Public connected integer graphs with real immutable parameter payloads.

The existing signed i8/i32 half-up/saturating boundary is unchanged. Width and
parameter ownership vary; no target schedule or model identity selects policy.
"""

from __future__ import annotations

import numpy as np

from merlin.model_check.kernel_reference import _int_matmul

from . import development_sources as D

SCHEMA = "merlin.phase0.public_parameter_compile_stress.v1"


def declaration(name, *, width, depth, shared=False):
    return dict(
        schema=SCHEMA,
        scope="public_development",
        name=name,
        pattern="parameter_projection_chain",
        geometry=dict(width=width, depth=depth, shared=shared),
        arithmetic=dict(contract=D.CONTRACT, readout_shift=3),
        max_logical_elements=2**30,
    )


def describe(d):
    if (
        set(d) != {"schema", "scope", "name", "pattern", "geometry", "arithmetic", "max_logical_elements"}
        or d["schema"] != SCHEMA
        or d["scope"] != "public_development"
        or d["pattern"] != "parameter_projection_chain"
        or not D.workload_roster._label(d["name"])
        or d["arithmetic"] != dict(contract=D.CONTRACT, readout_shift=3)
    ):
        raise ValueError("select an explicit public parameter graph and the existing exact integer contract")
    g = d["geometry"]
    if (
        set(g) != {"width", "depth", "shared"}
        or type(g["width"]) is not int
        or not 2 <= g["width"] <= 4096
        or type(g["depth"]) is not int
        or not 1 <= g["depth"] <= 8
        or type(g["shared"]) is not bool
        or type(d["max_logical_elements"]) is not int
        or not 1 <= d["max_logical_elements"] <= 2**30
    ):
        raise ValueError("parameter graph needs finite width/depth and explicit immutable sharing")
    width, depth, shared = g["width"], g["depth"], g["shared"]
    count = 1 if shared else depth
    parameter_bytes = count * width**2
    if parameter_bytes + 3 * width > d["max_logical_elements"]:
        raise ValueError("complete parameter and live-out payload exceeds source ceiling")
    return dict(
        declaration=d,
        declaration_sha256=D._digest(d),
        inputs=[dict(name="x", shape=[1, width], dtype="int8")],
        outputs=[
            dict(name="wide", shape=[1, width], dtype="int32"),
            dict(name="state", shape=[1, width], dtype="int8"),
        ],
        parameters=[dict(name=f"w{i}", shape=[width, width], dtype="int8", immutable=True) for i in range(count)],
        parameter_bytes=parameter_bytes,
        unique_parameter_tensors=count,
        parameter_uses=depth,
        logical_elements=parameter_bytes + 3 * width,
        scalar_products=depth * width**2,
        contractions=depth,
        parameter_ownership="one shared immutable Parameter" if shared else "distinct immutable Parameters",
        optimization_mechanisms=[
            "parameter_payload_packaging",
            "shared_parameter_dedup",
            "streamed_weight_blob",
            "connected_readout_lifetime",
        ],
        pending_contract=[],
    )


def render(d):
    info, g = describe(d), d["geometry"]
    width, depth = g["width"], g["depth"]
    lines = [
        '"""Public deterministic parameter compilation stress; no performance admission."""',
        "import torch",
        "",
        *D._integer_saturation_source(),
        f"# Declaration: {info['declaration_sha256']}",
        "class Program(torch.nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(info["unique_parameter_tensors"]):
        lines += [
            (
                f"        w{i} = (((torch.arange({width**2}, dtype=torch.int64) + {3 * i + 1}) % 17) - 8)"
                f".to(torch.int8).reshape({width}, {width})"
            ),
            f"        self.register_parameter('w{i}', torch.nn.Parameter(w{i}, requires_grad=False))",
        ]
    lines += ["    def forward(self, x):", "        state = x"]
    for i in range(depth):
        lines += [
            f"        wide = torch._int_mm(state, self.w{0 if g['shared'] else i})",
            "        state = _saturate_i8(torch.div(wide + 4, 8, rounding_mode='floor'))",
        ]
    lines += [
        "        return wide, state",
        "",
        "def get_model_and_inputs():",
        f"    x = (((torch.arange({width}, dtype=torch.int64)) % 17) - 8).to(torch.int8).reshape(1, {width})",
        "    return Program().eval(), (x,)",
    ]
    return "\n".join(lines) + "\n"


def inputs(d):
    width = d["geometry"]["width"]
    describe(d)
    return [((np.arange(width, dtype=np.int64) % 17) - 8).astype(np.int8).reshape(1, width)]


def parameter(d, i):
    describe(d)
    g = d["geometry"]
    if type(i) is not int or not 0 <= i < (1 if g["shared"] else g["depth"]):
        raise ValueError("parameter ordinal outside complete source ownership")
    width = g["width"]
    # Independent NumPy source reconstruction; no captured weight bytes are trusted.
    return (((np.arange(width**2, dtype=np.int64) + 3 * i + 1) % 17) - 8).astype(np.int8).reshape(width, width)


def evaluate(d, values):
    describe(d)
    if (
        len(values) != 1
        or not isinstance(values[0], np.ndarray)
        or values[0].dtype != np.dtype("int8")
        or not np.array_equal(values[0], inputs(d)[0])
    ):
        raise ValueError("parameter stress reference requires the unchanged independent public input")
    state = values[0]
    for i in range(d["geometry"]["depth"]):
        wide = _int_matmul(state, parameter(d, 0 if d["geometry"]["shared"] else i))
        if np.any(wide < -(2**31)) or np.any(wide >= 2**31):
            raise ValueError("complete source exceeds its exact i32 accumulation")
        state = np.clip(np.floor_divide(wide + 4, 8), -128, 127).astype(np.int8)
    return [wide.astype(np.int32), state]
