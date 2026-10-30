"""Exact test-only MLIR translations of the supplied sixteen kernel contracts.

Runtime arrays remain function arguments. This builder neither imports TVM nor
generates goldens, downloads models, selects compiler algorithms or grants proof
authority. The handoff-to-MLIR translation is a trusted test-fixture boundary;
ordinary parser checks and native comparisons qualify it separately from Lean.
All consumers must explicitly select index_bits=64. No dimensions are reduced.
"""

from __future__ import annotations

import hashlib

from merlin.common.paths import repo_root

PINNED_COMMIT = "ebf58f2c11816370ee27f99557a6bd0b5363643f"
HANDOFF_PATH = "examples/gemmini/comparisons/tvm/KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md"
HANDOFF_SHA256 = "4a4b77899e5b43cc1c63452452a2a83a8f1a72b35287b1f90603520cbc6d9640"
DATA_PATH = "examples/gemmini/comparisons/tvm/kernel_benchmark_data.py"
DATA_SHA256 = "2ece855c1438b0aceacb58313ee7f45430f6547d5ecef90f1ba76ebc089587b2"

_GEMMS = {
    "G01": (512, 512, 512),
    "G02": (12544, 256, 64),
    "G03": (12544, 64, 256),
    "G04": (3136, 512, 128),
    "G05": (3136, 128, 512),
    "G06": (784, 256, 1024),
    "T03": (16, 5632, 2048),
    "T05": (16, 2048, 2048),
    "T07": (1, 2048, 32000),
}
_CONVS = {"C01": (56, 64, 64, 0x6000), "C02": (28, 128, 128, 0x5000), "C03": (14, 256, 256, 0x4000)}


def case_ids() -> tuple[str, ...]:
    return tuple(
        [f"G{i:02d}" for i in range(1, 7)] + [f"C{i:02d}" for i in range(1, 4)] + [f"T{i:02d}" for i in range(1, 8)]
    )


def _tensor(shape, element):
    return "tensor<" + "x".join([*(str(size) for size in shape), element]) + ">"


def _argument(name, shape, dtype="int8"):
    return {"name": name, "canonical_input_name": name, "shape": tuple(shape), "dtype": dtype}


def _specification(case):
    if case in _GEMMS:
        m, k, n = _GEMMS[case]
        return dict(
            id=case,
            kind="matmuls",
            roster=[("W", (k, n)), ("A0", (m, k))],
            products=[("A0", "W", "Y0")],
            scale=1.0 if case.startswith("G") else None,
        )
    if case in ("T01", "T02"):
        k, n = (64, 16) if case == "T01" else (16, 64)
        return dict(
            id=case,
            kind="matmuls",
            scale=None if case == "T01" else 1 / 128,
            roster=[(f"K{i}", (k, n)) for i in range(4)] + [(f"A{i}", (16, k)) for i in range(32)],
            products=[(f"A{i}", f"K{i // 8}", f"Y{i}") for i in range(32)],
        )
    if case in ("T04", "T06"):
        weights = (
            [("Wg", 5632, "Yg"), ("Wu", 5632, "Yu")]
            if case == "T04"
            else [
                ("Wq", 2048, "Yq"),
                ("Wk", 256, "Yk"),
                ("Wv", 256, "Yv"),
            ]
        )
        return dict(
            id=case,
            kind="matmuls",
            scale=1 / 128,
            roster=[(name, (2048, n)) for name, n, _ in weights] + [("A0", (16, 2048))],
            products=[("A0", name, output) for name, _, output in weights],
        )
    if case in _CONVS:
        h, ci, co, seed = _CONVS[case]
        return dict(
            id=case,
            kind="conv",
            scale=1.0,
            bias_seed=seed,
            roster=[("IFM", (4, h + 2, h + 2, ci)), ("W", (9 * ci, co))],
        )
    raise ValueError("unknown supplied kernel case: " + str(case))


class _Builder:
    def __init__(self):
        self.lines = ["%zero_i32 = arith.constant 0 : i32"]

    def emit(self, text):
        self.lines.extend(text.strip().splitlines())

    def zeros(self, name, shape):
        kind = _tensor(shape, "i32")
        self.emit(f"""
%empty_{name} = tensor.empty() : {kind}
%init_{name} = linalg.fill ins(%zero_i32 : i32) outs(%empty_{name} : {kind}) -> {kind}
""")
        return "%init_" + name

    def matmul(self, a, w, output, a_shape, w_shape):
        m, k = a_shape
        reduction, n = w_shape
        if k != reduction:
            raise ValueError("supplied contraction has inconsistent reduction dimensions")
        shape = (m, n)
        initial = self.zeros(output, shape)
        self.emit(f"""
%acc_{output} = linalg.generic {{indexing_maps = [affine_map<(m,n,k)->(m,k)>,
    affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
  iterator_types = ["parallel", "parallel", "reduction"]}}
  ins(%{a}, %{w} : {_tensor(a_shape, "i8")}, {_tensor(w_shape, "i8")})
  outs({initial} : {_tensor(shape, "i32")}) {{
  ^bb0(%a_{output}: i8, %w_{output}: i8, %sum_{output}: i32):
    %signed_a_{output} = arith.extsi %a_{output} : i8 to i32
    %signed_w_{output} = arith.extsi %w_{output} : i8 to i32
    %product_{output} = arith.muli %signed_a_{output}, %signed_w_{output} : i32
    %next_{output} = arith.addi %sum_{output}, %product_{output} : i32
    linalg.yield %next_{output} : i32
}} -> {_tensor(shape, "i32")}
""")
        return "%acc_" + output, shape

    def readout(self, source, output, shape, scale):
        if scale is None:
            return source, "int32"
        rank = len(shape)
        dimensions = ", ".join(f"d{i}" for i in range(rank))
        identity = f"affine_map<({dimensions}) -> ({dimensions})>"
        iterators = ", ".join('"parallel"' for _ in shape)
        before, after = _tensor(shape, "i32"), _tensor(shape, "i8")
        self.emit(f"""
%out_empty_{output} = tensor.empty() : {after}
%result_{output} = linalg.generic {{indexing_maps = [{identity}, {identity}], iterator_types = [{iterators}]}}
  ins({source} : {before}) outs(%out_empty_{output} : {after}) {{
  ^bb0(%value_{output}: i32, %unused_{output}: i8):
""")
        if scale == 1.0:
            self.emit(f"""
  %low_{output} = arith.constant -128 : i32
  %high_{output} = arith.constant 127 : i32
  %at_least_{output} = arith.maxsi %value_{output}, %low_{output} : i32
  %clamped_{output} = arith.minsi %at_least_{output}, %high_{output} : i32
  %narrow_{output} = arith.trunci %clamped_{output} : i32 to i8
""")
        elif scale == 1 / 128:
            self.emit(f"""
  %float_{output} = arith.sitofp %value_{output} : i32 to f32
  %scale_{output} = arith.constant 0.0078125 : f32
  %scaled_{output} = arith.mulf %float_{output}, %scale_{output} : f32
  %rounded_{output} = math.roundeven %scaled_{output} : f32
  %low_{output} = arith.constant -128.0 : f32
  %high_{output} = arith.constant 127.0 : f32
  %at_least_{output} = arith.maximumf %rounded_{output}, %low_{output} : f32
  %clamped_{output} = arith.minimumf %at_least_{output}, %high_{output} : f32
  %narrow_{output} = arith.fptosi %clamped_{output} : f32 to i8
""")
        else:
            raise ValueError("unrecognized handoff readout scale")
        self.emit(f"linalg.yield %narrow_{output} : i8\n}} -> {after}")
        return "%result_" + output, "int8"

    def convolution(self, specification):
        shapes = dict(specification["roster"])
        batch, height, width, ci = shapes["IFM"]
        co = shapes["W"][1]
        ho, wo = height - 2, width - 2
        shape = (batch, ho, wo, co)
        accumulator = _tensor(shape, "i32")
        weights = _tensor((3, 3, ci, co), "i8")
        self.emit(f"""
%weights4 = tensor.expand_shape %W [[0, 1, 2], [3]] output_shape [3, 3, {ci}, {co}]
  : {_tensor(shapes["W"], "i8")} into {weights}
%bias_empty = tensor.empty() : {accumulator}
%bias_init = linalg.generic {{indexing_maps = [affine_map<(b,y,x,o)->(o)>, affine_map<(b,y,x,o)->(b,y,x,o)>],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
  ins(%B : {_tensor((co,), "i32")}) outs(%bias_empty : {accumulator}) {{
  ^bb0(%bias: i32, %old_bias: i32):
    linalg.yield %bias : i32
}} -> {accumulator}
%conv = linalg.generic {{indexing_maps = [
    affine_map<(b,y,x,o,r,s,c)->(b,y+r,x+s,c)>,
    affine_map<(b,y,x,o,r,s,c)->(r,s,c,o)>,
    affine_map<(b,y,x,o,r,s,c)->(b,y,x,o)>],
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}}
  ins(%IFM, %weights4 : {_tensor(shapes["IFM"], "i8")}, {weights}) outs(%bias_init : {accumulator}) {{
  ^bb0(%input_value: i8, %weight_value: i8, %sum: i32):
    %signed_input = arith.extsi %input_value : i8 to i32
    %signed_weight = arith.extsi %weight_value : i8 to i32
    %product = arith.muli %signed_input, %signed_weight : i32
    %next = arith.addi %sum, %product : i32
    linalg.yield %next : i32
}} -> {accumulator}
""")
        result, dtype = self.readout("%conv", "Y0", shape, 1.0)
        flat = (batch * ho * wo, co)
        self.emit(
            f"%Y0 = tensor.collapse_shape {result} [[0, 1, 2], [3]] : {_tensor(shape, 'i8')} into {_tensor(flat, 'i8')}"
        )
        return [("%Y0", {"name": "Y0", "shape": flat, "dtype": dtype})]


def fixture(case_id: str) -> dict:
    """Return exact runtime signature, source and source-bound fixture metadata."""
    specification = _specification(case_id)
    arguments = [_argument(name, shape) for name, shape in specification["roster"]]
    if specification["kind"] == "conv":
        arguments.append(_argument("B", (dict(specification["roster"])["W"][1],), "int32"))
    builder = _Builder()
    if specification["kind"] == "conv":
        returned = builder.convolution(specification)
    else:
        returned = []
        shapes = dict(specification["roster"])
        for a, w, name in specification["products"]:
            acc, shape = builder.matmul(a, w, name, shapes[a], shapes[w])
            result, dtype = builder.readout(acc, name, shape, specification["scale"])
            returned.append((result, {"name": name, "shape": shape, "dtype": dtype}))
    elements = {"int8": "i8", "int32": "i32"}
    parameters = ", ".join(f"%{row['name']}: {_tensor(row['shape'], elements[row['dtype']])}" for row in arguments)
    result_types = ", ".join(_tensor(row["shape"], elements[row["dtype"]]) for _, row in returned)
    return_signature = result_types if len(returned) == 1 else "(" + result_types + ")"
    source = "// Supplied kernel contract; selected consumer index_bits=64.\nmodule {\n"
    source += f"func.func @forward({parameters}) -> {return_signature} {{\n"
    source += "\n".join("  " + line for line in builder.lines)
    source += "\n  func.return " + ", ".join(value for value, _ in returned) + " : " + result_types + "\n}\n}\n"
    translation = repo_root() / "merlin/tests/ir/kernel_benchmark_handoff_inputs.py"
    return {
        "case_id": case_id,
        "entry": "forward",
        "index_bits": 64,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "arguments": tuple(arguments),
        "outputs": tuple(row for _, row in returned),
        "case_spec": specification,
        "input_manifest_mapping": tuple(
            dict(argument=index, canonical_input_name=row["canonical_input_name"], fixture_array=f"input_{index}")
            for index, row in enumerate(arguments)
        ),
        "provenance": {
            "repository": "https://github.com/ucb-bar/merlin",
            "commit": PINNED_COMMIT,
            "handoff": {"path": HANDOFF_PATH, "sha256": HANDOFF_SHA256},
            "generator": {"path": DATA_PATH, "sha256": DATA_SHA256},
            "translation": {"path": str(translation), "sha256": hashlib.sha256(translation.read_bytes()).hexdigest()},
            "translation_boundary": (
                "Trusted test-only handoff-to-MLIR transcription; no TVM/IREE or native equivalence proof"
            ),
            "array_policy": "Runtime signed-nibble arguments from pinned generator; references never embedded in IR",
        },
    }
