"""The EL1 input adapter preserves mathematics without supplying an execution plan."""
import json
from pathlib import Path

import pytest
import yaml
import numpy as np

from merlin_experiments.phase1 import neutral_program_inputs as N


def module(*lines):
    return ('module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", '
            'merlin_iface.abi_version = "0.1"} {\n' + '\n'.join(lines) + '\n}\n').encode()


def tensor(name, shape, dtype="i8", role="input"):
    return f'%{name} = merlin_iface.tensor {{name = "{name}", role = "{role}"}} : tensor<{shape}x{dtype}>'


def product(output="Y0", extra="", epilogue="[]", shape="2x4", dtype="i32"):
    return [
        '%packed = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<3x4xi8>) -> !merlin_iface.resident',
        '%acc = merlin_iface.matmul %A, %packed : (tensor<2x3xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>',
        f'%{output} = merlin_iface.commit %acc {{name = "{output}", epilogue = {epilogue}, output_dtype = "{dtype}"{extra}}} : (!merlin_iface.acc<i32>) -> tensor<{shape}x{dtype}>',
        'merlin_iface.evict %packed : (!merlin_iface.resident) -> ()',
    ]


def matmul(**kwargs):
    return module(tensor("W", "3x4", role="weight"), tensor("A", "2x3"), *product(**kwargs))


def decoded(raw):
    return json.loads(N.encode_source(raw))


def test_product_is_a_mathematical_expression_not_a_command_sequence():
    raw = matmul()
    out = N.encode_source(raw)
    d = json.loads(out)
    assert d["schema"] == "numerical_program.v1"
    assert d["expressions"] == [{"id": "v0", "operator": "matmul", "operands": ["i1", "i0"],
                                  "shape": [2, 4], "dtype": "i32"}]
    assert d["outputs"] == [{"name": "Y0", "value": "v0", "shape": [2, 4], "dtype": "i32"}]
    for forbidden in (b"merlin_iface", b"packed_rhs", b"resident", b"COMMIT", b"RES_PACK", b"gemmini", b"abi_version"):
        assert forbidden not in out
    assert N.encode_source(raw) == out


def test_bias_scale_relu_then_saturation_preserve_order_and_rounding():
    raw = module(tensor("W", "3x4", role="weight"), tensor("A", "2x3"),
                 tensor("B", "4", "i32", "bias"),
                 *product(epilogue='["bias_add", "acc_scale", "relu"]', dtype="i8",
                          extra=', bias = "B", acc_scale = 0.125 : f32'))
    expressions = decoded(raw)["expressions"]
    assert [x["operator"] for x in expressions] == ["matmul", "add_bias", "scale_round", "relu", "cast"]
    assert expressions[2]["parameters"] == {"factor": 0.125, "arithmetic_dtype": "f32", "rounding": "ties_to_even"}
    assert expressions[-1]["parameters"]["conversion"] == "saturate"


def test_column_scales_are_numeric_parameters_not_a_scale_schedule():
    out = decoded(matmul(epilogue='["acc_scale"]', dtype="i8",
                         extra=', acc_scale_by_column = [0.5 : f32, 0.25 : f32, 0.125 : f32, 1.0 : f32]'))
    parameters = out["expressions"][1]["parameters"]
    assert parameters == {"factors": [0.5, 0.25, 0.125, 1.0], "axis": -1,
                          "arithmetic_dtype": "f32", "rounding": "ties_to_even"}


def test_scaled_add_keeps_one_rounding_and_tolerance_and_discards_redundant_provenance():
    raw = module(tensor("X0", "2x4"), tensor("X1", "2x4"),
                 '%Y0 = merlin_iface.residual_add %X0, %X1 {name = "Y0", epilogue = ["relu"], lhs_scale = 0.5 : f32, rhs_scale = 0.25 : f32, bound_lsb = 1 : i64, source_scales = [0.5 : f32, 0.25 : f32, 1.0 : f32], output_dtype = "i8"} : (tensor<2x4xi8>, tensor<2x4xi8>) -> tensor<2x4xi8>')
    expressions = decoded(raw)["expressions"]
    assert [row["operator"] for row in expressions] == ["scaled_add", "relu", "cast"]
    assert expressions[0]["parameters"]["rounding"] == "ties_to_even_once_after_sum"
    assert expressions[0]["parameters"]["allowed_error_lsb"] == 1


def test_intermediate_named_outputs_are_preserved_for_chain_abi():
    raw = module(tensor("W", "3x4", role="weight"), tensor("A", "2x3"), tensor("X1", "2x4"),
                 *product(output="Yc", dtype="i8", epilogue='["acc_scale"]', extra=', acc_scale = 0.5 : f32'),
                 '%Y0 = merlin_iface.residual_add %Yc, %X1 {name = "Y0", epilogue = [], lhs_scale = 1.0 : f32, rhs_scale = 1.0 : f32, bound_lsb = 0 : i64, output_dtype = "i8"} : (tensor<2x4xi8>, tensor<2x4xi8>) -> tensor<2x4xi8>')
    d = decoded(raw)
    assert [x["name"] for x in d["outputs"]] == ["Yc", "Y0"]
    intermediate = d["outputs"][0]["value"]
    assert d["expressions"][3]["operands"][0] == intermediate


def test_asymmetric_convolution_padding_uses_top_left_bottom_right():
    raw = module(tensor("IFM", "2x5x7x3"), tensor("W", "18x4", role="weight"),
                 '%Y0 = merlin_iface.conv2d %IFM, %W {name = "Y0", kernel = [3, 2, 3, 4], stride = [2, 3], padding = [1, 2, 3, 4], dilation = [1, 2], layout = "nhwc", epilogue = [], output_dtype = "i32"} : (tensor<2x5x7x3xi8>, tensor<18x4xi8>) -> tensor<32x4xi32>')
    d = decoded(raw)
    assert d["expressions"][0]["shape"] == [32, 4]
    assert d["expressions"][0]["parameters"]["padding"] == [1, 2, 3, 4]


def test_unpadded_pool_does_not_invent_padding_identity():
    raw = module(tensor("W", "3x4", role="weight"), tensor("A", "2x3"),
                 *product(epilogue='["maxpool"]', shape="1x4", dtype="i8",
                          extra=', pool_in_dims = [1, 2], pool_size = [1, 2], pool_stride = [1, 2]'))
    pool = decoded(raw)["expressions"][1]
    assert pool["operator"] == "max_pool"
    assert pool["parameters"]["padding_value"] is None
    assert pool["shape"] == [1, 4]


def test_nonzero_pool_padding_without_value_fails_closed():
    raw = matmul(epilogue='["maxpool"]', extra=', pool_in_dims = [1, 2], pool_size = [1, 2], pool_stride = [1, 2], pool_padding = [1, 0, 0, 0]')
    with pytest.raises(N.UnsupportedProgram, match="padding"):
        N.encode_source(raw)


@pytest.mark.parametrize("raw", [
    matmul(extra=', invented_attribute = 42'),
    matmul().replace(b"merlin_iface.matmul", b"merlin_iface.unknown"),
    matmul().replace(b"  ", b"").replace(b"\n}", b"\n%x = arith.constant 7 : i32\n}"),
    matmul(dtype="f32"),
    matmul(shape="9x4"),
    matmul(epilogue='["acc_scale"]', extra=', acc_scale = nan : f32'),
    matmul(epilogue='["acc_scale"]', extra=', acc_scale = 1.0 : f32, acc_scale_by_column = [1.0]'),
    matmul(epilogue='["unknown"]'),
    matmul().replace(b"tensor<3x4xi8>", b"tensor<0x4xi8>"),
    matmul().replace(b"tensor<3x4xi8>", b"tensor<3x4xi8, #encoded>"),
])
def test_unknown_or_inconsistent_math_fails_without_returning_original_source(raw):
    with pytest.raises(N.UnsupportedProgram):
        N.encode_source(raw)


GENERIC = b'module {\nfunc.func @forward(%a: tensor<2x3xi8>) -> tensor<2x3xi8> {\n %0 = tensor.cast %a : tensor<2x3xi8> to tensor<2x3xi8>\n return %0 : tensor<2x3xi8>\n}\n}\n'


def test_generic_mathematical_mlir_preserves_every_byte():
    assert N.encode_source(GENERIC) is GENERIC
    projected = N.project_input({}, GENERIC)
    assert projected.format == "generic_math_mlir_v1"
    assert projected.payload == GENERIC


def test_author_capsule_and_standalone_sources_equal_runtime_encoder(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    source = matmul()
    (case / "input.interface.mlir").write_bytes(source)
    manifest = yaml.safe_dump({"id": "case", "interface_mlir": "input.interface.mlir", "linalg_mlir": "aux.mlir"}).encode()
    (case / "capsule.yaml").write_bytes(manifest)
    transform = N.file_transform_for(tmp_path)
    _, view_manifest = transform("case/capsule.yaml", manifest)
    name, view_source = transform("case/input.interface.mlir", source)
    assert yaml.safe_load(view_manifest)["program_input"] == Path(name).name
    assert "interface_mlir" not in yaml.safe_load(view_manifest)
    assert "linalg_mlir" not in yaml.safe_load(view_manifest)
    assert view_source == N.encode_source(source)
    assert transform("public_sources/no-capsule.interface.mlir", source) == (
        "public_sources/no-capsule.interface.program.json", N.encode_source(source))
    assert transform("source.generic.mlir", GENERIC) == ("source.generic.mlir", GENERIC)


@pytest.mark.parametrize("declared", ["../outside.mlir", "/outside.mlir"])
def test_capsule_source_escape_is_rejected(tmp_path, declared):
    transform = N.file_transform_for(tmp_path)
    manifest = yaml.safe_dump({"interface_mlir": declared}).encode()
    with pytest.raises(N.UnsupportedProgram, match="escapes"):
        transform("capsule.yaml", manifest)


def test_capsule_symlink_escape_is_rejected(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (tmp_path / "outside.mlir").write_bytes(matmul())
    (case / "link.mlir").symlink_to(tmp_path / "outside.mlir")
    with pytest.raises(N.UnsupportedProgram, match="escapes"):
        N.file_transform_for(tmp_path)("case/capsule.yaml", b"interface_mlir: link.mlir\n")


def test_public_contract_explains_numeric_semantics_without_a_dialect_or_schedule():
    contract = N.contract_text()
    assert "ties to even" in contract
    assert "[top,left,bottom,right]" in contract
    assert "Every named\noutput is observable" in contract
    for token in ("merlin_iface", "resident_pack", "COMMIT", "CONFIG_ST", "tiled_matmul", "scheduling-api"):
        assert token not in contract


def _evaluate_math(document, inputs):
    """Independent test-only mathematical evaluator: no compiler, command or target model."""
    env = {row["id"]: np.asarray(inputs[row["name"]], dtype=np.int64).reshape(row["shape"])
           for row in document["inputs"]}
    for row in document["expressions"]:
        args = [env[key] for key in row["operands"]]
        p, op = row.get("parameters", {}), row["operator"]
        if op == "matmul":
            result = np.matmul(*args)
        elif op == "add_bias":
            result = args[0] + args[1]
        elif op == "scale_round":
            factor = np.asarray(p.get("factor", p.get("factors")), dtype=np.float32)
            result = np.rint(args[0].astype(np.float32) * factor).astype(np.int64)
        elif op == "scaled_add":
            lhs = args[0].astype(np.float32) * np.float32(p["lhs_factor"])
            rhs = args[1].astype(np.float32) * np.float32(p["rhs_factor"])
            result = np.rint(lhs + rhs).astype(np.int64)
        elif op == "round_shift_right":
            shift = p["shift"]
            result = args[0] if shift <= 0 else (args[0] + (1 << (shift - 1))) >> shift
        elif op == "relu":
            result = np.maximum(args[0], 0)
        elif op == "cast":
            result = np.clip(args[0], -128, 127) if row["dtype"] == "i8" else args[0]
        elif op == "max_pool":
            h, w = p["spatial_shape"]
            ph, pw = p["window"]
            sh, sw = p["stride"]
            pt, pl, pb, pr = p["padding"]
            channels = args[0].shape[-1]
            source = args[0].reshape(-1, h, w, channels)
            oh, ow = (h + pt + pb - ph) // sh + 1, (w + pl + pr - pw) // sw + 1
            result = np.empty((len(source), oh, ow, channels), dtype=np.int64)
            for batch in range(len(source)):
                for y in range(oh):
                    for x in range(ow):
                        cells = []
                        for dy in range(ph):
                            for dx in range(pw):
                                iy, ix = y * sh - pt + dy, x * sw - pl + dx
                                if 0 <= iy < h and 0 <= ix < w:
                                    cells.append(source[batch, iy, ix])
                                else:
                                    assert p["padding_value"] is not None
                                    cells.append(np.full(channels, p["padding_value"], dtype=np.int64))
                        result[batch, y, x] = np.max(cells, axis=0)
        elif op == "conv2d":
            source, weight = args
            kh, kw = p["kernel"]
            sh, sw = p["stride"]
            dh, dw = p["dilation"]
            pt, pl, pb, pr = p["padding"]
            n, h, w, ci = source.shape
            co = weight.shape[-1]
            oh = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
            ow = (w + pl + pr - dw * (kw - 1) - 1) // sw + 1
            weight = weight.reshape(kh, kw, ci, co)
            result = np.zeros((n, oh, ow, co), dtype=np.int64)
            for batch in range(n):
                for y in range(oh):
                    for x in range(ow):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy, ix = y * sh - pt + ky * dh, x * sw - pl + kx * dw
                                if 0 <= iy < h and 0 <= ix < w:
                                    result[batch, y, x] += np.matmul(source[batch, iy, ix], weight[ky, kx])
        else:
            raise AssertionError("unmodeled test operation: " + op)
        env[row["id"]] = np.asarray(result, dtype=np.int64).reshape(row["shape"])
    return {row["name"]: env[row["value"]].reshape(row["shape"]).tolist() for row in document["outputs"]}


_NUMERIC_CASES = [
    matmul(),
    module(tensor("W", "3x4", role="weight"), tensor("A", "2x3"), tensor("B", "4", "i32", "bias"),
           *product(epilogue='["bias_add", "acc_scale", "relu"]', dtype="i8",
                    extra=', bias = "B", acc_scale = 0.5 : f32')),
    matmul(epilogue='["requant"]', dtype="i8", extra=', requant_shift = 1 : i64'),
    matmul(epilogue='["requant"]', dtype="i8", extra=', requant_shift = -1 : i64'),
    matmul(epilogue='["acc_scale"]', dtype="i8",
           extra=', acc_scale_by_column = [0.5 : f32, 0.25 : f32, 0.125 : f32, 1.0 : f32]'),
    matmul(epilogue='["maxpool"]', shape="1x4", dtype="i8",
           extra=', pool_in_dims = [1, 2], pool_size = [1, 2], pool_stride = [1, 2]'),
    matmul(epilogue='["maxpool"]', shape="4x4", dtype="i8",
           extra=', pool_in_dims = [1, 2], pool_size = [1, 2], pool_stride = [1, 1], pool_padding = [1, 1, 0, 0], pool_pad_value = 0 : i64'),
    module(tensor("X0", "2x4"), tensor("X1", "2x4"),
           '%Y0 = merlin_iface.residual_add %X0, %X1 {name = "Y0", epilogue = ["relu"], lhs_scale = 0.5 : f32, rhs_scale = 0.25 : f32, bound_lsb = 1 : i64, output_dtype = "i8"} : (tensor<2x4xi8>, tensor<2x4xi8>) -> tensor<2x4xi8>'),
    module(tensor("IFM", "2x5x7x3"), tensor("W", "18x4", role="weight"),
           '%Y0 = merlin_iface.conv2d %IFM, %W {name = "Y0", kernel = [3, 2, 3, 4], stride = [2, 3], padding = [1, 2, 3, 4], dilation = [1, 2], layout = "nhwc", epilogue = [], output_dtype = "i32"} : (tensor<2x5x7x3xi8>, tensor<18x4xi8>) -> tensor<32x4xi32>'),
]


@pytest.mark.parametrize("raw", _NUMERIC_CASES)
def test_independent_numpy_math_matches_private_integer_reference(raw):
    from merlin.runtime.reference import reference_outputs
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    document = decoded(raw)
    inputs = {}
    for row in document["inputs"]:
        size = int(np.prod(row["shape"]))
        # Includes negative odd values, large biases and saturation cases.
        values = (np.arange(size, dtype=np.int64) * 17 + 3) % 255 - 127
        if row["name"] == "B":
            values *= 100
        inputs[row["name"]] = values.reshape(row["shape"]).tolist()
    mathematical = _evaluate_math(document, inputs)
    private_reference = reference_outputs(parse_interface_mlir(raw.decode()), inputs)
    assert mathematical == private_reference
