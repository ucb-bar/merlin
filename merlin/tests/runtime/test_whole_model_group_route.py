"""A whole model reaches the device GROUP BY GROUP, and every group is accounted for by name.

The gap these cover. A whole-model FSM-free ResNet-50 ran on FireSim at 43.9 M cycles, and it was
produced by hand-written schedules -- the compiler an agent optimises has never emitted that program,
because it compiles CAPSULES. Three things kept it from doing so, and all three are checked here:

* the ``Placement`` was computed AFTER the build had already run, so the decision could not reach the
  emission however it came out (``test_the_route_is_decided_before_the_build``);
* the build was never given a ``device=``, so the offload path was unreachable from a compile
  (``test_the_build_is_handed_the_routing_the_placement_produced``);
* the device side synthesized a bare ``M x K x N`` per contraction, which drops the bias, the
  requantize, the activation and the pooling a captured layer carries
  (``test_a_device_build_given_stated_programs_refuses_an_unstated_symbol``, and the epilogue
  assertions below).

And the honesty property, which is the one worth having a test for at all: a group the package
cannot emit is NAMED and refused. Never dropped, never quietly left on the host with the groups
nobody could take -- those are different facts, and a census that merged them would let a model
running a third of its layers on the device read exactly like one running all of them.
"""

from __future__ import annotations

import pytest
import selected_driver

from merlin.common import mlir_query as mq
from merlin.llvmlower import group_offload as GO

pytestmark = pytest.mark.target("gemmini")

#: A target that closes a quantized linear layer's readout onto its unit. The route under test is
#: target-agnostic -- the device is a parameter throughout -- but a test of "did two layers become
#: two device calls" needs a target whose oracle actually admits them, so one real target is named
#: here and nowhere in the library code it exercises.
_TARGET = "gemmini"

_ID = "affine_map<(d0, d1) -> (d0, d1)>"
_COL = "affine_map<(d0, d1) -> (d1)>"
_ROW = "affine_map<(d0, d1) -> (d0)>"


def _layer(tag: str, activation: str, m: int, k: int, n: int, *, bias: str, bias_map: str = _COL) -> list[str]:
    """dq(x), dq(w) -> matmul -> bias -> relu -> quantize, in the fake-quantized form a capture has."""
    t = tag
    return [
        f'    %wd{t} = "quant_ext.dequantize_per_tensor"(%w{t}, %s, %z) <{{quant_min = -127 : i64, '
        f"quant_max = 127 : i64}}> : (tensor<{k}x{n}xi8>, tensor<f32>, tensor<i64>) -> tensor<{k}x{n}xf32>",
        f'    %xd{t} = "quant_ext.dequantize_per_tensor"({activation}, %s, %z) <{{quant_min = -128 : i64, '
        f"quant_max = 127 : i64}}> : (tensor<{m}x{k}xi8>, tensor<f32>, tensor<i64>) -> tensor<{m}x{k}xf32>",
        f"    %e0{t} = tensor.empty() : tensor<{m}x{n}xf32>",
        f"    %f{t} = linalg.fill ins(%c0 : f32) outs(%e0{t} : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>",
        f'    %mm{t} = linalg.matmul {{prov.region_id = "matmul_{t}", prov.op = "matmul", '
        f'prov.family = "contraction", prov.orig_dtype = "int8"}} '
        f"ins(%xd{t}, %wd{t} : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) "
        f"outs(%f{t} : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>",
        f"    %e1{t} = tensor.empty() : tensor<{m}x{n}xf32>",
        f"    %ba{t} = linalg.generic {{indexing_maps = [{_ID}, {bias_map}, {_ID}], "
        f'iterator_types = ["parallel", "parallel"]}} ins(%mm{t}, {bias} : tensor<{m}x{n}xf32>, '
        f"tensor<{n if bias_map == _COL else m}xf32>) outs(%e1{t} : tensor<{m}x{n}xf32>) {{",
        "    ^bb0(%p: f32, %q0: f32, %o: f32):",
        "      %r = arith.addf %p, %q0 : f32",
        "      linalg.yield %r : f32",
        f"    }} -> tensor<{m}x{n}xf32>",
        f"    %e2{t} = tensor.empty() : tensor<{m}x{n}xf32>",
        f"    %relu{t} = linalg.generic {{indexing_maps = [{_ID}, {_ID}], "
        f'iterator_types = ["parallel", "parallel"]}} ins(%ba{t} : tensor<{m}x{n}xf32>) '
        f"outs(%e2{t} : tensor<{m}x{n}xf32>) {{",
        "    ^bb0(%p: f32, %o: f32):",
        "      %zero = arith.constant 0.000000e+00 : f32",
        "      %r = arith.maximumf %p, %zero : f32",
        "      linalg.yield %r : f32",
        f"    }} -> tensor<{m}x{n}xf32>",
        f'    %q{t} = "quant_ext.quantize_per_tensor"(%relu{t}, %s, %z) <{{quant_min = -128 : i64, '
        f'quant_max = 127 : i64, output_dtype = "int8"}}> : (tensor<{m}x{n}xf32>, tensor<f32>, '
        f"tensor<i64>) -> tensor<{m}x{n}xi8>",
    ]


def _two_layers(*, second_bias_map: str = _COL) -> str:
    """Two chained quantized linear layers: the smallest module that can tell one call from two."""
    rows = second_bias_map == _ROW
    return "\n".join(
        [
            "builtin.module {",
            "  func.func @forward(%x: tensor<4x8xi8>, %wa: tensor<8x16xi8>, %biasa: tensor<16xf32>, "
            "%wb: tensor<16x32xi8>, %biasb: tensor<32xf32>, %biasr: tensor<4xf32>) -> tensor<4x32xi8> {",
            "    %s = arith.constant dense<5.000000e-01> : tensor<f32>",
            "    %z = arith.constant dense<0> : tensor<i64>",
            "    %c0 = arith.constant 0.000000e+00 : f32",
            *_layer("a", "%x", 4, 8, 16, bias="%biasa"),
            *_layer("b", "%qa", 4, 16, 32, bias="%biasr" if rows else "%biasb", bias_map=second_bias_map),
            "    func.return %qb : tensor<4x32xi8>",
            "  }",
            "}",
        ]
    )


#: Which model arguments a weights manifest would call stored: the two weights and the biases.
_WEIGHT_ARGS = {1, 2, 3, 4, 5}


def _plan(text: str) -> GO.GroupOffload:
    return GO.plan(mq.parse(text), _TARGET, weight_args=_WEIGHT_ARGS, model="two_layer")


# ------------------------------------------------------------------ one call per closed group


def test_a_closed_group_becomes_one_device_call_carrying_its_whole_program() -> None:
    offload = _plan(_two_layers())
    census = offload.census()
    assert census["mechanism"] == {GO.ON_DEVICE: 2, GO.ON_HOST: 0, GO.REFUSED: 0}
    assert census["uniform"] and census["accounted"] == census["groups"] == 2
    # One call per group, and the calls are DISTINCT symbols: two layers of the same shape are two
    # calls in one program, and a route that collapsed them would run one layer's kernel for both.
    assert len({call.symbol for call in offload.calls}) == 2
    for call in offload.calls:
        # THE READOUT IS IN THE CALL. This is the whole difference from the contraction-granular
        # route: a bare `M x K x N` would carry none of these, and the bias, the requantize and the
        # activation would silently stay on the host.
        assert call.epilogue == ["bias_add", "acc_scale", "relu"]
        assert call.entry["op"] == "matmul"
        assert call.entry["acc_scale"] == pytest.approx(0.5 * 0.5 / 0.5)


def test_the_entries_a_group_route_hands_a_backend_commit_the_integer_type_the_layer_does() -> None:
    """The capsule each entry builds commits i8 -- the type the capture's quantize produces.

    Read off the BUILT capsule rather than the entry, because the commit type is resolved from the
    epilogue against the target's declared readout, and an entry that merely named a dtype would be
    asserting the test's opinion instead of the generator's.
    """
    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.targetgen import corpus_spec as CS

    binding = _mesh_tile_binding(_TARGET, "int8", "int32")
    for call in _plan(_two_layers()).calls:
        capsule, iface = CS.build(dict(call.entry), binding)
        attributes = (capsule.get("operation") or {}).get("attributes") or {}
        assert attributes.get("epilogue") == ["bias_add", "acc_scale", "relu"]
        assert str(attributes.get("output_dtype")) == "i8"
        assert 'output_dtype = "i8"' in iface


# ------------------------------------------------------------------ nothing leaves unnamed


def test_a_group_that_cannot_be_stated_is_refused_by_name_and_not_left_to_the_host() -> None:
    """A bias along the activations' axis cannot be applied by a per-output readout.

    The group is still CLOSED -- the planner admitted it -- so this is precisely the case a silent
    route would mishandle: dropping it would shrink the model, and filing it with the host regions
    would make a gap in the statement look like a placement decision.
    """
    offload = _plan(_two_layers(second_bias_map=_ROW))
    census = offload.census()
    assert census["mechanism"][GO.REFUSED] == 1
    assert census["accounted"] == census["groups"]
    (refusal,) = offload.refused
    assert refusal.name.endswith("group1") and refusal.name in {row["name"] for row in census["refusals"]}
    assert "bias" in refusal.why
    # ...and it is in NEITHER of the other two buckets.
    assert refusal.index not in {call.index for call in offload.calls}
    assert refusal.index not in {region.index for region in offload.host}


def test_a_group_in_none_of_the_three_buckets_is_a_defect_and_raises() -> None:
    offload = _plan(_two_layers())
    GO.require_every_group_accounted(offload)
    import dataclasses

    dropped = dataclasses.replace(offload, calls=offload.calls[:1])
    with pytest.raises(GO.SilentGroupError, match="only 1 are accounted"):
        GO.require_every_group_accounted(dropped)


def test_a_package_that_declines_a_group_is_recorded_by_that_groups_name(tmp_path, monkeypatch) -> None:
    """And an exit status of zero is not taken for an answer.

    Measured on a real submission: the entrypoint that emits the target artifact returns 0 and
    prints an EMPTY entry function for a layer it refused, so counting exit statuses scored 71 of
    71 on a capture where the package actually emits 54. The verdict is the command buffer.
    """
    import json
    import subprocess

    from merlin.llvmlower import group_offload

    offload = _plan(_two_layers())
    refused_group = offload.calls[1].entry["name"]

    def _fake_run(_package, _name, _source, output_json=None, timeout=0):
        buffer = {"abi_version": "0.1", "target": _TARGET, "commands": [{"opcode": "COMMIT"}]}
        if refused_group in str(_source):
            buffer = {**buffer, "commands": [], "declined": {"reason": "no lowering for this operation"}}
        assert output_json is not None
        output_json.write_text(json.dumps(buffer), encoding="utf-8")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(group_offload, "load_package", lambda _d: object(), raising=False)
    monkeypatch.setattr("merlin.targetgen.oot_runner.load_package", lambda *a, **k: object())
    monkeypatch.setattr("merlin.targetgen.oot_runner.run_entrypoint", _fake_run)

    from merlin.compile.mesh import _mesh_tile_binding

    report = GO.emit_group_artifacts(
        offload,
        package_dir=tmp_path,
        binding=_mesh_tile_binding(_TARGET, "int8", "int32"),
        workdir=tmp_path / "emit",
    )
    assert [row["name"] for row in report["emitted"]] == [offload.calls[0].entry["name"]]
    assert report["declined_by_package"] == {refused_group: "no lowering for this operation"}


# ------------------------------------------------------------------ the route reaches the build


def _compiled(capture=None, order: list | None = None, *, offload: bool = False, seen: dict | None = None) -> dict:
    """``compile_model`` with the build stubbed out, so the test is about the route, not a toolchain."""
    from merlin import compile_cli

    real_route = compile_cli._route_before_build

    def _route(*args, **kwargs):
        if order is not None:
            order.append("route")
        return real_route(*args, **kwargs)

    def _build(*_args, **kwargs):
        if order is not None:
            order.append("build")
            order.append("device" if "device" in kwargs else "no-device-parameter")
        if seen is not None:
            seen["device"] = kwargs.get("device", "<absent>")
        return {"status": "compiled"}

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(compile_cli, "_route_before_build", _route)
        monkey.setattr(compile_cli, "compile_rvv", _build)
        return compile_cli.compile_model(
            "two_layer",
            "int8",
            target=_TARGET,
            run="none",
            verify=False,
            package=None,
            auto_capture=False,
            timeout=5,
            linalg_mlir=_two_layers(),
            capture_bundle=capture,
            offload=offload,
        )
    finally:
        monkey.undo()


def _bundle_with_manifest(tmp_path):
    """A capture whose weights manifest names the stored arguments, as a real bundle carries."""
    import json

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text(_two_layers(), encoding="utf-8")
    (bundle / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": {"kind": "input"},
                **{str(i): {"kind": "weight"} for i in sorted(_WEIGHT_ARGS)},
            }
        ),
        encoding="utf-8",
    )
    return bundle


def test_the_route_is_decided_before_the_build(tmp_path) -> None:
    """Placement is an INPUT to a build or it is a commentary on one.

    ``compile_model`` used to compute its ``Placement`` after ``compile_rvv`` had already lowered,
    built and run the model, so a routing that put every contraction on an accelerator was recorded
    beside an image that ran all of them on the host, and the two read as one result.
    """
    order: list[str] = []
    out = _compiled(capture=_bundle_with_manifest(tmp_path), order=order)
    assert order[:3] == ["route", "build", "device"], "the placement must be decided before the build, and handed to it"
    # ...and the compile reports the group-by-group census, ONE ENTRY PER CLOSED GROUP, each
    # carrying the readout its layer carries rather than a bare contraction.
    census = out["device_program"]
    assert census["mechanism"] == {GO.ON_DEVICE: 2, GO.ON_HOST: 0, GO.REFUSED: 0}
    assert [row["epilogue"] for row in census["calls"]] == [["bias_add", "acc_scale", "relu"]] * 2
    assert len({row["symbol"] for row in census["calls"]}) == 2


def test_a_compile_that_cannot_tell_the_stored_operand_refuses_that_group_by_name() -> None:
    """Handed a module as text there is no weights manifest to read, so a first layer whose two
    operands are both model arguments cannot be stated -- and it says so, per group, instead of
    guessing which side holds the weight."""
    census = _compiled()["device_program"]
    assert census["mechanism"] == {GO.ON_DEVICE: 1, GO.ON_HOST: 0, GO.REFUSED: 1}
    (refusal,) = census["refusals"]
    assert refusal["name"].endswith("group0") and "weights manifest" in refusal["why"]


def test_the_build_is_handed_the_routing_the_placement_produced() -> None:
    """``device=`` exists on the routine a compile actually calls, and carries the decision.

    It existed on the bare-metal path and on ``prepare_for_lowering`` while the entrypoint between
    them had no parameter for it at all, so no compile could ask for an offload however the
    placement decided.
    """
    import inspect

    from merlin import compile_cli
    from merlin.runtime.backends import zephyr_model

    assert "device" in inspect.signature(compile_cli.compile_rvv).parameters
    assert "device" in inspect.signature(zephyr_model.build_app).parameters
    # The parameter is not decorative: the build passes it on to the preparation pass that performs
    # the rewrite. A parameter accepted and dropped is worse than an absent one.
    source = inspect.getsource(zephyr_model.build_app)
    assert "device=device" in source


def test_deriving_a_routing_is_not_the_same_act_as_building_against_one(tmp_path) -> None:
    """The build is given the routing when the caller asks for it, and nothing when it does not.

    Every caller that already passes a backend package would otherwise have started emitting device
    calls the moment the derivation became possible -- a change to what those builds compute, made by
    a function nobody asked to change them. The record says what WOULD move either way.
    """
    bundle = _bundle_with_manifest(tmp_path)
    seen: dict = {}
    _compiled(capture=bundle, seen=seen)
    assert seen["device"] is None, "an unrequested offload must not reach the build"

    seen_on: dict = {}
    out = _compiled(capture=bundle, offload=True, seen=seen_on)
    # No backend package is named here, so there is nothing to build against and the routing is
    # honestly unavailable -- but the census still reports the two groups that would have moved.
    assert seen_on["device"] is None
    assert out["device_routing"]["status"] == "unavailable"
    assert out["device_program"]["mechanism"][GO.ON_DEVICE] == 2


def test_a_compile_that_cannot_derive_a_routing_says_why_rather_than_offloading_nothing() -> None:
    """An absent routing and a routing that decided against offloading are different facts.

    No backend package was named here, so there is nothing to build the device side with -- and the
    record says exactly that instead of leaving the key absent.
    """
    routing = _compiled()["device_routing"]
    assert routing["status"] == "unavailable" and "package" in routing["why"]


# ------------------------------------------------------------------ the device build keeps the readout


def test_a_device_build_given_stated_programs_refuses_an_unstated_symbol(tmp_path) -> None:
    """Fail closed. A caller routing stated programs that hands over an unstated symbol has a gap in
    its statement, and substituting a bare contraction for it would drop that layer's whole readout
    while the build still reported a kernel."""
    from merlin.llvmlower.device_build import build_device_objects

    built = build_device_objects(
        _TARGET,
        {"d0": (16, 16, 32)},
        {"d0": ("i8", "i8", "i32")},
        package_dir=tmp_path / "nonexistent",
        workdir=tmp_path / "work",
        operand_dtype="int8",
        accum_dtype="i32",
        entries={},
    )
    assert not built.ok
    reasons = dict(built.skipped)
    assert "d0" in reasons and "stated group program" in reasons["d0"]


def test_the_two_ways_a_kernel_can_be_built_are_named_apart() -> None:
    """A mixed archive must not read as one mechanism: the two compute different functions."""
    from merlin.llvmlower import device_build

    assert device_build.FROM_GROUP != device_build.FROM_EXTENTS
    assert "built_from" in device_build.DeviceBuild.__dataclass_fields__


# ------------------------------------------------- the rewrite: the group becomes the device call


def _capture(directory, shapes: dict) -> object:
    """A capture beside which the route finds a weights manifest and its tensors (the prepack's input).

    ``shapes`` maps each stored model argument, in argument order from 1, to its shape: 2-D int8
    weights and 1-D float biases (multiples of the folded scale, so the fold is exact)."""
    import json
    import struct

    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    header, payload, manifest = {}, b"", {"0": {"kind": "input"}}
    for index, (name, shape) in enumerate(shapes.items(), start=1):
        if len(shape) == 2:
            array, spelled = rng.integers(-8, 8, shape).astype(np.int8), "I8"
        else:
            array, spelled = (rng.integers(-8, 8, shape) * 0.25).astype(np.float32), "F32"
        raw = array.tobytes()
        header[name] = {"dtype": spelled, "shape": list(shape), "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload += raw
        manifest[str(index)] = {"kind": "weight", "weight": name}
    blob = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + payload)
    (directory / "model.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory / "model.mlir"


_TWO_LAYER_STORED = {"wa": (8, 16), "biasa": (16,), "wb": (16, 32), "biasb": (32,), "biasr": (4,)}


def _rewritten(text: str, **kw):
    """The module after the whole-program rewrite, with what it routed and what it refused."""
    from merlin.common.ir_lock import IR_LOCK
    from merlin.llvmlower import device_offload as DO

    with IR_LOCK:
        module = mq.parse(text)
        rewrite = DO.rewrite_groups_to_device(
            module, _TARGET, select=lambda _s: True, weight_args=_WEIGHT_ARGS, model="two_layer", **kw
        )
        module.verify()  # a call whose declaration or types disagreed would fail here
        from merlin.xdsl_dialects._common import text as to_text

        return rewrite, to_text(module)


def test_each_closed_group_becomes_exactly_one_device_call_and_the_layer_leaves_the_host(tmp_path) -> None:
    """The route the phase-2 loop needs: the model's own text no longer contains the layers.

    Counting the calls is not enough on its own -- a rewrite that added calls and left the linalg in
    place would pass that -- so the contraction, its bias, its activation and its requantize are all
    asserted GONE from the driver.
    """
    rewrite, text = _rewritten(_two_layers(), capture=_capture(tmp_path, _TWO_LAYER_STORED))
    assert rewrite.moved == 2, rewrite.skipped
    assert rewrite.granularity == "group"
    assert text.count("func.call @merlin_dev_") == 2
    assert "linalg.matmul" not in text
    assert "quant_ext.quantize_per_tensor" not in text, "the requantize stayed on the host"
    assert "arith.maximumf" not in text, "the activation stayed on the host"


def test_the_call_takes_the_stored_integers_and_commits_the_integer_the_layer_does(tmp_path) -> None:
    """(activation, stored tensor, folded bias, destination) in the device's own precision.

    The group READS its scale splats and zero points too; the scale is baked into the stated program,
    so a call carrying them would declare a callee no kernel ABI can define. Under the logical kernel
    ABI the call passes every pointer the kernel's interface declares: the bias, folded into the
    accumulator's integer domain, is one of them.
    """
    rewrite, text = _rewritten(_two_layers(), capture=_capture(tmp_path, _TWO_LAYER_STORED))
    for routed in rewrite.routed:
        assert rewrite.arg_access[routed.symbol] == ("read", "read", "read", "write")
        assert [op["role"] for op in rewrite.call_buffers[routed.symbol]] == ["input_0", "weight_0", "bias_0", "out_0"]
        assert routed.dtypes == ("i8", "i8", "i8"), "the call must carry the integer datapath, not the capture's f32"
        assert routed.group is not None
    assert "(tensor<4x8xi8>, tensor<8x16xi8>, tensor<16xi32>, tensor<4x16xi8>) -> tensor<4x16xi8>" in text


@selected_driver.requires_support(_TARGET)
def test_a_bias_with_no_host_source_is_declined_by_name() -> None:
    """Without the capture's weights there is no folded bias to pass, and the group says so."""
    rewrite, _text = _rewritten(_two_layers())
    assert rewrite.moved == 0
    assert all("interface value bias_0 has no host source" in why for _name, why in rewrite.skipped), rewrite.skipped


def test_the_entries_the_rewrite_records_carry_the_layers_epilogue_and_multiplier() -> None:
    """What separates this route from the contraction one. A bare `M x K x N` carries none of it."""
    rewrite, _text = _rewritten(_two_layers())
    for entry in rewrite.entries.values():
        assert entry["epilogue"] == ["bias_add", "acc_scale", "relu"]
        assert entry["acc_scale"] == pytest.approx(0.5 * 0.5 / 0.5)
        assert entry["operand_dtype"] == "int8"
    assert set(rewrite.programs) == set(rewrite.entries)
    assert all(program["entry"]["op"] == "matmul" for program in rewrite.programs.values())


def test_a_group_that_escapes_as_a_float_is_refused_by_name_rather_than_routed(tmp_path) -> None:
    """An integer kernel cannot produce the capture's float accumulation.

    Measured on the public `SY_micro_model` capsule: all four of its device groups stop at the
    contraction, because the matmul's result is read twice. Routing them would emit a call whose
    result type disagrees with what the kernel computes -- which links, runs, and is wrong.
    """
    # The second layer WITHOUT its closing requantize: the group then escapes as f32.
    text = "\n".join(
        [
            "builtin.module {",
            "  func.func @forward(%x: tensor<4x8xi8>, %wa: tensor<8x16xi8>, %biasa: tensor<16xf32>, "
            "%wb: tensor<16x32xi8>, %biasb: tensor<32xf32>) -> tensor<4x32xf32> {",
            "    %s = arith.constant dense<5.000000e-01> : tensor<f32>",
            "    %z = arith.constant dense<0> : tensor<i64>",
            "    %c0 = arith.constant 0.000000e+00 : f32",
            *_layer("a", "%x", 4, 8, 16, bias="%biasa"),
            *_layer("b", "%qa", 4, 16, 32, bias="%biasb")[:-1],
            "    func.return %relub : tensor<4x32xf32>",
            "  }",
            "}",
        ]
    )
    stored = {"wa": (8, 16), "biasa": (16,), "wb": (16, 32), "biasb": (32,)}
    rewrite, _out = _rewritten(text, capture=_capture(tmp_path, stored))
    assert rewrite.moved == 1, "only the layer that still commits an integer may move"
    assert any("escapes as a float" in why for _name, why in rewrite.skipped), rewrite.skipped


def test_two_layers_asking_for_the_same_program_share_one_device_kernel(tmp_path) -> None:
    """Two calls, one symbol. A route that minted a symbol per CALL SITE would build the same kernel
    twice, put both in the archive, and price the model as if it needed two."""
    text = "\n".join(
        [
            "builtin.module {",
            "  func.func @forward(%x: tensor<4x8xi8>, %wa: tensor<8x8xi8>, %biasa: tensor<8xf32>, "
            "%wb: tensor<8x8xi8>, %biasb: tensor<8xf32>) -> tensor<4x8xi8> {",
            "    %s = arith.constant dense<5.000000e-01> : tensor<f32>",
            "    %z = arith.constant dense<0> : tensor<i64>",
            "    %c0 = arith.constant 0.000000e+00 : f32",
            *_layer("a", "%x", 4, 8, 8, bias="%biasa"),
            *_layer("b", "%qa", 4, 8, 8, bias="%biasb"),
            "    func.return %qb : tensor<4x8xi8>",
            "  }",
            "}",
        ]
    )
    stored = {"wa": (8, 8), "biasa": (8,), "wb": (8, 8), "biasb": (8,)}
    rewrite, out = _rewritten(text, capture=_capture(tmp_path, stored))
    assert rewrite.moved == 2, rewrite.skipped
    assert out.count("func.call @merlin_dev_") == 2
    assert len(rewrite.signatures) == 1, f"the same program was given {len(rewrite.signatures)} kernels"


def test_two_layers_differing_only_in_their_multiplier_are_not_given_one_kernel() -> None:
    """The corpus calls them the same DEMAND; the emission bakes the multiplier into the artifact.

    `group_capsules._identity` drops `acc_scale` on purpose -- a unit's command stream is the same
    for every positive multiplier -- and a kernel keyed on it alone would hand one layer the other
    layer's scale, computing a scaled version of itself with nothing in the build saying so.
    """
    from merlin.llvmlower.device_offload import _kernel_identity

    base = {"op": "matmul", "kind": "op", "M": 4, "K": 8, "N": 16, "epilogue": ["acc_scale"]}
    assert _kernel_identity({**base, "name": "a", "acc_scale": 0.25}) != _kernel_identity(
        {**base, "name": "b", "acc_scale": 0.5}
    )
    assert _kernel_identity({**base, "name": "a", "acc_scale": 0.25}) == _kernel_identity(
        {**base, "name": "b", "acc_scale": 0.25}
    )


# ------------------------------------------------- the sidecar is what the build actually reads


def test_the_sidecar_hands_the_build_the_statements_and_not_only_the_signatures(tmp_path) -> None:
    """The rewrite runs in the lowering subprocess and the device build runs outside it.

    A build that read the signatures and dropped the statements would emit the same NUMBER of kernels
    from bare extents, link identically, and lose every layer's readout.
    """
    from merlin.common.ir_lock import IR_LOCK
    from merlin.llvmlower.device_offload import BY_GROUP, build_arguments, load_sidecar, rewrite_prepared_file

    prepared = tmp_path / "prepared.mlir"
    prepared.write_text(_two_layers(), encoding="utf-8")
    with IR_LOCK:
        rewrite = rewrite_prepared_file(
            prepared,
            tmp_path,
            _TARGET,
            select=lambda _s: True,
            granularity=BY_GROUP,
            weight_args=_WEIGHT_ARGS,
            model="two_layer",
            capture=_capture(tmp_path / "capture", _TWO_LAYER_STORED),
        )
    assert rewrite.moved == 2, rewrite.skipped
    sidecar = load_sidecar(tmp_path)
    assert sidecar["granularity"] == BY_GROUP
    arguments = build_arguments(sidecar)
    assert set(arguments["entries"]) == set(arguments["signatures"]) == set(rewrite.signatures)
    assert all(entry["epilogue"] == ["bias_add", "acc_scale", "relu"] for entry in arguments["entries"].values())
    # The printer drops arg_attrs from a bodyless declaration; the seam patches them back, and a
    # declaration without them makes one-shot-bufferize copy the weight operand of every layer.
    # Under the logical kernel ABI each call also passes its folded bias: four accesses per callee.
    assert prepared.read_text(encoding="utf-8").count("bufferization.access") == 4 * len(rewrite.signatures)
    assert arguments["call_buffers"] == rewrite.call_buffers


def test_a_contraction_sidecar_states_no_program_and_says_so_rather_than_an_empty_one(tmp_path) -> None:
    """`None` and `{}` are different instructions to the build: "nothing was stated" versus "stated
    programs were routed and none was supplied", and the second is refused."""
    from merlin.common.ir_lock import IR_LOCK
    from merlin.llvmlower.device_offload import BY_CONTRACTION, build_arguments, load_sidecar, rewrite_prepared_file

    prepared = tmp_path / "prepared.mlir"
    prepared.write_text(_two_layers(), encoding="utf-8")
    with IR_LOCK:
        rewrite_prepared_file(prepared, tmp_path, _TARGET, select=lambda _s: True, granularity=BY_CONTRACTION)
    assert build_arguments(load_sidecar(tmp_path))["entries"] is None


def test_a_stated_program_is_not_asked_for_a_contraction_triple_it_does_not_have() -> None:
    """A convolution states taps and strides, not `M x K x N`.

    Demanding the triple of a statement would decline exactly the layers this route exists to carry:
    measured on the public `SY_model_resnet50` capsule, 54 of the 69 groups it routes are
    convolutions, and none of them carries an extent triple.
    """
    from merlin.llvmlower.device_build import FROM_EXTENTS, FROM_GROUP, kernel_entry

    conv = {"op": "conv2d", "kind": "op", "ci": 3, "N": 64, "Himg": 8, "Wimg": 8, "kh": 3, "kw": 3}
    entry, provenance, refusal = kernel_entry("d0", (), conv, _TARGET)
    assert refusal == "" and provenance == FROM_GROUP
    assert entry is not None and entry["op"] == "conv2d" and entry["name"] == "d0"
    # ...and with NO statement the triple is still required, because then it is the only shape there is.
    entry, provenance, refusal = kernel_entry("d0", (), None, _TARGET)
    assert entry is None and provenance == FROM_EXTENTS and "neither 3 nor 4 extents" in refusal
    # The synthesized form is a BARE contraction: none of the layer's readout is in it.
    entry, provenance, _refusal = kernel_entry("d0", (16, 16, 32), None, _TARGET)
    assert provenance == FROM_EXTENTS and entry is not None
    assert (entry["M"], entry["N"], entry["K"]) == (16, 16, 32) and "epilogue" not in entry


# ------------------------------------------------- the granularity is carried, never assumed


def test_the_routing_carries_the_granularity_the_caller_asked_for() -> None:
    """A contraction route and a group route build different programs. Which one ran has to be a
    property of the routing the build was handed, not of whichever rewrite it happened to call."""
    from merlin.llvmlower import device_build
    from merlin.llvmlower.device_offload import BY_CONTRACTION, BY_GROUP

    assert device_build.DeviceRouting("d", "p", "int8", "i32").granularity == BY_CONTRACTION
    assert device_build.DeviceRouting("d", "p", "int8", "i32", granularity=BY_GROUP).granularity == BY_GROUP
