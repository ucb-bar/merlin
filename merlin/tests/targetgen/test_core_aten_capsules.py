"""Single-call capsules keep private answers and reject incomplete boundary evidence."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from targetgen_fixtures import measured_routing

from merlin.targetgen import capsule_common
from merlin.targetgen import core_aten_capsules as C
from merlin.targetgen.core_aten_capture import case_capture_name


def synthetic_routing(case_id):
    return measured_routing(case_id, "synthetic", 10, [{"source": "test"}])


@pytest.fixture
def packaged(tmp_path):
    tensor = dict(
        kind="tensor", dtype="float32", shape=[2], stride=[1], storage_offset=0, requires_grad=False, values=[1.0, 2.0]
    )
    arguments = dict(args=dict(kind="tuple", items=[tensor]), kwargs={})
    case = dict(
        overload="aten.alias.default",
        case_id="heldout alias case",
        arguments=arguments,
        post_arguments=copy.deepcopy(arguments),
        expected=tensor,
        comparison="torch_close",
        comparison_parameters={"rtol": 0, "atol": 0},
        mutated_arguments=[],
        output_input_aliases=["result->args[0]"],
    )
    source = tmp_path / "captures" / case_capture_name(case)
    source.mkdir(parents=True)
    meta = {key: value for key, value in tensor.items() if key not in {"kind", "values"}}
    capture = dict(
        overload=case["overload"],
        status="captured_exact",
        capture_meta=dict(
            ok=True,
            opaque=0,
            input_abi=[dict(dtype="f32", shape=[2])],
            output_abi=[dict(dtype="f32", shape=[2])],
            result_contract=dict(
                authority="same_conversion_exported_program",
                inputs=[meta],
                results=[dict(meta, role="user_output", alias_inputs=[0])],
            ),
        ),
    )
    (source / "capture.json").write_text(json.dumps(capture))
    (source / "inputs.json").write_text("[[1.0, 2.0]]")
    (source / "golden.json").write_text('{"private_answer": 987654321}')
    (source / "capsule.pytorch.py").write_text("# input loader fixture\n")
    (source / "capsule.linalg.mlir").write_text(
        "builtin.module {\nfunc.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {\n"
        '%y = "tensor.cast"(%x) {prov.aten = "aten.alias.default"} : (tensor<2xf32>) -> tensor<2xf32>\n'
        "func.return %y : tensor<2xf32>\n}\n}\n"
    )
    public, private = tmp_path / "public", tmp_path / "_private" / "public"
    manifest = C.write_capsules({"cases": [case]}, source.parent, public, private)
    capsule = capsule_common.load_capsule(public / manifest["capsules"][0]["id"])
    raw = [np.array([1, 2], np.float32).tobytes()] * 2
    readback = dict(metadata=[meta, meta], aliases=[[1], [1]], pre_bytes={"0": raw[1].hex()})
    return capsule, raw, readback, public, private


def test_writer_private_answers_and_schema_safe_additive_id(packaged):
    capsule, _, _, public, private = packaged
    assert capsule["name"].startswith("case_")
    assert "987654321" not in "".join(p.read_text() for p in public.rglob("*") if p.is_file() and p.suffix != ".npz")
    assert not list(public.rglob("golden*"))
    assert private.stat().st_mode & 0o777 == 0o700
    assert (private / capsule["name"] / "golden.yaml").stat().st_mode & 0o777 == 0o600
    assert C.load_private(capsule)["batch_map"]["output_count"] == 2


def test_legacy_view_recovers_committed_same_conversion_layout(packaged, tmp_path):
    from merlin.llvmlower.result_layout import read_layout

    capsule, _, _, public, _ = packaged
    original = (public / capsule["name"] / "capsule.interface.mlir").read_bytes()
    C.runtime_bundle(capsule, tmp_path / "runtime")
    layout = read_layout((tmp_path / "runtime/model.mlir").read_text())
    assert layout["results"][0]["alias_inputs"] == [0]
    assert layout["results"][0]["stride"] == [1]
    assert layout["results"][1]["alias_inputs"] == [0]
    assert (public / capsule["name"] / "capsule.interface.mlir").read_bytes() == original


@pytest.mark.parametrize("defect", ["authority", "alias", "shape", "abi", "overload", "count"])
def test_legacy_view_rejects_unbound_capture_metadata(packaged, tmp_path, defect):
    capsule, _, _, _, private = packaged
    path = private / capsule["name"] / "capture.json"
    capture = json.loads(path.read_text())
    contract = capture["capture_meta"]["result_contract"]
    if defect == "authority":
        contract["authority"] = "eager_golden"
    elif defect == "alias":
        contract["results"][0]["alias_inputs"] = []
    elif defect == "shape":
        contract["results"][0]["shape"] = [1, 2]
    elif defect == "abi":
        capture["capture_meta"]["output_abi"][0]["dtype"] = "i32"
    elif defect == "overload":
        capture["overload"] = "aten.clone.default"
    else:
        contract["inputs"] = []
    path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="legacy result contract"):
        C.runtime_bundle(capsule, tmp_path / "runtime")


@pytest.mark.parametrize("missing", ["capture", "contract"])
def test_legacy_missing_capture_contract_does_not_invent_alias(packaged, tmp_path, missing):
    from merlin.llvmlower.result_layout import read_layout

    capsule, _, _, _, private = packaged
    path = private / capsule["name"] / "capture.json"
    if missing == "capture":
        path.unlink()
    else:
        capture = json.loads(path.read_text())
        del capture["capture_meta"]["result_contract"]
        path.write_text(json.dumps(capture))
    C.runtime_bundle(capsule, tmp_path / "runtime")
    assert read_layout((tmp_path / "runtime/model.mlir").read_text())["results"][0] is None


def test_legacy_result_contract_requires_public_source_binding(packaged, tmp_path):
    capsule, _, _, public, _ = packaged
    source = public / capsule["name"] / "capsule.interface.mlir"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="public full-call input changed"):
        C.runtime_bundle(capsule, tmp_path / "runtime")


def test_legacy_fresh_copy_keeps_natural_result_descriptor(packaged, tmp_path):
    from merlin.llvmlower.result_layout import read_layout

    capsule, _, _, public, _ = packaged
    original = C.load_private(capsule)["batch_map"]["cases"][0]
    case = copy.deepcopy(original["case"])
    case.update(overload="aten.clone.default", case_id="fresh-copy", output_input_aliases=[])
    capture = json.loads(
        (public.parent / "captures" / case_capture_name(original["case"]) / "capture.json").read_text()
    )
    capture["overload"] = case["overload"]
    capture["capture_meta"]["result_contract"]["results"][0]["alias_inputs"] = []
    source = tmp_path / "fresh-captures" / case_capture_name(case)
    source.mkdir(parents=True)
    (source / "capture.json").write_text(json.dumps(capture))
    (source / "inputs.json").write_text("[[1.0,2.0]]")
    (source / "golden.json").write_text("[1.0,2.0]")
    (source / "capsule.pytorch.py").write_text("# fresh clone fixture\n")
    (source / "capsule.linalg.mlir").write_text("""builtin.module {
      func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {
        %out = tensor.empty() : tensor<2xf32>
        %copy = linalg.copy {prov.aten = "aten.clone.default"}
          ins(%x : tensor<2xf32>) outs(%out : tensor<2xf32>) -> tensor<2xf32>
        func.return %copy : tensor<2xf32>
      }
    }""")
    manifest = C.write_capsules({"cases": [case]}, source.parent, tmp_path / "fresh-public", tmp_path / "fresh-private")
    fresh = capsule_common.load_capsule(tmp_path / "fresh-public" / manifest["capsules"][0]["id"])
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MERLIN_CORE_ATEN_PRIVATE_ROOT", str(tmp_path / "fresh-private"))
        C.runtime_bundle(fresh, tmp_path / "fresh-runtime")
    assert read_layout((tmp_path / "fresh-runtime/model.mlir").read_text())["results"][0] is None


def test_writer_refuses_lost_exact_overload_provenance(packaged):
    capsule, _, _, public, _ = packaged
    case = C.load_private(capsule)["batch_map"]["cases"][0]["case"]
    captures = public.parent / "captures"
    source = captures / case_capture_name(case) / "capsule.linalg.mlir"
    source.write_text(source.read_text().replace("aten.alias.default", "aten.clone.default"))
    with pytest.raises(ValueError, match="exact overload provenance"):
        C.write_capsules({"cases": [case]}, captures, public.parent / "new-public", public.parent / "new-private")


@pytest.mark.parametrize("defect", ["missing_readback", "alias", "output", "post", "snapshot"])
def test_full_call_rejects_independent_boundary_defects(packaged, defect):
    capsule, raw, readback, _, _ = packaged
    assert C.grade_readback(capsule, raw, semantic_readback=readback, provenance={"scope": "test"})["passed_count"] == 1
    if defect == "missing_readback":
        readback = None
    elif defect == "alias":
        readback["aliases"][0] = []
    elif defect in {"output", "post"}:
        raw[0 if defect == "output" else 1] = np.zeros(2, np.float32).tobytes()
    else:
        readback["pre_bytes"] = {}
    assert C.grade_readback(capsule, raw, semantic_readback=readback, provenance={"scope": "test"})["passed_count"] == 0


@pytest.mark.parametrize("surface", ["capsule.interface.mlir", "inputs.npz", "golden.yaml"])
def test_bound_inputs_and_answers_fail_closed(packaged, surface):
    capsule, _, _, public, private = packaged
    root = private if surface == "golden.yaml" else public
    path = root / capsule["name"] / surface
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        C.load_private(capsule)


def test_readback_preserves_round2_result_count_and_absent_shape_frames(packaged):
    capsule, raw, readback, _, _ = packaged
    assert (
        C.grade_readback(capsule, raw, output_shapes=[], semantic_readback=readback, provenance={"scope": "test"})[
            "passed_count"
        ]
        == 1
    )
    assert (
        C.grade_readback(capsule, raw + raw, semantic_readback=readback, provenance={"scope": "test"})["passed_count"]
        == 0
    )
    assert (
        C.grade_readback(capsule, raw, output_shapes=[[2]], semantic_readback=readback, provenance={"scope": "test"})[
            "passed_count"
        ]
        == 0
    )


def test_runner_grades_candidate_emission_and_requires_rtl(packaged, tmp_path, monkeypatch):
    capsule, raw, readback, _, _ = packaged
    config = SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers={"L3"})
    paths = SimpleNamespace(run_path=tmp_path / "runs")
    seen = []

    def entrypoints(*args, **kwargs):
        return None, {}, "submitted LLVM"

    def execute(**kwargs):
        seen.append(kwargs["llvm_mlir"])
        return dict(
            output_bytes=raw,
            semantic_readback=readback,
            provenance={"scope": "test"},
            engine="test",
            derived_from_rtl=False,
        )

    monkeypatch.setattr(capsule_common, "run_entrypoints", entrypoints)
    adapter = SimpleNamespace(run_full_call=execute)
    options = dict(paths=paths, config=config, pkg=None, contract=None, timeout=1, no_oracle=False)
    result = C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)
    assert result["status"] == "pass" and seen == ["submitted LLVM"]
    raw[0] = np.zeros(2, np.float32).tobytes()
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)["status"] == "fail"
    raw[0] = raw[1]
    capsule["required_oracle_tiers"] = ["L2", "L3"]
    assert (
        C.run_capsule(capsule, tmp_path, adapters={"L2": adapter, "L3": adapter}, **options)["status"] == "incomplete"
    )
    capsule["required_oracle_tiers"] = ["L2"]
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": object()}, **options)["status"] == "incomplete"


def test_operator_tier_floor_adds_mandatory_rtl_tier_to_scored_capsules_only(packaged, tmp_path, monkeypatch):
    capsule, raw, readback, _, _ = packaged
    capsule.update(cohort="public", scored=True)
    config = SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers={"L3"})
    paths = SimpleNamespace(run_path=tmp_path / "runs")
    options = dict(paths=paths, config=config, pkg=None, contract=None, timeout=1, no_oracle=False)
    calls = []
    monkeypatch.setattr(capsule_common, "run_entrypoints", lambda *a, **kw: (None, {}, "submitted LLVM"))

    def adapter(tier, rtl):
        def execute(**kwargs):
            calls.append(tier)
            observation = dict(output_bytes=raw, semantic_readback=readback, provenance={"tier": tier})
            return dict(observation, engine=tier, derived_from_rtl=rtl)

        return SimpleNamespace(run_full_call=execute)

    adapters = {"L2": adapter("L2", False), "L3": adapter("L3", True)}
    monkeypatch.delenv("MERLIN_CAPSULE_TIER_FLOOR", raising=False)
    result = C.run_capsule(capsule, tmp_path, adapters=adapters, **options)
    assert result["status"] == "pass" and set(result["tiers"]) == {"L2"} and calls == ["L2"]

    monkeypatch.setenv("MERLIN_CAPSULE_TIER_FLOOR", "L3")
    calls.clear()
    result = C.run_capsule(capsule, tmp_path, adapters=adapters, **options)
    assert result["status"] == "pass" and calls == ["L2", "L3"]
    assert result["tiers"]["L3"]["status"] == "pass" and result["tiers"]["L3"]["derived_from_rtl"] is True
    assert capsule["required_oracle_tiers"] == ["L2"]  # the declaration itself is untouched

    for missing in ({"L2": adapters["L2"]}, {**adapters, "L3": object()}):
        result = C.run_capsule(capsule, tmp_path, adapters=missing, **options)
        assert result["status"] == "incomplete" and result["tiers"]["L3"]["not_run_is_not_pass"] is True
        assert result["failure"] == {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS", "tier": "L3"}

    capsule.update(cohort="host_guard", scored=False)
    calls.clear()
    result = C.run_capsule(capsule, tmp_path, adapters={"L2": adapters["L2"]}, **options)
    assert result["status"] == "pass" and set(result["tiers"]) == {"L2"} and calls == ["L2"]

    monkeypatch.setenv("MERLIN_CAPSULE_TIER_FLOOR", "rtl")
    with pytest.raises(ValueError, match="MERLIN_CAPSULE_TIER_FLOOR"):
        C.run_capsule(capsule | {"scored": True}, tmp_path, adapters=adapters, **options)


def test_feedback_grade_defers_the_floor_tier_without_running_it(packaged, tmp_path, monkeypatch):
    from merlin.targetgen import capsule_runner as CR

    capsule, raw, readback, _, _ = packaged
    capsule.update(cohort="public", scored=True)
    config = SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers={"L3"})
    options = dict(paths=SimpleNamespace(run_path=tmp_path / "runs"), config=config, pkg=None, contract=None)
    monkeypatch.setattr(capsule_common, "run_entrypoints", lambda *a, **kw: (None, {}, "submitted LLVM"))
    monkeypatch.setenv("MERLIN_CAPSULE_TIER_FLOOR", "L3")
    observation = dict(output_bytes=raw, semantic_readback=readback, provenance={}, engine="L2", derived_from_rtl=False)
    adapters = CR.defer_tiers({"L2": SimpleNamespace(run_full_call=lambda **kw: observation)}, ["L3"])
    result = C.run_capsule(capsule, tmp_path, adapters=adapters, timeout=1, no_oracle=False, **options)
    assert result["status"] == "pass" and result["tiers"]["L3"]["status"] == "skipped"
    assert result["tiers"]["L3"]["not_run_is_not_pass"] is True  # recorded, never counted as an L3 pass


@pytest.mark.parametrize(
    ("floor", "capsule", "expected"),
    [
        (None, {"required_oracle_tiers": ["L2"]}, ["L2"]),
        ("L3", {"required_oracle_tiers": ["L2"]}, ["L2", "L3"]),
        ("L3", {"required_oracle_tiers": ["L2"], "scored": False}, ["L2"]),
        ("L3", {"required_oracle_tiers": ["L2", "L3"]}, ["L2", "L3"]),
        ("L4", {"required_oracle_tiers": ["L0", "L2"]}, ["L0", "L2", "L3", "L4"]),
        ("L2", {"required_oracle_tiers": ["L3"]}, ["L3"]),
    ],
)
def test_required_tiers_raise_declared_ladder_to_operator_floor(monkeypatch, floor, capsule, expected):
    from merlin.targetgen.tier_policy import required_tiers

    if floor is None:
        monkeypatch.delenv("MERLIN_CAPSULE_TIER_FLOOR", raising=False)
        assert required_tiers(capsule) is capsule["required_oracle_tiers"]
    else:
        monkeypatch.setenv("MERLIN_CAPSULE_TIER_FLOOR", floor)
    assert required_tiers(capsule) == expected


def test_source_bundle_scoring_requires_device_evidence_and_reports_guards(packaged, tmp_path, monkeypatch):
    capsule, raw, readback, _, _ = packaged
    capsule.update(cohort="public", scored=True, lane_expectation="device")
    config = SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers={"L3"})
    options = dict(
        paths=SimpleNamespace(run_path=tmp_path / "runs"),
        config=config,
        pkg=None,
        contract=None,
        timeout=1,
        no_oracle=False,
    )
    observation = dict(
        output_bytes=raw,
        semantic_readback=readback,
        provenance={"scope": "test"},
        engine="test",
        derived_from_rtl=False,
        lane="device",
        executed_instructions=0,
    )

    def execute(**kwargs):
        assert kwargs["package_dir"] == tmp_path
        assert kwargs["capsule"] is capsule
        return observation

    adapter = SimpleNamespace(run_full_call=execute, compiles_source_bundle=True)
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)["status"] == "fail"
    observation["executed_instructions"] = 3
    result = C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)
    assert result["status"] == "pass"
    assert (result["lane"], result["executed_instructions"], result["cohort"]) == ("device", 3, "public")
    capsule.update(cohort="host_guard", scored=False, lane_expectation="host-guard")
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)["status"] == "fail"
    observation.update(lane="host", executed_instructions=0)
    result = C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)
    assert result["status"] == "pass" and result["cohort"] == "guard" and result["scored"] is False


@pytest.fixture
def synthetic_execution_provider(tmp_path, monkeypatch):
    from merlin.targetgen.core_aten_device import load_execution_provider

    path = tmp_path / "provider.py"
    path.write_text(
        "import hashlib\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "from merlin.targetgen.plugins import load_module\n"
        "TARGET = 'test_target'\n"
        "def routing(directory, *, target, package, **kwargs):\n"
        "    identity = hashlib.sha256(str(Path(package).resolve()).encode()).hexdigest()\n"
        "    backend = load_module(package, 'mlir_oot.golden_device_catalog',\n"
        "                          package_name='test_snapshot_' + identity)\n"
        "    _, inventory = backend.build_catalog((Path(directory) / 'model.mlir').read_text())\n"
        "    assert inventory['covered_contractions']\n"
        "    return SimpleNamespace(catalog_builder=backend.merlin_builder(None))\n"
    )
    # Exercise explicit provider selection even when no external provider is installed.
    monkeypatch.setenv("MERLIN_CORE_ATEN_EXECUTION_PROVIDER", str(path))
    return load_execution_provider("test_target")


def test_two_snapshot_backends_grade_consecutively_in_one_process(
    packaged,
    tmp_path,
    synthetic_execution_provider,
):
    provider = synthetic_execution_provider
    capsule, raw, readback, _, _ = packaged
    packages = []
    for marker in ("first", "second"):
        package = tmp_path / marker / "submission"
        module = package / "mlir_oot"
        module.mkdir(parents=True)
        (module / "identity.py").write_text(f"MARKER = {marker!r}\n")
        (module / "golden_device_catalog.py").write_text(
            "from .identity import MARKER\n"
            "def build_catalog(source):\n"
            "    return None, {'covered_contractions': [MARKER]}\n"
            "def merlin_builder(tools):\n"
            "    return MARKER\n"
        )
        packages.append(package)
    loaded = []

    def execute(*, bundle, package_dir, **kwargs):
        routing = provider.routing(
            bundle,
            target="test_target",
            package=package_dir,
            facts={},
            eligible=[(None, SimpleNamespace(dtypes=("i8", "i8", "i32")))],
        )
        loaded.append(routing.catalog_builder)
        return dict(output_bytes=raw, semantic_readback=readback, provenance={"scope": "test"})

    config = SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers=set())
    adapter = SimpleNamespace(run_full_call=execute, compiles_source_bundle=True)
    # Revisit the first snapshot too: cached imports must retain their original identity.
    for index, package in enumerate([*packages, packages[0]]):
        result = C.run_capsule(
            capsule,
            package,
            paths=SimpleNamespace(run_path=tmp_path / f"grade-{index}"),
            config=config,
            adapters={"L2": adapter},
            pkg=None,
            contract=None,
            timeout=1,
            no_oracle=False,
        )
        assert result["status"] == "pass", result["failure"]
    assert loaded == ["first", "second", "first"]


@pytest.mark.parametrize(
    "error", ["PluginError: two packages claim the namespace 'backend'", "ValueError: build refused", None]
)
def test_execution_refusal_preserves_tier_reason_and_compile_category(packaged, tmp_path, error):
    from merlin.targetgen.tier_integrity import reason_for

    capsule, _, _, _, _ = packaged
    adapter = SimpleNamespace(
        compiles_source_bundle=True,
        run_full_call=lambda **_: dict(output_bytes=None, execution_error=error, provenance={"scope": "test"}),
    )
    result = C.run_capsule(
        capsule,
        tmp_path,
        paths=SimpleNamespace(run_path=tmp_path / "grade"),
        config=SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers=set()),
        adapters={"L2": adapter},
        pkg=None,
        contract=None,
        timeout=1,
        no_oracle=False,
    )
    reason = error or "no hardware output"
    assert result["status"] == "fail"
    assert result["failure"] == dict(plane="compile", category="PROTOCOL_VIOLATION", tier="L2", detail=reason)
    assert result["numeric"]["status"] == "skipped"
    assert result["tiers"]["L2"]["reason"] == reason
    assert reason_for("L2", result["tiers"]["L2"]) == reason
    assert json.loads((tmp_path / "grade" / "capsule_result.json").read_text())["tiers"]["L2"]["reason"] == reason


def test_packaging_and_sealed_runtime_preserve_transposed_caller(tmp_path):
    values = np.arange(6, dtype=np.float32).reshape(2, 3).T
    tensor = dict(
        kind="tensor",
        dtype="float32",
        shape=[3, 2],
        stride=[1, 3],
        storage_offset=0,
        requires_grad=False,
        values=values.tolist(),
    )
    arguments = dict(args=dict(kind="tuple", items=[tensor]), kwargs={})
    expected = {**tensor, "stride": [2, 1], "values": (-values).tolist()}
    case = dict(
        overload="aten.neg.default",
        arguments=arguments,
        post_arguments=copy.deepcopy(arguments),
        expected=expected,
        comparison="torch_close",
        comparison_parameters={"rtol": 0, "atol": 0},
        mutated_arguments=[],
        output_input_aliases=[],
    )
    source = tmp_path / "captures" / case_capture_name(case)
    source.mkdir(parents=True)
    meta = {key: value for key, value in expected.items() if key not in {"kind", "values"}}
    capture = dict(
        overload=case["overload"],
        status="captured_exact",
        capture_meta=dict(
            ok=True,
            opaque=0,
            input_abi=[dict(dtype="f32", shape=[3, 2])],
            output_abi=[dict(dtype="f32", shape=[3, 2])],
            result_contract=dict(
                authority="same_conversion_exported_program",
                inputs=[meta],
                results=[dict(meta, role="user_output", alias_inputs=[])],
            ),
        ),
    )
    (source / "capture.json").write_text(json.dumps(capture))
    (source / "inputs.json").write_text(json.dumps([values.tolist()]))
    (source / "golden.json").write_text(json.dumps(expected["values"]))
    (source / "capsule.pytorch.py").write_text("# loader fixture\n")
    (source / "capsule.linalg.mlir").write_text("""builtin.module {
      func.func @forward(%x: tensor<3x2xf32>) -> tensor<3x2xf32> {
        %out = tensor.empty() : tensor<3x2xf32>
        %y = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
          affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"],
          prov.aten = "aten.neg.default"}
          ins(%x : tensor<3x2xf32>) outs(%out : tensor<3x2xf32>) {
          ^bb0(%a: f32, %b: f32):
            %n = arith.negf %a : f32
            linalg.yield %n : f32
          } -> tensor<3x2xf32>
        func.return %y : tensor<3x2xf32>
      }
    }""")
    public, private = tmp_path / "public", tmp_path / "_private" / "public"
    manifest = C.write_capsules({"cases": [case]}, source.parent, public, private)
    capsule = capsule_common.load_capsule(public / manifest["capsules"][0]["id"])
    golden = C.load_private(capsule)
    assert golden["semantic_io"]["outputs"][1]["stride"] == [1, 3]
    # Simulate an older sealed contract without editing its files or commitments.
    legacy = copy.deepcopy(golden)
    legacy["semantic_io"]["outputs"][1]["stride"] = [2, 1]
    from unittest.mock import patch

    with patch.object(C, "load_private", return_value=legacy):
        C.runtime_bundle(capsule, tmp_path / "runtime")
    boundary = json.loads((tmp_path / "runtime" / "semantic_io.json").read_text())
    assert boundary["outputs"][1]["stride"] == [1, 3]
    raw = [(-values).tobytes(), values.tobytes()]
    readback = dict(metadata=boundary["outputs"], aliases=[[], [1]], pre_bytes={"0": raw[1].hex()})
    verdict = C.grade_readback(capsule, raw, semantic_readback=readback, provenance={"scope": "test"})
    assert verdict["passed_count"] == 1
    assert next(iter(verdict["cases"].values()))["semantic_scope"] == "full"
    assert legacy["semantic_io"]["outputs"][1]["stride"] == [2, 1]


@pytest.mark.parametrize("lane", ["device", "any"])
def test_selected_numeric_flow_requires_exact_reference(packaged, monkeypatch, lane):
    capsule, raw, readback, _, _ = packaged
    capsule["lane_expectation"] = lane
    golden = C.load_private(capsule)
    record = golden["batch_map"]["cases"][0]
    record["numeric_policy"] = "spec_int8"
    record["numeric_admission"] = {
        "integer_reference": {
            "status": "prepared",
            "engine": "independent-test-reference",
            "expected_output_bytes": {"0": raw[0].hex()},
        }
    }
    capsule["operation"]["attributes"]["numeric_flow"] = "spec_int8"
    monkeypatch.setattr(C, "load_private", lambda *args: copy.deepcopy(golden))
    routing = synthetic_routing(record["case_id"])
    monkeypatch.setattr("merlin.targetgen.core_aten_device.verify_execution_evidence", lambda evidence: True)
    assert C.grade_readback(capsule, raw, semantic_readback=readback, routing=routing)["passed_count"] == 1
    assert C.grade_readback(capsule, raw, semantic_readback=readback)["passed_count"] == 0
    record["numeric_admission"]["integer_reference"]["expected_output_bytes"]["0"] = bytes(len(raw[0])).hex()
    verdict = C.grade_readback(capsule, raw, semantic_readback=readback, routing=routing)
    assert verdict["passed_count"] == 0
    assert next(iter(verdict["cases"].values()))["integer_reference"]["status"] == "mismatch"
    del record["numeric_admission"]["integer_reference"]
    assert C.grade_readback(capsule, raw, semantic_readback=readback)["passed_count"] == 0


def test_full_call_rejects_unattributed_device_execution(packaged, monkeypatch):
    capsule, raw, readback, _, _ = packaged
    routing = synthetic_routing("wrong-case")
    verdict = C.grade_readback(capsule, raw, semantic_readback=readback, routing=routing)
    assert verdict["passed_count"] == 0
    assert next(iter(verdict["cases"].values()))["evidence_error"]


def test_parallel_writer_preserves_bytes_order_and_private_permissions(packaged, tmp_path):
    """Real spawned workers produce the same commitments as the serial writer."""
    capsule, _, _, public, private = packaged
    source_root = tmp_path / "captures"
    case = C.load_private(capsule)["batch_map"]["cases"][0]["case"]
    # Deliberately reverse identifier order: workers must retain declaration order.
    cases = [dict(case, case_id=name) for name in ("z-last", "a-first")]
    import shutil

    for item in cases:
        shutil.copytree(source_root / case_capture_name(case), source_root / case_capture_name(item))
    roots = []
    manifests = []
    for workers in (1, 2):
        root = tmp_path / f"workers-{workers}"
        manifests.append(
            C.write_capsules(
                {"cases": cases},
                source_root,
                root / "hidden",
                root / "_private/hidden",
                label="hidden",
                workers=workers,
            )
        )
        roots.append(root)
    assert manifests[0] == manifests[1]
    assert [row["case_id"] for row in manifests[1]["capsules"]] == [case["case_id"] for case in cases]
    trees = [{p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} for root in roots]
    assert trees[0] == trees[1]
    for owner in (roots[1] / "hidden", roots[1] / "_private/hidden"):
        for path in owner.rglob("*"):
            assert path.stat().st_mode & 0o077 == 0


def test_parallel_writer_collects_every_refusal_without_a_manifest(packaged, tmp_path):
    capsule, _, _, _, _ = packaged
    case = C.load_private(capsule)["batch_map"]["cases"][0]["case"]
    cases = [dict(case, case_id=str(n)) for n in range(5)]
    with pytest.raises(C.CapsuleBuildError) as caught:
        C.write_capsules(
            {"cases": cases},
            tmp_path / "captures",
            tmp_path / "failed/public",
            tmp_path / "failed/_private/public",
            workers=2,
        )
    assert [row["case_id"] for row in caught.value.failures] == [str(n) for n in range(5)]
    assert not (tmp_path / "failed/public/manifest.json").exists()


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_writer_worker_budget_is_explicit(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        C.write_capsules(
            {"cases": []}, tmp_path / "captures", tmp_path / "public", tmp_path / "private", workers=workers
        )


def test_native_numeric_capsule_requires_complete_admission_and_independent_bits(packaged, monkeypatch):
    capsule, raw, readback, _, _ = packaged
    golden = C.load_private(capsule)
    record = golden["batch_map"]["cases"][0]
    record["numeric_policy"] = "spec_int8"
    plan = dict(
        numerical_family="F1",
        status="admitted",
        exact_dtype_sites=1,
        host_floor=dict(status="required", comparison="native_eager_bits_every_boundary_result"),
        sites=[dict(status="admitted")],
    )
    golden["batch_map"]["numeric_preparation"] = plan
    record["numeric_admission"] = plan
    record["native_eager_boundary"] = dict(
        authority="untransformed_native_cpu_eager", output_bytes=[raw[0].hex()], post_input_bytes=[raw[1].hex()]
    )
    capsule["operation"]["attributes"]["numeric_flow"] = "spec_int8"
    monkeypatch.setattr(C, "load_private", lambda *args: copy.deepcopy(golden))
    routing = synthetic_routing(record["case_id"])
    monkeypatch.setattr("merlin.targetgen.core_aten_device.verify_execution_evidence", lambda evidence: True)
    verdict = C.grade_readback(capsule, raw, semantic_readback=readback, routing=routing)
    row = next(iter(verdict["cases"].values()))
    assert row["host_floor"]["status"] == "pass"
    assert row["integer_reference"] == {}
    assert verdict["passed_count"] == 1
    assert C.grade_readback(capsule, raw, semantic_readback=readback)["passed_count"] == 0
    record["native_eager_boundary"]["output_bytes"] = [bytes(len(raw[0])).hex()]
    assert C.grade_readback(capsule, raw, semantic_readback=readback, routing=routing)["passed_count"] == 0
    record["native_eager_boundary"]["output_bytes"] = [raw[0].hex()]
    for key, value in (("numerical_family", "F3"), ("status", "refused"), ("exact_dtype_sites", 2)):
        previous = plan[key]
        plan[key] = value
        assert C.grade_readback(capsule, raw, semantic_readback=readback, routing=routing)["passed_count"] == 0
        plan[key] = previous


def test_numeric_refusal_diagnostic_omits_repeated_inputs_and_expected_bytes():
    plan = dict(
        status="host",
        refusal_code="not_profitable",
        selection={"spec": "large shared contract"},
        provider_catalog={"assembly": "large catalog"},
        integer_reference=dict(status="prepared", expected_output_bytes={"0": "private-answer"}),
    )
    error = C.NumericFixtureError("declined", plan)
    assert error.plan["refusal_code"] == "not_profitable"
    assert error.plan["integer_reference"] == {"status": "prepared"}
    assert "selection" not in error.plan and "provider_catalog" not in error.plan
    assert "private-answer" not in str(error)
    assert plan["integer_reference"]["expected_output_bytes"] == {"0": "private-answer"}


def test_writer_requires_native_storage_comparison(packaged):
    capsule, raw, _, _, _ = packaged
    assert capsule["numeric_policy"]["compare"] == "exact_bits"
    assert capsule["numeric_policy"]["rtol"] == capsule["numeric_policy"]["atol"] == 0
    record = C.load_private(capsule)["batch_map"]["cases"][0]
    assert record["case"]["comparison_parameters"] == dict(rtol=0, atol=0, storage_bits=True)
    from merlin.targetgen.core_aten_batch_grade import _grade_output

    # A sub-ULP-looking error must not pass because the corpus carried a tolerance.
    wrong = np.array([1.0, np.nextafter(np.float32(2), np.float32(3))], np.float32).tobytes()
    expected = record["case"]["expected"]
    assert _grade_output(raw[0], record["output_abi"][0], expected, record["case"])["status"] == "pass"
    assert _grade_output(wrong, record["output_abi"][0], expected, record["case"])["status"] == "mismatch"


def test_decoded_float_comparison_cannot_claim_storage_bits():
    from merlin.targetgen.capsule_golden import compare

    with pytest.raises(ValueError, match="storage-boundary"):
        compare({"x": [0.0]}, {"x": [-0.0]}, dict(compare="exact_bits", dtype="f32"))


@pytest.mark.parametrize("device", [False, True])
def test_admission_lanes_preserve_host_abi_and_publish_only_proved_numeric_flow(packaged, tmp_path, monkeypatch, device):
    capsule, _, _, public, _ = packaged
    golden = C.load_private(capsule)
    case = golden["batch_map"]["cases"][0]["case"]
    captures = tmp_path / "captures"
    expected = dict(status="admitted", numerical_family="F1", exact_dtype_sites=1,
                    sites=[dict(status="admitted")],
                    host_floor=dict(status="required", comparison="native_eager_bits_every_boundary_result")) if device else dict(status="host", refusal_code="non_native_unit_construction")

    def prepare(bundle, report, flow, *, require_device):
        assert not require_device
        if not device:
            # A failed candidate must not leak its modified host input sidecars.
            (bundle / "inputs.npz").write_bytes(b"refused candidate")
        return expected

    monkeypatch.setattr(C, "prepare_numeric_fixture", prepare)
    manifest = C.write_capsules({"cases": [case]}, captures, tmp_path / "admission-public",
                                tmp_path / "admission-private", numeric_flow={"policy": "spec_int8"},
                                admission_lanes=True)
    row = manifest["capsules"][0]
    assert row["native_admission"] == expected
    assert row["lane_expectation"] == ("device" if device else "host-guard")
    member = tmp_path / "admission-public" / row["id"]
    created = capsule_common.load_capsule(member)
    assert bool(created["operation"]["attributes"].get("numeric_flow")) is device
    assert (member / "inputs.npz").read_bytes() == (public / capsule["name"] / "inputs.npz").read_bytes()
