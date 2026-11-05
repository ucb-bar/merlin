"""Rolling scheduling custody, not simulated target execution or certification."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_runner as runner
from merlin.targetgen.model_gate_continuation import ModelGatePrerequisites, ModelGateContinuation
from merlin_experiments.phase1.feedback import rolling_model_gate as R
from test_tier_boundary_join import observations, prepare_observation
from test_candidate_execution_products import retained


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    modeldir = tmp_path / "model"; modeldir.mkdir()
    declaration = modeldir / "capsule.yaml"
    declaration.write_text("name: model\nkind: model\ngate: {after_op_pass_fraction: 0.8}\n")
    model = dict(name="model", kind="model", __dir__=str(modeldir),
                 gate={"after_op_pass_fraction": .8})
    caps = {name: dict(name=name, kind="layer", required_oracle_tiers=["L0","L1","L2","L3"])
            for name in "ABCDE"}
    scope = dict(compiler_sha256="c"*64, target="fixture", profile_sha256="p",
                 bundle_snapshot_sha256="b", instruments={"L2":"i2", "L3":"i3"},
                 declarations={name: name for name in caps})
    def new_store(selected=None):
        gate = ModelGatePrerequisites("fixture", {"model": sha256(declaration.read_bytes()).hexdigest()},
            frozenset(caps), set(), "s")
        return R.CompletionStore(tmp_path / "workspace", scope=selected or scope,
            capsules=caps, gate=gate, source_check=lambda: None)
    def observation(path, *, tier):
        left, right = observations()
        record = left if tier == "L2" else right
        record["common"]["capsule"] = json.loads(Path(path).read_text())["capsule"]
        record["result_sha256"] = sha256(Path(path).read_bytes()).hexdigest()
        return record
    monkeypatch.setattr(R.tier_boundary_join, "optional_observation", observation)
    def close(store, name, **changes):
        row = dict(capsule=name, kind="layer", status="pass", numeric={"status":"pass"},
            tiers={t:{"status":"pass"} for t in caps[name]["required_oracle_tiers"]})
        row.update(changes)
        path = tmp_path / (name + ".json"); path.write_text(json.dumps(row))
        store.complete(row, path)
        return path
    return new_store, close, model, scope


def model_request(store, model):
    calls = []
    def execute(cap, workers):
        calls.append(cap["name"])
        return {"capsule": cap["name"], "status":"incomplete"}
    rows = runner._run_streaming_model_gate([], [model], execute,
        max_workers=1, target="fixture", oracle_adapters={"L2":object(), "L3":object()},
        model_gate_prerequisites=store.gate)
    return calls, rows


def test_single_model_requires_four_of_five_exact_completed_ops(fixture):
    new, close, model, _ = fixture
    store = new()
    for name in "ABC": close(store, name)
    next_request = new(); next_request.load()
    calls, rows = model_request(next_request, model)
    assert calls == [] and rows[0]["status"] == "gated"
    close(store, "D")
    next_request = new(); next_request.load()
    calls, rows = model_request(next_request, model)
    assert calls == ["model"]
    assert rows[0]["status"] == "incomplete"  # Admission grants no model pass.
    assert model["_driver_model_gate_admission"]["fraction"] == .8
    assert "E" not in next_request.gate.qualified


@pytest.mark.parametrize("changes", [
    {"status":"fail", "failure":{"plane":"placement"}},
    {"numeric":{"status":"fail"}},
    {"status":"incomplete", "tiers":{"L2":{"status":"pass"}}},
    {"tiers":{t:{"status":"pass" if t != "L3" else "unavailable"} for t in ["L0","L1","L2","L3"]}},
])
def test_failed_missing_or_partial_tiers_do_not_count_and_replace_success(fixture, changes):
    new, close, _, _ = fixture
    store = new(); close(store, "A")
    assert store.gate.qualified == {"A"}
    close(store, "A", **changes)
    later = new(); later.load()
    assert later.gate.qualified == set()


@pytest.mark.parametrize("field", ["compiler_sha256", "profile_sha256", "bundle_snapshot_sha256", "instruments", "declarations", "target"])
def test_foreign_scope_is_not_a_snapshot_union(fixture, field):
    new, close, _, scope = fixture
    store = new(); close(store, "A")
    changed = deepcopy(scope); changed[field] = "changed"
    foreign = new(changed); foreign.load()
    assert not foreign.gate.qualified
    foreign._path("A").write_bytes(store._path("A").read_bytes())
    with pytest.raises(ValueError, match="selected identities"): foreign.load()


def test_original_receipt_and_products_reopened_at_admission(fixture, monkeypatch):
    new, close, _, _ = fixture
    store = new(); path = close(store, "A")
    original = path.read_bytes(); path.write_text("{}")
    with pytest.raises(ValueError, match="completion changed"): new().load()
    path.write_bytes(original)
    monkeypatch.setattr(R.tier_boundary_join, "optional_observation", lambda *a, **k: None)
    with pytest.raises(ValueError, match="execution products changed"): new().load()


def test_mutated_model_declaration_and_frozen_source_refuse(fixture):
    new, close, model, _ = fixture
    store = new()
    (Path(model["__dir__"]) / "capsule.yaml").write_text("changed")
    with pytest.raises(ValueError, match="declaration changed"): model_request(store, model)
    def changed(): raise ValueError("frozen compiler changed")
    store.source_check = changed
    with pytest.raises(ValueError, match="frozen compiler changed"): close(store, "A")


def test_old_partial_model_voucher_cannot_bypass_full_rolling_cohort(fixture):
    new, _, model, _ = fixture
    store = new()
    token = ModelGateContinuation("model", store.gate.models["model"], "fixture", "a"*64, "b"*64)
    rows = runner._run_streaming_model_gate([], [model], lambda *a: pytest.fail("must remain gated"),
        max_workers=1, target="fixture", oracle_adapters={"L2":1,"L3":1},
        model_gate_prerequisites=store.gate, model_gate_continuations=(token,))
    assert rows[0]["status"] == "gated"


def test_real_producer_reopening_remains_required(retained, monkeypatch):
    # The existing neutral fixture has one real invocation record, not an NPU
    # execution. The second role here is explicitly a unit-test checker stub.
    path, root, row = prepare_observation(retained)
    row["status"] = "pass"
    row["tiers"]["L3"] = {"status":"pass"}
    path.write_text(json.dumps(row))
    native_observe = R.tier_boundary_join.observe
    def observe(path, *, tier):
        real = native_observe(path, tier="L2")
        if tier == "L3": real.update(tier="L3", engine="second-fixture")
        return real
    monkeypatch.setattr(R.tier_boundary_join, "observe", observe)
    cap = dict(name="A", required_oracle_tiers=["L0","L1","L2","L3"])
    scope = dict(target="fixture", compiler_sha256=row["compiler_source"]["candidate_sha256"])
    assert R._qualified(path, cap, scope) is not None
    program = row["compiler_source"]["executed_programs"][0]
    (root / program["elf"]["path"]).write_bytes(b"mutated executable")
    assert R._qualified(path, cap, scope) is None


def test_broker_preparation_uses_full_declared_original_cohort_and_selected_owner(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import compiler_profile_runtime, joint_physical_correctness
    from merlin.targetgen import capsule_common, target_experiment
    workspace = tmp_path / "workspace"; source = workspace / "submission"
    source.mkdir(parents=True); (source / "compiler.py").write_text("# neutral compiler source\n")
    root = tmp_path / "corpus"; root.mkdir()
    caps = []
    for name, kind in [("op", "layer"), ("ineligible", "layer"), ("model", "model"), ("calibration", "layer")]:
        directory = root / name; directory.mkdir()
        (directory / "capsule.yaml").write_text("name: " + name)
        caps.append(dict(name=name, kind=kind, __dir__=str(directory), required_oracle_tiers=["L0","L1","L2","L3"]))
    monkeypatch.setattr(joint_physical_correctness, "roster", lambda: {"op","ineligible","model"})
    public_root = tmp_path / "public"; public_root.mkdir()
    public_caps = deepcopy(caps)
    for capsule in public_caps:
        directory = public_root / capsule["name"]; directory.mkdir()
        (directory / "capsule.yaml").write_text("name: " + capsule["name"] + "\noracle_tier_ceiling: L3\n")
        capsule["__dir__"] = str(directory)
    monkeypatch.setattr(capsule_common, "discover_capsules",
        lambda roots, **kw: public_caps if roots == (public_root,) else caps)
    monkeypatch.setattr(runner, "_gate_counts", lambda row, *a: row["capsule"] != "ineligible")
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda p: SimpleNamespace(target="fixture"))
    owner = SimpleNamespace(snapshot_digest="selected-corpus-library", profile={"sha256":"selected-profile"})
    monkeypatch.setattr(compiler_profile_runtime, "active_original_execution",
        lambda: SimpleNamespace(owner=owner, source_sha256="exact-profile-source"))
    monkeypatch.setattr(R.promotion, "cert_instrument", lambda tier, **kw: "checker-" + tier)
    monkeypatch.setattr(R.vouchers, "_context", lambda *a: {"contract_sha256":"selected-contract"})
    for k, v in R.broker_environment(workspace, root, source).items(): monkeypatch.setenv(k, v)
    store, complete = R.prepare(context=SimpleNamespace(descriptor="descriptor"), contract=root,
        source=source, runs=tmp_path / "runs", graded_root=public_root)
    assert store.gate.denominator == {"op"}
    assert set(store.gate.models) == {"model"}
    assert set(store.scope["declarations"]) == {"op","ineligible","model"}
    assert store.scope["profile_sha256"] == "selected-profile"
    assert store.scope["bundle_snapshot_sha256"] == "selected-corpus-library"
    assert set(store.scope["instruments"]) == {"L0","L1","L2","L3"}
    assert callable(complete)
    model = next(c for c in public_caps if c["name"] == "model")
    assert store.scope["declarations"]["model"] != store.scope["graded_model_declarations"]["model"]
    assert store.gate.fraction(model, target="fixture") == 0.0
    store.gate.qualified.add("op")
    assert store.gate.fraction(model, target="fixture") == 1.0
    # Both original denominator metadata and the graded model remain pinned.
    for declaration in (root / "model" / "capsule.yaml", root / "op" / "capsule.yaml",
                        public_root / "model" / "capsule.yaml"):
        original = declaration.read_bytes(); declaration.write_bytes(original + b"changed")
        with pytest.raises(ValueError, match="source declaration changed"):
            store.gate.fraction(model, target="fixture")
        with pytest.raises(ValueError, match="source declaration changed"):
            store.load()
        declaration.write_bytes(original)
    # The actual worker's subset copy must equal the pinned public projection.
    copied = tmp_path / "copy"; copied.mkdir()
    (copied / "capsule.yaml").write_bytes((public_root / "model" / "capsule.yaml").read_bytes())
    assert store.gate.fraction({**model, "__dir__": str(copied)}, target="fixture") == 1.0
    (copied / "capsule.yaml").write_text("changed")
    with pytest.raises(ValueError, match="declaration changed"):
        store.gate.fraction({**model, "__dir__": str(copied)}, target="fixture")
    # Projection is not permission to change the selected gate or tier policy.
    model["gate"] = {"after_op_pass_fraction": 0}
    with pytest.raises(ValueError, match="public model policy differs"):
        R.prepare(context=SimpleNamespace(descriptor="descriptor"), contract=root,
            source=source, runs=tmp_path / "runs", graded_root=public_root)
    model.pop("gate")
    (source / "compiler.py").write_text("# changed source\n")
    with pytest.raises(ValueError, match="frozen compiler changed"):
        R.prepare(context=SimpleNamespace(descriptor="descriptor"), contract=root,
            source=source, runs=tmp_path / "runs", graded_root=public_root)


def test_live_observation_preserves_checked_l2_before_next_engine_overwrites_input(retained, tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    path, generated, row = prepare_observation(retained)
    scope = dict(target="fixture", compiler_sha256=row["compiler_source"]["candidate_sha256"])
    caps = {"A": dict(name="A", required_oracle_tiers=["L0","L1","L2","L3"])}
    gate = ModelGatePrerequisites("fixture", {}, frozenset({"A"}), set(), "scope")
    store = R.CompletionStore(tmp_path / "workspace", scope=scope, capsules=caps, gate=gate, source_check=lambda: None)
    tier = row["tiers"]["L2"]
    result = SimpleNamespace(status="pass", to_dict=lambda: deepcopy(tier))
    kwargs = dict(generated=generated, package_dir=Path(row["compiler_source"]["package"]),
                  shas=row["toolchain_shas"])
    with store.observing():
        R.capture_tier("A", "L2", result, **kwargs)
    record = store.live_l2[("A", str(generated))]
    assert record["common"]["compiler_sha256"] == scope["compiler_sha256"]
    # A later engine may reuse the original build path. The actual earlier
    # invocation was independently verified while those bytes were still live.
    (generated / "package_kernel.elf").write_bytes(b"next engine executable")
    with pytest.raises(ValueError):
        R.tier_boundary_join.observe(path, tier="L2")
    R._reopen_live_l2(record)
    stderr = Path(record["observed"]["stderr"]["path"])
    original_stderr = stderr.read_bytes(); stderr.write_bytes(b"changed stderr")
    with pytest.raises(ValueError, match="dependency or output changed"):
        R._reopen_live_l2(record)
    stderr.write_bytes(original_stderr)
    # Persisting a full completion consumes this exact live observation; later
    # admission reopens it after the original ELF path has been reused.
    right = {"schema":"merlin.original_tier_boundary.v1", "common":record["common"],
             "tier":"L3", "engine":"second-fixture", "execution_digest":"1"*64,
             "elf_sha256":"2"*64, "harness_sha256":"3"*64, "result_sha256":"4"*64}
    original_observe = R.tier_boundary_join.optional_observation
    monkeypatch.setattr(R.tier_boundary_join, "optional_observation",
        lambda p, *, tier: right if tier == "L3" else original_observe(p, tier=tier))
    row.update(status="pass")
    row["tiers"]["L3"] = {"status":"pass"}
    path.write_text(json.dumps(row))
    store.complete(row, path)
    reopened = R.CompletionStore(tmp_path / "workspace", scope=scope, capsules=caps,
        gate=ModelGatePrerequisites("fixture", {}, frozenset({"A"}), set(), "scope"), source_check=lambda: None)
    reopened.load()
    assert reopened.gate.qualified == {"A"}
    # Only the original verified ELF is relocatable; its retained bytes and
    # original invocation/console must still remain exact.
    retained_elf = Path(record["retained_elf"]["path"])
    retained_elf.write_bytes(b"changed retained executable")
    with pytest.raises(ValueError, match="retained executable changed"):
        R._reopen_live_l2(record)
    store.live_l2.clear()
    with store.observing():
        R.capture_tier("A", "L2", SimpleNamespace(status="fail"), **kwargs)
    assert store.live_l2 == {}


def test_partial_screen_does_not_erase_previous_complete_same_source_proof(fixture, tmp_path):
    new, close, _, _ = fixture
    store = new()
    # Separate request paths preserve the actual earlier full receipt.
    original = close(store, "A")
    retained = tmp_path / "earlier.json"; retained.write_bytes(original.read_bytes())
    record_path = store._path("A"); record = json.loads(record_path.read_bytes())
    record["receipt"] = str(retained); record_path.write_text(json.dumps(record))
    close(store, "A", status="incomplete",
          failure={"plane":"oracle_unavailable", "category":"NOT_RUN_IS_NOT_PASS", "tier":"L3"},
          tiers={t:{"status":"pass" if t != "L3" else "unavailable"} for t in ["L0","L1","L2","L3"]})
    later = new(); later.load()
    assert later.gate.qualified == {"A"}
