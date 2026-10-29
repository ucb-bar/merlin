import json
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_final_coverage as F
from merlin_experiments.phase1 import joint_grade_shards as S
from test_joint_final_coverage import _native_fixture, _score  # noqa: E402

from merlin.targetgen import capsule_grade as CG


def _caps(n, extends=None):
    caps = [{"name": f"c{i:03d}", "kind": "layer", "label": "public", "__dir__": f"/x/c{i:03d}"} for i in range(n)]
    for child, parent in (extends or {}).items():
        caps[child]["extends"] = caps[parent]["name"]
    return caps


def _admission(caps):
    names = sorted(c["name"] for c in caps)
    return {"n_admitted_capsules": len(names), "admitted_name_set_sha256": CG._name_set_sha256(names)}


def _shard_score(assigned, status="pass"):
    s = _score([c["name"] for c in assigned])
    s["cohort_admission"] = _admission(assigned)
    s["host_guard_total"] = 0
    for row in s["per_capsule"]:
        row["status"] = status
    s["n_passed"] = len(assigned) if status == "pass" else 0
    return s


def test_flag_off_is_the_default(monkeypatch):
    for raw in (None, "", "0", "off"):
        env = {} if raw is None else {S.ENV: raw}
        assert S.configured_processes(env, slots=46) == 0


def test_processes_are_clamped_to_half_the_slots():
    assert S.configured_processes({S.ENV: "auto"}, slots=46) == 23
    assert S.configured_processes({S.ENV: "auto"}, slots=47) == 24
    assert S.configured_processes({S.ENV: "99"}, slots=46) == 23
    assert S.configured_processes({S.ENV: "5"}, slots=46) == 5
    with pytest.raises(ValueError):
        S.configured_processes({S.ENV: "lots"}, slots=46)


def test_partition_is_deterministic_closed_under_extends_and_exact():
    caps = _caps(40, extends={5: 3, 9: 5, 30: 3})
    a, b = S.partition(caps, 6), S.partition(list(reversed(caps)), 6)
    assert [[c["name"] for c in s] for s in a] == [[c["name"] for c in s] for s in b]
    flat = [c["name"] for s in a for c in s]
    assert sorted(flat) == sorted(c["name"] for c in caps) and len(set(flat)) == 40
    home = {c["name"]: i for i, s in enumerate(a) for c in s}
    assert home["c003"] == home["c005"] == home["c009"] == home["c030"]
    assert max(map(len, a)) - min(map(len, a)) <= 4


def test_partition_refuses_models_and_trivial_rosters():
    caps = _caps(10)
    caps[4]["kind"] = "model"
    assert S.partition(caps, 4) is None
    assert S.partition(_caps(10), 1) is None
    assert S.partition(_caps(1), 4) is None
    assert S.partition(_caps(5, extends={i: 0 for i in range(1, 5)}), 4) is None  # one component


def test_merge_is_deterministic_and_sums_counters():
    caps = _caps(9)
    shards = S.partition(caps, 3)
    scores = [(a, _shard_score(a), None) for a in shards]
    m1 = S.merge_scores(scores, caps=caps, admission=_admission(caps))
    m2 = S.merge_scores(list(scores), caps=caps, admission=_admission(caps))
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert [r["capsule"] for r in m1["per_capsule"]] == sorted(c["name"] for c in caps)
    assert m1["n_capsules"] == 9 and m1["n_passed"] == 9
    group = F._capsule_group(caps)
    F._apply_score(group, m1)
    assert group["passed"] is True and all(r["status"] == "pass" for r in group["obligations"].values())


def test_failed_missing_or_foreign_shard_never_passes():
    caps = _caps(9)
    shards = S.partition(caps, 3)
    foreign = _shard_score(shards[1])
    foreign["per_capsule"][0]["capsule"] = "zzz"
    for bad in (
        (shards[1], None, "shard exited rc=1"),
        (shards[1], foreign, None),
        (shards[1], {"failure": {"plane": "build"}}, None),
    ):
        rows = [(shards[0], _shard_score(shards[0]), None), bad, (shards[2], _shard_score(shards[2]), None)]
        merged = S.merge_scores(rows, caps=caps, admission=_admission(caps))
        group = F._capsule_group(caps)
        F._apply_score(group, merged)
        status = {n: r["status"] for n, r in group["obligations"].items()}
        for cap in shards[1]:
            assert status[cap["name"]] == "skipped"
        assert group["passed"] is False and merged["n_gated_deferred"] == len(shards[1])
        assert sum(v == "pass" for v in status.values()) == 6
        assert merged["sharded_grade"]["shard_failures"]


def test_failed_shard_rows_carry_the_native_fail_of_other_shards():
    caps = _caps(6)
    shards = S.partition(caps, 2)
    rows = [(shards[0], _shard_score(shards[0], "fail"), None), (shards[1], _shard_score(shards[1]), None)]
    merged = S.merge_scores(rows, caps=caps, admission=_admission(caps))
    group = F._capsule_group(caps)
    F._apply_score(group, merged)
    assert {r["status"] for r in group["obligations"].values()} == {"pass", "fail"}


class _Proc:
    def __init__(self, directory, behaviour, spec):
        self.pid, self.returncode, self.directory, self.spec = 1 << 30, None, directory, spec
        self.behaviour = behaviour
        self.started = time.time()

    def poll(self):
        if self.returncode is None:
            if self.behaviour == "hang":
                return None
            self.returncode = 3 if self.behaviour == "crash" else 0
            if self.behaviour == "ok":
                assigned = [{"name": n} for n in self.spec["capsules"]]
                (self.directory / "score.json").write_text(json.dumps(_shard_score(assigned)))
        return self.returncode


def _grade_sharded(tmp_path, monkeypatch, behaviours, end_in=60, grace=0):
    caps = _caps(12)
    monkeypatch.setattr(S, "GRACE_S", grace)
    monkeypatch.setattr(S, "_POLL_S", 0.01)
    monkeypatch.setattr(S, "_KILL_WAIT_S", 0.05)
    killed = []
    monkeypatch.setattr(S.os, "killpg", lambda pid, sig: killed.append(sig))
    seen = []

    def spawn(spec_path, log):
        spec = json.loads(spec_path.read_text())
        seen.append(spec)
        return _Proc(spec_path.parent, behaviours[spec["index"] % len(behaviours)], spec)

    monkeypatch.setattr(S, "_spawn", spawn)
    monkeypatch.setattr(S, "_admission", lambda caps, root, d: _admission(caps))
    context = SimpleNamespace(descriptor="d", repo="r", harness=None, readback_policy=None, candidate_policy=None)
    inputs = SimpleNamespace(arm="a", public_root=tmp_path, additional_forbidden=())
    checks = []
    merged = S.grade_sharded(
        caps=caps,
        root=tmp_path,
        tag="core",
        output=tmp_path,
        frozen=tmp_path / "f",
        run_dir=tmp_path / "r",
        context=context,
        inputs=inputs,
        contract="c",
        descriptor_sha256="x",
        end_epoch=time.time() + end_in,
        timeout_s=30,
        processes=4,
        floor_exempt=False,
        verify=lambda: checks.append(1),
    )
    return caps, merged, seen, killed, checks


def test_grade_sharded_merges_all_shards_and_reverifies_inputs(tmp_path, monkeypatch):
    caps, merged, seen, killed, checks = _grade_sharded(tmp_path, monkeypatch, ["ok"])
    assert len(seen) == 4 and checks == [1] and not killed
    assert all(s["workers"] == 2 and s["timeout_s"] == 30 for s in seen)
    assert sorted(n for s in seen for n in s["capsules"]) == sorted(c["name"] for c in caps)
    assert merged["n_passed"] == 12 and merged["sharded_grade"]["n_shards_merged"] == 4
    assert SimpleNamespace(**seen[0]).run_dir == str(tmp_path / "r")


def test_crashed_shard_leaves_deferred_rows(tmp_path, monkeypatch):
    caps, merged, *_ = _grade_sharded(tmp_path, monkeypatch, ["ok", "crash"])
    group = F._capsule_group(caps)
    F._apply_score(group, merged)
    assert merged["sharded_grade"]["n_shards_merged"] == 2 and group["passed"] is False
    assert {r["status"] for r in group["obligations"].values()} == {"pass", "skipped"}


def test_deadline_kills_hung_shards_and_defers_their_rows(tmp_path, monkeypatch):
    caps, merged, seen, killed, _ = _grade_sharded(tmp_path, monkeypatch, ["ok", "hang"], end_in=-1)
    assert killed  # killpg only on the shard process group, never a pattern kill
    deferred = [r for r in merged["per_capsule"] if r["status"] == "gated"]
    assert len(deferred) == 6 and all("deadline" in r["background_deferred"] for r in deferred)


def test_validate_flag_off_runs_in_process_and_flag_on_shards(tmp_path, monkeypatch):
    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv(S.ENV, raising=False)
    report = F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert report["n_passed"] == 246 and len([c for c in calls if c[0] == "capsules"]) == 2
    assert not list((prepared.run_dir / "joint_final_coverage").glob("*-shards"))

    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path / "on", monkeypatch)
    inputs.context.harness = inputs.context.candidate_policy = None
    monkeypatch.setenv(S.ENV, "4")
    monkeypatch.setattr(S, "slot_capacity", lambda: 46)
    fake = []

    def sharded(**kw):
        fake.append(kw["tag"])
        return _score([c["name"] for c in kw["caps"]])

    monkeypatch.setattr(S, "grade_sharded", sharded)
    report = F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert fake == ["core", "scalability"] and [c for c in calls if c[0] == "capsules"] == []
    assert report["n_passed"] == 246 and report["passed"] is True
    assert (prepared.run_dir / "joint_final_coverage" / "core-score.json").is_file()


def test_validate_unshardable_group_falls_back_in_process(tmp_path, monkeypatch):
    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv(S.ENV, "auto")
    monkeypatch.setattr(S, "slot_capacity", lambda: 46)
    monkeypatch.setattr(S, "grade_sharded", lambda **kw: None)
    report = F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert report["n_passed"] == 246 and len([c for c in calls if c[0] == "capsules"]) == 2


def test_unmeasured_shard_cannot_veto_and_nested_aggregates_are_recomputed():
    caps = _caps(8)
    shards = S.partition(caps, 2)
    a, b = _shard_score(shards[0]), _shard_score(shards[1])
    for score, assigned in ((a, shards[0]), (b, shards[1])):
        score["public_passed"] = f"{len(assigned)}/{len(assigned)}"
        score["highest_tier"] = "L2"
        score["pass_evidence"] = {
            "n_passed": len(assigned),
            "rtl_tiers_seen": [],
            "fidelity_seen": ["functional_model"],
        }
        score["tier_reuse"] = {"executed": {"L2": len(assigned)}, "carried": {}, "capsules_with_a_carried_tier": 0}
    empty = _shard_score(shards[1])  # a shard that measured nothing reports false closure flags
    empty.update(n_capsules=0, n_passed=0, structural_evidence_all_pass=False, gradeable=False, numeric_all_exact=None)
    merged = S.merge_scores([(shards[0], a, None), (shards[1], b, None)], caps=caps, admission=_admission(caps))
    assert merged["public_passed"] == "8/8" and merged["pass_evidence"]["n_passed"] == 8
    assert merged["tier_reuse"]["executed"] == {"L2": 8} and merged["highest_tier"] == "L2"
    assert merged["coverage"] is None if "coverage" in merged else True
    veto = S.merge_scores([(shards[0], a, None), (shards[1], empty, None)], caps=caps, admission=_admission(caps))
    assert veto["structural_evidence_all_pass"] is True and veto["gradeable"] is True
