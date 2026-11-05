import pytest
from merlin_experiments.phase1 import joint_calibration_admission as A


def test_selected_deny_applies_only_exact_model_and_parameter_scope(monkeypatch):
    path=['target','prices','compute_ws','latency_cycles']
    row=dict(model_sha256='a'*64,price_paths=[path],reason='opposing structural residuals',evidence=[{'id':'evidence'}])
    review=dict(schema='merlin.native_calibration_admission_review.v1',deny=[row])
    monkeypatch.setenv(A.ENV,'{"id":"review"}')
    def read(pin):
        return review if pin['id']=='review' else {'model_sha256':pin['id']} if pin['id']!='evidence' else {'public':'diagnostic'}
    monkeypatch.setattr(A.PERF,'_read',read)
    plan=dict(runtime={'id':'a'*64},parameters=[{'path':path}])
    with pytest.raises(ValueError,match='blocked_structural_model_admission'):A.assert_allowed(plan)
    A.assert_allowed(dict(plan,runtime={'id':'b'*64}))
    A.assert_allowed(dict(plan,parameters=[{'path':['different','price']}]))
    monkeypatch.delenv(A.ENV)
    A.assert_allowed(plan)


def test_changed_matching_evidence_fails_closed(monkeypatch):
    monkeypatch.setenv(A.ENV,'{"id":"review"}')
    def read(pin):
        if pin['id']=='review':return dict(schema='merlin.native_calibration_admission_review.v1',deny=[dict(model_sha256='a'*64,price_paths=[['p']],reason='reviewed',evidence=[{'id':'missing'}])])
        if pin['id']=='runtime':return dict(model_sha256='a'*64)
        raise ValueError('evidence changed')
    monkeypatch.setattr(A.PERF,'_read',read)
    with pytest.raises(ValueError,match='evidence changed'):
        A.assert_allowed(dict(runtime={'id':'runtime'},parameters=[{'path':['p']}]))


def test_denied_round_cannot_reuse_cached_labels_or_selection(monkeypatch, tmp_path):
    from merlin_experiments.phase1 import joint_calibration as C
    plan = dict(schema=C.SCHEMA, runtime={"id":"runtime"}, parameters=[], gates=dict(C.GATES))
    (tmp_path / "selection.json").write_text('{"cached":true}')
    (tmp_path / "old-label.json").write_text('{"cycles":123}')
    monkeypatch.setattr(C.PERF, "_read", lambda pin: plan)
    def deny(actual):
        assert actual == plan
        raise ValueError("blocked_structural_model_admission: reviewed")
    monkeypatch.setattr(A, "assert_allowed", deny)
    with pytest.raises(ValueError, match="blocked_structural_model_admission"):
        C.run_round({}, context=None, backend=None, output=tmp_path)
    assert not (tmp_path / "probe-0").exists()


def test_cached_selection_cannot_bypass_current_accuracy_gates(monkeypatch, tmp_path):
    from merlin_experiments.phase1 import joint_calibration as C
    plan = dict(schema=C.SCHEMA, runtime={"id": "runtime"}, parameters=[],
                gates=dict(C.GATES, max_case=.30))
    (tmp_path / "selection.json").write_text('{"cached":true}')
    monkeypatch.setattr(C.PERF, "_read", lambda pin: plan)
    with pytest.raises(ValueError, match="current strict per-case gates"):
        C.run_round({}, context=None, backend=None, output=tmp_path)
    assert not (tmp_path / "probe-0").exists()
