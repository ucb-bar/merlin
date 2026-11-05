"""Synthetic ancestry guards; these tests grant no model accuracy qualification."""

import copy
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_readout_revision as R
from test_joint_provider_revision import case as provider_case  # noqa: F401 -- pytest fixture registration


@pytest.fixture
def case(request):
    c = request.getfixturevalue("provider_case")
    c.before["target"].update(
        hardware={"normalizations": 0}, scheduling={"version": 5, "accumulator_streamed_readout": 1}
    )
    c.before["memory"] = {"caches": {"l1d": {"line_bytes": 64, "coherence_probe_service": {}}}}
    c.after = copy.deepcopy(c.before)
    c.after["libraries"][1] = c.provider["sha256"]
    c.after["target"]["hardware"]["readout_extra_command_registered_stages"] = 6
    c.after["target"]["scheduling"].update(R.SELECTORS | {"readout_probe_request_cycles": 6})
    c.after["target"]["assumptions"] = dict(R.ASSUMPTIONS)
    response = {
        "schema": "merlin.provisional_probe_write_response.v1",
        "scope": "offset_zero_wormhole_no_backpressure",
        "block_bytes": 64,
        "pipeline_cycles": 8,
    }
    c.after["memory"]["caches"]["l1d"]["coherence_probe_service"]["write_response"] = response
    c.old_mp = c.put("old-model.json", c.before)
    c.new_mp = c.put("model.json", c.after)
    c.old_spec.update(model_sha256=c.old_mp["sha256"])
    c.new_spec.update(model_sha256=c.new_mp["sha256"])
    observer = c.put("spike", b"synthetic observer")
    c.new_spec["observer"] = observer["path"]
    c.old = c.put("old-runtime.json", c.old_spec)
    c.new = c.put("runtime.json", c.new_spec)
    audit = c.put(
        "audit.json",
        {
            "schema": "merlin.public_readout_staging_facts.v1",
            "qualification_issued": False,
            "parameter_origin": "source_frontend_and_public_gsim_response_stages",
            "additions": {"frontend_stages": 6, "probe_request_cycles": 6, "write_response": response},
            "rtl_sources": [c.source],
            "public_observations": [c.row["physical_evidence"]],
        },
    )
    build = c.put(
        "build.json",
        {
            "provider": c.provider,
            "observer": observer,
            "source_identity": c.after["domain"]["feature_engine_sha256"],
            "sources": [c.source],
        },
    )
    for name, model in [("baseline_report", c.old_mp), ("report", c.new_mp)]:
        report = json.loads(open(c.row[name]["path"]).read())
        report["model_sha256"] = model["sha256"]
        c.row[name] = c.put(name + ".json", report)
    replays = c.put(
        "replays.json",
        {
            "schema": "merlin.provider_public_replays.v1",
            "private_labels_used": False,
            "qualification_issued": False,
            "rows": [c.row],
        },
    )
    denial = c.put(
        "denial.json",
        {
            "schema": "merlin.native_calibration_admission_review.v1",
            "deny": [c.denial_row, dict(c.denial_row, model_sha256=c.new_mp["sha256"], evidence=[audit])],
        },
    )
    c.evidence.update(
        schema=R.SCHEMA,
        previous_runtime=c.old,
        runtime=c.new,
        previous_model=c.old_mp,
        model=c.new_mp,
        source_manifests={"previous": c.source, "current": c.source},
        structural_audit=audit,
        build=build,
        public_replays=replays,
        continued_denial=denial,
    )
    return c


def verify(c):
    R.verify(
        SimpleNamespace(model_json=json.dumps(c.before)),
        SimpleNamespace(model_json=json.dumps(c.after)),
        c.evidence,
        previous=c.old,
        current=c.new,
    )


def test_explicit_structural_successor_remains_provisional(case):
    verify(case)


@pytest.mark.parametrize("change", ["price", "capacity", "scope", "selector", "geometry"])
def test_delta_cannot_smuggle_existing_semantic_changes(case, change):
    after = copy.deepcopy(case.after)
    if change == "price":
        after["target"]["prices"]["ws"] += 1
    if change == "capacity":
        after["target"]["hardware"]["normalizations"] = 1
    if change == "scope":
        after["scope"]["kind"] = "trace"
    if change == "selector":
        after["target"]["scheduling"]["unreviewed"] = 1
    if change == "geometry":
        after["memory"]["caches"]["l1d"]["line_bytes"] = 128
    audit = json.loads(open(case.evidence["structural_audit"]["path"]).read())
    with pytest.raises(ValueError, match="unreviewed semantics"):
        R._delta(case.before, after, audit)


@pytest.mark.parametrize("change", ["denial", "window", "source", "promotion"])
def test_no_missing_evidence_or_authority_bypass(case, change):
    if change == "denial":
        case.evidence["continued_denial"] = case.previous_denial
    if change == "window":
        case.new_spec["timing_selection"] = {"window": "different"}
        case.new = case.put("runtime.json", case.new_spec)
        case.evidence["runtime"] = case.new
    if change == "source":
        open(case.source["path"], "w").write("{}")
    if change == "promotion":
        case.evidence["promotion_authorized"] = True
    with pytest.raises((ValueError, RuntimeError)):
        verify(case)


def test_admission_requires_successor_denial_and_never_authorizes_fit(case, monkeypatch):
    from merlin_experiments.phase1 import joint_calibration_admission as A
    from merlin_experiments.phase1 import joint_scoped_revision as S

    monkeypatch.setattr(S, "verify", lambda *args, **kwargs: None)
    origin = dict(
        kind="scoped_observer_diagnostics",
        previous={"path": "initial", "sha256": "initial"},
        runtime=case.old,
        evidence=case.put(
            "origin.json",
            dict(
                schema=S.SCHEMA,
                runtime=case.old,
                model=case.old_mp,
                continued_denial=case.previous_denial,
                calibration_profile=case.profile,
            ),
        ),
    )
    revision = dict(
        schema="merlin.runtime_implementation_revision.v1",
        kind=R.KIND,
        previous=case.old,
        runtime=case.new,
        evidence=case.put("review.json", case.evidence),
        review_id="synthetic",
    )
    selection = dict(runtime=case.new, runtime_revisions=[origin, revision])
    plan = dict(runtime=case.new, selection_document=selection, parameters=case.params)
    monkeypatch.setattr(
        R.PERF,
        "_runtime",
        lambda pin: SimpleNamespace(model_json=json.dumps(case.after if pin == case.new else case.before)),
    )
    monkeypatch.setenv(A.ENV, json.dumps(case.previous_denial))
    with pytest.raises(ValueError, match="active admission review differs"):
        A.assert_allowed(plan)
    monkeypatch.setenv(A.ENV, json.dumps(case.evidence["continued_denial"]))
    with pytest.raises(ValueError, match="blocked_structural_model_admission"):
        A.assert_allowed(plan)


@pytest.fixture
def partial_case(case):
    c = case
    c.before = copy.deepcopy(c.after)
    c.before['libraries'][1] = 'previous-provider'
    c.before['target']['scheduling'].update(scratchpad_row_loads=1, scratchpad_request_admission=1)
    c.before['target']['hardware']['reader_beat_bytes'] = 16
    c.after = copy.deepcopy(c.before)
    c.after['libraries'][1] = c.provider['sha256']
    c.after['target']['scheduling'].update(R.PARTIAL_SELECTORS)
    c.after['target']['hardware']['reader_request_capture_stages'] = 1
    c.old_mp, c.new_mp = c.put('partial-old-model.json', c.before), c.put('partial-model.json', c.after)
    c.old_spec.update(model=c.old_mp['path'], model_sha256=c.old_mp['sha256'], observer=c.new_spec['observer'])
    c.new_spec.update(model=c.new_mp['path'], model_sha256=c.new_mp['sha256'])
    c.old, c.new = c.put('partial-old-runtime.json', c.old_spec), c.put('partial-runtime.json', c.new_spec)
    audit = json.loads(open(c.evidence['structural_audit']['path']).read())
    audit.update(schema='merlin.public_partial_row_staging_facts.v1',
                 parameter_origin='source_derived_row_protocol_no_price_changes',
                 additions={'reader_request_capture_stages': 1})
    c.evidence.update(schema=R.PARTIAL_SCHEMA, previous_runtime=c.old, runtime=c.new,
                      previous_model=c.old_mp, model=c.new_mp,
                      structural_audit=c.put('partial-audit.json', audit))
    for key, model in [('baseline_report', c.old_mp), ('report', c.new_mp)]:
        report = json.loads(open(c.row[key]['path']).read())
        report['model_sha256'] = model['sha256']
        c.row[key] = c.put('partial-' + key + '.json', report)
    replay = json.loads(open(c.evidence['public_replays']['path']).read())
    replay['rows'] = [c.row]
    c.evidence['public_replays'] = c.put('partial-replays.json', replay)
    denial = json.loads(open(c.evidence['continued_denial']['path']).read())
    denial['deny'][-1]['model_sha256'] = c.new_mp['sha256']
    c.evidence['continued_denial'] = c.put('partial-denial.json', denial)
    return c


def verify_partial(c):
    R.verify(SimpleNamespace(model_json=json.dumps(c.before)), SimpleNamespace(model_json=json.dumps(c.after)),
             c.evidence, previous=c.old, current=c.new, partial_rows=True)


def test_partial_protocol_exact_provisional_successor(partial_case):
    verify_partial(partial_case)


@pytest.mark.parametrize('change', ['price', 'scope', 'observer', 'fact', 'denial', 'promotion'])
def test_partial_protocol_cannot_widen_authority(partial_case, change):
    c = partial_case
    if change in {'price', 'scope', 'fact'}:
        after = copy.deepcopy(c.after)
        if change == 'price':
            after['target']['prices']['ws'] += 1
        elif change == 'scope':
            after['scope']['kind'] = 'whole_trace'
        else:
            after['target']['hardware']['reader_request_capture_stages'] = 2
        audit = json.loads(open(c.evidence['structural_audit']['path']).read())
        with pytest.raises(ValueError, match='unreviewed semantics'):
            R._partial_delta(c.before, after, audit)
        return
    if change == 'observer':
        c.new_spec['observer'] = 'different-observer'
        c.new = c.put('partial-runtime.json', c.new_spec)
        c.evidence['runtime'] = c.new
    if change == 'denial':
        c.evidence['continued_denial'] = c.previous_denial
    if change == 'promotion':
        c.evidence['promotion_authorized'] = True
    with pytest.raises(ValueError):
        verify_partial(c)


@pytest.fixture
def load_case(partial_case):
    c = partial_case
    c.before = copy.deepcopy(c.after)
    c.before['libraries'][1] = 'previous-load-provider'
    c.after = copy.deepcopy(c.before)
    c.after['libraries'][1] = c.provider['sha256']
    c.after['target']['scheduling']['scratchpad_load_frontend_arrival'] = 1
    c.after['target']['hardware']['load_extra_command_registered_stages'] = 6
    c.after['target']['assumptions']['load_command_frontend_no_backpressure'] = 1
    c.old_mp, c.new_mp = c.put('load-old-model.json', c.before), c.put('load-model.json', c.after)
    c.old_spec.update(model=c.old_mp['path'], model_sha256=c.old_mp['sha256'])
    c.new_spec.update(model=c.new_mp['path'], model_sha256=c.new_mp['sha256'])
    c.old, c.new = c.put('load-old-runtime.json', c.old_spec), c.put('load-runtime.json', c.new_spec)
    audit = json.loads(open(c.evidence['structural_audit']['path']).read())
    audit.update(schema='merlin.public_load_frontend_arrival_facts.v1',
                 parameter_origin='source_derived_load_frontend_no_price_changes',
                 additions={'load_extra_command_registered_stages': 6})
    c.evidence.update(schema=R.LOAD_SCHEMA, previous_runtime=c.old, runtime=c.new,
                      previous_model=c.old_mp, model=c.new_mp,
                      structural_audit=c.put('load-audit.json', audit))
    for key, model in [('baseline_report', c.old_mp), ('report', c.new_mp)]:
        report = json.loads(open(c.row[key]['path']).read())
        report['model_sha256'] = model['sha256']
        c.row[key] = c.put('load-' + key + '.json', report)
    replay = json.loads(open(c.evidence['public_replays']['path']).read())
    replay['rows'] = [c.row]
    c.evidence['public_replays'] = c.put('load-replays.json', replay)
    denial = json.loads(open(c.evidence['continued_denial']['path']).read())
    denial['deny'][-1]['model_sha256'] = c.new_mp['sha256']
    c.evidence['continued_denial'] = c.put('load-denial.json', denial)
    return c


def verify_load(c):
    R.verify(SimpleNamespace(model_json=json.dumps(c.before)), SimpleNamespace(model_json=json.dumps(c.after)),
             c.evidence, previous=c.old, current=c.new, load_frontend=True)


def test_load_frontend_exact_provisional_successor(load_case):
    verify_load(load_case)


@pytest.mark.parametrize('change', ['price', 'memory', 'scope', 'stage_count', 'backpressure', 'capacity'])
def test_load_frontend_cannot_change_existing_accounting(load_case, change):
    c = load_case
    after = copy.deepcopy(c.after)
    if change == 'price':
        after['target']['prices']['ws'] += 1
    elif change == 'memory':
        after['memory']['caches']['l1d']['line_bytes'] *= 2
    elif change == 'scope':
        after['scope']['kind'] = 'trace'
    elif change == 'stage_count':
        after['target']['hardware']['load_extra_command_registered_stages'] += 1
    elif change == 'backpressure':
        after['target']['assumptions']['load_command_frontend_no_backpressure'] = 0
    else:
        after['target']['hardware']['normalizations'] = 1
    audit = json.loads(open(c.evidence['structural_audit']['path']).read())
    with pytest.raises(ValueError, match='unreviewed semantics'):
        R._load_frontend_delta(c.before, after, audit)


@pytest.mark.parametrize('change', ['observer', 'denial', 'promotion', 'source'])
def test_load_frontend_preserves_provenance_and_denial(load_case, change):
    c = load_case
    if change == 'observer':
        c.new_spec['observer'] = 'unreviewed-observer'
        c.new = c.put('load-runtime.json', c.new_spec)
        c.evidence['runtime'] = c.new
    elif change == 'denial':
        c.evidence['continued_denial'] = c.previous_denial
    elif change == 'promotion':
        c.evidence['promotion_authorized'] = True
    else:
        open(c.source['path'], 'w').write('{}')
    with pytest.raises((ValueError, RuntimeError)):
        verify_load(c)


def test_load_frontend_active_admission_still_denies_fitting(load_case, monkeypatch):
    from merlin_experiments.phase1 import joint_calibration_admission as A
    from merlin_experiments.phase1 import joint_scoped_revision as S

    c = load_case
    monkeypatch.setattr(S, 'verify', lambda *args, **kwargs: None)
    origin = dict(kind='scoped_observer_diagnostics', previous=c.old, runtime=c.old,
                  evidence=c.put('load-origin.json', dict(schema=S.SCHEMA, runtime=c.old,
                    continued_denial=c.previous_denial, calibration_profile=c.profile)))
    revision = dict(schema='merlin.runtime_implementation_revision.v1', kind=R.LOAD_KIND,
                    previous=c.old, runtime=c.new, evidence=c.put('load-review.json', c.evidence),
                    review_id='synthetic')
    plan = dict(runtime=c.new, parameters=c.params,
                selection_document=dict(runtime=c.new, runtime_revisions=[origin, revision]))
    monkeypatch.setattr(R.PERF, '_runtime', lambda pin: SimpleNamespace(
        model_json=json.dumps(c.after if pin == c.new else c.before)))
    monkeypatch.setenv(A.ENV, json.dumps(c.previous_denial))
    with pytest.raises(ValueError, match='active admission review differs'):
        A.assert_allowed(plan)
    monkeypatch.setenv(A.ENV, json.dumps(c.evidence['continued_denial']))
    with pytest.raises(ValueError, match='blocked_structural_model_admission'):
        A.assert_allowed(plan)


@pytest.fixture
def bootstrap_case(load_case):
    c = load_case
    c.before = copy.deepcopy(c.after)
    c.before['memory']['caches']['l2'] = dict(sets=128, ways=8, line_bytes=64)
    c.after = copy.deepcopy(c.before)
    receipt = c.put('loader-source-receipt.json', {'loader': 'synthetic_source_receipt'})
    state = R.BOOTSTRAP_STATE | {'source_receipt_sha256': receipt['sha256']}
    c.after['memory']['initial_state'] = state
    c.old_mp, c.new_mp = c.put('boot-old-model.json', c.before), c.put('boot-model.json', c.after)
    c.old_spec.update(model=c.old_mp['path'], model_sha256=c.old_mp['sha256'])
    c.old_spec['libraries'] = copy.deepcopy(c.new_spec['libraries'])
    c.new_spec.update(model=c.new_mp['path'], model_sha256=c.new_mp['sha256'])
    c.old, c.new = c.put('boot-old-runtime.json', c.old_spec), c.put('boot-runtime.json', c.new_spec)
    audit = json.loads(open(c.evidence['structural_audit']['path']).read())
    audit.update(schema='merlin.public_loader_cache_bootstrap_facts.v1',
                 parameter_origin='source_derived_loader_state_no_price_changes',
                 additions={'initial_state': state}, loader_source_receipt=receipt,
                 loader_sources=[c.source], private_or_firesim_labels_used=False)
    c.evidence.update(schema=R.BOOTSTRAP_SCHEMA, previous_runtime=c.old, runtime=c.new,
                      previous_model=c.old_mp, model=c.new_mp,
                      structural_audit=c.put('boot-audit.json', audit))
    for key, model in [('baseline_report', c.old_mp), ('report', c.new_mp)]:
        report = json.loads(open(c.row[key]['path']).read())
        report['model_sha256'] = model['sha256']
        c.row[key] = c.put('boot-' + key + '.json', report)
    replay = json.loads(open(c.evidence['public_replays']['path']).read())
    replay['rows'] = [c.row]
    c.evidence['public_replays'] = c.put('boot-replays.json', replay)
    denial = json.loads(open(c.evidence['continued_denial']['path']).read())
    denial['deny'][-1]['model_sha256'] = c.new_mp['sha256']
    c.evidence['continued_denial'] = c.put('boot-denial.json', denial)
    return c


def test_loader_bootstrap_exact_review(bootstrap_case):
    c = bootstrap_case
    R.verify(SimpleNamespace(model_json=json.dumps(c.before)), SimpleNamespace(model_json=json.dumps(c.after)),
             c.evidence, previous=c.old, current=c.new, loader_bootstrap=True)


@pytest.mark.parametrize('change', ['address_hints', 'replacement', 'price', 'scope', 'provider', 'receipt'])
def test_loader_bootstrap_cannot_smuggle_warm_state_or_authority(bootstrap_case, change):
    c = bootstrap_case
    after = copy.deepcopy(c.after)
    if change == 'address_hints':
        after['memory']['initial_state']['warm_addresses'] = [4096]
    elif change == 'replacement':
        after['memory']['initial_state']['replacement_scope'] = 'lru'
    elif change == 'price':
        after['target']['prices']['ws'] += 1
    elif change == 'scope':
        after['scope']['kind'] = 'trace'
    elif change == 'provider':
        after['libraries'][1] = 'unreviewed-provider'
    else:
        after['memory']['initial_state']['source_receipt_sha256'] = '0' * 64
    audit = json.loads(open(c.evidence['structural_audit']['path']).read())
    with pytest.raises(ValueError, match='unreviewed semantics'):
        R._bootstrap_delta(c.before, after, audit)
