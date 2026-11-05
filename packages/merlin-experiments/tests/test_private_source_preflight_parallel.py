"""Real owned CLI fixture workers prove parallel custody/strict source-only joins."""

import json
from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_source_preflight_parallel as P


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    roster = ("first", "second", "third")
    spec, descriptor = tmp_path / "spec.json", tmp_path / "descriptor.json"
    spec.write_text(json.dumps({"models": list(roster)}))
    descriptor.write_text("{}")
    monkeypatch.setattr(P.models, "requirements_for", lambda _: roster)
    monkeypatch.setattr(P.models, "program_requirements_for", lambda _: {name: ["body"] for name in roster})

    def binding(spec, record, *, root, target):
        return {
            "schema": P.freeze.SCHEMA,
            "spec_sha256": sha256(spec.read_bytes()).hexdigest(),
            "record_sha256": "b" * 64,
            "root": str(root),
        }

    monkeypatch.setattr(P.freeze, "binding", binding)
    monkeypatch.setattr(P.freeze, "stage", lambda spec, root, **_: {"root": str(root), "archives": [], "sources": []})
    script = tmp_path / "source_fixture.py"
    script.write_text("""import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('spec','descriptor','out','target','model'):p.add_argument('--'+name,required=True)
a=p.parse_args();spec=Path(a.spec);doc=json.loads(spec.read_bytes());out=Path(a.out);out.mkdir(parents=True)
(out/'sources').mkdir();markers=out.parent.parent/'markers';markers.mkdir(exist_ok=True)
(markers/a.model).touch();end=time.monotonic()+3
while len(list(markers.iterdir()))<3 and time.monotonic()<end:time.sleep(.01)
if len(list(markers.iterdir()))<3:raise SystemExit(3)
time.sleep(doc.get('delay',0))
roster=doc['models'];sha=hashlib.sha256(spec.read_bytes()).hexdigest()
r={'schema':'unused_fixture','scope':'scope_fixture','partition_passed':False,'target':a.target,'mode':'source_preflight_partition','selected_model':a.model,
'source_partition_passed':True,'source_preflight_passed':False,'passed':False,'candidate_tree_sha256':None,
'private_spec_sha256':sha,'required_models':roster,'required_programs':{n:['body'] for n in roster},
'models':[{'model':a.model,'status':'source_inputs_verified','checks':{'source':{'body':{}}}}],
'authored_source_freeze':{'schema':'merlin.phase1.private_source_freeze.v1','spec_sha256':sha,
'record_sha256':'b'*64,'root':str(out/'sources')}}
if a.model=='second':
 d=doc.get('damage')
 if d=='candidate':r['candidate_tree_sha256']='a'*64
 if d=='full':r['source_preflight_passed']=True
 if d=='program':r['required_programs']['second']=[]
 if d=='freeze':r['authored_source_freeze']['root']='/wrong'
(out/'private_full_model_source_preflight.json').write_text(json.dumps(r))
""")
    script.write_text(
        script.read_text()
        .replace("'unused_fixture'", repr(P.models.RESULT_SCHEMA))
        .replace("'scope_fixture'", repr(P.models.SOURCE_PREFLIGHT_SCOPE))
    )
    original = P._command

    def command(*args):
        argv = original(*args)
        return [argv[0], str(script), *argv[4:]]

    monkeypatch.setattr(P, "_command", command)
    return spec, descriptor


def test_three_real_owned_workers_join_complete_source_roster_without_candidate(fixture, tmp_path):
    spec, descriptor = fixture
    report = P.preflight(spec, descriptor=descriptor, out=tmp_path / "out", target="fixture", timeout_s=8)
    assert report["source_preflight_passed"] is True and report["passed"] is False
    assert report["candidate_tree_sha256"] is None and report["source_workers"] == 3
    assert len(report["models"]) == len(report["source_partitions"]) == 3
    assert all(worker["cleanup"]["complete"] and worker["cleanup"]["guardian_reaped"] for worker in report["workers"])


@pytest.mark.parametrize("damage", ["candidate", "full", "program", "freeze"])
def test_source_partition_cannot_change_scope_roster_or_freeze(fixture, tmp_path, damage):
    spec, descriptor = fixture
    spec.write_text(json.dumps({**json.loads(spec.read_text()), "damage": damage}))
    with pytest.raises(ValueError, match="roster/source ownership"):
        P.preflight(spec, descriptor=descriptor, out=tmp_path / "out", target="fixture", timeout_s=8)


def test_timeout_refuses_source_pass_and_reaps_every_owned_worker(fixture, tmp_path):
    spec, descriptor = fixture
    spec.write_text(json.dumps({**json.loads(spec.read_text()), "delay": 8}))
    report = P.preflight(spec, descriptor=descriptor, out=tmp_path / "out", target="fixture", timeout_s=0.3)
    assert report["source_preflight_passed"] is False
    assert all(worker["error"] == "TimeoutExpired" for worker in report["workers"])
    assert all(worker["cleanup"]["complete"] and worker["cleanup"]["guardian_reaped"] for worker in report["workers"])
