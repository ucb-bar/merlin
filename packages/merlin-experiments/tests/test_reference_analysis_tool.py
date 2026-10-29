"""Real analyzer calls through the existing standalone selfcheck diagnostic channel."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile

import pytest
from merlin_experiments.phase1.brokers import selfcheck as broker
from merlin_experiments.phase1.brokers import selfcheck_lanes
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_experiments as reference
from merlin_experiments.phase1.tools import selfcheck as client
from test_reference_experiment_feedback import _fixture

LISTING = '''0x100 0x000000ef jal 200
0x200 0x0200307b .insn r 0x7b,3,1,x0,x0,x0
0x204 0x00008082 ret
0x104 0x00008082 ret
'''


def _world(tmp_path):
    fixture = _fixture(tmp_path)
    refs = tmp_path / 'reference-kernels'
    refs.mkdir()
    (refs / 'trace.txt').write_text(LISTING)
    (refs / 'schedule.py').write_text('def schedule(x):\n    for i in range(4):\n        tile(x, i)\n')
    facts = tmp_path / 'facts.json'
    facts.write_text(json.dumps(dict(interfaces=[dict(name='funct_decode_table', custom_opcode=123,
                                                       legal_funct=[1], names={'1':'load'})])))
    fixture['facts'] = facts
    return fixture


def _request(operation, path, **kw):
    return dict(action=analysis.ACTION, reference_analysis=dict(operation=operation, path=path, **kw))


def test_off_refuses_before_reading_even_identical_raws(tmp_path):
    fixture = _world(tmp_path)
    response = analysis.handle(_request('lift', 'reference-kernels/trace.txt'), workspace=tmp_path,
                               facts=fixture['facts'])
    assert response['status'] == 'denied' and response['reason'] == 'reference_guidance_disabled'
    assert 'report' not in response and response['qualification_issued'] is False


def test_source_and_listing_use_real_core_analyzers_with_unknown_runtime_authority(tmp_path):
    fixture = _world(tmp_path)
    with reference.invocation(fixture['path'], fixture['sha']):
        source = analysis.handle(_request('lift', 'reference-kernels/schedule.py'), workspace=tmp_path)
        assert source['report']['parser'] == 'python.ast'
        assert source['report']['functions'][0]['symbol'] == 'schedule'
        assert source['report']['functions'][0]['loop_count'] == 1
        assert source['report']['semantic_region_status'] == 'unknown'
        listed = analysis.handle(_request('lift', 'reference-kernels/trace.txt'), workspace=tmp_path,
                                  facts=fixture['facts'], compiler_sha256=fixture['compiler'])
    report = listed['report']
    assert listed['status'] == 'analyzed' and listed['all_pass'] is None
    assert report['total'] == 2 and report['listed_instructions'] == 4
    leaf = report['regions'][1]
    assert leaf['selected_decode']['counts']['load'] == 1
    assert leaf['schedule_model']['source_kind'] == 'static_disassembly'
    assert leaf['schedule_model']['complete'] is False
    assert leaf['cca']['dispatch'] is None
    assert 'register_and_memory_values_not_supplied' in leaf['schedule_model']['unknowns']
    assert listed['artifact_to_current_compiler_binding'] == 'unknown'
    assert listed['semantic_equivalence'] == listed['performance'] == 'unknown'
    assert str(tmp_path) not in json.dumps(listed)
    assert len(json.dumps(listed).encode()) < 64 << 10


def test_current_candidate_bytes_reanalyzed_and_reference_paths_cannot_masquerade(tmp_path):
    fixture = _world(tmp_path)
    candidate = tmp_path / 'selfcheck_out/requests/owned/kernel.dis'
    candidate.parent.mkdir(parents=True)
    candidate.write_text('100: 0200307b .insn r 0x7b,3,1,x0,x0,x0\n104: 8082 ret\n')
    with reference.invocation(fixture['path'], fixture['sha']):
        value = analysis.handle(_request('candidate', candidate.relative_to(tmp_path).as_posix()),
            workspace=tmp_path, facts=fixture['facts'], compiler_sha256=fixture['compiler'])
        wrong = analysis.handle(_request('candidate', 'reference-kernels/trace.txt'), workspace=tmp_path)
    assert value['report']['regions'][0]['schedule_model']['command_sites'] == 1
    assert value['report']['input']['sha256'] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert value['current_compiler_sha256'] == fixture['compiler']
    assert value['artifact_to_current_compiler_binding'] == 'unknown'
    assert wrong['status'] == 'unknown'


@pytest.mark.parametrize('path', ['/etc/passwd','../private','reference-kernels/../private',
                                  '.qa_channel/resp_grader.json', 'submission/backend.py'])
def test_reference_inputs_never_select_host_or_grader_surfaces(tmp_path, path):
    fixture = _world(tmp_path)
    with reference.invocation(fixture['path'], fixture['sha']):
        assert analysis.handle(_request('lift', path), workspace=tmp_path)['status'] == 'unknown'


def test_symlink_and_archive_escape_are_refused_without_extraction(tmp_path):
    fixture = _world(tmp_path)
    (tmp_path / 'reference-kernels/link').symlink_to(fixture['facts'])
    archive = tmp_path / 'reference-kernels/source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('schedule.py', 'def actual(x):\n    return x\n')
        z.writestr('../outside.py', 'private')
    with reference.invocation(fixture['path'], fixture['sha']):
        bad = analysis.handle(_request('lift','reference-kernels/link'),workspace=tmp_path)
        member = analysis.handle(_request('lift','reference-kernels/source.zip',member='schedule.py'),
                                 workspace=tmp_path)
        escape = analysis.handle(_request('lift','reference-kernels/source.zip',member='../outside.py'),
                                 workspace=tmp_path)
        roster = analysis.handle(_request('inventory','reference-kernels/source.zip'),workspace=tmp_path)
    assert bad['status'] == escape['status'] == 'unknown'
    assert member['report']['functions'][0]['symbol'] == 'actual'
    assert member['report']['input']['container']['sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert roster['report']['total'] == 2
    assert not (tmp_path / 'outside.py').exists()


def test_diagnostic_dispatch_and_real_standalone_client_roundtrip(tmp_path):
    fixture = _world(tmp_path)
    shim = tmp_path / 'agent_selfcheck.py'
    shutil.copyfile(client.__file__, shim)
    channel = tmp_path / '.qa_channel'
    channel.mkdir()
    command = [sys.executable, str(shim),'--reference-analysis','lift','--reference-path',
               'reference-kernels/trace.txt','--reference-limit','2','--wait-budget','5']
    process = subprocess.Popen(command, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        requests = []
        while not requests and time.monotonic() < deadline:
            requests = list(channel.glob('req_*.json'))
            time.sleep(.01)
        assert requests
        request_path = requests[0]
        assert selfcheck_lanes._diagnostic_request(request_path)
        request = json.loads(request_path.read_text())
        assert request['timeout'] == 1800
        rid = request['request_id']
        with reference.invocation(fixture['path'], fixture['sha']):
            response = broker._reference_analysis_response(request, workspace=tmp_path, facts=fixture['facts'],
                compiler_sha256=fixture['compiler'], rid=rid, started_ns=time.time_ns())
        (channel / f'resp_{rid}.json').write_text(json.dumps(response))
        (channel / f'done_{rid}').write_text('ok')
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr + stdout
        value = json.loads(stdout)
        assert value['schema'] == analysis.SCHEMA and value['status'] == 'analyzed'
        assert value['selfcheck_request_id'] == rid
        assert value['official'] is False and value['qualification_issued'] is False
        assert value['all_pass'] is None and value['numeric_assessment'] == 'not_run'
        assert value['report']['regions'][1]['schedule_model']['command_sites'] == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_real_workspace_alias_reads_verified_snapshot_not_mutable_live_source(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import workspace_transport as transport
    from merlin_experiments.phase1.context import InvocationContext

    owner = tmp_path / 'owner'
    owner.mkdir()
    fixture = _world(owner)
    repo, workspace = tmp_path / 'operator', tmp_path / 'author'
    source = repo / 'raw-reference-grant'
    source.mkdir(parents=True)
    original = 'def admitted(x):\n    return x\n'
    (source / 'schedule.py').write_text(original)
    descriptor = tmp_path / 'descriptor.yaml'
    descriptor.write_text('target: fixture\n')
    context = InvocationContext(repo, descriptor, tmp_path, 'fixture', tmp_path / 'runs', tmp_path / 'reports',
                                tmp_path / 'bundles', ())
    bundle = dict(allowed=[dict(path=str(source), mode='ro', **{'as':'reference-kernels'})])
    manifest = tmp_path / 'original-bundle.yaml'
    manifest.write_text(json.dumps(bundle))
    monkeypatch.setenv('MERLIN_OUT_ROOT', str(tmp_path / 'out'))
    transport.assemble_workspace(bundle, workspace, context=context)
    try:
        alias = workspace / 'reference-kernels'
        assert alias.is_symlink() and alias.readlink() == source
        (source / 'schedule.py').write_text('def not_admitted(x):\n    return 123\n')
        with reference.invocation(fixture['path'], fixture['sha']):
            response = analysis.handle(_request('lift','reference-kernels/schedule.py'), workspace=workspace,
                                       bundle_manifest=manifest, repo=repo)
        assert response['status'] == 'analyzed'
        assert response['report']['functions'][0]['symbol'] == 'admitted'
        assert response['report']['input']['sha256'] == hashlib.sha256(original.encode()).hexdigest()
        assert str(repo) not in json.dumps(response)
        alias.unlink()
        alias.symlink_to(owner / 'reference-kernels')
        with reference.invocation(fixture['path'], fixture['sha']):
            refused = analysis.handle(_request('lift','reference-kernels/schedule.py'), workspace=workspace,
                                      bundle_manifest=manifest, repo=repo)
        assert refused['status'] == 'unknown'
    finally:
        transport.BW.remove_bundle_snapshot(workspace)
