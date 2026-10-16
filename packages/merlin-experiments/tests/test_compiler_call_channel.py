"""Per-invocation existing client transport controls; no model or simulator runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.brokers import compiler_call_channel as C
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2 import campaign as PC

from merlin.common.paths import repo_root
from merlin.targetgen import package_runtime as P
from merlin.targetgen.sandbox import bwrap as B
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3, attach_common_substrate


@pytest.fixture
def original(tmp_path):
    repo = repo_root()
    source = repo / "src"
    workspace = tmp_path / "original"
    workspace.mkdir()
    bundle = {"arm":"raw_baseline","tools":[],"allowed":[],"denied":[],"host_inputs":[]}
    attach_common_substrate(bundle,version=COMMON_SUBSTRATE_V3,python_source_root=source,host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(source / row["path"].removeprefix("merlin/python/"))
    archive = tmp_path / "archive.yaml"
    archive.write_text(yaml.safe_dump(bundle))
    B.materialize_bundle_inputs(workspace,bundle,repo=repo)
    owner = R.CommonTransformService(workspace=workspace,bundle_manifest=archive,repo=repo)
    package = tmp_path / "package"
    package.mkdir()
    (package / "compiler.py").write_text("# source pinned for transport controls\n")
    staging = tmp_path / "call"
    staging.mkdir()
    source = staging / "input.mlir"
    source.write_text("module {}\n")
    return SimpleNamespace(owner=owner,package=package,workspace=staging,source=source,root=tmp_path)


def request(*,action="common-check"):
    rid = "1_2_" + "a"*32
    payload = {"schema":"merlin.common_check_request.v1","before_ir":"before","after_ir":"after","entry":"entry"}
    return rid,{"protocol":3,"request_id":rid,"requester":{},"requested_at_unix_ns":time.time_ns(),
        "deadline_unix_ns":time.time_ns()+5_000_000_000,"action":action,"transform":payload}


def test_dispatch_calls_only_original_service_with_actual_hashes_and_remaining_deadline(tmp_path,monkeypatch):
    owner = R.CommonTransformService.__new__(R.CommonTransformService)
    owner.profile = {"common_substrate":COMMON_SUBSTRATE_V3,"sha256":"a"*64}
    seen = []
    def check(self,payload,*,work_root,timeout_s):
        seen.append((payload,work_root,timeout_s))
        return {"status":"unknown","reason":"synthetic transport boundary; no proof issued"}
    monkeypatch.setattr(R.CommonTransformService,"check",check)
    rid,row = request()
    answer = C._dispatch(owner,row,rid,deadline=time.monotonic()+3,proof_root=tmp_path,
        package_sha256="b"*64,input_sha256="c"*64)
    assert len(seen)==1 and 1 <= seen[0][2] <= 3
    assert seen[0][0] is row["transform"] and seen[0][1]==tmp_path
    assert answer["selfcheck_request_id"]==rid
    assert answer["submission_sha256"]=="b"*64 and answer["invocation_input_sha256"]=="c"*64
    assert answer["source_sha256"]==hashlib.sha256(b"before").hexdigest()
    assert answer["target_sha256"]==hashlib.sha256(b"after").hexdigest()


@pytest.mark.parametrize("change",[
    {"action":"selfcheck"},{"request_id":"another"},{"protocol":2},
    {"semantics":{}},{"deadline_unix_ns":0},{"toolchain":"candidate"},
])
def test_action_identity_policy_overrides_and_expired_requests_never_reach_service(tmp_path,monkeypatch,change):
    owner = R.CommonTransformService.__new__(R.CommonTransformService)
    owner.profile = {"common_substrate":COMMON_SUBSTRATE_V3,"sha256":"a"*64}
    monkeypatch.setattr(R.CommonTransformService,"check",lambda *a,**k:pytest.fail("invalid request dispatched"))
    rid,row = request()
    row.update(change)
    answer = C._dispatch(owner,row,rid,deadline=time.monotonic()+3,proof_root=tmp_path,
        package_sha256="b"*64,input_sha256="c"*64)
    assert answer["status"]=="unknown" and answer["all_pass"] is False
    assert answer["selfcheck_request_id"]==rid


def test_legacy_profile_does_not_gain_authored_check(tmp_path,monkeypatch):
    owner = R.CommonTransformService.__new__(R.CommonTransformService)
    owner.profile = {"common_substrate":"merlin.common_compiler_substrate.v1","sha256":"a"*64}
    monkeypatch.setattr(R.CommonTransformService,"check",lambda *a,**k:pytest.fail("legacy capability widened"))
    rid,row = request()
    answer=C._dispatch(owner,row,rid,deadline=time.monotonic()+3,proof_root=tmp_path,
        package_sha256="b"*64,input_sha256="c"*64)
    assert answer["status"]=="unavailable"


def test_channel_reads_reject_links_oversize_and_write_ignores_response_alias(tmp_path):
    target=tmp_path / "outside"
    target.write_text("private")
    (tmp_path / "linked").symlink_to(target)
    (tmp_path / "huge").write_bytes(b"x"*40)
    descriptor=os.open(tmp_path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        with pytest.raises(OSError):
            C._read(descriptor,"linked",100)
        with pytest.raises(ValueError,match="budget"):
            C._read(descriptor,"huge",20)
        (tmp_path / "response").symlink_to(target)
        C._write(descriptor,"response",{"own":"reply"})
        assert target.read_text()=="private"
        assert json.loads((tmp_path / "response").read_text())=={"own":"reply"}
    finally:
        os.close(descriptor)


def test_actual_host_channel_accepts_existing_client_and_closes_private_owner(original):
    from merlin.transforms.client import check_transform

    value=original
    before=set(value.root.iterdir())
    with C.compiler_call_transport(value.owner,workspace=value.workspace,package=value.package,
            input_path=value.source,command_name="compile",timeout_s=15) as transport:
        assert transport.environment()==(("MERLIN_COMMON_WORKSPACE",str(value.workspace)),)
        answer=check_transform(value.workspace,"module {}","module {}",entry="absent",wait_seconds=10)
        # This fixture deliberately selects no Lean toolchain. Transport success
        # establishes neither mathematical acceptance nor numerical execution.
        assert answer["status"]=="unavailable"
        assert answer["source_sha256"]==hashlib.sha256(b"module {}").hexdigest()
        assert answer["invocation_input_sha256"]==C._sha(value.source)
        assert answer["submission_sha256"]==C._sources(value.package)
        assert not any(str(path) in json.dumps(answer) for path in (value.owner.bundle_manifest,value.owner.workspace))
        assert transport.remaining_timeout() > 0
    assert (value.workspace / ".qa_channel/STOP").is_file()
    assert set(value.root.iterdir())==before


def test_separately_readonly_selected_input_needs_no_parent_grant(original):
    from merlin.transforms.client import check_transform

    value=original
    external=value.root / "readonly-input"
    external.write_text("module {}")
    external.chmod(0o444)
    with C.compiler_call_transport(value.owner,workspace=value.workspace,package=value.package,
            input_path=external,command_name="compile",timeout_s=15):
        answer=check_transform(value.workspace,"module {}","module {}",entry="absent",wait_seconds=10)
        assert answer["status"]=="unavailable" and answer["invocation_input_sha256"]==C._sha(external)
        assert str(external) not in json.dumps(answer)


def test_original_transport_borrows_explicit_cpu_and_reaps_before_release(original, monkeypatch):
    from merlin.common.cpu_pool import selected_pool
    from merlin.transforms.client import check_transform

    value = original
    cpu = next(iter(os.sched_getaffinity(0)))
    root = value.root / "cpu-slots"
    before = os.sched_getaffinity(0)
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(root))
    with C.compiler_call_transport(value.owner, workspace=value.workspace, package=value.package,
                                  input_path=value.source, command_name="compile", timeout_s=15,
                                  borrow_compute=True):
        assert os.sched_getaffinity(0) == {cpu}
        answer = check_transform(value.workspace, "module {}", "module {}", entry="absent", wait_seconds=10)
        assert answer["status"] == "unavailable"  # no selected proof toolchain
    assert os.sched_getaffinity(0) == before
    with selected_pool((cpu,), slot_root=root, timeout_s=0):
        pass  # original broker has closed its duplicate slot before release


def test_actual_original_compiler_call_uses_existing_client_inside_its_box(original,monkeypatch):
    """Real compilation-command/mount/client control; selected proof tool is absent."""
    if shutil.which("bwrap") is None:
        pytest.skip("bwrap is unavailable")
    value=original
    script=value.package / "compiler.py"
    script.write_text('''import hashlib,json,os,sys
from pathlib import Path
from merlin.transforms.client import check_transform
workspace=Path(os.environ["MERLIN_COMMON_WORKSPACE"])
assert workspace != Path.cwd()
response=check_transform(workspace,"module {}","module {}",entry="absent",wait_seconds=10)
assert response["status"]=="unavailable"
assert response["invocation_input_sha256"]==hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
Path(sys.argv[2]).write_text(json.dumps(response))
''')
    def policy(target,workspace,package,*,inputs):
        prefix=["bwrap","--unshare-user","--unshare-pid","--unshare-net","--ro-bind","/usr","/usr",
            "--ro-bind","/lib","/lib","--ro-bind","/lib64","/lib64","--proc","/proc","--dev","/dev",
            "--bind",str(workspace),str(workspace),"--ro-bind",str(package),str(package)]
        for root in sorted({Path(sys.prefix),Path(sys.base_prefix)}):
            prefix += ["--ro-bind",str(root),str(root)]
        executable = Path(sys.executable)
        if executable.is_symlink():
            alias = executable.readlink().parent.parent
            prefix += ["--ro-bind",str(alias.resolve()),str(alias)]
        basic=PC.PackageSandboxPolicy(tuple(prefix),(),(),Path(workspace),Path(package),target,
            "export PYTHONDONTWRITEBYTECODE=1; ")
        return R.active_public_package_policy(basic,target=target)
    monkeypatch.setattr(PC,"package_sandbox_policy",policy)
    monkeypatch.setattr(B,"apply_final_answer_masks",lambda argv,*a,**k:argv)
    monkeypatch.setattr(P,"_resolve_argv",lambda pkg,name,inp,out:[sys.executable,str(script),str(inp),str(out)])
    monkeypatch.setattr(P,"_needs_interpreter",lambda *a:False)
    pkg=SimpleNamespace(directory=value.package,manifest={},integrity_exempt=False)
    destination=value.workspace / "result.json"
    with R.original_compiler_execution(value.owner,target_experiment=None,candidate=value.package,
            work_root=value.root / "private-calls"):
        result=P.active_package_executor().run_entrypoint(pkg,"compile",value.source,destination,timeout=15)
    assert result.returncode==0,result.stderr
    answer=json.loads(destination.read_bytes())
    assert answer["status"]=="unavailable" and answer["submission_sha256"]==C._sources(value.package)
    assert not any(path.name.startswith("original-proof-") for path in value.root.rglob("*"))


def _cleanup_probe(selection, control):
    """Dedicated test host with a native-boundary stub; no proof is issued."""
    def checking(self, request, *, work_root, timeout_s):
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
        (control / "owned-child.json").write_text(json.dumps({"pid":child.pid}))
        time.sleep(30)
        return {"status":"unknown"}
    C.CommonTransformService.apply = checking
    raise SystemExit(C.main(["--selection",str(selection)]))


def test_cancellation_reaps_own_session_proof_child_before_final_identity_work(original):
    """Real Linux ownership/cleanup control with one sleeping native-boundary stub."""
    import multiprocessing

    value = original
    control = value.root / "cleanup-control"
    control.mkdir()
    channel = value.workspace / ".qa_channel"
    channel.mkdir()
    selection = control / "selection.json"
    selection.write_text(json.dumps({"schema":C.SCHEMA,"workspace":str(value.workspace),
        "package":str(value.package),"package_sha256":C._sources(value.package),
        "input_path":str(value.source),"input_sha256":C._sha(value.source),"command_name":"compile",
        "deadline":time.monotonic()+15,"parent_pid":os.getpid(),
        "owner":{"workspace":str(value.owner.workspace),"bundle_manifest":str(value.owner.bundle_manifest),
            "repo":str(value.owner.repo),"archive_sha256":hashlib.sha256(value.owner.bundle_bytes).hexdigest(),
            "snapshot_sha256":value.owner.snapshot_digest,"profile_sha256":value.owner.profile["sha256"],
            "snapshot_marker_sha256":value.owner.snapshot_marker_digest}}))
    process=multiprocessing.get_context("fork").Process(target=_cleanup_probe,args=(selection,control))
    process.start()
    deadline=time.monotonic()+5
    try:
        while not (control / "ready.json").exists():
            assert time.monotonic()<deadline and process.is_alive()
            time.sleep(0.02)
        rid,row=request(action="common-transform")
        row["transform"]={"schema":"merlin.common_transform_request.v1","source_ir":"module {}","options":{}}
        (channel / ("req_"+rid+".json")).write_text(json.dumps(row))
        while not (control / "owned-child.json").exists():
            assert time.monotonic()<deadline and process.is_alive()
            time.sleep(0.02)
        pid=json.loads((control / "owned-child.json").read_bytes())["pid"]
        assert Path(f"/proc/{pid}").exists()
        process.terminate()
        process.join(4)
        assert not process.is_alive() and process.exitcode==0
        closed=json.loads((control / "closed.json").read_bytes())
        assert closed["complete"] is True and closed["reaped_children"]>=1
        assert not Path(f"/proc/{pid}").exists()
    finally:
        if process.is_alive():
            process.kill()
            process.join(2)


def test_phase2_normal_box_reuses_original_explicit_runtime_selection(original,monkeypatch):
    """Ordinary invocation routing control; actual compiler call is stubbed."""
    value=original
    selected=PC.PackageSandboxInputs(PC.TC.ToolchainPaths(value.root,*[str(value.root / str(i)) for i in range(5)]),
        PC.TC.SimToolchain(),"",())
    policy=PC.PackageSandboxPolicy(("not-launched",),(),(),value.workspace,value.package,None,"",
        selected_inputs=selected)
    pkg=SimpleNamespace(directory=value.package,manifest={},integrity_exempt=False)
    seen=[]
    def call(self,actual,name,inp,out,**kwargs):
        assert self.inputs is selected
        assert actual is pkg and inp==value.source
        assert not self.work_root.is_relative_to(value.workspace)
        seen.append(name)
        return subprocess.CompletedProcess([],0,"actual call stub","")
    monkeypatch.setattr(R._Executor,"run_entrypoint",call)
    monkeypatch.setattr(PC,"select_package_sandbox_inputs",lambda *a:pytest.fail("runtime rediscovery"))
    with value.owner.compiler_imports(),PC.boxed_entrypoints(policy):
        result=P.run_entrypoint(pkg,"compile",value.source,value.workspace / "out",timeout=5)
    assert result.returncode==0 and seen==["compile"]


def _lazy_selection(value):
    return {"schema": C.SCHEMA, "workspace": str(value.workspace),
        "package": str(value.package), "package_sha256": C._sources(value.package),
        "input_path": str(value.source), "input_sha256": C._sha(value.source),
        "command_name": "compile", "deadline": time.monotonic() + 15, "parent_pid": os.getpid(),
        "owner": {"workspace": str(value.owner.workspace), "bundle_manifest": str(value.owner.bundle_manifest),
            "repo": str(value.owner.repo), "archive_sha256": hashlib.sha256(value.owner.bundle_bytes).hexdigest(),
            "snapshot_sha256": value.owner.snapshot_digest, "profile_sha256": value.owner.profile["sha256"],
            "snapshot_marker_sha256": value.owner.snapshot_marker_digest}}


def test_idle_channel_readiness_does_not_construct_full_snapshot_service(original, monkeypatch):
    value = original
    channel = value.workspace / '.qa_channel'
    channel.mkdir()
    (channel / 'STOP').write_text('stop')
    control = value.root / 'lazy-control'
    control.mkdir()
    monkeypatch.setattr(C, 'CommonTransformService', lambda **kwargs: pytest.fail('idle SDK scan'))
    C._serve(_lazy_selection(value), control)
    assert json.loads((control / 'ready.json').read_bytes())['ready'] is True


def test_first_request_requires_full_original_service_before_dispatch(original, monkeypatch):
    value = original
    channel = value.workspace / '.qa_channel'
    channel.mkdir()
    rid, row = request()
    (channel / ('req_' + rid + '.json')).write_text(json.dumps(row))
    control = value.root / 'lazy-control'
    control.mkdir()
    calls = []

    def construct(**kwargs):
        calls.append(kwargs)
        raise ValueError('full frozen payload refused')

    monkeypatch.setattr(C, 'CommonTransformService', construct)
    monkeypatch.setattr(C, '_dispatch', lambda *a, **kw: pytest.fail('unverified request dispatched'))
    with pytest.raises(ValueError, match='full frozen payload refused'):
        C._serve(_lazy_selection(value), control)
    assert len(calls) == 1
    assert not (channel / ('resp_' + rid + '.json')).exists()


@pytest.mark.parametrize('changed', ['archive', 'marker'])
def test_lazy_readiness_refuses_mutated_protected_metadata(original, changed):
    value = original
    selection = _lazy_selection(value)
    if changed == 'archive':
        value.owner.bundle_manifest.write_bytes(value.owner.bundle_bytes + b'\n')
    else:
        from merlin.common.bundle_snapshot import MARKER, bundle_snapshot_root
        marker = bundle_snapshot_root(value.owner.workspace) / MARKER
        marker.chmod(0o600)
        marker.write_bytes(marker.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed its original'):
        C._verify_owner_selection(selection['owner'])
