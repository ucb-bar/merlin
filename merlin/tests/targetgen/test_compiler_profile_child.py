"""Original profile restoration in a real Python child, without model execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2 import campaign as C
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common import bundle_snapshot as BW
from merlin.common.paths import repo_root
from merlin.targetgen import compiler_profile_admission as API
from merlin.targetgen import package_runtime as P
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
from merlin.targetgen.target_experiment import load_target_experiment
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3, attach_common_substrate

_CONTROL = """import json,os,sys
from pathlib import Path
from merlin.targetgen import package_runtime as P, compiler_profile_admission as API
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD
from types import SimpleNamespace
spec = Path(sys.argv[2]); raw = spec.read_bytes()
with CHILD.restore_model_child(int(sys.argv[1]), spec_path=spec, spec_bytes=raw) as receipt:
    job=json.loads(raw)
    assert P.active_package_executor() is not None
    assert API.active_compiler_imports().permits('merlin.transforms.normalization')
    assert not API.active_compiler_imports().permits('merlin.verify.lean_checked')
    P.integrity_scan(SimpleNamespace(directory=Path(job['package_dir']),integrity_exempt=False,manifest={}))
    assert not Path('/proc/self/fd/'+sys.argv[1]).exists()
print(json.dumps({'status':'source-scope-control-only','original_compiler_child':receipt}))
"""


@pytest.fixture
def original(tmp_path, monkeypatch):
    repo = repo_root()
    roots = [repo / "src", *sorted((repo / "packages").glob("*/src"))]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, roots)))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V3, python_source_root=repo / "src", host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(repo / "src" / row["path"].removeprefix("merlin/python/"))
    manifest = tmp_path / "input_bundle_manifest.yaml"
    manifest.write_text(yaml.safe_dump(bundle))
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo)
    owner = CommonTransformService(workspace=workspace, bundle_manifest=manifest, repo=repo)
    candidate = workspace / "submission"
    candidate.mkdir()
    (candidate / "tool.py").write_text("from merlin.transforms.normalization import simplify_integers\n")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: scope_fixture\ncapsule_corpus: capsules\n")
    target = load_target_experiment(descriptor, source_root=repo)
    # Explicit typed tool selection is carried unchanged; no namespace/tool is
    # executed or qualified by these import-scope controls.
    paths = TC.ToolchainPaths(repo, str(repo), str(repo), str(repo), str(repo), str(repo), ())
    inputs = C.PackageSandboxInputs(paths, TC.SimToolchain(probes=(TC.ToolProbe("diagnostic", "true"),)), "", ())
    private = tmp_path / "private"
    private.mkdir()
    spec = private / "spec.json"
    spec.write_text(
        json.dumps({"package_dir": str(candidate), "target": target.target, "capsule": {"name": "synthetic"}})
    )
    frozen = next(
        path
        for row in bundle["allowed"]
        if (path := BW.snapshot_path(BW.bundle_snapshot_root(workspace), Path(row["path"]), repo)).is_file()
        and path.stat().st_size > 1
    )
    return SimpleNamespace(
        owner=owner,
        candidate=candidate,
        target=target,
        inputs=inputs,
        private=private,
        spec=spec,
        frozen=frozen,
    )


def _scope(original):
    return R.original_compiler_execution(
        original.owner,
        target_experiment=original.target,
        candidate=original.candidate,
        work_root=original.private / "executor",
        sandbox_inputs=original.inputs,
    )


def _run(transport, original, *, environment=None, script=_CONTROL):
    return subprocess.run(
        [sys.executable, "-c", script, str(transport.fd), str(original.spec)],
        pass_fds=transport.pass_fds,
        capture_output=True,
        text=True,
        timeout=20,
        env=dict(os.environ) if environment is None else environment,
    )


def _additional_library(original):
    from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library

    root = original.private / "additional"
    (root / "merlin").mkdir(parents=True)
    (root / "merlin/__init__.py").write_text("")
    (root / "merlin/portable.py").write_text("def lower(value):\n return value + 1\n")
    (root / "private-answer.txt").write_text("private leaf is not granted")
    contract = freeze_compiler_library(
        root,
        review_id="child-scope-test",
        public_modules=("merlin.portable",),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
    )
    return CompilerLibrarySelection(contract, root)


def test_fresh_child_reconstructs_same_reviewed_library_without_changing_original_profile(original):
    from merlin.targetgen.compiler_library import active_compiler_library, scoped_compiler_library

    selection = _additional_library(original)
    (original.candidate / "tool.py").write_text("from merlin.portable import lower\n")
    script = _CONTROL.replace(
        "    job=json.loads(raw)",
        """    job=json.loads(raw)
    from merlin.targetgen.compiler_library import active_compiler_library
    library=active_compiler_library()
    assert library is not None and library.contract.review_id == 'child-scope-test'
    assert not API.active_compiler_imports().permits('merlin.portable')
    assert 'private-answer.txt' not in {member.path for member in library.contract.members}
    assert receipt['schema'] == CHILD.LIBRARY_SCHEMA
    assert receipt['compiler_library_sha256'] == CHILD._digest(library.record())""",
    )
    with scoped_compiler_library(selection), _scope(original):
        profile_before = original.owner.profile.copy()
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            assert transport.document["schema"] == CHILD.LIBRARY_SCHEMA
            assert transport.document["compiler_library"] == selection.record()
            result = _run(transport, original, script=script)
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))
            assert active_compiler_library() is selection
        assert original.owner.profile == profile_before
    assert active_compiler_library() is None


@pytest.mark.parametrize("change", ["before", "during", "missing", "legacy_upgrade"])
def test_sealed_child_rejects_stale_missing_or_legacy_library_injection(original, change):
    from merlin.targetgen.compiler_library import scoped_compiler_library

    selection = _additional_library(original)
    with scoped_compiler_library(selection), _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            script = _CONTROL
            if change == "before":
                (selection.root / "merlin/portable.py").write_text("changed = True\n")
            elif change == "during":
                script = script.replace(
                    "    job=json.loads(raw)",
                    "    job=json.loads(raw)\n"
                    "    from merlin.targetgen.compiler_library import active_compiler_library\n"
                    "    (active_compiler_library().root/'merlin/portable.py').write_text('changed = True\\n')",
                )
            else:
                document = dict(transport.document)
                if change == "missing":
                    document.pop("compiler_library")
                else:
                    document["schema"] = CHILD.SCHEMA
                with CHILD._sealed_transport(
                    document, executor=transport.executor, boundary=transport.boundary
                ) as forged:
                    result = _run(forged, original)
                assert result.returncode != 0 and "original parent/spec" in result.stderr
                return
            result = _run(transport, original, script=script)
            assert result.returncode != 0 and "library" in result.stderr
            # Restore selected bytes so the parent scope can close normally;
            # the failed child never issues a successful result receipt.
            (selection.root / "merlin/portable.py").write_text("def lower(value):\n return value + 1\n")


def test_actual_fresh_interpreter_restores_exact_original_profile_source_and_sealed_scope(original):
    with _scope(original):
        assert CHILD.model_child_temporary_root().is_relative_to(original.private)
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            with pytest.raises(OSError):
                os.write(transport.fd, b"changed")
            result = _run(transport, original)
            assert result.returncode == 0, result.stderr
            checked = transport.verify_result(json.loads(result.stdout))
            assert checked["status"] == "source-scope-control-only"
            assert checked["original_compiler_child"]["candidate_sha256"] == P.active_package_executor().source_sha256
    assert P.active_package_executor() is None and API.active_compiler_imports() is None
    assert list((original.private / "compiler").iterdir()) == []


def test_child_setup_batches_two_full_reads_then_reopens_original_checks(original):
    prefix = """from pathlib import Path
from merlin.common import bundle_snapshot as BW
reads = []
digest = BW.hashlib.file_digest
def observe(stream, *args, **kwargs):
    if Path(stream.name) == Path(PAYLOAD):
        reads.append(stream.name)
    return digest(stream, *args, **kwargs)
BW.hashlib.file_digest = observe
""".replace("PAYLOAD", repr(str(original.frozen)))
    script = prefix + _CONTROL.replace(
        "    assert P.active_package_executor() is not None",
        """    assert len(reads) == 2
    assert BW._ACTIVE_VERIFICATION.get() is None
    CHILD._owner_record(P.active_package_executor().owner)
    assert len(reads) == 3
    assert P.active_package_executor() is not None""",
    )
    with _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec,
            environment=dict(os.environ),
            package_dir=original.candidate,
        ) as transport:
            result = _run(transport, original, script=script)
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))


def test_completed_case_checkpoint_does_not_add_snapshot_reads_and_expires_with_worker(original):
    prefix = """from pathlib import Path
from merlin.common import bundle_snapshot as BW
reads=[]
digest=BW.hashlib.file_digest
def measured(stream,*args,**kwargs):
    if Path(stream.name)==Path(PAYLOAD): reads.append(stream.name)
    return digest(stream,*args,**kwargs)
BW.hashlib.file_digest=measured
""".replace("PAYLOAD", repr(str(original.frozen)))
    script = prefix + _CONTROL.replace(
        "    assert P.active_package_executor() is not None",
        """    assert len(reads)==2
    with CHILD.restored_case_checkpoint() as closed:
        assert closed=={}
    assert closed['original_compiler_child']==receipt
    assert closed['worker']['pid']==os.getpid()
    assert len(reads)==2, 'case checkpoint repeated a whole snapshot scan'
    assert P.active_package_executor() is not None""",
    )
    with _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            result = _run(transport, original, script=script)
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))


@pytest.mark.parametrize("when", ["setup", "grading"])
def test_child_same_size_restored_mtime_mutation_cannot_cross_its_boundary(original, when):
    mutation = """def mutate():
    path = Path(PAYLOAD)
    metadata = path.stat()
    path.chmod(0o600)
    with path.open('r+b') as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
""".replace("PAYLOAD", repr(str(original.frozen)))
    prefix = "import contextlib, os\nfrom pathlib import Path\n" + mutation
    if when == "setup":
        prefix += """from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
original_lifetime = R._original_execution_lifetime
@contextlib.contextmanager
def altered_lifetime(*args, **kwargs):
    with original_lifetime(*args, **kwargs):
        mutate()
        yield
R._original_execution_lifetime = altered_lifetime
"""
        script = prefix + _CONTROL.replace(
            "    assert P.active_package_executor() is not None",
            "    print('GRADER ENTERED', flush=True)",
        )
    else:
        script = prefix + _CONTROL.replace(
            "    assert P.active_package_executor() is not None",
            """    print('GRADER ENTERED', flush=True)
    mutate()
    CHILD._owner_record(P.active_package_executor().owner)
    print('MUTATED OWNER ACCEPTED', flush=True)""",
        )
    with pytest.raises((RuntimeError, ValueError), match="content differs"):
        with _scope(original):
            with CHILD.prepare_model_child(
                spec_path=original.spec,
                environment=dict(os.environ),
                package_dir=original.candidate,
            ) as transport:
                result = _run(transport, original, script=script)
                assert result.returncode != 0 and "content differs" in result.stderr
                assert "MUTATED OWNER ACCEPTED" not in result.stdout
                if when == "setup":
                    assert not result.stdout.strip()
                else:
                    assert "GRADER ENTERED" in result.stdout


def test_child_archive_mutation_after_setup_cannot_yield_grader_capability(original):
    prefix = """import contextlib
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
original_lifetime = R._original_execution_lifetime
@contextlib.contextmanager
def changed_archive(service, **kwargs):
    with original_lifetime(service, **kwargs):
        service.bundle_manifest.write_bytes(service.bundle_manifest.read_bytes()+b'\\n')
        yield
R._original_execution_lifetime = changed_archive
"""
    script = prefix + _CONTROL.replace(
        "    assert P.active_package_executor() is not None",
        "    print('GRADER ENTERED', flush=True)",
    )
    with pytest.raises(ValueError, match="changed"):
        with _scope(original):
            with CHILD.prepare_model_child(
                spec_path=original.spec,
                environment=dict(os.environ),
                package_dir=original.candidate,
            ) as transport:
                result = _run(transport, original, script=script)
                assert result.returncode != 0 and "authority changed" in result.stderr
                assert not result.stdout.strip()


def test_large_original_runtime_selection_restores_through_sealed_bounded_fd(original):
    surfaces = tuple(
        AnswerSurface("private runtime", original.private / f"member-{index:05d}", "file", "hidden")
        for index in range(3000)
    )
    original.inputs = replace(original.inputs, surfaces=surfaces)
    with _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            assert os.fstat(transport.fd).st_size <= CHILD.MAX_BYTES
            assert len(json.dumps(transport.document).encode()) > CHILD.MAX_BYTES
            result = _run(
                transport,
                original,
                script=_CONTROL.replace(
                    "assert P.active_package_executor() is not None",
                    "assert len(P.active_package_executor().inputs.surfaces) == 3000",
                ),
            )
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))


@pytest.mark.parametrize("payload", ["oversized", "truncated", "suffix"])
def test_compressed_selection_refuses_expansion_and_unclosed_streams(payload):
    compressed = (
        zlib.compress(b"x" * (CHILD.MAX_SELECTION_BYTES + 1)) if payload == "oversized" else zlib.compress(b"{}")
    )
    if payload == "truncated":
        compressed = compressed[:-1]
    elif payload == "suffix":
        compressed += b"unbound trailing bytes"
    with pytest.raises(StageGateError, match="closed transport bound"):
        CHILD._decode_selection(CHILD._COMPRESSED_SELECTION + compressed)


@pytest.mark.parametrize("change", ["environment", "source", "descriptor", "spec_bytes"])
def test_child_rejects_changed_original_inputs_before_execution(original, change):
    with _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            environment, script = dict(os.environ), _CONTROL
            if change == "environment":
                environment["OTHER_UNSEALED_SELECTION"] = "changed"
            elif change == "source":
                (original.candidate / "tool.py").write_text("another source revision")
            elif change == "descriptor":
                original.target.path.write_text("target: another_fixture\n")
            elif change == "spec_bytes":
                script = _CONTROL.replace("raw = spec.read_bytes()", "raw = spec.read_bytes()+b' '")
            result = _run(transport, original, environment=environment, script=script)
            assert result.returncode != 0
            assert "StageGateError" in result.stderr
            assert not result.stdout.strip()


def test_child_job_cannot_select_another_target_even_inside_a_valid_seal(original):
    original.spec.write_text(json.dumps({"package_dir": str(original.candidate), "target": "unrelated_fixture"}))
    with _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            result = _run(transport, original)
            assert result.returncode != 0 and "another original compiler or target" in result.stderr


def test_foreign_executor_and_import_only_context_cannot_silently_drop(original):
    with P.scoped_package_executor(SimpleNamespace(run_entrypoint=lambda: None, build_package=lambda: None)):
        with pytest.raises(StageGateError, match="foreign or incomplete"):
            CHILD.model_child_temporary_root()
    with original.owner.compiler_imports():
        with pytest.raises(StageGateError, match="foreign or incomplete"):
            CHILD.model_child_temporary_root()


def test_saved_child_result_cannot_replace_original_completion(original):
    with _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            with pytest.raises(StageGateError, match="did not complete"):
                transport.verify_result({"status": "pass", "original_compiler_child": {"status": "verified"}})


def test_legacy_unselected_path_keeps_exact_original_behavior(original):
    assert CHILD.model_child_temporary_root() is None
    with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ), package_dir=None) as selected:
        assert selected is None


def test_unsealed_or_writable_descriptors_cannot_select_execution(original):
    fd = CHILD._memfd()
    try:
        os.write(fd, b"{}")
        with pytest.raises(StageGateError, match="immutable and sealed"):
            with CHILD.restore_model_child(fd, spec_path=original.spec, spec_bytes=original.spec.read_bytes()):
                pytest.fail("unsealed capability entered")
    finally:
        os.close(fd)


def test_ordinary_budget_supervisor_and_child_entry_restore_the_original_scope(original, monkeypatch):
    from merlin.targetgen import capsule_runner as CR

    # Actual supervisor and CLI child, with the numerical grader deliberately
    # replaced by an import-scope control. No model or simulator is executed.
    script = """import sys
from pathlib import Path
from types import SimpleNamespace
from merlin.targetgen import capsule_runner as CR, package_runtime as P, compiler_profile_admission as API
def control(capsule, **kwargs):
    assert P.active_package_executor() is not None
    assert API.active_compiler_imports().permits('merlin.transforms.normalization')
    P.integrity_scan(SimpleNamespace(directory=Path(kwargs['package_dir']),integrity_exempt=False,manifest={}))
    return {'capsule':capsule['name'],'status':'source-scope-control-only'}
CR._grade_candidate_model_capsule_inline=control
raise SystemExit(CR.main(sys.argv[1:]))
"""
    monkeypatch.setattr(CR, "_CHILD_PREAMBLE", CR._CHILD_GUARD + script)
    with _scope(original):
        result = CR._grade_model_capsule_unlocked(
            {"name": "synthetic_scope", "kind": "model"},
            target=original.target.target,
            timeout=10,
            budget_s=10,
            package_dir=original.candidate,
            candidate_context={"scope_control": True},
            candidate_source_pinned=CR._SUITE_MODEL_PIN,
        )
    assert result["status"] == "source-scope-control-only"
    assert result["original_compiler_child"]["scope"].endswith("no numerical or timing authority")


def test_serialized_resource_claims_cannot_replace_actual_kernel_controls(original):
    with _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            document = dict(transport.document)
            document["resources"] = {
                "cpu_ids": [0],
                "ram_bytes": 1,
                "disk_bytes": 1,
                "writable_roots": [str(original.private)],
                "boundary_sha256": "candidate-assertion",
            }
            raw = json.dumps(document).encode()
            writable = CHILD._memfd()
            readonly = None
            try:
                os.write(writable, raw)
                CHILD.fcntl.fcntl(writable, CHILD._ADD_SEALS, CHILD._SEALS)
                readonly = os.open(f"/proc/self/fd/{writable}", os.O_RDONLY | os.O_CLOEXEC)
                fake = CHILD.ModelChildTransport(
                    fd=readonly, document=document, executor=transport.executor, boundary=None
                )
                result = _run(fake, original)
                assert result.returncode != 0 and "StageGateError" in result.stderr
                assert not result.stdout.strip()
            finally:
                os.close(writable)
                if readonly is not None:
                    os.close(readonly)
