"""Closed compiler data transport and actual private namespace canaries."""

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from merlin_experiments.phase1 import candidate_products as P
from merlin_experiments.phase2 import campaign, qualification_policy

from merlin.common import invocation_record
from merlin.targetgen.sandbox import preflight
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
from merlin.targetgen.target_experiment import load_target_experiment


@pytest.fixture
def selected(tmp_path, monkeypatch):
    """Actual copied tiny Python selection, not compiler/hardware qualification."""
    repo = tmp_path / "operator"
    repo.mkdir()
    descriptor = repo / "target.yaml"
    descriptor.write_text("target: product_fixture\nsim_via: synthetic\n")
    target = load_target_experiment(descriptor, source_root=repo)
    runtime = tmp_path / "runtime"
    paths = TC.ToolchainPaths(
        repo,
        str(runtime / "venv"),
        str(runtime / "llvm"),
        str(runtime / "compat"),
        str(runtime / "clang"),
        str(runtime / "uv"),
        python_import_roots=(),
    )
    for path in (paths.venv, paths.llvm, paths.compat_lib, paths.clang_bin, paths.clang_resource, paths.uv_python):
        Path(path).mkdir(parents=True, exist_ok=True)
    (Path(paths.venv) / "bin").mkdir()
    (Path(paths.llvm) / "bin").mkdir()
    (Path(paths.llvm) / "bin/runtime-data.bin").write_bytes(b"selected ordinary runtime resource")
    bundle = Path(paths.venv) / "private-bundle"
    bundle.mkdir()
    (bundle / "secret").write_text("SYNTHETIC_INVOCATION_PRIVATE_INPUT")
    shutil.copyfile(Path("/usr/bin/python3").resolve(), Path(paths.venv) / "bin/python")
    (Path(paths.venv) / "bin/python").chmod(0o755)
    secret = repo / "private-answer.txt"
    secret.write_text("SYNTHETIC_PRIVATE_ANSWER")
    inputs = campaign.PackageSandboxInputs(
        paths,
        TC.SimToolchain(),
        "",
        (AnswerSurface("private fixture", secret, "file", "golden"),),
    )
    monkeypatch.setattr(
        TC, "required_tool_probes", lambda *a, **k: (TC.ToolProbe("selected-python", "python --version", paths.venv),)
    )
    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    record = qualification_policy.freeze(frozen_root, target, inputs)
    frozen = qualification_policy.restore(frozen_root, record)
    workspaces = tmp_path / "workspaces"
    workspaces.mkdir()
    bwrap = Path(shutil.which("bwrap") or "/usr/bin/bwrap").resolve()
    transport = P.CandidateProductTransport(
        target,
        inputs=frozen,
        bwrap_binary=bwrap,
        private_paths=(secret,),
        workspace_root=workspaces,
    )
    source = tmp_path / "prepared.mlir"
    source.write_text("module { /* exact prepared source */ }\n")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    return transport, candidate, source, secret, tmp_path


def invoke(selected, program, *, products=(P.ProductSpec("product.bin", 4096),), **options):
    transport, candidate, source, _, root = selected
    (candidate / "compile.py").write_text(program)
    return transport.compile(
        candidate,
        source,
        argv=("{python}", "{package}/compile.py", "{input_mlir}", "{output_dir}"),
        products=products,
        substitutions=transport.tool_substitutions(),
        destination=root / "retained",
        timeout_s=15,
        **options,
    )


def require_namespace():
    probe = preflight.probe_sandbox(network_isolation=True)
    if not probe.usable:
        pytest.skip("actual networkless bwrap unavailable: " + probe.reason)


def test_actual_copied_candidate_has_no_private_environment_parent_or_network(selected, monkeypatch):
    require_namespace()
    transport, candidate, _, secret, root = selected
    monkeypatch.setenv("MERLIN_CORE_ATEN_PRIVATE_ROOT", str(secret.parent))
    monkeypatch.setenv("PYTHONPATH", "/private/ungranted/grader")
    parent_marker = root / "parent-only.txt"
    parent_marker.write_text("another private file")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    program = f"""
import os, socket, sys
from pathlib import Path
assert "MERLIN_CORE_ATEN_PRIVATE_ROOT" not in os.environ
assert "/private/ungranted/grader" not in os.environ.get("PYTHONPATH", "")
assert not Path({str(secret)!r}).is_file() or not Path({str(secret)!r}).read_bytes()
assert not Path({str(parent_marker)!r}).exists()
assert not Path({str(candidate)!r}).exists()
assert os.getppid() < 100
assert not Path("/proc/{os.getpid()}/environ").exists()
s = socket.socket()
s.settimeout(0.2)
try:
 s.connect(("127.0.0.1", {listener.getsockname()[1]}))
except OSError:
 pass
else:
 raise AssertionError("network grant leaked")
source = Path(sys.argv[1]).read_bytes()
try:
 Path(__file__).write_text("mutated compiler")
except OSError:
 pass
else:
 raise AssertionError("candidate was writable")
try:
 Path(sys.argv[1]).write_bytes(b"mutated prepared source")
except OSError:
 pass
else:
 raise AssertionError("prepared source was writable")
Path(sys.argv[2], "product.bin").write_bytes(source)
"""
    try:
        receipt = invoke(selected, program)
    finally:
        listener.close()
    assert not any(
        getattr(module, "__file__", "") and str(getattr(module, "__file__", "")).startswith(receipt["workspace"])
        for module in sys.modules.values()
    )
    product = receipt["products"]["product.bin"]
    assert Path(product["path"]).read_bytes() == selected[2].read_bytes()
    observed = invocation_record.verify(Path(receipt["invocation_record"]["path"]))
    assert observed["environment"]["keys"] == []
    assert observed["executable"]["path"] == str(transport.bwrap_binary)
    assert any(pin["path"].startswith(receipt["workspace"] + "/compiler/") for pin in observed["dependencies"])


@pytest.mark.parametrize(
    "defect", ["symlink", "hardlink", "fifo", "extra", "cache", "directory", "oversize", "missing"]
)
def test_actual_malformed_products_refuse_and_keep_attempt(selected, defect):
    require_namespace()
    action = {
        "symlink": "path.symlink_to(sys.argv[1])",
        "hardlink": "other=out/'other'; other.write_bytes(b'aliased'); os.link(other,path)",
        "fifo": "os.mkfifo(path)",
        "extra": "path.write_bytes(b'ok'); (out/'unexpected').write_bytes(b'extra')",
        "cache": (
            "path.write_bytes(b'ok'); (out/'__pycache__').mkdir(); (out/'__pycache__/hidden.pyc').write_bytes(b'extra')"
        ),
        "directory": "path.write_bytes(b'ok'); (out/'empty').mkdir()",
        "oversize": "path.write_bytes(b'x'*4097)",
        "missing": "pass",
    }[defect]
    products = (P.ProductSpec("product.bin", 4096),)
    if defect == "hardlink":
        products += (P.ProductSpec("other", 4096),)
    with pytest.raises((ValueError, OSError)):
        invoke(
            selected,
            "import os,sys\nfrom pathlib import Path\nout=Path(sys.argv[2]);path=out/'product.bin'\n" + action,
            products=products,
        )
    observations = list((selected[4] / "retained").rglob("invocation.json"))
    assert len(observations) == 1
    with pytest.raises(ValueError):
        invocation_record.verify(observations[0])


def test_actual_timeout_kills_detached_namespace_descendant(selected):
    require_namespace()
    transport, candidate, source, _, root = selected
    script = candidate / "compile.py"
    script.write_text("""
import subprocess,sys,time
from pathlib import Path
out=Path(sys.argv[2])
(out/'started').write_text('started')
child="import time;from pathlib import Path;time.sleep(2);Path("+repr(str(out/'late'))+").write_text('escaped')"
subprocess.Popen([sys.executable,'-c',child],start_new_session=True,
                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
time.sleep(30)
""")
    with pytest.raises((TimeoutError, subprocess.TimeoutExpired)):
        transport.compile(
            candidate,
            source,
            argv=("{python}", "{package}/compile.py", "{input_mlir}", "{output_dir}"),
            products=(P.ProductSpec("product.bin", 4096),),
            substitutions=transport.tool_substitutions(),
            destination=root / "retained",
            timeout_s=1,
        )
    stage = next(transport.workspace_root.iterdir())
    assert (stage / "products/started").exists(), "timeout must exercise an actually started compiler"
    time.sleep(2.1)
    assert not (stage / "products/late").exists()


def test_actual_success_closes_detached_namespace_descendant(selected):
    require_namespace()
    program = """
import subprocess,sys
from pathlib import Path
out=Path(sys.argv[2])
child="import time;from pathlib import Path;time.sleep(0.5);Path("+repr(str(out/'late'))+").write_text('escaped')"
subprocess.Popen([sys.executable,'-c',child],start_new_session=True,
                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
(out/'product.bin').write_bytes(b'actual')
"""
    receipt = invoke(selected, program)
    time.sleep(0.6)
    assert not (Path(receipt["workspace"]) / "products/late").exists()
    invocation_record.verify(Path(receipt["invocation_record"]["path"]))


@pytest.mark.parametrize("owner", ["candidate", "prepared", "runtime"])
def test_actual_source_or_runtime_drift_during_compile_refuses(selected, owner):
    require_namespace()
    transport, candidate, source, _, root = selected
    tool = candidate / "compile.py"
    tool.write_text("""
import sys,time
from pathlib import Path
out=Path(sys.argv[2]); (out/'product.bin').write_bytes(b'actual')
time.sleep(0.6)
""")
    started = threading.Event()
    errors = []

    def change():
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                stages = list(transport.workspace_root.iterdir())
                if stages and (stages[0] / "products/product.bin").exists():
                    started.set()
                    if owner == "candidate":
                        (candidate / "new.py").write_text("NEW = 1\n")
                    elif owner == "prepared":
                        source.write_text("module { /* changed */ }")
                    else:
                        snapshot = P.BW.bundle_snapshot_root(
                            transport.inputs.root / transport.inputs.record["workspace"]
                        )
                        member = next(path for path in snapshot.rglob("runtime-data.bin") if path.is_file())
                        member.chmod(0o600)
                        member.write_bytes(b"changed frozen tool")
                    return
                time.sleep(0.01)
            raise AssertionError("compiler did not reach the actual execution boundary")
        except BaseException as error:
            errors.append(error)

    mutation = threading.Thread(target=change)
    mutation.start()
    try:
        with pytest.raises((ValueError, RuntimeError)):
            transport.compile(
                candidate,
                source,
                argv=("{python}", "{package}/compile.py", "{input_mlir}", "{output_dir}"),
                products=(P.ProductSpec("product.bin", 4096),),
                substitutions=transport.tool_substitutions(),
                destination=root / "retained",
                timeout_s=15,
            )
    finally:
        mutation.join(timeout=11)
    assert started.is_set() and not errors and not mutation.is_alive()


def test_actual_output_budget_and_private_diagnostic_refusal(selected):
    require_namespace()
    with pytest.raises(ValueError, match="output exceeds") as error:
        invoke(selected, "print('SYNTHETIC_PRIVATE_DIAGNOSTIC'*10000)", max_output_bytes=256)
    assert "SYNTHETIC_PRIVATE_DIAGNOSTIC" not in str(error.value)


def test_actual_retained_product_drift_invalidates_invocation(selected):
    require_namespace()
    receipt = invoke(
        selected, "import sys;from pathlib import Path;Path(sys.argv[2],'product.bin').write_bytes(b'actual')"
    )
    product = Path(receipt["products"]["product.bin"]["path"])
    product.chmod(0o600)
    product.write_bytes(b"forged")
    with pytest.raises(ValueError, match="changed"):
        invocation_record.verify(Path(receipt["invocation_record"]["path"]))


def test_actual_later_host_lowering_preserves_compiler_input_identity(selected):
    require_namespace()
    _, _, prepared, _, _ = selected
    receipt = invoke(
        selected,
        "import sys;from pathlib import Path;"
        "Path(sys.argv[2],'product.bin').write_bytes(Path(sys.argv[1]).read_bytes())",
    )
    record = Path(receipt["invocation_record"]["path"])
    original = prepared.read_bytes()
    prepared.write_text("module { /* later host lowering */ }\n")
    observed = invocation_record.verify(record)
    retained_input = Path(observed["inputs"][0]["path"])
    assert retained_input != prepared and retained_input.read_bytes() == original
    retained_input.chmod(0o600)
    retained_input.write_text("module { /* forged compiler input */ }\n")
    with pytest.raises(ValueError, match="changed"):
        invocation_record.verify(record)


def test_actual_candidate_preserves_empty_directory_and_execution_modes(selected):
    require_namespace()
    _, candidate, _, _, _ = selected
    (candidate / "empty").mkdir(mode=0o700)
    (candidate / "helper").write_text("ordinary executable source")
    (candidate / "helper").chmod(0o751)
    receipt = invoke(
        selected,
        """
import sys,stat
from pathlib import Path
root=Path(__file__).parent
assert (root/'empty').is_dir()
assert stat.S_IMODE((root/'empty').stat().st_mode)==0o500
assert stat.S_IMODE((root/'helper').stat().st_mode)==0o551
Path(sys.argv[2],'product.bin').write_bytes(b'actual')
""",
    )
    assert receipt["package"]["directories"]["empty"] == 0o500
    assert receipt["package"]["members"]["helper"]["mode"] == 0o551


def test_actual_invocation_private_bundle_is_masked_inside_frozen_runtime_grant(selected):
    require_namespace()
    transport = selected[0]
    private = Path(transport.inputs.record["inputs"]["paths"]["venv"]) / "private-bundle"
    receipt = invoke(
        selected,
        f"""
import sys
from pathlib import Path
assert not Path({str(private / "secret")!r}).exists()
Path(sys.argv[2],'product.bin').write_bytes(b'actual')
""",
        private_paths=(private,),
    )
    assert str(private) in receipt["private_paths"]


def test_invocation_private_hardlink_is_refused_before_execution(selected, monkeypatch):
    transport, candidate, source, _, root = selected
    private = root / "invocation-private"
    private.mkdir()
    (private / "golden").write_text("SYNTHETIC_PRIVATE_VALUE")
    os.link(private / "golden", candidate / "stolen")
    monkeypatch.setattr(P, "_capture", lambda *a, **k: pytest.fail("private alias reached compiler"))
    with pytest.raises((ValueError, RuntimeError)):
        transport.compile(
            candidate,
            source,
            argv=("{python}",),
            products=(P.ProductSpec("product.bin", 100),),
            substitutions=transport.tool_substitutions(),
            destination=root / "retained",
            timeout_s=10,
            private_paths=(private,),
        )


@pytest.mark.parametrize("name", ["", "../escape", "/absolute", "x/../y", "x//y", "./x", "x\\y"])
def test_unsafe_product_roster_refuses_without_execution(name):
    with pytest.raises(ValueError):
        P.ProductSpec(name, 100)


def test_live_or_reconstructed_selection_is_not_a_transport(tmp_path):
    with pytest.raises(ValueError, match="existing frozen"):
        P.CandidateProductTransport(
            None, inputs={}, bwrap_binary=tmp_path / "bwrap", private_paths=(), workspace_root=tmp_path
        )


def test_private_hardlink_intake_refuses_before_copy_or_execution(selected, monkeypatch):
    transport, candidate, source, secret, root = selected
    os.link(secret, candidate / "stolen.txt")
    monkeypatch.setattr(P, "_capture", lambda *a, **k: pytest.fail("unadmitted compiler executed"))
    with pytest.raises((RuntimeError, ValueError)):
        transport.compile(
            candidate,
            source,
            argv=("{python}",),
            products=(P.ProductSpec("product.bin", 100),),
            substitutions=transport.tool_substitutions(),
            destination=root / "retained",
            timeout_s=10,
        )
    assert not list(transport.workspace_root.iterdir())


def test_runtime_record_mutation_refuses_before_execution(selected, monkeypatch):
    transport = selected[0]
    transport.inputs.record["env_prefix"] += " dangerous"
    monkeypatch.setattr(P, "_capture", lambda *a, **k: pytest.fail("runtime drift executed"))
    with pytest.raises(ValueError, match="selection changed"):
        transport.tool_substitutions()
