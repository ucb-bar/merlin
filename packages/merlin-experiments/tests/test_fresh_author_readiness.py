"""No-model normal readiness controls confer no fresh origin or target runtime."""

import json
import os
import shlex
import subprocess
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import component_origin as O
from merlin_experiments.phase1 import component_tool_readiness as T
from merlin_experiments.phase1.providers import codex_agent as CA
from merlin_experiments.phase2.component_experiment import ApprovedInput, RuntimeGrant, materialize_component_view
from merlin_experiments.phase2.component_runtime import inventory_runtime
from merlin_experiments.phase2.contracts import StageGateError, sha256_file

from merlin.common import invocation_record
from merlin.common.digest import sha256_bytes
from merlin.targetgen.compiler_library import freeze_compiler_library


def test_missing_runtime_inputs_refuse_before_either_readiness_or_paid_author(tmp_path, monkeypatch):
    values = {field.name: None for field in fields(O.FreshPhase1Inputs)}
    values.update(candidate=tmp_path / "candidate", output=tmp_path / "output")
    inputs = O.FreshPhase1Inputs(**values)
    monkeypatch.setattr(O, "probe_native_author_tools", lambda *_: pytest.fail("native probe preceded admission"))
    monkeypatch.setattr(O, "_probe_shared_tools", lambda *_: pytest.fail("shared probe preceded admission"))
    monkeypatch.setattr(CA, "run_round", lambda *_args, **_kwargs: pytest.fail("paid author preceded admission"))
    with pytest.raises(StageGateError, match="independently issued RTL"):
        O.run_fresh_component_phase1(inputs, model="no-model-control", effort="low", wall_budget_seconds=1)
    assert not inputs.output.exists() and not inputs.candidate.exists()


@pytest.mark.parametrize("change", ["none", "source", "destination", "control", "bytes"])
def test_outer_launcher_requires_exact_selected_identity_with_no_inner_fallback(tmp_path, change):
    outer, inner = tmp_path / "outer", tmp_path / "inner"
    outer.write_text("private source-membership fixture only")
    inner.write_text("different private nested helper fixture")
    selected = RuntimeGrant(outer, "/usr/bin/author-bwrap", sha256_file(outer))
    nested = RuntimeGrant(inner, "/usr/bin/bwrap", sha256_file(inner))
    values = {field.name: None for field in fields(O.FreshPhase1Inputs)}
    values.update(control_runtime=(selected, nested), author_sandbox=selected)
    inputs = O.FreshPhase1Inputs(**values)
    assert inputs.sandbox_binary == outer
    if change == "none":
        inputs = replace(inputs, author_sandbox=None)
    elif change == "source":
        clone = tmp_path / "clone"
        clone.write_bytes(outer.read_bytes())
        inputs = replace(inputs, author_sandbox=replace(selected, source=clone))
    elif change == "destination":
        inputs = replace(inputs, author_sandbox=replace(selected, destination=nested.destination))
    elif change == "control":
        inputs = replace(inputs, control_runtime=(nested,))
    else:
        outer.write_text("changed selected outer bytes")
    with pytest.raises(StageGateError):
        inputs.sandbox_binary


@pytest.fixture
def native_inputs(tmp_path):
    names = (
        "MERLIN_TEST_CODEX",
        "MERLIN_TEST_CODEX_BWRAP",
        "MERLIN_TEST_BWRAP",
        "MERLIN_TEST_COMPONENT_PYTHON",
        "MERLIN_TEST_COMPONENT_STDLIB",
        "MERLIN_TEST_SHELL",
    )
    selectors = {name: os.environ.get(name) for name in names}
    if not all(selectors.values()):
        pytest.skip("requires explicitly selected public native client/helper and interpreter/std-library inventory")
    codex, sandbox, outer_sandbox, python, stdlib, shell = (
        Path(value).resolve(strict=True) for value in selectors.values()
    )
    runtime = inventory_runtime(
        files=(),
        trees=((stdlib, str(stdlib)),),
        executables=(
            (sandbox, "/usr/bin/bwrap"),
            (outer_sandbox, "/usr/bin/author-bwrap"),
            (python, "/usr/bin/python3"),
            (python, str(python)),
            (shell, "/bin/sh"),
            *((path, str(path)) for path in sorted((stdlib / "lib-dynload").glob("*.so"))),
        ),
    )
    control_runtime = (*runtime, RuntimeGrant(codex, "/usr/bin/codex", sha256_file(codex)))
    library = tmp_path / "library"
    library.mkdir()
    (library / "api.py").write_text('"""Private public-input control, not compiler support."""\n')
    contract = freeze_compiler_library(
        library, review_id="private-no-model-readiness", public_modules=("api",), sources=(("api.py", "api"),)
    )
    public = tmp_path / "public.txt"
    public.write_text("OWNED_ADMITTED_INPUT")
    view = materialize_component_view(
        tmp_path / "view",
        library=contract,
        library_root=library,
        generation_sha256="1" * 64,
        inputs=(ApprovedInput(public, "generated_input/public.txt", sha256_file(public), "generated_input"),),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name, text in O.inert_scaffold("synthetic").items():
        (candidate / name).write_text(text)
    output = tmp_path / "output"
    output.mkdir()
    inputs = SimpleNamespace(
        view=view,
        candidate=candidate,
        output=output,
        runtime=runtime,
        control_runtime=control_runtime,
        sandbox_binary=outer_sandbox,
        codex_destination="/usr/bin/codex",
        readiness=(),
        compiler_transport=None,
    )
    return inputs, python, selectors


def _probe(inputs, command, expected):
    inputs.readiness = (O.FreshToolProbe("host_compiler", command, sha256_bytes(expected)),)
    O._verify_author_tool_containment(
        runtime=inputs.runtime, control_runtime=inputs.control_runtime, readiness=inputs.readiness
    )


def test_actual_native_readiness_uses_exact_public_input_and_preserves_inert_oot(native_inputs):
    inputs, _, selectors = native_inputs
    initial = O._tree(inputs.candidate)
    _probe(
        inputs,
        (
            "/usr/bin/python3",
            "-I",
            "-B",
            "-c",
            'from pathlib import Path; print(Path("/component-inputs/generated_input/public.txt").read_text())',
        ),
        b"OWNED_ADMITTED_INPUT\n",
    )
    T.probe_native_author_tools(inputs)
    assert O._tree(inputs.candidate) == initial
    home = inputs.output / "readiness" / "native_author" / "home"
    assert not (home / "auth.json").exists()
    records = tuple(inputs.output.rglob("invocation.json"))
    assert len(records) == 1
    observed = invocation_record.require_environment(records[0], environment={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    assert observed["stage"] == "fresh_phase1_native_author_host_compiler"
    assert observed["returncode"] == 0
    (inputs.output / "scope.json").write_text(
        json.dumps(
            {
                "scope": "real config-only no-model native readiness; no origin/compiler/target runtime authority",
                "selectors": selectors,
                "test_source_sha256": sha256_file(Path(__file__)),
                "initial_oot_sha256": initial["sha256"],
                "final_oot_sha256": O._tree(inputs.candidate)["sha256"],
            }
        )
    )


def test_host_readiness_success_cannot_mask_author_tool_private_path_refusal(native_inputs, tmp_path):
    inputs, python, _ = native_inputs
    private = tmp_path / "unadmitted.txt"
    private.write_text("OWNED_UNADMITTED_CANARY")
    script = "from pathlib import Path; print(Path(" + repr(str(private)) + ").read_text())"
    host_command = (str(python), "-I", "-B", "-c", script)
    inputs.readiness = (O.FreshToolProbe("host_compiler", host_command, sha256_bytes(b"OWNED_UNADMITTED_CANARY\n")),)
    # Actual old selected transport succeeds. No synthetic success callback or
    # authority is supplied. The identical arguments in the native author must refuse.
    O._probe_shared_tools(inputs, ("/usr/bin/env",))
    observed = invocation_record.verify(
        next((inputs.output / "readiness").glob("host_compiler/invocations/*/invocation.json"))
    )
    assert observed["returncode"] == 0
    # Reuse the exact original command; native membership has the same source
    # and destination, so the failure cannot be explained by an alias change.
    _probe(inputs, host_command, b"OWNED_UNADMITTED_CANARY\n")
    with pytest.raises(StageGateError, match="native author tool readiness failed"):
        T.probe_native_author_tools(inputs)
    failed = json.loads(next((inputs.output / "readiness" / "native_author").rglob("invocation.json")).read_text())
    assert failed["returncode"] != 0 and failed["status"] == "failed"
    assert failed["inputs_unchanged"] and failed["dependencies_unchanged"] and failed["executable_unchanged"]
    assert b"FileNotFoundError" in Path(failed["stderr"]["path"]).read_bytes()
    assert not (inputs.output / "compiler_origin.json").exists()


def test_actual_native_preflight_cannot_seed_a_compiler_from_tool_probe(native_inputs):
    inputs, _, _ = native_inputs
    before = O._tree(inputs.candidate)
    script = (
        "from pathlib import Path; Path("
        + repr(str(inputs.candidate / "driver.py"))
        + ').write_text("injected compiler seed")'
    )
    _probe(inputs, ("/usr/bin/python3", "-I", "-B", "-c", script), b"")
    with pytest.raises(StageGateError, match="native author tool readiness failed"):
        T.probe_native_author_tools(inputs)
    assert O._tree(inputs.candidate) == before
    failed = json.loads(next((inputs.output / "readiness" / "native_author").rglob("invocation.json")).read_text())
    assert failed["returncode"] != 0 and failed["status"] == "failed"
    assert b"Read-only file system" in Path(failed["stderr"]["path"]).read_bytes()
    assert failed["inputs_unchanged"] and failed["dependencies_unchanged"] and failed["executable_unchanged"]
    assert not (inputs.output / "compiler_origin.json").exists()


@pytest.mark.parametrize("payload", ["source", "directory", "symlink"])
def test_probe_cannot_add_members_to_disposable_readiness_copy(native_inputs, payload):
    inputs, _, _ = native_inputs
    before = O._tree(inputs.candidate)
    destination = repr(str(inputs.candidate / "extra"))
    operation = {
        "source": '.write_text("injected source")',
        "directory": ".mkdir()",
        "symlink": '.symlink_to("driver.py")',
    }[payload]
    script = "from pathlib import Path; Path(" + destination + ")" + operation
    _probe(inputs, ("/usr/bin/python3", "-I", "-B", "-c", script), b"")
    with pytest.raises(StageGateError, match="source membership|symlink"):
        T.probe_native_author_tools(inputs)
    assert O._tree(inputs.candidate) == before
    assert not (inputs.candidate / "extra").exists()
    observed = invocation_record.verify(next((inputs.output / "readiness" / "native_author").rglob("invocation.json")))
    assert observed["returncode"] == 0  # Refused by actual membership proof, not process failure.
    assert not (inputs.output / "compiler_origin.json").exists()


def test_actual_ordinary_provider_preflight_denies_owned_auth_without_requesting_model(native_inputs, monkeypatch):
    inputs, _, _ = native_inputs
    initial = O._tree(inputs.candidate)
    home = inputs.output / "ordinary-preflight-home"
    home.mkdir()
    config = home / "config.toml"
    config.write_text(
        CA._candidate_permission_config(
            home, read_paths=("/component-inputs", *(grant.destination for grant in inputs.control_runtime))
        )
    )
    auth = home / "auth.json"
    auth.write_text('{"private_owned_canary":"NOT_A_CREDENTIAL"}')
    policy = list(
        T.strict_tool_policy(
            inputs.view,
            inputs.candidate,
            runtime=inputs.control_runtime,
            candidate_destination=str(inputs.candidate),
            bwrap_binary=inputs.sandbox_binary,
        )
    )
    policy.insert(policy.index("--unshare-all") + 1, "--share-net")
    expected_binds = ["--bind", str(home), str(home), "--setenv", "CODEX_HOME", str(home)]

    def command_owner(inner, workspace, bundle, *, extra_binds):
        assert workspace == inputs.candidate and bundle == {} and extra_binds == expected_binds
        return shlex.join([*policy, *extra_binds, "--", "/bin/sh", "-c", inner])

    original_run = subprocess.run
    records = []

    def observed_actual_process(command, **kwargs):
        # Instrument the ordinary fixed provider without substituting its
        # process or result. This diagnostic cannot issue experimental origin.
        with invocation_record.observe(
            inputs.output / "ordinary-provider-observation",
            stage="actual_ordinary_provider_preflight",
            argv=command,
            cwd=kwargs["cwd"],
            inputs=(Path(command[1]), config, auth),
            dependencies=(Path(CA.__file__), Path(__file__), *(grant.source for grant in inputs.control_runtime)),
        ) as observation:
            result = original_run(command, **kwargs)
            observation.complete(result)
            records.append(observation.path)
            return result

    monkeypatch.setattr(CA.subprocess, "run", observed_actual_process)
    CA._preflight_candidate_sandbox(
        inputs.candidate,
        home,
        inputs.codex_destination,
        {},
        command_owner,
        inputs.output / "ordinary-rounds",
        0,
        runtime_binds=lambda selected: expected_binds,
    )
    assert len(records) == 1
    observed = invocation_record.verify(records[0])
    assert observed["returncode"] == 0 and observed["stage"] == "actual_ordinary_provider_preflight"
    assert auth.read_text() == '{"private_owned_canary":"NOT_A_CREDENTIAL"}'
    assert O._tree(inputs.candidate) == initial
    assert not tuple(path for path in inputs.candidate.rglob("*") if path.is_dir())
    assert not (inputs.output / "compiler_origin.json").exists()
