"""Keep candidate imports separate from the upstream host-lowering Python."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


def driver():
    source = repo_root() / "examples/gemmini/model-study/full_model.py"
    spec = importlib.util.spec_from_file_location("full_forward_study", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_policy_must_match_both_host_recipe_and_tolerances():
    module = driver()
    policy = {"atol": 2e-6, "rtol": 1e-5, "equal_nan": False, "finite_required": True}
    settings = SimpleNamespace(host_math_policy="portable_exp_trig")
    document = {"numeric_contract": {"host_math": {"policy": "portable_exp_trig"}}, "comparison_policy": policy}
    module.verify_reference_policy(document, settings, policy)
    with pytest.raises(ValueError):
        module.verify_reference_policy(document, SimpleNamespace(host_math_policy="native"), policy)
    with pytest.raises(ValueError):
        module.verify_reference_policy(document, settings, {**policy, "atol": 0.5})


def fixture_wrapper(tmp_path, compiler_source):
    compiler = tmp_path / "compiler"
    overlay = compiler / "compiler_library"
    generic = tmp_path / "generic"
    foreign = tmp_path / "foreign"
    origins = tmp_path / "origins"
    for path in (overlay / "merlin", generic / "merlin", foreign / "merlin", origins):
        path.mkdir(parents=True)
    (overlay / "merlin/__init__.py").write_text(
        "from pkgutil import extend_path\n__path__=extend_path(__path__,__name__)\n"
    )
    (overlay / "merlin/local.py").write_text("VALUE='overlay'\n")
    (generic / "merlin/shared.py").write_text("VALUE='generic'\n")
    (foreign / "merlin/forbidden.py").write_text("VALUE='foreign'\n")
    marker = tmp_path / "pth-executed"
    (foreign / "intrusion.pth").write_text("import pathlib; pathlib.Path(" + repr(str(marker)) + ").touch()\n")
    entry = compiler / "compiler.py"
    entry.write_text(compiler_source)
    bootstrap = repo_root() / "examples/gemmini/model-study/candidate_python.py"
    config = tmp_path / "candidate.json"
    settings = {
        "python": sys.executable,
        "outer_python": sys.executable,
        "entrypoint": str(entry),
        "compiler_roots": [str(compiler), str(overlay), str(generic)],
        "dependency_roots": [str(foreign)],
        "origins": str(origins),
    }
    config.write_text(json.dumps(settings))
    wrapper = tmp_path / "python3"
    wrapper.write_text(driver().candidate_wrapper(settings, bootstrap, config))
    wrapper.chmod(0o755)
    return wrapper, entry, origins, marker


def test_candidate_uses_both_frozen_roots_and_does_not_process_pth(tmp_path):
    wrapper, entry, origins, marker = fixture_wrapper(
        tmp_path, "import merlin.local,merlin.shared\nprint(merlin.local.VALUE,merlin.shared.VALUE)\n"
    )
    result = subprocess.run([str(wrapper), str(entry)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "overlay generic"
    assert not marker.exists()
    observations = [json.loads(path.read_text()) for path in origins.glob("*.json")]
    assert len(observations) == 1
    rows = observations[0]["modules"]
    assert {row["module"] for row in rows} >= {"merlin", "merlin.local", "merlin.shared"}
    assert all(row["allowed"] and len(row["sha256"]) == 64 for row in rows)


def test_external_namespace_child_fails_even_with_origin_receipt(tmp_path):
    wrapper, entry, origins, marker = fixture_wrapper(tmp_path, "import merlin.forbidden\n")
    result = subprocess.run([str(wrapper), str(entry)], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "outside frozen roots" in result.stderr
    assert len(list(origins.glob("*.json"))) == 1
    assert not marker.exists()


def test_nonentry_python_keeps_outer_environment_even_with_entrypoint_later(tmp_path):
    wrapper, entry, origins, _ = fixture_wrapper(tmp_path, "raise RuntimeError('must not run')\n")
    result = subprocess.run(
        [str(wrapper), "-c", "import os; print(os.environ['PYTHONPATH'])", str(entry)],
        env=dict(os.environ, PYTHONPATH="outer-checker-root"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "outer-checker-root"
    assert not list(origins.glob("*.json"))


def selection(kind="independent_heads", **changes):
    return {
        "kind": kind,
        "source_ir_sha256": "a" * 64,
        "effects": {
            "qualification_identity": "fixture explicitly inspected synchronous call",
            "no_retained_aliases": True,
            "no_releases": True,
            "complete_before_return": True,
        },
        **changes,
    }


def test_source_route_build_forwards_typed_selection_and_baseline_has_no_new_key():
    from merlin.llvmlower.head_source_route import HeadBatchSourceRoute
    from merlin.llvmlower.sibling_source_route import SiblingSourceRoute

    module = driver()
    calls = []

    def builder(stage, **kwargs):
        calls.append((stage, kwargs))
        return "built"

    assert module.build_selected(builder, "stage", offload=True) == "built"
    assert calls[-1] == ("stage", {"offload": True})
    heads = module.source_route(selection(whole_batch=True, shared_rhs_row_views=True))
    assert type(heads) is HeadBatchSourceRoute
    module.build_selected(builder, "stage", source_route=heads, offload=True)
    assert calls[-1][1]["source_route"] is heads
    assert heads.whole_batch and heads.shared_rhs_row_views
    siblings = module.source_route(selection("shared_input", partial=True))
    assert type(siblings) is SiblingSourceRoute and siblings.partial
    module.build_selected(builder, "stage", source_route=siblings, offload=True)
    assert calls[-1][1]["source_route"] is siblings


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "typo"},
        {"kind": []},
        {"unknown": True},
        {"source_ir_sha256": "A" * 64},
        {"whole_batch": 1},
        {"shared_rhs_row_views": "true"},
        {"effects": {}},
        {
            "effects": {
                "qualification_identity": "premise",
                "no_retained_aliases": True,
                "no_releases": True,
                "complete_before_return": False,
            }
        },
    ],
)
def test_source_route_rejects_untyped_or_unknown_selection(changes):
    with pytest.raises(ValueError):
        driver().source_route(selection(**changes))


def test_sibling_route_rejects_head_only_flag_and_duplicate_json_fields():
    module = driver()
    with pytest.raises(ValueError):
        module.source_route(selection("shared_input", whole_batch=True))
    with pytest.raises(ValueError, match="duplicate"):
        json.loads('{"effects":{"no_releases":true,"no_releases":false}}', object_pairs_hook=module.unique_object)


def test_retained_producer_bytes_match_pins_and_reject_change_during_copy(tmp_path):
    import hashlib

    source = tmp_path / "driver.py"
    source.write_bytes(b"print('fixture')\n")
    output = tmp_path / "output"
    output.mkdir()

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def pin(path):
        return {"path": str(Path(path).resolve()), "sha256": sha(path)}

    rows = driver().retain_producers(output, [source], SimpleNamespace(pin=pin, sha=sha))
    assert rows == [{"original": pin(source), "retained": pin(output / "producer/driver.py")}]
    assert source.read_bytes() == (output / "producer/driver.py").read_bytes()
    second = tmp_path / "changed-output"
    second.mkdir()
    calls = []

    def changing_pin(path):
        row = pin(path)
        calls.append(row)
        if len(calls) == 1:
            source.write_bytes(b"changed")
        return row

    with pytest.raises(ValueError, match="producer changed"):
        driver().retain_producers(second, [source], SimpleNamespace(pin=changing_pin, sha=sha))


def test_selected_input_snapshot_rejects_bytes_changed_between_pin_and_parse(tmp_path):
    import hashlib

    path = tmp_path / "selected.json"
    path.write_bytes(b'{"selected":1}')

    def pin(path):
        return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}

    module = driver()
    payload, identity = module.selected_bytes(path, SimpleNamespace(pin=pin))
    assert payload == b'{"selected":1}' and identity == pin(path)
    calls = []

    def drifting_pin(path):
        record = pin(path)
        calls.append(record)
        if len(calls) == 1:
            Path(path).write_bytes(b'{"selected":2}')
        return record

    with pytest.raises(ValueError, match="changed while reading"):
        module.selected_bytes(path, SimpleNamespace(pin=drifting_pin))
