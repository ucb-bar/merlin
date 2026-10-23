"""Native full-call discovery consumes explicit support ownership, never overlays."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.yaml import write_yaml
from merlin.targetgen import spike_full_call as F
from merlin.targetgen import target_registry as registry

PROGRAM = """
def full_call_oracle(target):
    def run(*args, **kwargs):
        raise AssertionError("construction must not execute a grade")
    def execute(*args, **kwargs):
        return {"target": target, "marker": MARKER}
    def qualify(*args, **kwargs):
        return {"marker": MARKER}
    run.run_full_call = execute
    run.qualify_full_call = qualify
    return run
"""


def support(root, *, marker="owned", declaration="oracle.py", program=PROGRAM):
    write_yaml(
        root / "provider.yaml",
        {"schema": "merlin.provider.v1", "id": "fixture", "target": "full_call_fixture", "role": "support"},
    )
    write_yaml(
        root / "contracts/target_contract.yaml",
        {"name": "full_call_fixture", "runner": {"full_call_provider": declaration}},
    )
    (root / "oracle.py").write_text("MARKER = " + repr(marker) + "\n" + program)
    return root


@pytest.fixture
def selection(tmp_path, monkeypatch, request):
    # The repository retains failed pytest temp trees only; successful parameter
    # variants may otherwise reuse a deleted canonical root in the same process.
    root = support(tmp_path / ("support-" + hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]))
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setattr(registry, "reference_targets", lambda: {})
    return root


@pytest.mark.parametrize("absolute", [False, True])
def test_relative_and_absolute_owned_resources_construct_full_call(selection, absolute):
    if absolute:
        write_yaml(
            selection / "contracts/target_contract.yaml",
            {"name": "full_call_fixture", "runner": {"full_call_provider": str(selection / "oracle.py")}},
        )
    adapter = F.declared("full_call_fixture")
    assert adapter.run_full_call()["marker"] == "owned"
    assert adapter.qualify_full_call()["marker"] == "owned"
    assert F.declared("full_call_fixture").run_full_call()["marker"] == "owned"


def test_native_l2_selection_uses_actual_declared_factory(selection, monkeypatch):
    from merlin.targetgen import capsule_runner as C

    monkeypatch.setattr(
        C._oracle_policy,
        "chipyard_tier_plan",
        lambda target: SimpleNamespace(selection=None, unavailable_reason="L2-only"),
    )
    adapters = C._sim_engine_adapters("chipyard", "full_call_fixture")
    assert adapters["L2"].run_full_call()["marker"] == "owned"
    assert callable(adapters["L2"].qualify_full_call)


def test_support_factory_may_delegate_to_trusted_core_adapter(selection):
    (selection / "oracle.py").write_text(
        "from merlin.targetgen.spike_full_call import full_call_adapter\n"
        "def full_call_oracle(target):\n"
        ' return full_call_adapter(target, isa="rv64imafdc")\n'
    )
    adapter = F.declared("full_call_fixture")
    assert adapter.__module__ == F.__name__
    assert adapter.run_full_call.__module__ == F.__name__
    assert adapter.qualify_full_call.__module__ == F.__name__


def test_capability_overlay_neither_grants_nor_redirects_execution(selection, tmp_path, monkeypatch):
    evil = tmp_path / "evil.py"
    evil.write_text("raise AssertionError('overlay must never import')\n")
    overlay = tmp_path / "overlay.yaml"
    write_yaml(overlay, {"name": "full_call_fixture", "runner": {"full_call_provider": str(evil)}})
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(overlay))
    assert F.declared("full_call_fixture").run_full_call()["marker"] == "owned"
    write_yaml(selection / "contracts/target_contract.yaml", {"name": "full_call_fixture"})
    assert F.declared("full_call_fixture") is None


def test_absent_reference_declaration_preserves_generic_without_authority(tmp_path, monkeypatch):
    root = support(tmp_path / "reference")
    write_yaml(root / "contracts/target_contract.yaml", {"name": "full_call_fixture"})
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    monkeypatch.setattr(registry, "reference_targets", lambda: {"full_call_fixture": root})
    assert F.declared("full_call_fixture") is None
    write_yaml(
        root / "contracts/target_contract.yaml",
        {"name": "full_call_fixture", "runner": {"full_call_provider": "oracle.py"}},
    )
    with pytest.raises(ValueError):
        F.declared("full_call_fixture")


@pytest.mark.parametrize(
    "declaration", [None, "", {}, "../outside.py", "/tmp/unowned.py", "missing.py", "oracle", "oracle.py:factory"]
)
def test_invalid_present_declaration_is_not_generic_fallback(selection, declaration):
    write_yaml(
        selection / "contracts/target_contract.yaml",
        {"name": "full_call_fixture", "runner": {"full_call_provider": declaration}},
    )
    with pytest.raises(ValueError):
        F.declared("full_call_fixture")


def test_symlinked_module_refuses_before_import(selection, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("raise AssertionError('unadmitted import')\n")
    (selection / "oracle.py").unlink()
    (selection / "oracle.py").symlink_to(outside)
    with pytest.raises(ValueError, match="contained"):
        F.declared("full_call_fixture")


@pytest.mark.parametrize(
    "program",
    [
        "raise RuntimeError('SYNTHETIC_PRIVATE_PAYLOAD')",
        "full_call_oracle = None",
        "def full_call_oracle(target):\n raise RuntimeError('SYNTHETIC_PRIVATE_PAYLOAD')",
        "def full_call_oracle(target):\n return lambda: None",
        "def full_call_oracle(target):\n return None",
    ],
)
def test_broken_factories_refuse_without_private_error_payload(selection, program):
    (selection / "oracle.py").write_text(program + "\n")
    with pytest.raises(ValueError) as error:
        F.declared("full_call_fixture")
    assert "SYNTHETIC_PRIVATE_PAYLOAD" not in str(error.value)


def test_foreign_callback_owner_is_not_a_supported_adapter(selection, tmp_path):
    outside = tmp_path / "foreign.py"
    outside.write_text("def execute(*args, **kwargs): return None\n")
    program = f"""
import importlib.util
spec = importlib.util.spec_from_file_location("foreign_callback", {str(outside)!r})
foreign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foreign)
def full_call_oracle(target):
 def run(*args, **kwargs): return None
 run.run_full_call = foreign.execute
 run.qualify_full_call = foreign.execute
 return run
"""
    (selection / "oracle.py").write_text(program)
    with pytest.raises(ValueError, match="owned execution"):
        F.declared("full_call_fixture")


def test_candidate_role_cannot_supply_executable_declaration(selection, monkeypatch):
    resolved_support = registry.resolve("full_call_fixture")
    write_yaml(
        selection / "provider.yaml",
        {
            "schema": "merlin.provider.v1",
            "id": "candidate",
            "target": "full_call_fixture",
            "role": "candidate_compiler",
        },
    )
    # Discovery does not select candidate compilers as support. Even a resolved
    # legacy contract cannot turn that candidate into executable authority.
    assert F.declared("full_call_fixture") is None
    monkeypatch.setattr(registry, "resolve", lambda target: resolved_support)
    with pytest.raises(ValueError):
        F.declared("full_call_fixture")


def test_namespaces_do_not_mix_multiple_provider_roots(selection, tmp_path, monkeypatch):
    first = F.declared("full_call_fixture")
    other = support(tmp_path / "other", marker="second")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(other))
    second = F.declared("full_call_fixture")
    assert first.run_full_call()["marker"] == "owned"
    assert second.run_full_call()["marker"] == "second"
    assert first.__module__ != second.__module__


@pytest.mark.parametrize("change", ["factory", "resource", "added", "removed", "symlink"])
def test_changed_support_owner_refuses_before_reusing_factory(selection, change):
    resource = selection / "lookup.bin"
    resource.write_bytes(b"before")
    first = F.declared("full_call_fixture")
    imported = importlib.import_module(first.__module__)
    imported.full_call_oracle = lambda target: pytest.fail("cached factory was reused")
    if change == "factory":
        (selection / "oracle.py").write_text("raise AssertionError('new source must not import')\n")
    elif change == "resource":
        resource.write_bytes(b"after")
    elif change == "added":
        (selection / "new.py").write_text("VALUE = 1\n")
    elif change == "removed":
        resource.unlink()
    else:
        (selection / "new.py").symlink_to(resource)
    with pytest.raises(ValueError, match="source (changed|inventory)"):
        F.declared("full_call_fixture")


@pytest.mark.parametrize("change", ["helper", "resource", "added", "removed"])
def test_changed_declared_dependency_refuses_across_provider_roots(selection, tmp_path, monkeypatch, change):
    dependency = tmp_path / "compiler"
    dependency.mkdir()
    (dependency / "helper.py").write_text("def execute(): return 'first'\ndef qualify(): return 'qualified'\n")
    (dependency / "table.bin").write_bytes(b"before")
    dependency_program = f"""
import importlib.util
spec = importlib.util.spec_from_file_location("cached_dependency", {str(dependency / "helper.py")!r})
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
def full_call_oracle(target):
 def run(*args, **kwargs): return None
 run.run_full_call = helper.execute
 run.qualify_full_call = helper.qualify
 return run
"""
    for root in (selection, support(tmp_path / "second")):
        (root / "oracle.py").write_text(dependency_program)
        write_yaml(
            root / "provider.yaml",
            {
                "schema": "merlin.provider.v1",
                "id": "fixture",
                "target": "full_call_fixture",
                "role": "support",
                "source_dependencies": {"compiler": "../compiler"},
            },
        )
    first = F.declared("full_call_fixture")
    assert first.run_full_call() == "first"
    assert first.qualify_full_call() == "qualified"
    if change == "helper":
        (dependency / "helper.py").write_text("def execute(): return 'changed'\n")
    elif change == "resource":
        (dependency / "table.bin").write_bytes(b"after")
    elif change == "added":
        (dependency / "new.py").write_text("VALUE = 1\n")
    else:
        (dependency / "helper.py").unlink()
    importlib.import_module(first.__module__).full_call_oracle = lambda target: pytest.fail("cached factory reused")
    with pytest.raises(ValueError, match="source changed"):
        F.declared("full_call_fixture")
    # The dependency's verifier namespace may be shared even with a new factory
    # namespace, so a fresh support root cannot authorize changed dependency bytes.
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(tmp_path / "second"))
    with pytest.raises(ValueError, match="source changed"):
        F.declared("full_call_fixture")


def test_generated_bytecode_does_not_change_source_membership(selection):
    F.declared("full_call_fixture")
    cache = selection / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "extra.pyc").write_bytes(b"generated")
    assert F.declared("full_call_fixture").run_full_call()["marker"] == "owned"


def test_factory_cannot_change_its_source_inventory_during_construction(selection):
    (selection / "oracle.py").write_text(
        "from pathlib import Path\n"
        + PROGRAM.replace(
            "def full_call_oracle(target):",
            'def full_call_oracle(target):\n    Path(__file__).with_name("new.bin").write_bytes(b"changed")',
        )
    )
    with pytest.raises(ValueError, match="source changed during"):
        F.declared("full_call_fixture")
    with pytest.raises(ValueError, match="source changed after"):
        F.declared("full_call_fixture")


def test_failed_construction_still_pins_partially_imported_owner(selection):
    (selection / "oracle.py").write_text("raise RuntimeError('synthetic failure')\n")
    with pytest.raises(ValueError, match="could not be constructed"):
        F.declared("full_call_fixture")
    (selection / "oracle.py").write_text("MARKER = 'repaired'\n" + PROGRAM)
    with pytest.raises(ValueError, match="source changed after"):
        F.declared("full_call_fixture")


def test_frozen_root_stays_owned_after_live_root_disappears(selection, tmp_path, monkeypatch):
    from merlin_experiments import source_snapshot as S

    source = tmp_path / "source"
    (source / "python").mkdir(parents=True)
    (source / "python/main.py").write_text("VALUE = 1\n")
    frozen = tmp_path / "frozen"
    S.create(
        source,
        frozen,
        output_root=tmp_path / "out",
        source_roots=("python",),
        python_roots=("python",),
        legacy_roots=(),
        target_name="full_call_fixture",
        provider=S.selected_provider("full_call_fixture"),
        runtime_links=False,
    )
    receipt = S.verify(frozen)
    owned = frozen / receipt["selected_provider"]["root"]
    (selection / "oracle.py").unlink()
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(owned))
    adapter = F.declared("full_call_fixture")
    assert Path(importlib.util.find_spec(adapter.__module__).origin).is_relative_to(owned)
    assert adapter.run_full_call()["marker"] == "owned"
