"""A freshly onboarded target must LOAD, not merely resolve.

THE DEFECT THIS IS THE REGRESSION FOR. ``merlin-onboard`` produced a target package that resolved
through ``target_registry`` — its contract parsed, its capability manifest loaded through the spine,
the CLI printed "OK — the target routes through the capability spine" — and then
``base.get_backend(<name>)`` raised a bare ``KeyError``. Not one generated package in this repo's
history had a ``plugin`` block, because the function whose docstring promised to write one never did.
A whole target could be onboarded, reported as routed, and be unreachable by every runtime caller.

WHAT "LOADS" MEANS HERE, PRECISELY. It means the package's backend module is imported, self-registers,
and satisfies the ``@runtime_checkable`` ``Backend`` protocol — the same shape every other backend in
the repo has. It does NOT mean the package computes anything: a generated package has no toolchain and
no device, so every execution entry declines by name, and the test asserts that too. The tempting
alternative — running the buffer through Merlin's own reference runtime and reporting those numbers as
the target's — is what the retired ``runtime_adapter`` did, and it is why that module is on the
sandbox's oracle-callable deny list.

AND THE NEGATIVE, WHICH IS THE HALF THAT ROTS. A target whose contract declares no compute engines has
no derivable target class, and ``None`` is a real answer there: nobody has said what silicon this is.
The package must become a NAMED ABSENCE — present in ``load_failures()`` with ``compute_units`` in its
reason — and must NOT appear in ``list_backends()``. Registering it under a guessed class would file an
undeclared accelerator as a CPU, and a result attributed to the wrong kind of device gets cited.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from merlin.runtime.backends import base
from merlin.targetgen import package as pkg

#: Neutral synthetic names. Distinct per test because backend registration is process-global and
#: idempotent: a name loaded once stays loaded, so reusing one would make the second test a no-op.
ONBOARDED = "synth_loadable_npu"
UNDECLARED = "synth_undeclared_npu"


def _descriptor(root, target: str):
    """A minimal descriptor + the residual side-input onboarding derives a manifest from."""
    package = root / "out" / "artifacts" / "targets" / target / "contracts"
    package.mkdir(parents=True)
    (package / "residual.yaml").write_text(f"target: {target}\nkind: systolic\n", encoding="utf-8")
    descriptor = root / "target_experiment.yaml"
    descriptor.write_text(f"target: {target}\n", encoding="utf-8")
    return descriptor, root / "out"


def test_an_onboarded_target_is_reachable_through_get_backend(tmp_path, monkeypatch):
    from merlin.targetgen import onboard as onboard_mod

    descriptor, out_root = _descriptor(tmp_path, ONBOARDED)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(out_root))
    result = onboard_mod.onboard(descriptor)
    root = result.oot_root
    assert root is not None, "onboarding must say where it wrote the package"

    # The plugin block the generator used to promise and never write.
    contract = yaml.safe_load((root / "contracts" / "target_contract.yaml").read_text(encoding="utf-8"))
    assert contract["plugin"]["backend"], "no plugin.backend means get_backend raises KeyError"
    assert pkg.capability(root, "backend").provided is True

    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    backend = base.get_backend(ONBOARDED)
    assert isinstance(backend, base.Backend), (
        "the generated backend must satisfy the module-level Backend protocol, not a bespoke "
        "adapter class the registry cannot reach"
    )
    assert ONBOARDED in base.list_backends()
    # The class is DERIVED from the engines the contract declares, never defaulted.
    assert base.info(ONBOARDED).target_class is base.target_class_for(ONBOARDED)


def test_the_onboarded_backend_declines_instead_of_borrowing_the_oracle(tmp_path, monkeypatch):
    """It loads; it does not pretend to compute. Both halves are the point."""
    from merlin.targetgen import onboard as onboard_mod

    name = ONBOARDED  # same package; registration is idempotent and this asserts its behaviour
    descriptor, out_root = _descriptor(tmp_path, name)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(out_root))
    root = onboard_mod.onboard(descriptor).oot_root
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    backend = base.get_backend(name)

    assert backend.available() is False
    result = backend.run_command_buffer({"target": name, "commands": []})
    declined = result["declined"]
    assert declined["reason"] and result["outputs"] == {}, (
        "an empty result with no stated reason is SILENT_NO_WORK — a grader reads it as a program "
        "that ran and produced zeros"
    )
    with pytest.raises(NotImplementedError):
        backend.compile_command_buffer({}, tmp_path)
    # The opcode map is DERIVED or it is UNKNOWN; it is never a borrowed literal.
    assert backend.opcodes() or backend.derivation().startswith("UNKNOWN(")


def test_a_target_with_no_compute_units_is_a_named_absence(tmp_path, monkeypatch):
    """No derivable class -> recorded failure naming ``compute_units``, and NOT in list_backends()."""
    root = tmp_path / UNDECLARED
    (root / "contracts").mkdir(parents=True)
    (root / "contracts" / "target_contract.yaml").write_text(
        textwrap.dedent(f"""\
            name: {UNDECLARED}
            version: '0.1'
            capabilities: {{ops: [matmul]}}
            memory_model: {{}}
            compiler_obligations: []
            hardware_promises: []
            runtime_promises: []
            legality: []
            plugin:
              backend: backend
            """),
        encoding="utf-8",
    )
    from merlin.targetgen.generate import oot_package

    backend_dir = root / oot_package.BACKEND_DIR
    backend_dir.mkdir()
    (backend_dir / "__init__.py").write_text(oot_package.backend_module_source(UNDECLARED), encoding="utf-8")

    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    assert base.target_class_for(UNDECLARED) is None, "the fixture must actually declare no engines"

    with pytest.raises(KeyError) as exc:
        base.get_backend(UNDECLARED)
    assert UNDECLARED not in base.list_backends(), "a target of unknown class must not be registered"
    reason = base.load_failures().get(UNDECLARED, "")
    assert "compute_units" in reason, f"the absence must name what is missing, got: {reason!r}"
    assert "compute_units" in str(exc.value), "get_backend must carry the recorded reason, not a bare name"


def test_a_plugin_backend_that_points_nowhere_is_recorded_not_ignored(tmp_path, monkeypatch):
    """MUTATION: a typo'd plugin path used to be silently skipped — no backend, and no way to tell.

    ``_oot_plugin_modules`` resolved the declared path itself and simply did not append when it missed,
    which is the exact failure ``targetgen.plugins`` exists to stop, reproduced one layer down.
    """
    name = "synth_dangling_npu"
    root = tmp_path / name
    (root / "contracts").mkdir(parents=True)
    (root / "contracts" / "target_contract.yaml").write_text(
        f"name: {name}\nversion: '0.1'\ncapabilities: {{}}\nmemory_model: {{}}\n"
        "compiler_obligations: []\nhardware_promises: []\nruntime_promises: []\nlegality: []\n"
        "plugin:\n  backend: backend_typo\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    base.list_backends()  # trigger discovery
    reason = base.load_failures().get(name, "")
    assert "backend_typo" in reason and "resolves to nothing" in reason, (
        f"a declared-but-missing plugin module must be recorded with its path, got: {reason!r}"
    )
    assert name not in base.list_backends()
