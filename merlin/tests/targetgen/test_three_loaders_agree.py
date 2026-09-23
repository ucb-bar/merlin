"""The convergence gate: one generated package, three loaders, three NAMED answers.

WHY THIS FILE EXISTS SEPARATELY FROM THE PER-LOADER TESTS. Each loader already has a test that it
behaves (``test_generated_package_loads`` for the runtime backend, ``test_targetgen_toy`` for the
generated compiler manifest, ``test_package_capabilities`` for the classifier). None of them asserts
the property the whole change is for: that a SECOND, GENERATED target gets a definite answer from
EVERY loader — loaded, or refused by name — rather than one loader working and the other two
discovering the mismatch by failing somewhere inside themselves. Three green per-loader tests and a
package no loader agrees about is exactly the state this repo was in.

THE SHARPEST FORM OF THE DEFECT, WHICH IS WHAT THE SECOND TEST PINS. ``manifest.yaml`` names two
disjoint schemas in sibling directories: the registry format (``target``/``run_id``/``outputs``) and
the frozen experiment ABI (``artifact_type``/``entrypoints``/``commands``). Given ``<dir>/manifest.yaml``
alone, no code could say which contract it claimed. Pointing a loader at the wrong one surfaced as
``KeyError: 'target'`` or a ``FileNotFoundError`` on ``dialect.py`` — an error about a field, from
inside a load, when the honest answer was "this package does not do that". The mutation guard is
literally that: the pre-fix exception types must NOT be what comes out.

WHAT IS NOT ASSERTED HERE, ON PURPOSE. That the generated package COMPILES anything. Emitting a
working out-of-tree compiler is phase 1's entire job; a generator that could do it would make the
thesis unfalsifiable. The generated ``compiler`` capability is ABSENT and says so, and the value of
that is precisely that ``oot_runner`` can refuse before invoking anything — the difference between
"does not claim to be a compiler" and "claims to be one and is bad at it".
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

from merlin.common.paths import artifacts_dir
from merlin.runtime.backends import base
from merlin.targetgen import oot_runner, registry
from merlin.targetgen import package as pkg
from merlin.targetgen.generate import oot_package

#: Neutral synthetic name. Distinct from every other test's because backend registration is
#: process-global and idempotent — a name loaded by another module would make this a no-op.
GENERATED = "synth_three_loader_npu"


def _onboard(tmp_path, monkeypatch, target: str):
    """Onboard ``target`` from a minimal descriptor and return its package root."""
    from merlin.targetgen import onboard as onboard_mod

    contracts = tmp_path / "out" / "artifacts" / "targets" / target / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "residual.yaml").write_text(f"target: {target}\nkind: systolic\n", encoding="utf-8")
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(f"target: {target}\n", encoding="utf-8")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    root = onboard_mod.onboard(descriptor).oot_root
    assert root is not None, "onboarding must say where it wrote the package"
    return root


# --------------------------------------------------------------------------------------------------
def test_a_generated_target_gets_a_named_answer_from_every_loader(tmp_path, monkeypatch):
    """Loaded or refused-by-name, three times over. No loader may fail inside itself."""
    root = _onboard(tmp_path, monkeypatch, GENERATED)
    caps = pkg.capabilities_of(root)
    assert set(caps) == set(pkg.CAPABILITIES), "every capability is settled; 'not in the dict' is not an answer"

    # Loader 1 — runtime backends. PROVIDED: the package must actually import and register.
    assert caps["backend"].provided is True, caps["backend"].explain(root)
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    backend = base.get_backend(GENERATED)
    assert isinstance(backend, base.Backend)
    assert GENERATED not in base.load_failures(), "a package that loads must not also be a recorded failure"

    # Loader 2 — the frozen experiment ABI. The manifest is a WORK ORDER: it loads, and it carries the
    # absent-compiler verdict rather than dropping it. A gate whose answer is computed and then thrown
    # away is indistinguishable from one that never ran.
    assert caps["compiler"].provided is False
    package = oot_runner.load_package(root)
    assert package.compiler_capability is not None, "the verdict must ride on the Package, not vanish"
    assert package.compiler_capability.provided is False
    assert package.compiler_capability.missing, "an absence with no named cause is the shape being retired"

    # Loader 3 — the staged pipeline's dialect registry. ABSENT, and it says which file is missing.
    assert caps["dialect"].provided is False
    with pytest.raises(pkg.PackageCapabilityMissing) as refusal:
        registry.load_target(root)
    assert "dialect.py" in str(refusal.value), f"the refusal must name the missing file: {refusal.value}"


def test_the_generated_backend_declines_with_a_reason_rather_than_returning_zeros(tmp_path, monkeypatch):
    """SILENT_NO_WORK guard: empty outputs with no stated reason read to a grader as a program that ran."""
    root = _onboard(tmp_path, monkeypatch, GENERATED)
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    # OOT ownership is process-immutable. Another test selected a package of the same name
    # at a different path, so this qualification must start in a fresh worker.
    code = """
from merlin.runtime.backends import base
backend = base.get_backend('synth_three_loader_npu')
assert backend.available() is False
assert backend.why_unavailable()
result = backend.run_command_buffer({'target': 'synth_three_loader_npu', 'commands': []})
assert result['outputs'] == {}
assert result['declined']['reason'] and result['declined']['op']
assert backend.opcodes() or backend.derivation().startswith('UNKNOWN(')
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_both_generators_emit_the_same_capability_shape(tmp_path, monkeypatch):
    """Two materialisers, one shape. A second entrance to the same hole is not a convergence.

    ``capability_manifests.write_oot_target`` (the zero-env path ``merlin-onboard`` takes) and
    ``pipeline.build`` (the staged path, which assembles its repo out of ``Artifact`` objects rather
    than writing YAML directly) both produce a target package. They used to produce different ones, and
    a classifier that agreed with only one of them would have left every package from the other
    unclassifiable — which is the defect, relocated rather than closed.
    """
    from merlin.common.paths import repo_root
    from merlin.targetgen import pipeline

    toy = repo_root() / "examples/toy_npu/target"
    built = pipeline.build(
        target_name=toy.name,
        source_dir=str(toy / "docs"),
        examples_dir=str(toy / "examples"),
        out=tmp_path / "staged",
        emit=["xdsl", "mlir", "zephyr", "llvm-plan", "runtime"],
    ).out
    onboarded = _onboard(tmp_path / "zero_env", monkeypatch, GENERATED)

    def shape(root):
        return {name: (cap.provided, cap.source) for name, cap in pkg.capabilities_of(root).items()}

    assert shape(built) == shape(onboarded), (
        f"the two generators disagree about what they produced: {shape(built)} vs {shape(onboarded)}"
    )
    # And it is the shape phase 1 is handed: reachable at run time, with the compiler left to write.
    assert shape(built) == {
        "backend": (True, "inferred"),
        "compiler": (False, "declared"),
        "dialect": (False, "inferred"),
    }


# --------------------------------------------------------------------------------------------------
def _registry_format_package(root):
    """A registry-format package: ``manifest.yaml`` with ``target``/``run_id``/``outputs``, no artifact_type."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.yaml").write_text(
        yaml.safe_dump({"target": root.name, "run_id": "r0", "outputs": {"dialect_module": "dialect.py"}}),
        encoding="utf-8",
    )
    (root / "dialect.py").write_text("DIALECT_NAME = 'x'\nSPEC_OPS = {}\n", encoding="utf-8")
    (root / "lowering.yaml").write_text("interface_to_target: {}\ntarget_to_opcode: {}\n", encoding="utf-8")
    return root


def _abi_format_package(root):
    """An experiment-ABI package: ``manifest.yaml`` with artifact_type/entrypoints/commands, no run_id.

    Built from the generator's own manifest so this fixture cannot drift from the frozen schema, with
    the explicit ``package_capabilities`` declaration removed so the capability is INFERRED — which is
    what every one of the thirteen tracked certified packages relies on.
    """
    root.mkdir(parents=True, exist_ok=True)
    manifest = oot_package.compiler_manifest(root.name)
    manifest.pop(pkg.DECLARATION_KEY, None)
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    tool = root / manifest["entrypoints"]["tool"]
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text(oot_package.compiler_tool_source(root.name), encoding="utf-8")
    return root


def test_two_manifests_one_filename_are_told_apart_without_trial_parsing(tmp_path):
    """The collision, as siblings. Each is classified for what it IS, and the other loader says so.

    MUTATION: drop ``package.require`` from ``registry.load_target`` and the ABI half of this raises
    ``KeyError: 'target'`` from inside the load instead of ``PackageCapabilityMissing`` — which is the
    behaviour that shipped, and the reason nothing could tell the two formats apart.
    """
    reg = _registry_format_package(tmp_path / "as_registry")
    abi = _abi_format_package(tmp_path / "as_abi")

    reg_caps, abi_caps = pkg.capabilities_of(reg), pkg.capabilities_of(abi)
    assert reg_caps["dialect"].provided is True and reg_caps["compiler"].provided is False
    assert abi_caps["compiler"].provided is True and abi_caps["dialect"].provided is False
    # The absence names the discriminating key, not a sentence about it.
    assert any("artifact_type" in m for m in reg_caps["compiler"].missing), reg_caps["compiler"].missing
    assert any("dialect.py" in m for m in abi_caps["dialect"].missing), abi_caps["dialect"].missing

    # Wrong loader, right answer: a named capability refusal, NOT the field error that used to come out.
    with pytest.raises(pkg.PackageCapabilityMissing) as refusal:
        registry.load_target(abi)
    assert not isinstance(refusal.value, (KeyError, FileNotFoundError))
    assert "dialect" in str(refusal.value)


def test_the_abi_loader_refuses_with_the_exception_its_callers_already_catch(tmp_path, monkeypatch):
    """At ``fail`` phase the refusal is the EXISTING CertFailure — no caller sees a new exception type.

    ``load_package`` is the live certification path: every capsule of every in-flight session goes
    through it. A refusal that arrived as a new exception class would escape every ``except CertFailure``
    already written and surface as a tool crash, which grades worse than the thing being refused.
    """
    from merlin.perf import gate_phase

    reg = _registry_format_package(tmp_path / "as_registry")  # no ABI manifest -> compiler absent

    # At the phase the declaration actually carries, this loader must not be the thing that refuses:
    # the manifest is missing, so the pre-existing schema check owns the failure.
    declared = gate_phase.configured_phase(oot_runner.CAPABILITY_GATE)
    assert declared in gate_phase.PHASES, f"the gate must be declared, got {declared!r}"

    monkeypatch.setattr(gate_phase, "configured_phase", lambda _name: gate_phase.PHASE_FAIL)
    with pytest.raises(oot_runner.CertFailure) as refusal:
        oot_runner.load_package(reg)
    assert refusal.value.plane == "contract"
    assert refusal.value.category is oot_runner.FailureCategory.STRUCTURAL_INVARIANT_VIOLATION
    assert "compiler" in refusal.value.detail

    # And a package that DOES provide the capability is not refused by the gate at the same phase.
    abi = _abi_format_package(tmp_path / "as_abi")
    assert oot_runner.load_package(abi).compiler_capability.provided is True


# --------------------------------------------------------------------------------------------------
def _tracked_packages():
    """Every tracked package directory under the codegen-package home. Data, never a name list."""
    root = artifacts_dir() / "targets"
    if not root.is_dir():
        return []
    return [
        d
        for d in sorted(root.glob("*/*"))
        if d.is_dir() and ((d / "manifest.yaml").is_file() or (d / "contracts" / "target_contract.yaml").is_file())
    ]


def test_no_package_that_already_certifies_is_invalidated(monkeypatch):
    """Nothing migrates. Every ABI manifest on disk still classifies as a compiler, from inference alone.

    This is the constraint that shaped the whole change: the ABI manifest is a FROZEN experiment
    contract, mirrored and depended on by certified artifacts and in-flight agent sessions. If the
    classifier's inference rules ever disagree with the schema the runner validates against, packages
    that already carry verdicts would start being refused — so the population is checked against the
    real tree rather than a fixture that can be written to agree.
    """
    monkeypatch.delenv("MERLIN_OUT_ROOT", raising=False)
    packages = _tracked_packages()
    if not packages:
        pytest.skip("no packages on disk to classify")

    abi = []
    for d in packages:
        doc = (
            yaml.safe_load((d / "manifest.yaml").read_text(encoding="utf-8"))
            if (d / "manifest.yaml").is_file()
            else None
        )
        if isinstance(doc, dict) and "artifact_type" in doc:
            abi.append(d)
    assert abi, "the ABI population must not be empty, or this assertion is vacuous"

    for d in abi:
        cap = pkg.capability(d, "compiler")
        assert cap.provided is True, f"a certified package stopped classifying as a compiler: {cap.explain(d)}"

    # And every package answers all three questions without raising, whatever shape it is.
    for d in packages:
        caps = pkg.capabilities_of(d)
        assert set(caps) == set(pkg.CAPABILITIES)
        for cap in caps.values():
            assert cap.provided or cap.missing, f"{d}: {cap.name} is absent with no named cause"
