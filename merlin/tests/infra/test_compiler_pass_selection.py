"""Host pass selection prepares actual packages before original source freeze."""

import json
import shutil
import threading
from pathlib import Path

import pytest

from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_pass_selection import (
    CompilerPassSelection,
    OriginalPassFactory,
    active_compiler_passes,
    restore_original_selection,
    scoped_compiler_passes,
)

BINDING = PassCapabilityBinding("exact-hardware", "exact-software", "unchanged-i32", "native-test", "fixed-library")


class CheckedPreparation:
    """Neutral source-only fixture; it provides no target qualification."""

    def __init__(self, evidence):
        self.evidence = Path(evidence)
        self.identity = sha256_file(self.evidence)
        self.checks = 0

    def revalidate(self):
        if sha256_file(self.evidence) != self.identity:
            raise ValueError("independent evidence changed")
        self.checks += 1

    def prepare_compiler(self, source, destination, *, policy):
        self.revalidate()
        shutil.copytree(source, destination)
        (destination / "policy.json").write_text(json.dumps(policy, sort_keys=True))
        self.revalidate()
        return destination


def independent_factory(inputs, *, work_root):
    """Replay an actual original fixture check, rather than a saved success."""
    path = Path(inputs["evidence"])
    if path.read_text() != "independently checked neutral fixture\n":
        raise ValueError("original independent admission refused")
    work_root.mkdir(exist_ok=False)
    (work_root / "original-check-ran").write_text(sha256_file(path))
    owner = CheckedPreparation(path)
    return ReviewedPassCapabilities((ReviewedPassCapability("configuration_cleanup", BINDING, owner),))


def selection(tmp_path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("independently checked neutral fixture\n")
    factory = OriginalPassFactory(independent_factory, {"evidence": str(evidence)})
    registry = factory.create(work_root=tmp_path / "initial-check")
    return CompilerPassSelection(registry, BINDING, mechanisms=("configuration_cleanup",), factory=factory)


def test_prepare_separate_package_then_reconstruct_original_admission(tmp_path):
    selected = selection(tmp_path)
    source = tmp_path / "original"
    source.mkdir()
    (source / "compiler.py").write_text("print('original compiler')\n")
    original = (source / "compiler.py").read_bytes()
    results = [
        selected.prepare_compiler("configuration_cleanup", source, tmp_path / name, policy={"enabled": enabled})
        for name, enabled in (("baseline", False), ("candidate", True))
    ]
    assert (source / "compiler.py").read_bytes() == original
    assert results[0]["source_sha256"] == results[1]["source_sha256"]
    assert results[0]["prepared_sha256"] != results[1]["prepared_sha256"]
    assert results[0]["selection"] == results[1]["selection"]
    restored = restore_original_selection(selected.record(), work_root=tmp_path / "child-check")
    assert restored.record() == selected.record()
    assert (tmp_path / "child-check/original-check-ran").is_file()
    with scoped_compiler_passes(restored):
        assert active_compiler_passes() is restored
    assert active_compiler_passes() is None


@pytest.mark.parametrize("damage", ["evidence", "source", "binding", "owner", "factory", "missing_factory"])
def test_forged_or_stale_transport_never_recreates_saved_success(tmp_path, damage):
    selected = selection(tmp_path)
    record = selected.record()
    if damage == "evidence":
        Path(record["factory"]["inputs"]["evidence"]).write_text("copied successful verdict\n")
    elif damage == "source":
        record["factory"]["callback"]["sha256"] = "0" * 64
    elif damage == "binding":
        record["binding"]["runtime"] = "foreign-runtime"
    elif damage == "owner":
        record["registry"]["capabilities"][0]["owner_identity"] = "copied-success"
    elif damage == "factory":
        record["factory"]["callback"]["symbol"] = "selection"
    else:
        record["factory"] = None
    with pytest.raises((ValueError, TypeError)):
        restore_original_selection(record, work_root=tmp_path / "refused-child")


def test_missing_or_unselected_capability_never_prepares_compiler(tmp_path):
    selected = CompilerPassSelection(ReviewedPassCapabilities(), BINDING, mechanisms=("configuration_cleanup",))
    with pytest.raises(ValueError, match="no exact reviewed"):
        selected.prepare_compiler("configuration_cleanup", tmp_path / "absent", tmp_path / "unused", policy={})
    with pytest.raises(ValueError, match="not originally selected"):
        selected.resolve("dense")
    assert not (tmp_path / "unused").exists()


def test_source_freeze_preparation_cannot_overlap_original(tmp_path):
    selected = selection(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="fresh disjoint"):
        selected.prepare_compiler("configuration_cleanup", source, source / "copy", policy={})


def test_preparation_or_saved_success_never_substitutes_for_final_program_check(tmp_path):
    selected = selection(tmp_path)
    with pytest.raises(ValueError, match="final emitted-program checker"):
        selected.check_functional_guard(None, None, {"status": "clean"}, {"status": "clean"}, grade_root=tmp_path)


def test_threads_must_carry_original_context_and_changed_selection_refuses(tmp_path):
    selected = selection(tmp_path)
    failures = []

    def escape():
        try:
            active_compiler_passes()
        except RuntimeError as error:
            failures.append(str(error))

    with scoped_compiler_passes(selected):
        thread = threading.Thread(target=escape)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(failures) == 1 and "escaped" in failures[0]
    selected.registry._capabilities = ()
    with pytest.raises(ValueError, match="changed"):
        selected.record()


def test_explicit_metadata_protocol_checks_original_owner_at_both_boundaries(tmp_path):
    from merlin.targetgen.compiler_pass_selection import METADATA_PROTOCOL

    evidence = tmp_path / "evidence.txt"
    evidence.write_text("independently checked neutral fixture\n")
    factory = OriginalPassFactory(
        independent_factory,
        {
            "evidence": str(evidence),
            "metadata_verification_protocol": METADATA_PROTOCOL,
        },
    )
    registry = factory.create(work_root=tmp_path / "initial-check")
    selected = CompilerPassSelection(registry, BINDING, mechanisms=("configuration_cleanup",), factory=factory)
    owner = registry._capabilities[0]._owner
    before = owner.checks
    record = selected.record()
    assert owner.checks == before + 2
    with selected.metadata_operation():
        for _ in range(5):
            assert selected.record() == record
            assert selected.resolve("configuration_cleanup").owner is owner
    assert owner.checks == before + 4
    restored = restore_original_selection(record, work_root=tmp_path / "child-check")
    assert restored.record() == record
    source = tmp_path / "source"
    source.mkdir()
    (source / "compiler.py").write_text("unchanged compiler\n")
    before = owner.checks
    prepared = selected.prepare_compiler("configuration_cleanup", source, tmp_path / "prepared", policy={})
    assert prepared["selection"] == record
    # Two direct preparation checks plus fresh registry entry and exit checks.
    assert owner.checks == before + 4
    with pytest.raises(ValueError, match="independent evidence changed"):
        with selected.metadata_operation():
            evidence.write_text("changed original proof")
            selected.record()


def test_unknown_metadata_protocol_cannot_skip_original_owner_checks(tmp_path):
    selected = selection(tmp_path)
    factory = OriginalPassFactory(
        independent_factory,
        {
            **selected.factory.inputs,
            "metadata_verification_protocol": "unknown",
        },
    )
    with pytest.raises(ValueError, match="unknown original pass metadata"):
        CompilerPassSelection(selected.registry, BINDING, mechanisms=("configuration_cleanup",), factory=factory)
