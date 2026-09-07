from __future__ import annotations

from dataclasses import dataclass

from merlin.perf import artifact_activity as A


@dataclass(frozen=True)
class _Endpoint:
    engine: str = "spatial"

    _roles = {
        "CFG": ("config",),
        "DMA": ("dma",),
        "LOAD": ("operand_load",),
        "MAC": ("accumulate",),
        "WAIT": ("sync",),
        "LOOP": ("loop_descriptor",),
        "UNMAPPED": (),
    }

    def roles_of(self, name: str) -> tuple[str, ...]:
        return self._roles.get(name, ())


def _facts(monkeypatch):
    monkeypatch.setattr(A, "funct_table_for", lambda _target: {
        "names": {"1": "CFG", "2": "DMA", "3": "LOAD", "4": "MAC",
                  "5": "WAIT", "6": "LOOP", "7": "UNMAPPED"},
    })
    monkeypatch.setattr(A, "endpoints_for", lambda _target: (_Endpoint(),))
    # Keep this unit test about the artifact join. Role ownership itself has independent contract tests.
    monkeypatch.setattr(A.asm_provenance, "provenance_of_role", lambda role: type(
        "P", (), {"to_dict": lambda self: {"role": role, "actionable": True}})())
    monkeypatch.setattr(A.asm_provenance, "opportunities", lambda *_args, **_kwargs: [])


def test_it_lifts_issued_activity_and_visible_overlap(monkeypatch):
    _facts(monkeypatch)
    trace = {
        "source": "candidate.mlir",
        "instructions": [
            {"index": 0, "class": "FENCE", "funct": None},
            {"index": 1, "class": "CONFIG_EX", "funct": 1},
            {"index": 2, "class": "DMA", "funct": 2},
            {"index": 3, "class": "MVIN", "funct": 3},
            {"index": 4, "class": "COMPUTE", "funct": 4},
            {"index": 5, "class": "WAIT", "funct": 5},
            {"index": 6, "class": "LOOP", "funct": 6},
            {"index": 7, "class": "FENCE", "funct": None},
        ],
    }

    got = A.analyze_artifact_activity(trace, target="derived-target")

    assert got["status"] == "decoded"
    assert got["issued"] == {
        "movement_instructions": 2,
        "compute_instructions": 1,
        "configuration_instructions": 1,
        "loop_descriptor_instructions": 1,
        "synchronization_instructions": 3,
        "dma_instructions": 1,
    }
    assert got["encoding_resolution"]["status"] == "complete"
    cca = got["program_cca"]
    assert cca["scope"] == "program"
    assert cca["dispatch"]["loop_offloaded"] is True
    assert cca["dispatch"]["dma_overlap"] is True
    assert cca["dispatch"]["dma_issue_to_wait"] == 2
    assert cca["communication"]["copy_compute_overlap"] is True


def test_it_never_turns_missing_role_or_unknown_encoding_into_zero(monkeypatch):
    _facts(monkeypatch)
    trace = {
        "source": "candidate.mlir",
        "instructions": [
            {"index": 0, "class": "SOMETHING", "funct": 7},
            {"index": 1, "class": "UNKNOWN", "funct": 99},
            {"index": 2, "class": "UNKNOWN", "funct": None},
        ],
    }

    got = A.analyze_artifact_activity(trace, target="derived-target")

    assert got["encoding_resolution"]["status"] == "partial"
    assert got["encoding_resolution"]["named_without_role"] == [
        {"index": 0, "identity": "UNMAPPED"},
    ]
    assert got["encoding_resolution"]["unknown_instruction_indices"] == [1, 2]
    assert got["program_cca"]["provenance"]["confidence"] == "low"
    assert got["dynamic_unknowns"]["occupancy"].startswith("UNKNOWN")


def test_empty_artifact_is_unknown_not_zero_or_complete(monkeypatch):
    _facts(monkeypatch)

    got = A.analyze_artifact_activity({"source": "empty", "instructions": []},
                                      target="derived-target")

    assert got["status"] == "UNKNOWN"
    assert got["encoding_resolution"]["status"] == "UNKNOWN"
    assert got["issued"]["movement_instructions"] is None
    assert got["issued"]["compute_instructions"] is None
    assert got["program_cca"] is None
