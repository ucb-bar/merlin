"""Immutable per-round compiler snapshots and mechanism attribution.

The controller retains experiment-wide admission, candidate/snapshot scope checks
and scientific execution order. This owner records evidence without compiling or
launching a candidate and receives its shared compiler source root explicitly.
"""

from __future__ import annotations

import copy
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree

from . import contracts as P2_CONTRACTS
from . import static_identity as SI
from .mechanism_program import MechanismProgram


@dataclass(frozen=True)
class PreparedMechanismRound:
    """Captured snapshot awaiting the caller's candidate-scope admission."""

    snapshot: Path


class MechanismRounds:
    """Own the open/closed round lifecycle and exact attribution receipts."""

    def __init__(self, program: MechanismProgram) -> None:
        self.program = program
        self._active: dict[str, Any] | None = None
        self._prepared: PreparedMechanismRound | None = None
        self._pending_start: dict[str, Any] | None = None
        self._pending_candidate: Path | None = None

    @property
    def configured(self) -> bool:
        return self.program.has_catalog or self.program.catalog_binding is not None

    @property
    def active(self) -> dict[str, Any] | None:
        """A detached view of the most recent round's retained state."""
        return copy.deepcopy(self._active)

    @staticmethod
    def validate_index(round_index: int) -> None:
        if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0:
            raise ValueError("compiler mechanism round index must be a nonnegative integer")

    def _write(self, name: str, record: Mapping[str, Any]) -> Path:
        path = self.program.output / name
        with path.open("xb") as stream:
            stream.write(P2_CONTRACTS.canonical_json(record))
        path.chmod(0o444)
        return path

    def capture(
        self,
        candidate: Path,
        *,
        round_index: int,
        compiler_shared_source_root: Path,
    ) -> PreparedMechanismRound:
        """Capture after candidate admission; publication awaits snapshot admission."""
        self.validate_index(round_index)
        self.program.check_integrity()
        if not self.configured:
            raise ValueError("compiler mechanism round requires a frozen mechanism catalog")
        if self._active is not None and self._active.get("closed") is not True:
            raise ValueError("preceding compiler mechanism round is not closed")
        before = hash_tree(candidate)["sha256"]
        dependencies = SI.compiler_dependency_record(candidate, shared_source_root=compiler_shared_source_root)
        snapshot = self.program.output / f"mechanism_round_start_{round_index:04d}"
        shutil.copytree(candidate, snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if hash_tree(snapshot)["sha256"] != before:
            raise ValueError("compiler changed while capturing the mechanism round start")
        for path in snapshot.rglob("*"):
            if path.is_symlink():
                raise ValueError("compiler mechanism round start contains a symlink")
            path.chmod(path.stat().st_mode & ~0o222)
        snapshot.chmod(0o555)
        prepared = PreparedMechanismRound(snapshot)
        self._prepared = prepared
        self._pending_candidate = candidate
        self._pending_start = {
            "round": round_index,
            "candidate_sha256": before,
            "compiler_dependencies": dependencies,
        }
        return prepared

    def publish_start(self, prepared: PreparedMechanismRound) -> dict[str, Any]:
        """Publish the start receipt only after the caller admits its snapshot."""
        self.program.check_integrity()
        if prepared is not self._prepared or self._pending_start is None or self._pending_candidate is None:
            raise ValueError("compiler mechanism round start requires its original capture")
        binding = {
            "schema": "global_compiler_mechanism_round_start_v1",
            **self._pending_start,
            "candidate_path": str(self._pending_candidate.resolve()),
            "round_start_path": str(prepared.snapshot.resolve()),
            "mechanism_catalog_sha256": self.program.catalog_binding_sha256,
            "mechanism_work_order_sha256": self.program.work_order_binding_sha256,
            "edit_authority_sha256": P2_CONTRACTS.document_sha256(self.program.edit_authority.binding),
        }
        receipt = self._write(f"mechanism_round_start_{binding['round']:04d}.json", binding)
        self._active = {
            **copy.deepcopy(binding),
            "closed": False,
            "round_start_receipt": {"path": str(receipt.resolve()), "sha256": P2_CONTRACTS.sha256_file(receipt)},
        }
        self._prepared = None
        self._pending_start = None
        self._pending_candidate = None
        return copy.deepcopy(self._active)

    def inspect(
        self,
        candidate: Path,
        *,
        require_semantic_edit: bool,
        compiler_shared_source_root: Path,
    ) -> dict[str, Any] | None:
        """Attribute current bytes before compilation; initial seed analysis has no round delta."""
        from merlin.perf.compiler_edit_scope import inspect_round_mechanism_edits

        self.program.check_integrity()
        if not self.configured:
            return None
        active = self._active
        candidate_sha256 = hash_tree(candidate)["sha256"]
        if active is None:
            if candidate_sha256 != self.program.edit_authority.binding["initial_candidate_sha256"]:
                raise ValueError("edited compiler analysis has no immutable mechanism round start")
            return {
                "schema": "global_compiler_mechanism_seed_analysis_v1",
                "status": "initial_seed",
                "candidate_sha256": candidate_sha256,
                "mechanism_catalog_sha256": self.program.catalog_binding_sha256,
            }
        if (
            self.program.work_order_binding is not None
            and self.program.analysis_binding is None
            and (require_semantic_edit or candidate_sha256 != active.get("candidate_sha256"))
        ):
            raise ValueError("compiler mechanism work order has no current static-analysis binding")
        start = Path(active["round_start_path"])
        start_receipt = Path(active["round_start_receipt"]["path"])
        if (
            Path(candidate).resolve() != Path(active["candidate_path"])
            or start.is_symlink()
            or not start.is_dir()
            or start.stat().st_mode & 0o222
            or hash_tree(start)["sha256"] != active["candidate_sha256"]
            or SI.compiler_dependency_record(start, shared_source_root=compiler_shared_source_root)
            != active["compiler_dependencies"]
            or start_receipt.is_symlink()
            or not start_receipt.is_file()
            or P2_CONTRACTS.sha256_file(start_receipt) != active["round_start_receipt"]["sha256"]
            or P2_CONTRACTS.mapping_file(start_receipt)
            != {
                key: value
                for key, value in active.items()
                if key not in ("closed", "round_start_receipt", "finalized_candidate_sha256", "final_status")
            }
        ):
            raise ValueError("immutable compiler mechanism round-start binding changed")
        finalized = active.get("finalized_candidate_sha256")
        if finalized is not None and candidate_sha256 != finalized:
            raise ValueError("compiler changed after final mechanism attribution")
        if active.get("closed") is True and active.get("final_status") != "allowed":
            raise ValueError("refused compiler mechanism round cannot be analyzed")
        result = inspect_round_mechanism_edits(
            self.program.edit_authority.seed,
            start,
            candidate,
            self.program.edit_authority.contract,
            self.program.catalog,
        )
        result.update(
            {
                "round": active["round"],
                "round_start_path": str(start),
                "round_start_sha256": active["candidate_sha256"],
                "round_start_receipt": copy.deepcopy(active["round_start_receipt"]),
                "candidate_sha256": candidate_sha256,
                "candidate_compiler_dependencies": SI.compiler_dependency_record(
                    candidate, shared_source_root=compiler_shared_source_root
                ),
                "mechanism_catalog_binding_sha256": self.program.catalog_binding_sha256,
                "mechanism_work_order_binding_sha256": self.program.work_order_binding_sha256,
                "mechanism_work_order_analysis_sha256": (self.program.analysis_binding or {}).get("sha256"),
            }
        )
        if require_semantic_edit and result["semantic_noop"]:
            result["status"] = "refused"
            result["violations"].append({"reason": "authored round has no semantic compiler mechanism delta"})
        return result

    def require_open(self, round_index: int) -> None:
        """Refuse invalid finalization before the caller checks experiment inputs."""
        active = self._active
        if active is None or active.get("round") != round_index or active.get("closed") is True:
            raise ValueError("compiler mechanism round finalization has no matching open round")

    def finalize(
        self,
        candidate: Path,
        *,
        round_index: int,
        compiler_shared_source_root: Path,
    ) -> dict[str, Any] | None:
        """Persist attribution and close the round for these exact candidate bytes."""
        if not self.configured:
            self.program.check_integrity()
            return None
        self.require_open(round_index)
        result = self.inspect(
            candidate,
            require_semantic_edit=True,
            compiler_shared_source_root=compiler_shared_source_root,
        )
        assert result is not None
        path = self._write(f"mechanism_round_{round_index:04d}.json", result)
        self._active["closed"] = True
        self._active["finalized_candidate_sha256"] = result["candidate_sha256"]
        self._active["final_status"] = result["status"]
        return {
            **copy.deepcopy(result),
            "receipt": {"path": str(path.resolve()), "sha256": P2_CONTRACTS.sha256_file(path)},
        }
