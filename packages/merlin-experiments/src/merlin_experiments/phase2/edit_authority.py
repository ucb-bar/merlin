"""Frozen compiler edit authority, cumulative edit inspection and host guidance.

The caller retains experiment-wide input admission and its observation timing.
This owner binds the approved contract to its immutable initial source and uses
canonical compiler-edit and guidance primitives; mechanism attribution is separate.
"""

from __future__ import annotations

import copy
import json
import shutil
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.perf.agent_guidance import inspect_compiler_package

from . import contracts as P2_CONTRACTS


def _readonly_seed_identity(root: Path) -> tuple[str, tuple[tuple[str, int], ...]]:
    """Bind all seed bytes and entries, including digest-excluded and empty directories."""
    entries = []
    for path in (root, *sorted(root.rglob("*"))):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("host-frozen compiler edit seed is absent, linked, or changed")
        mode = path.stat().st_mode
        if mode & 0o222:
            raise ValueError("host-frozen compiler edit seed permissions changed")
        entries.append((path.relative_to(root).as_posix(), stat.S_IFMT(mode) | stat.S_IMODE(mode)))
    return P2_CONTRACTS.exact_tree_record(root)["sha256"], tuple(entries)


def _guidance_contract_identity(path: Path | None) -> tuple[Path, str] | None:
    if path is None:
        return None
    try:
        if path.is_symlink() or not path.is_dir() or path.resolve() != path:
            raise ValueError("host-frozen compiler guidance contract is linked or changed")
        return path, P2_CONTRACTS.sha256_file(path / "schemas/manifest.schema.json")
    except (OSError, P2_CONTRACTS.StageGateError) as exc:
        raise ValueError("host-frozen compiler guidance contract is absent or unreadable") from exc


@dataclass(frozen=True)
class _AuthoritySeal:
    binding_bytes: bytes
    output: Path
    seed: Path
    initial_source: Path
    seed_identity: tuple[str, tuple[tuple[str, int], ...]]
    guidance_contract: tuple[Path, str] | None


class FrozenEditAuthority:
    """Own one host-approved contract and the seed against which edits are inspected."""

    def __init__(self, output: Path, *, guidance_contract: Path | None = None) -> None:
        """Select a contract root containing schemas/manifest.schema.json, or the legacy default."""
        self.output = output
        if guidance_contract is not None and Path(guidance_contract).is_symlink():
            raise ValueError("compiler guidance contract cannot be a symlink")
        self.guidance_contract = Path(guidance_contract).resolve() if guidance_contract is not None else None
        self.contract: dict[str, Any] | None = None
        self.seed: Path | None = None
        self.initial_source: Path | None = None
        self.guidance_inventory = None
        self.binding: dict[str, Any] | None = None
        self._freeze_started = False
        self._seal: _AuthoritySeal | None = None

    @property
    def configured(self) -> bool:
        """A freeze attempt cannot be disabled by clearing mutable public metadata."""
        return self._freeze_started

    def _write(self, name: str, record: Mapping[str, Any]) -> Path:
        path = self.output / name
        payload = P2_CONTRACTS.canonical_json(record)
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o444)
        return path

    def freeze(
        self,
        candidate: Path,
        contract: Mapping[str, Any],
        *,
        has_iterations: bool,
        source_pins: Mapping[str, str] | None = None,
        host_surface_declarations: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Host startup only: retain initial source and its externally approved edit authority."""
        from merlin.perf.compiler_edit_scope import validate_edit_contract

        if has_iterations or self.configured or self.contract is not None:
            raise ValueError("compiler edit authority must be frozen once before candidate execution")
        self.check_integrity()
        guidance_contract = _guidance_contract_identity(self.guidance_contract)
        if any(path.is_symlink() for path in candidate.rglob("*")):
            raise ValueError("initial compiler edit source contains a symlink")
        # A REVIEWED DECLARATION GOES THROUGH ITS OWN LOADER FIRST. `validate_edit_contract` checks
        # that every named symbol EXISTS; it cannot check the converse -- that the symbol which
        # DECIDES the thing under study was named -- because it has no idea what the experiment is
        # about. A declaration omitting that symbol validates, runs, produces rounds, edits and a
        # verdict, and was unwinnable the whole time, which in the artifact is indistinguishable
        # from an agent that tried and failed. `phase2_edit_contract.load` is the refusal for that,
        # and the reviewed declarations under merlin/contract/phase2_edit_contracts/ reached this
        # enforcement without ever passing it.
        #
        # Routed on the two annotation keys a DECLARATION carries and an inventory-built contract
        # does not, so the automatic path is untouched; `load` re-seals with the identical digest
        # `validate_edit_contract` recomputes, so a contract that was already valid is unchanged.
        if contract.get("target") and contract.get("package_id"):
            from merlin.perf import phase2_edit_contract as P2C

            contract = P2C.load(str(contract["target"]), str(contract["package_id"]), body=contract)
        validated = validate_edit_contract(contract, candidate)
        guidance = None
        if host_surface_declarations is not None:
            guidance = inspect_compiler_package(
                candidate,
                host_surface_declarations=host_surface_declarations,
                **({"contract": self.guidance_contract} if self.guidance_contract is not None else {}),
            )
            authorized = {
                (owner["surface_id"], owner["path"], owner["symbol"]) for owner in validated["existing_symbols"]
            }
            if any((surface.id, surface.path, surface.symbol) not in authorized for surface in guidance.surfaces):
                raise ValueError("host guidance surface lies outside frozen edit authority")
        if source_pins is not None:
            actual = {
                path.relative_to(candidate).as_posix(): P2_CONTRACTS.sha256_file(path)
                for path in candidate.rglob("*")
                if path.is_file()
                and ".git" not in path.relative_to(candidate).parts
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            }
            if actual != dict(source_pins):
                raise ValueError("host edit catalog source-file pins do not match the initial compiler")
        seed = self.output / "edit_scope_seed"
        # Once capture starts, failure requires a fresh owner/output root. Never reuse
        # partial evidence or reinterpret a failed freeze as development-only authority.
        self._freeze_started = True
        shutil.copytree(candidate, seed, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path in seed.rglob("*"):
            if path.is_symlink():
                raise ValueError("initial compiler edit source contains a symlink")
            path.chmod(path.stat().st_mode & ~0o222)
        seed.chmod(0o555)
        binding = {
            "schema": "host_frozen_compiler_edit_authority_v1",
            "initial_candidate_sha256": hash_tree(seed)["sha256"],
            "contract_document_sha256": P2_CONTRACTS.document_sha256(validated),
            "contract": validated,
            "seed_path": str(seed),
            "source_pins_checked": source_pins is not None,
        }
        if guidance is not None:
            binding["guidance_inventory"] = guidance.to_dict()
            binding["guidance_inventory_sha256"] = P2_CONTRACTS.document_sha256(guidance.to_dict())
        if guidance_contract is not None:
            binding["guidance_contract_sha256"] = guidance_contract[1]
        if _guidance_contract_identity(self.guidance_contract) != guidance_contract:
            raise ValueError("host-frozen compiler guidance contract changed during freeze")
        seal = _AuthoritySeal(
            P2_CONTRACTS.canonical_json(binding),
            self.output,
            seed,
            candidate.resolve(),
            _readonly_seed_identity(seed),
            guidance_contract,
        )
        self._write("compiler_edit_authority.json", binding)
        self.contract, self.seed = validated, seed
        self.initial_source = seal.initial_source
        self.guidance_inventory = guidance
        self.binding, self._seal = binding, seal
        return copy.deepcopy(self.binding)

    def check_integrity(self) -> None:
        """Rehash the frozen authority at the caller's immutable-input checkpoint."""
        if not self.configured:
            if any(
                value is not None
                for value in (
                    self.contract,
                    self.seed,
                    self.initial_source,
                    self.guidance_inventory,
                    self.binding,
                    self._seal,
                )
            ):
                raise ValueError("host-frozen compiler edit authority changed without a completed freeze")
            return
        seal = self._seal
        if seal is None:
            raise ValueError("host-frozen compiler edit authority freeze is incomplete")
        try:
            expected = json.loads(seal.binding_bytes)
            if (
                self.output != seal.output
                or self.seed != seal.seed
                or self.initial_source != seal.initial_source
                or self.guidance_contract != (seal.guidance_contract[0] if seal.guidance_contract is not None else None)
                or _guidance_contract_identity(self.guidance_contract) != seal.guidance_contract
                or P2_CONTRACTS.canonical_json(self.binding) != seal.binding_bytes
                or P2_CONTRACTS.document_sha256(self.contract) != expected["contract_document_sha256"]
            ):
                raise ValueError("host-frozen compiler edit authority changed")
            receipt = seal.output / "compiler_edit_authority.json"
            if (
                receipt.is_symlink()
                or not receipt.is_file()
                or receipt.stat().st_mode & 0o222
                or receipt.read_bytes() != seal.binding_bytes
                or _readonly_seed_identity(seal.seed) != seal.seed_identity
            ):
                raise ValueError("host-frozen compiler edit authority receipt or seed changed")
            guidance = expected.get("guidance_inventory_sha256")
            actual_guidance = (
                P2_CONTRACTS.document_sha256(self.guidance_inventory.to_dict())
                if self.guidance_inventory is not None
                else None
            )
            if guidance != actual_guidance:
                raise ValueError("host-frozen compiler guidance changed")
        except (OSError, TypeError, P2_CONTRACTS.StageGateError) as exc:
            raise ValueError("host-frozen compiler edit authority changed or is unreadable") from exc

    def validate_candidate(self, candidate: Path) -> dict[str, Any]:
        """Inspect cumulative edits after the caller admits experiment-wide inputs."""
        from merlin.perf.compiler_edit_scope import inspect_compiler_edits

        self.check_integrity()
        if self.contract is None:
            return {"status": "unconfigured_development_only"}
        result = inspect_compiler_edits(self.seed, candidate, self.contract)
        if result["status"] != "allowed":
            self._write(
                f"edit_scope_refusal_{time.time_ns()}.json",
                {**result, "candidate_sha256": hash_tree(candidate)["sha256"]},
            )
            raise ValueError("candidate edit exceeds host-frozen authority: " + str(result["violations"]))
        return result

    def inspect_optimization_surfaces(self, candidate: Path) -> dict[str, Any]:
        """Expose current AST locations with host-frozen semantics, never self-granted permissions."""
        self.check_integrity()
        if self.guidance_inventory is None:
            return inspect_compiler_package(
                candidate,
                **({"contract": self.guidance_contract} if self.guidance_contract is not None else {}),
            ).to_dict()
        inventory = inspect_compiler_package(
            candidate,
            host_surface_declarations=[surface.to_dict() for surface in self.guidance_inventory.surfaces],
            **({"contract": self.guidance_contract} if self.guidance_contract is not None else {}),
        )
        return {
            **inventory.to_dict(),
            "host_guidance_binding": {
                "inventory_sha256": self.binding["guidance_inventory_sha256"],
                "contract_document_sha256": self.binding["contract_document_sha256"],
                "permission_scope": "unchanged host-frozen edit contract",
                "source_scope": "current candidate AST locations; semantics from frozen host inventory",
            },
        }
