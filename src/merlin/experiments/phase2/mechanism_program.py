"""Host-frozen mechanism catalogs, portfolio work orders and static-analysis bindings.

The caller owns experiment-wide admission and its timing, candidate scope checks,
round attribution and revision publication. This owner grants no edit authority.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_bytes

from . import broker_evidence as BE
from . import contracts as P2_CONTRACTS
from .edit_authority import FrozenEditAuthority
from .revision_journal import RevisionJournal


@dataclass(frozen=True)
class PreparedWorkOrder:
    """Source preflight retained across the caller's candidate admission."""

    source: Path
    source_sha256: str


class MechanismProgram:
    """Own one frozen mechanism assignment and its exact portfolio evidence."""

    def __init__(
        self,
        output: Path,
        edit_authority: FrozenEditAuthority,
        *,
        portfolio_identity: Mapping[str, Any],
        portfolio_identity_sha256: str,
    ) -> None:
        self.output = output
        self.edit_authority = edit_authority
        self.portfolio_identity = copy.deepcopy(dict(portfolio_identity))
        self.portfolio_identity_sha256 = portfolio_identity_sha256
        self.catalog: dict[str, Any] | None = None
        self.catalog_binding: dict[str, Any] | None = None
        self._catalog_binding_sha256: str | None = None
        self.work_order: dict[str, Any] | None = None
        self.work_order_binding: dict[str, Any] | None = None
        self._work_order_binding_sha256: str | None = None
        self.analysis_binding: dict[str, Any] | None = None
        self._catalog_started = False
        self._work_order_started = False
        self._analysis_started = False
        self._prepared_work_order: PreparedWorkOrder | None = None

    @property
    def catalog_binding_sha256(self) -> str | None:
        """Original catalog receipt identity, independent of its mutable view."""
        return self._catalog_binding_sha256

    @property
    def work_order_binding_sha256(self) -> str | None:
        """Original work-order receipt identity, independent of its mutable view."""
        return self._work_order_binding_sha256

    @property
    def has_catalog(self) -> bool:
        """Retain configured state even if the public catalog metadata is cleared."""
        return self._catalog_started

    @property
    def has_work_order(self) -> bool:
        """Retain configured state even if public metadata is cleared."""
        return self._work_order_started

    def _write(self, name: str, record: Mapping[str, Any]) -> Path:
        path = self.output / name
        with path.open("xb") as stream:
            stream.write(P2_CONTRACTS.canonical_json(record))
        path.chmod(0o444)
        return path

    def check_integrity(self) -> None:
        """Verify retained program evidence independently of optional public views."""
        self.edit_authority.check_integrity()
        if P2_CONTRACTS.document_sha256(self.portfolio_identity) != self.portfolio_identity_sha256:
            raise ValueError("host-frozen mechanism portfolio identity changed")
        groups = (
            (self._catalog_started, (self.catalog, self.catalog_binding, self.catalog_binding_sha256)),
            (self._work_order_started, (self.work_order, self.work_order_binding, self.work_order_binding_sha256)),
            (self._analysis_started, (self.analysis_binding,)),
        )
        for started, values in groups:
            if (started and any(value is None for value in values)) or (
                not started and any(value is not None for value in values)
            ):
                raise ValueError("host-frozen mechanism program is incomplete or changed")
        if self.catalog_binding is not None:
            from merlin.perf.compiler_edit_scope import validate_mechanism_catalog

            binding = self.catalog_binding
            frozen = Path(binding["frozen_path"])
            source = Path(binding["source_path"])
            receipt = self.output / "compiler_mechanism_catalog_receipt.json"
            if (
                self.edit_authority.contract is None
                or self.catalog is None
                or self.catalog_binding_sha256 is None
                or binding.get("sha256") != self.catalog_binding_sha256
                or binding.get("sha256")
                != P2_CONTRACTS.document_sha256({key: value for key, value in binding.items() if key != "sha256"})
                or binding.get("contract_document_sha256") != self.edit_authority.binding["contract_document_sha256"]
                or binding.get("initial_candidate_sha256") != self.edit_authority.binding["initial_candidate_sha256"]
                or binding.get("catalog") != self.catalog
                or P2_CONTRACTS.document_sha256(self.catalog) != binding.get("catalog_document_sha256")
                or frozen != self.output / "compiler_mechanism_catalog.json"
                or frozen.is_symlink()
                or not frozen.is_file()
                or frozen.stat().st_mode & 0o222
                or P2_CONTRACTS.sha256_file(frozen) != binding.get("canonical_bytes_sha256")
                or P2_CONTRACTS.mapping_file(frozen) != self.catalog
                or receipt.is_symlink()
                or not receipt.is_file()
                or P2_CONTRACTS.mapping_file(receipt) != binding
                or not source.is_absolute()
                or source.resolve() != source
                or source.is_symlink()
                or not source.is_file()
                or source.stat().st_mode & 0o222
                or P2_CONTRACTS.sha256_file(source) != binding.get("source_file_sha256")
            ):
                raise ValueError("host-frozen compiler mechanism catalog changed")
            validate_mechanism_catalog(self.catalog, self.edit_authority.seed, self.edit_authority.contract)
        if self.work_order_binding is not None:
            binding = self.work_order_binding
            frozen = Path(binding["frozen_path"])
            source = Path(binding["source_path"])
            receipt = self.output / "compiler_mechanism_work_order_receipt.json"
            if (
                self.catalog_binding is None
                or self.work_order is None
                or self.work_order_binding_sha256 is None
                or binding.get("sha256") != self.work_order_binding_sha256
                or binding.get("sha256")
                != P2_CONTRACTS.document_sha256({key: value for key, value in binding.items() if key != "sha256"})
                or binding.get("mechanism_catalog_binding_sha256") != self.catalog_binding_sha256
                or binding.get("initial_candidate_sha256") != self.edit_authority.binding["initial_candidate_sha256"]
                or binding.get("portfolio_sha256") != self.portfolio_identity_sha256
                or binding.get("work_order") != self.work_order
                or P2_CONTRACTS.document_sha256(self.work_order) != binding.get("work_order_document_sha256")
                or frozen != self.output / "compiler_mechanism_work_order.json"
                or frozen.is_symlink()
                or not frozen.is_file()
                or frozen.stat().st_mode & 0o222
                or P2_CONTRACTS.sha256_file(frozen) != binding.get("canonical_bytes_sha256")
                or P2_CONTRACTS.mapping_file(frozen) != self.work_order
                or receipt.is_symlink()
                or not receipt.is_file()
                or P2_CONTRACTS.mapping_file(receipt) != binding
                or not source.is_absolute()
                or source.resolve() != source
                or source.is_symlink()
                or not source.is_file()
                or source.stat().st_mode & 0o222
                or P2_CONTRACTS.sha256_file(source) != binding.get("source_file_sha256")
            ):
                raise ValueError("host-frozen compiler mechanism work order changed")
            self.validate_work_order(
                self.work_order, candidate_sha256=self.edit_authority.binding["initial_candidate_sha256"]
            )
            if self.analysis_binding is not None:
                analysis_receipt = self.output / "compiler_mechanism_work_order_analysis.json"
                analysis_binding = self.analysis_binding
                if (
                    analysis_binding.get("sha256")
                    != P2_CONTRACTS.document_sha256(
                        {key: value for key, value in analysis_binding.items() if key != "sha256"}
                    )
                    or analysis_binding.get("work_order_binding_sha256") != self.work_order_binding_sha256
                    or analysis_receipt.is_symlink()
                    or not analysis_receipt.is_file()
                    or P2_CONTRACTS.mapping_file(analysis_receipt) != analysis_binding
                ):
                    raise ValueError("compiler mechanism work-order analysis binding changed")

    def freeze_catalog(self, source: Path, source_sha256: str, *, has_iterations: bool) -> dict[str, Any]:
        """Freeze one explicit read-only host catalog; its selectors grant no edit authority."""
        from merlin.perf.compiler_edit_scope import validate_mechanism_catalog

        source = Path(source)
        self.check_integrity()
        if self.edit_authority.contract is None:
            raise ValueError("compiler mechanism catalog requires frozen compiler edit authority")
        if has_iterations or self._catalog_started or self.catalog_binding is not None:
            raise ValueError("compiler mechanism catalog must be frozen once before candidate execution")
        if (
            not BE._is_sha256(source_sha256)
            or not source.is_absolute()
            or source.resolve() != source
            or source.is_symlink()
            or not source.is_file()
            or source.stat().st_mode & 0o222
            or P2_CONTRACTS.sha256_file(source) != source_sha256
        ):
            raise ValueError("compiler mechanism catalog is not an exact immutable absolute file")
        if source.is_relative_to(self.edit_authority.seed.resolve()) or source.is_relative_to(
            self.edit_authority.initial_source
        ):
            raise ValueError("compiler mechanism catalog cannot originate in candidate-editable source")
        raw = source.read_bytes()
        if len(raw) > 4_000_000:
            raise ValueError("compiler mechanism catalog exceeds the host metadata bound")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("compiler mechanism catalog must be one JSON object") from exc
        if not isinstance(document, Mapping):
            raise ValueError("compiler mechanism catalog must be one JSON object")
        validated = validate_mechanism_catalog(document, self.edit_authority.seed, self.edit_authority.contract)
        canonical = P2_CONTRACTS.canonical_json(validated)
        frozen = self.output / "compiler_mechanism_catalog.json"
        self._catalog_started = True
        with frozen.open("xb") as stream:
            stream.write(canonical)
        frozen.chmod(0o444)
        body = {
            "schema": "host_frozen_compiler_mechanism_catalog_v1",
            "source_path": str(source),
            "source_file_sha256": source_sha256,
            "frozen_path": str(frozen),
            "canonical_bytes_sha256": sha256_bytes(canonical),
            "catalog_document_sha256": P2_CONTRACTS.document_sha256(validated),
            "catalog_declared_sha256": validated["sha256"],
            "contract_document_sha256": self.edit_authority.binding["contract_document_sha256"],
            "initial_candidate_sha256": self.edit_authority.binding["initial_candidate_sha256"],
            "catalog": validated,
            "permission_scope": "mechanism attribution only; cumulative edit authority unchanged",
        }
        binding = {**body, "sha256": P2_CONTRACTS.document_sha256(body)}
        self.catalog = validated
        self.catalog_binding = binding
        self._catalog_binding_sha256 = binding["sha256"]
        self._write("compiler_mechanism_catalog_receipt.json", binding)
        self.check_integrity()
        return copy.deepcopy(binding)

    def validate_work_order(self, document: Mapping[str, Any], *, candidate_sha256: str) -> dict[str, Any]:
        """Validate the host's executable per-portfolio mechanism assignment."""
        if document.get("schema") != "host_prepared_mechanism_work_order_v1":
            raise ValueError("compiler mechanism work order has an unsupported schema")
        declared = document.get("sha256")
        if (
            not BE._is_sha256(declared)
            or P2_CONTRACTS.document_sha256({key: value for key, value in document.items() if key != "sha256"})
            != declared
        ):
            raise ValueError("compiler mechanism work-order document hash changed")
        if self.catalog is None or self.edit_authority.contract is None:
            raise ValueError("compiler mechanism work order requires the frozen catalog and authority")
        mechanisms = self.catalog.get("mechanisms")
        mechanism_id = document.get("mechanism_id")
        if not isinstance(mechanisms, list) or len(mechanisms) != 1 or mechanisms[0].get("id") != mechanism_id:
            raise ValueError("compiler mechanism work order does not exactly match the active catalog")
        if document.get("catalog_sha256") != self.catalog.get("sha256") or document.get(
            "contract_sha256"
        ) != self.edit_authority.contract.get("sha256"):
            raise ValueError("compiler mechanism work-order catalog or contract binding changed")
        if (
            document.get("initial_candidate_sha256") != candidate_sha256
            or document.get("round_start_candidate_sha256") != candidate_sha256
        ):
            raise ValueError("compiler mechanism work-order candidate binding changed")
        if (
            document.get("portfolio_sha256") != self.portfolio_identity_sha256
            or document.get("ordered_portfolio") != self.portfolio_identity["members"]
        ):
            raise ValueError("compiler mechanism work-order portfolio binding changed")
        if document.get("status") != "ready_for_authoring":
            raise ValueError("compiler mechanism work order is not host-ready for authoring")
        # Source-operation indices are graph-local and may repeat.  A flat list can silently bind
        # one model's index to another model, so only exact ordered per-member rows are executable.
        if document.get("source_operation_ids") not in (None, []):
            raise ValueError("compiler mechanism work order uses ambiguous flat source-operation IDs")
        rows = document.get("portfolio_site_bindings")
        if not isinstance(rows, list) or len(rows) != len(self.portfolio_identity["members"]):
            raise ValueError("compiler mechanism work order lacks every portfolio site binding")
        any_site = False
        for expected, row in zip(self.portfolio_identity["members"], rows, strict=True):
            if not isinstance(row, Mapping):
                raise ValueError("compiler mechanism work-order site binding is malformed")
            hashes = (
                "compiler_sha256",
                "source_sha256",
                "plan_digest",
                "candidate_command_buffer_sha256",
                "candidate_lowered_sha256",
            )
            operation_ids, chains = row.get("source_operation_ids"), row.get("chains")
            if (
                row.get("capsule") != expected["capsule"]
                or row.get("capsule_sha256") != expected["capsule_sha256"]
                or row.get("compiler_sha256") != candidate_sha256
                or any(not BE._is_sha256(row.get(name)) for name in hashes)
                or not isinstance(row.get("status"), str)
                or not row["status"]
                or not isinstance(row.get("inventory"), Mapping)
                or not isinstance(operation_ids, list)
                or not isinstance(chains, list)
                or any(
                    not isinstance(value, (str, int))
                    or isinstance(value, bool)
                    or (isinstance(value, str) and not value)
                    or (isinstance(value, int) and value < 0)
                    for value in operation_ids
                )
                or len({(type(value).__name__, value) for value in operation_ids}) != len(operation_ids)
                or any(not isinstance(chain, Mapping) for chain in chains)
            ):
                raise ValueError("compiler mechanism work-order per-member identity/hash/site binding changed")
            any_site = any_site or bool(operation_ids) or bool(chains)
        if not any_site:
            raise ValueError("compiler mechanism work order has no host-bound source sites")
        return copy.deepcopy(dict(document))

    def prepare_work_order(self, source: Path, source_sha256: str, *, has_iterations: bool) -> PreparedWorkOrder:
        """Check source/lifecycle before the caller admits the candidate; do not parse yet."""
        source = Path(source)
        self.check_integrity()
        if self.catalog_binding is None:
            raise ValueError("compiler mechanism work order requires a frozen mechanism catalog")
        if has_iterations or self._work_order_started or self.work_order_binding is not None:
            raise ValueError("compiler mechanism work order must be frozen once before candidate execution")
        if (
            not BE._is_sha256(source_sha256)
            or not source.is_absolute()
            or source.resolve() != source
            or source.is_symlink()
            or not source.is_file()
            or source.stat().st_mode & 0o222
            or P2_CONTRACTS.sha256_file(source) != source_sha256
        ):
            raise ValueError("compiler mechanism work order is not an exact immutable absolute file")
        if source.is_relative_to(self.edit_authority.seed.resolve()) or source.is_relative_to(
            self.edit_authority.initial_source
        ):
            raise ValueError("compiler mechanism work order cannot originate in candidate-editable source")
        prepared = PreparedWorkOrder(source, source_sha256)
        self._prepared_work_order = prepared
        return prepared

    def freeze_work_order(self, prepared: PreparedWorkOrder, *, candidate_sha256: str) -> dict[str, Any]:
        """Parse and freeze the prepared assignment after candidate admission."""
        self.check_integrity()
        if prepared is not self._prepared_work_order or self._work_order_started:
            raise ValueError("compiler mechanism work order requires its original source preflight")
        source, source_sha256 = prepared.source, prepared.source_sha256
        raw = source.read_bytes()
        if sha256_bytes(raw) != source_sha256:
            raise ValueError("compiler mechanism work-order source changed after preflight")
        if len(raw) > 8_000_000:
            raise ValueError("compiler mechanism work order exceeds the host metadata bound")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("compiler mechanism work order must be one JSON object") from exc
        if not isinstance(document, Mapping):
            raise ValueError("compiler mechanism work order must be one JSON object")
        validated = self.validate_work_order(document, candidate_sha256=candidate_sha256)
        canonical = P2_CONTRACTS.canonical_json(validated)
        frozen = self.output / "compiler_mechanism_work_order.json"
        self._work_order_started = True
        with frozen.open("xb") as stream:
            stream.write(canonical)
        frozen.chmod(0o444)
        body = {
            "schema": "host_frozen_compiler_mechanism_work_order_v1",
            "source_path": str(source),
            "source_file_sha256": source_sha256,
            "frozen_path": str(frozen),
            "canonical_bytes_sha256": sha256_bytes(canonical),
            "work_order_document_sha256": P2_CONTRACTS.document_sha256(validated),
            "work_order_declared_sha256": validated["sha256"],
            "mechanism_catalog_binding_sha256": self.catalog_binding_sha256,
            "initial_candidate_sha256": candidate_sha256,
            "portfolio_sha256": self.portfolio_identity_sha256,
            "work_order": validated,
            "permission_scope": "host work assignment only; grants no edit or measurement authority",
        }
        binding = {**body, "sha256": P2_CONTRACTS.document_sha256(body)}
        self.work_order = validated
        self.work_order_binding = binding
        self._work_order_binding_sha256 = binding["sha256"]
        self._write("compiler_mechanism_work_order_receipt.json", binding)
        self.check_integrity()
        return copy.deepcopy(binding)

    def bind_analysis(
        self, record: Mapping[str, Any], *, immutable_iteration: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Bind every assigned graph-local site row to the exact current static artifacts."""
        self.check_integrity()
        if self.work_order_binding is None:
            return None
        if self.analysis_binding is not None:
            immutable = immutable_iteration
            # Semantic and probe actions deliberately add elapsed time and receipt references to
            # the in-memory row after iteration_NNNN.json was sealed.  Those mutable action fields
            # are not static-analysis evidence. The caller resolves the persisted iteration through
            # its exact reuse verifier; require every static identity carried by
            # the caller to match it.  This permits continuation accounting without accepting a
            # substituted candidate, portfolio, analysis, or mechanism binding.
            static_fields = (
                "schema",
                "iteration",
                "candidate_sha256",
                "submitted_snapshot",
                "compiler_dependencies",
                "analysis_reuse_binding",
                "cross_run_static_analysis_binding",
                "baseline_sha256",
                "optimization_baseline_sha256",
                "optimization_baseline",
                "compiler_mechanism_catalog",
                "compiler_mechanism_work_order",
                "mechanism_work_order_analysis",
                "round_mechanism_attribution",
                "analysis",
                "readiness",
                "portfolio",
                "static_comparison",
                "static_analysis_bundle",
                "relative_semantic_evidence",
            )
            if (
                immutable is None
                or P2_CONTRACTS.document_sha256({name: record.get(name) for name in static_fields})
                != P2_CONTRACTS.document_sha256({name: immutable.get(name) for name in static_fields})
                or (immutable.get("portfolio") or {}).get("portfolio_sha256") != self.portfolio_identity_sha256
            ):
                raise ValueError("compiler mechanism work order has no immutable current portfolio analysis")
            return copy.deepcopy(self.analysis_binding)
        if (
            record.get("candidate_sha256") != self.work_order["round_start_candidate_sha256"]
            or (record.get("portfolio") or {}).get("portfolio_sha256") != self.portfolio_identity_sha256
        ):
            raise ValueError("compiler mechanism work order is not bound to the current portfolio analysis")
        members = []
        for index, (sentinel, site) in enumerate(
            zip(self.portfolio_identity["members"], self.work_order["portfolio_site_bindings"], strict=True)
        ):
            analysis = RevisionJournal.portfolio_member_analysis(record, index)
            plan = (analysis.get("diagnostics") or {}).get("verified_global_plan_emission") or {}
            emission = analysis.get("emission") or {}
            expected = {
                "compiler_sha256": record.get("candidate_sha256"),
                "source_sha256": plan.get("source_sha256"),
                "plan_digest": plan.get("plan_digest"),
                "candidate_command_buffer_sha256": emission.get("candidate_command_buffer_sha256"),
                "candidate_lowered_sha256": emission.get("candidate_lowered_sha256"),
            }
            if (
                analysis.get("candidate_sha256") != record.get("candidate_sha256")
                or (analysis.get("workload") or {}).get("capsule_sha256") != sentinel["capsule_sha256"]
                or any(site.get(name) != value or not BE._is_sha256(value) for name, value in expected.items())
            ):
                raise ValueError("compiler mechanism work-order member evidence differs from current analysis")
            members.append(
                {
                    "capsule": sentinel["capsule"],
                    "capsule_sha256": sentinel["capsule_sha256"],
                    **expected,
                    "analysis_sha256": P2_CONTRACTS.document_sha256(analysis),
                    "site_binding_sha256": P2_CONTRACTS.document_sha256(site),
                }
            )
        body = {
            "schema": "compiler_mechanism_work_order_analysis_binding_v1",
            "work_order_binding_sha256": self.work_order_binding_sha256,
            "candidate_sha256": record["candidate_sha256"],
            "portfolio_sha256": self.portfolio_identity_sha256,
            "iteration": record.get("iteration"),
            "members": members,
        }
        iteration = record.get("iteration")
        iteration_path = (
            self.output / f"iteration_{iteration:04d}.json"
            if isinstance(iteration, int) and not isinstance(iteration, bool)
            else None
        )
        if (
            iteration_path is None
            or iteration_path.is_symlink()
            or not iteration_path.is_file()
            or P2_CONTRACTS.document_sha256(P2_CONTRACTS.mapping_file(iteration_path))
            != P2_CONTRACTS.document_sha256(record)
        ):
            raise ValueError("compiler mechanism work order has no immutable analysis iteration")
        body["iteration_record"] = {
            "path": str(iteration_path.resolve()),
            "sha256": P2_CONTRACTS.sha256_file(iteration_path),
        }
        binding = {**body, "sha256": P2_CONTRACTS.document_sha256(body)}
        self._analysis_started = True
        self.analysis_binding = binding
        self._write("compiler_mechanism_work_order_analysis.json", binding)
        self.check_integrity()
        return copy.deepcopy(binding)
