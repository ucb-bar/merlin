"""Host-owned common transformation service on the original selfcheck channel."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import yaml

from merlin.common import bundle_snapshot as BW
from merlin.targetgen.tool_profile import verify_tool_profile

ACTION = "common-transform"
SCHEMA = "merlin.common_transform_request.v1"
MAX_REQUEST_BYTES = 256 * 1024
# V1--V5 request membership is frozen independently of the installed dataclass.
# Adding a host option must not silently grant it to historical experiments.
_FROZEN_PIPELINE_OPTIONS = frozenset(
    {
        "index_bits",
        "tile_iterations",
        "normalize_shapes",
        "simplify_integers",
        "reuse_buffers",
        "dependency_orders",
        "eliminate_redundancy",
        "hoist_invariants",
        "unroll_factors",
        "max_added_operations",
    }
)


def _frozen_import_root(root, originals, frozen):
    """Map only selected import files; an empty declared root supplies no imports."""
    matching = [
        target.parents[len(source.relative_to(root).parts) - 1]
        for source, target in zip(originals, frozen, strict=True)
        if source.is_relative_to(root)
    ]
    if not matching:
        return None
    if len(set(matching)) != 1:
        raise ValueError("common proof import root has no exact frozen file closure")
    return matching[0]


class CommonTransformService:
    """The launcher selects immutable inputs; requests cannot select proof policy."""

    def __init__(self, *, workspace: Path, bundle_manifest: Path, repo: Path, graph_binding=None):
        self.workspace, self.bundle_manifest, self.repo = workspace, bundle_manifest, repo
        if (
            bundle_manifest.is_symlink()
            or bundle_manifest.resolve() != bundle_manifest
            or not bundle_manifest.is_file()
        ):
            raise ValueError("common transform service requires the archived ordinary bundle manifest")
        if bundle_manifest.is_relative_to(workspace):
            raise ValueError("common transform authority cannot be an author-writable file")
        self.bundle_bytes = bundle_manifest.read_bytes()
        self.bundle = yaml.safe_load(self.bundle_bytes)
        from merlin.verify.lean_contract import document_digest

        self.bundle_identity = document_digest(self.bundle)
        self.profile = verify_tool_profile(self.bundle)
        if self.profile is None:
            raise ValueError("common transform service requires the newly versioned compiler substrate")
        marker = BW.verify_bundle_snapshot(workspace, self.bundle, repo=repo)
        self.snapshot_digest = marker["content_sha256"]
        self.snapshot_marker_digest = BW.snapshot_marker_digest(workspace)
        self.graph_binding = graph_binding
        if graph_binding is not None:
            from merlin.verify.lean_graph_pipeline_gate_service import validate_binding

            validate_binding(graph_binding, compiler_profile_sha256=self.profile["sha256"])

    def _verify(self) -> None:
        from merlin.common.paths import module_source_path
        from merlin.verify.lean_contract import document_digest

        if self.bundle_manifest.read_bytes() != self.bundle_bytes:
            raise ValueError("common transform bundle authority changed")
        if document_digest(self.bundle) != self.bundle_identity or verify_tool_profile(self.bundle) != self.profile:
            raise ValueError("common transform parsed bundle or profile authority changed")
        marker = BW.verify_bundle_snapshot(self.workspace, self.bundle, repo=self.repo)
        if (
            marker["content_sha256"] != self.snapshot_digest
            or BW.snapshot_marker_digest(self.workspace) != self.snapshot_marker_digest
        ):
            raise ValueError("common transform frozen source inputs changed")
        for row in self.profile["common_sources"]:
            module = row["module"].removesuffix(".__init__")
            actual = module_source_path(module)
            expected = row["sha256"]
            if self.profile["common_substrate"] in {
                "merlin.common_compiler_substrate.v6",
                "merlin.common_compiler_substrate.v7",
                "merlin.common_compiler_substrate.v8",
            }:
                from .tool_profile_shared import host_source_digest

                expected = host_source_digest(self.profile, row)
            if hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
                raise ValueError("common transform host source differs from the original admitted release")
        for row in self.profile["automatic_checker_sources"]:
            if row["module"] is not None:
                actual = module_source_path(row["module"])
            else:
                root = module_source_path("merlin.verify.lean_checker").parent.parent
                relative = Path(row["path"].removeprefix("merlin/python/merlin/"))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("automatic verifier data path is not a selected source leaf")
                actual = root / relative
            if hashlib.sha256(actual.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("automatic verifier host source differs from the original admitted release")
        if self.profile["common_substrate"] in {
            "merlin.common_compiler_substrate.v6",
            "merlin.common_compiler_substrate.v7",
            "merlin.common_compiler_substrate.v8",
        }:
            resources = self.profile["shared_scheduling_resources"]
            frozen = BW.snapshot_input_paths(
                self.workspace, self.bundle, [Path(row["path"]) for row in resources], repo=self.repo
            )
            for row, path in zip(resources, frozen, strict=True):
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError("shared compiler release resource differs from the original profile")
            if json.loads(frozen[0].read_bytes()) != self.profile["shared_scheduling_library"]:
                raise ValueError("shared compiler release manifest differs from the original profile")

    def _toolchain(self, work_root: Path):
        from merlin.verify.lean_contract import LeanToolchain

        selection = self.profile["lean_toolchain"]
        if selection.get("selection") != "explicit_manifest":
            raise FileNotFoundError("no explicitly selected common formal toolchain")
        candidates = [
            row["path"]
            for role in ("allowed", "host_inputs")
            for row in self.bundle[role]
            if row.get("note") == "selected formal verification toolchain manifest"
        ]
        if len(candidates) != 1:
            raise ValueError("common proof toolchain selection is ambiguous or absent")
        original_manifest = Path(candidates[0])
        [frozen_manifest] = BW.snapshot_input_paths(self.workspace, self.bundle, [original_manifest], repo=self.repo)
        payload = frozen_manifest.read_bytes()
        if hashlib.sha256(payload).hexdigest() != selection["manifest_sha256"]:
            raise ValueError("common proof toolchain manifest changed")
        document = json.loads(payload)

        def original(value):
            path = Path(value)
            return path if path.is_absolute() else original_manifest.parent / path

        executable = document["executable"]
        files = document["files"]
        originals = [original(executable["path"]), *(original(row["path"]) for row in files)]
        frozen = BW.snapshot_input_paths(self.workspace, self.bundle, originals, repo=self.repo)
        solver = document.get("sat_solver")
        if solver is not None:
            if type(solver) is not dict or set(solver) != {"path", "sha256"}:
                raise ValueError("original Lean SAT solver selection is malformed")
            matches = [
                selected
                for source, selected in zip(originals[1:], frozen[1:], strict=True)
                if source == original(solver["path"])
            ]
            if len(matches) != 1:
                raise ValueError("original Lean SAT solver is outside the frozen file inventory")
            solver = (matches[0], solver["sha256"])
        import_roots = []
        for value in document["library_roots"]:
            root = original(value)
            selected_root = _frozen_import_root(root, originals[1:], frozen[1:])
            if selected_root is not None:
                import_roots.append(selected_root)
        toolchain = LeanToolchain(
            executable=frozen[0],
            executable_sha256=executable["sha256"],
            library_roots=tuple(import_roots),
            files=tuple((path, row["sha256"]) for path, row in zip(frozen[1:], files, strict=True)),
            work_root=work_root,
            timeout_s=document.get("timeout_s", 120),
            memory_mb=document.get("memory_mb", 1024),
            dependency_revision=document.get("dependency_revision", ""),
            sat_solver=solver,
            allow_native_reflection=document.get("allow_native_reflection", False),
        )
        toolchain.verify()
        return toolchain

    def compiler_imports(self):
        """Admit exact original public API leaves during ordinary package checks."""
        from merlin.targetgen.compiler_profile_admission import original_compiler_imports

        self._verify()
        return original_compiler_imports(workspace=self.workspace, bundle_manifest=self.bundle_manifest, repo=self.repo)

    def apply(self, request: dict, *, work_root: Path, timeout_s: int | None = None) -> dict:
        """Run trusted transforms and independently check each actual IR change."""
        from merlin.transforms.certificates import ir_text, parse_ir
        from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
        from merlin.verify.lean_checker import accepted_pipeline_receipt, verify_pipeline_change
        from merlin.verify.lean_graph_pipeline_gate_service import (
            record_delivery, validate_binding, validate_options,
        )

        if set(request) != {"schema", "source_ir", "options"} or request["schema"] != SCHEMA:
            raise ValueError("common transform request cannot select semantics, coverage or proof results")
        if len(json.dumps(request).encode()) > MAX_REQUEST_BYTES:
            raise ValueError("common transform input exceeds the fixed byte budget")
        source, options = request["source_ir"], request["options"]
        if not isinstance(source, str) or not source.strip() or not isinstance(options, dict):
            raise ValueError("common transform requires actual structured IR and explicit parameters")
        if set(options) - _FROZEN_PIPELINE_OPTIONS:
            raise ValueError("common transform contains parameters outside the original frozen substrate")
        from merlin.targetgen.tool_profile import (
            COMMON_SUBSTRATE_V2,
            COMMON_SUBSTRATE_V3,
            COMMON_SUBSTRATE_V4,
            COMMON_SUBSTRATE_V5,
            COMMON_SUBSTRATE_V6,
            COMMON_SUBSTRATE_V7,
            COMMON_SUBSTRATE_V8,
        )

        supports_planning = self.profile.get("common_substrate") in {
            COMMON_SUBSTRATE_V2,
            COMMON_SUBSTRATE_V3,
            COMMON_SUBSTRATE_V4,
            COMMON_SUBSTRATE_V5,
            COMMON_SUBSTRATE_V6,
            COMMON_SUBSTRATE_V7,
            COMMON_SUBSTRATE_V8,
        }
        if not supports_planning and set(options) & {
            "eliminate_redundancy",
            "hoist_invariants",
            "unroll_factors",
            "max_added_operations",
        }:
            raise ValueError("transformation parameters are outside the original frozen substrate")
        index_bits = self.profile.get("host_index_bits")
        if type(index_bits) is not int:
            raise ValueError("common transform requires the independently reviewed host ABI index width")
        if "index_bits" in options and options["index_bits"] != index_bits:
            raise ValueError("common transform cannot change the frozen host ABI index width")
        selected = dict(options)
        selected["index_bits"] = index_bits
        for name in (
            "normalize_shapes",
            "simplify_integers",
            "reuse_buffers",
            "eliminate_redundancy",
            "hoist_invariants",
        ):
            if name in selected and type(selected[name]) is not bool:
                raise ValueError("common transform selectors must be booleans")
        tiles = selected.get("tile_iterations", ())
        if (
            not isinstance(tiles, (list, tuple))
            or len(tiles) > 32
            or any(type(tile) is not int or tile <= 0 for tile in tiles)
        ):
            raise ValueError("common transform tiling parameters exceed the admitted request budget")
        selected["tile_iterations"] = tuple(tiles)
        factors = selected.get("unroll_factors", ())
        growth = selected.get("max_added_operations", 4096)
        if (
            not isinstance(factors, (tuple, list))
            or len(factors) > 32
            or any(type(factor) is not int or not 1 <= factor <= 64 for factor in factors)
            or type(growth) is not int
            or not 1 <= growth <= 4096
        ):
            raise ValueError("unrolling exceeds the inherited host compilation budget")
        selected["unroll_factors"] = tuple(factors)
        selected["max_added_operations"] = growth
        orders = selected.get("dependency_orders", ())
        if not isinstance(orders, (list, tuple)) or len(orders) > 32:
            raise ValueError("common dependency order exceeds the admitted request budget")
        parsed_orders = []
        for row in orders:
            if not isinstance(row, (list, tuple)) or len(row) != 2 or type(row[0]) is not int:
                raise ValueError("common dependency order is malformed")
            order = row[1]
            if not isinstance(order, (list, tuple)) or any(type(index) is not int for index in order):
                raise ValueError("common dependency order is malformed")
            parsed_orders.append((row[0], tuple(order)))
        selected["dependency_orders"] = tuple(parsed_orders)
        # Host injection is separate from untrusted request data. No graph
        # selection means refusal, never automatic construction of old SDK policy.
        binding = self.graph_binding
        toolchain = validate_binding(binding, compiler_profile_sha256=self.profile["sha256"], work_root=work_root)
        validate_options(selected)
        if timeout_s is not None and (type(timeout_s) is not int or timeout_s < 1):
            raise ValueError("common transform host budget must be positive")
        started = time.monotonic()
        with BW.verified_snapshot_scope(self.workspace, self.bundle, repo=self.repo):
            self._verify()
            budget = min(timeout_s, binding.timeout_s) if timeout_s is not None else binding.timeout_s
            deadline = started + budget
            receipts = []
            live_receipts = []

            def verify(change):
                if time.monotonic() >= deadline:
                    raise TimeoutError("common transformation exhausted its original host work budget")
                receipt = verify_pipeline_change(change, toolchain=toolchain, deadline=deadline)
                if not accepted_pipeline_receipt(receipt, change, toolchain=toolchain):
                    raise ValueError(
                        "common covered transformation was not established by Lean: "
                        + str(getattr(receipt, "reason", "invalid checker completion"))
                    )
                receipts.append(receipt.to_dict())
                live_receipts.append(receipt)
                return receipt

            result = apply_common_pipeline(parse_ir(source), PipelineOptions(**selected), verifier=verify, plan_only=False)
            if result.plan_only is not False:
                raise ValueError("planning-only data cannot become a checked service result")
            if time.monotonic() > deadline:
                raise TimeoutError("common transformation exceeded its original host work budget")
            self._verify()
            document = {
                "status": "conditional_model_relation" if receipts else "unchanged",
                "scope": (
                    "per-boundary scalar model correspondence" if receipts else "structurally unchanged program"
                ),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "transformed_ir": ir_text(result.module),
                "receipts": receipts,
                "compiler_tool_profile_sha256": self.profile["sha256"],
                "hardware_conformance": "not_established",
                "numerical_execution": "not_established",
                "graph_profile": binding.describe(),
                "compiler_pass_chain": "not_established",
                "saved_report_is_authority": False,
            }
            if self.profile.get("common_substrate") in {
                COMMON_SUBSTRATE_V2,
                COMMON_SUBSTRATE_V3,
                COMMON_SUBSTRATE_V4,
                COMMON_SUBSTRATE_V5,
                COMMON_SUBSTRATE_V6,
                COMMON_SUBSTRATE_V7,
                COMMON_SUBSTRATE_V8,
            }:
                from merlin.perf.pass_observation import observe_changes

                document["pass_observations"] = observe_changes(
                    result.changes, profile_sha256=self.profile["sha256"], index_bits=index_bits
                )
            if time.monotonic() > deadline:
                raise TimeoutError("common transformation observations exceeded its original host work budget")
            if len(json.dumps(document).encode()) > 2 * MAX_REQUEST_BYTES:
                raise ValueError("common transformation output exceeds its inherited byte budget")
            validate_binding(binding, compiler_profile_sha256=self.profile["sha256"], work_root=work_root)
            if receipts:
                record_delivery(binding, document, source, result.changes, live_receipts, deadline=deadline)
            return document

    def compiler_eligible(self, document: dict, *, source_ir: str) -> bool:
        """Only the SAME live checked delivery is eligible under this graph profile.

        This is conditional graph eligibility, not hardware/numerical validation.
        Descriptive JSON/status fields, planning output and local proofs never
        recreate it. The default native hold still prevents admission here.
        """
        from merlin.verify.lean_graph_pipeline_gate_service import eligible_delivery, validate_binding

        try:
            self._verify()
            validate_binding(self.graph_binding, compiler_profile_sha256=self.profile["sha256"])
            return eligible_delivery(self.graph_binding, document, source_ir)
        except (ValueError, RuntimeError, OSError, TypeError, AttributeError, KeyError):
            return False

    def check(self, request: dict, *, work_root: Path, timeout_s: int | None = None) -> dict:
        """Check authored actual IR without accepting author-selected semantics."""
        if self.graph_binding is not None:
            raise ValueError("authored-check is outside the selected scalar graph service profile; no legacy fallback")
        from merlin.targetgen.tool_profile import (
            COMMON_SUBSTRATE_V3,
            COMMON_SUBSTRATE_V4,
            COMMON_SUBSTRATE_V5,
            COMMON_SUBSTRATE_V6,
            COMMON_SUBSTRATE_V7,
            COMMON_SUBSTRATE_V8,
        )
        from merlin.verify.lean_checked import accepted_authored_receipt, verify_authored_transform
        from merlin.verify.lean_contract import document_digest
        from merlin.verify.lean_isa_semantics import reconstruct_isa_semantics

        if self.profile.get("common_substrate") not in {
            COMMON_SUBSTRATE_V3,
            COMMON_SUBSTRATE_V4,
            COMMON_SUBSTRATE_V5,
            COMMON_SUBSTRATE_V6,
            COMMON_SUBSTRATE_V7,
            COMMON_SUBSTRATE_V8,
        }:
            raise ValueError("authored transformation checks are outside the original frozen substrate")
        if (
            not isinstance(request, dict)
            or set(request) != {"schema", "before_ir", "after_ir", "entry"}
            or (request["schema"] != "merlin.common_check_request.v1")
        ):
            raise ValueError("common check cannot select semantics, effects, resources, coverage or proof results")
        if len(json.dumps(request).encode()) > MAX_REQUEST_BYTES:
            raise ValueError("common check input exceeds the fixed byte budget")
        before, after, entry = (request[key] for key in ("before_ir", "after_ir", "entry"))
        if any(not isinstance(value, str) or not value.strip() for value in (before, after, entry)) or (
            len(entry.encode()) > 1024
        ):
            raise ValueError("common check requires actual before/after IR and a bounded entry symbol")
        width = self.profile.get("host_index_bits")
        if type(width) is not int:
            raise ValueError("common check requires the independently reviewed host ABI index width")
        if timeout_s is not None and (type(timeout_s) is not int or timeout_s < 1):
            raise ValueError("common check host budget must be positive")
        started = time.monotonic()
        with BW.verified_snapshot_scope(self.workspace, self.bundle, repo=self.repo):
            self._verify()
            toolchain = self._toolchain(work_root)
            deadline = started + min(timeout_s or toolchain.timeout_s, toolchain.timeout_s)
            if time.monotonic() >= deadline:
                raise TimeoutError("common authored check exhausted its original host work budget")

            def resolve_path(original):
                [selected] = BW.snapshot_input_paths(self.workspace, self.bundle, [Path(original)], repo=self.repo)
                return selected

            semantics = reconstruct_isa_semantics(self.profile["semantic_inputs"], resolve_path=resolve_path)
            registry, contract = semantics.registry, semantics.effect_contract(index_bits=width)
            receipt = verify_authored_transform(
                before, after, entry=entry, registry=registry, contract=contract, toolchain=toolchain, deadline=deadline
            )
            accepted = accepted_authored_receipt(
                receipt, before, after, entry=entry, registry=registry, contract=contract, toolchain=toolchain
            )
            if getattr(receipt, "verified", False) and not accepted:
                raise ValueError("common authored transformation did not obtain an original live formal completion")
            if time.monotonic() > deadline:
                raise TimeoutError("common authored check exceeded its original host work budget")
            self._verify()
            # Re-read the immutable semantics after checking; neither saved summaries
            # nor mutation of a source selection can re-attribute this completion.
            reconstruct_isa_semantics(self.profile["semantic_inputs"], resolve_path=resolve_path)
            document = {
                "status": receipt.status,
                "source_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "target_sha256": hashlib.sha256(after.encode()).hexdigest(),
                "scope": list(receipt.scope),
                "receipts": [receipt.to_dict()],
                "compiler_tool_profile_sha256": self.profile["sha256"],
                "admitted_semantics_sha256": document_digest(self.profile["semantic_inputs"]),
                "hardware_conformance": "not_established",
                "numerical_execution": "not_established",
                "compiler_pass_chain": "not_established",
                "outside_formal_coverage": "independent_checks_required",
            }
            if len(json.dumps(document).encode()) > 2 * MAX_REQUEST_BYTES:
                raise ValueError("common authored check output exceeds its inherited byte budget")
            return document
