"""Host-owned common transformation service on the original selfcheck channel."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path

import yaml

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.tool_profile import verify_tool_profile

ACTION = "common-transform"
SCHEMA = "merlin.common_transform_request.v1"
MAX_REQUEST_BYTES = 256 * 1024


class CommonTransformService:
    """The launcher selects immutable inputs; requests cannot select proof policy."""

    def __init__(self, *, workspace: Path, bundle_manifest: Path, repo: Path):
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
        self.profile = verify_tool_profile(self.bundle)
        if self.profile is None:
            raise ValueError("common transform service requires the newly versioned compiler substrate")
        marker = BW.verify_bundle_snapshot(workspace, self.bundle, repo=repo)
        self.snapshot_digest = marker["content_sha256"]

    def _verify(self) -> None:
        from merlin.common.paths import module_source_path

        if self.bundle_manifest.read_bytes() != self.bundle_bytes:
            raise ValueError("common transform bundle authority changed")
        marker = BW.verify_bundle_snapshot(self.workspace, self.bundle, repo=self.repo)
        if marker["content_sha256"] != self.snapshot_digest:
            raise ValueError("common transform frozen source inputs changed")
        for row in self.profile["common_sources"]:
            module = row["module"].removesuffix(".__init__")
            actual = module_source_path(module)
            if hashlib.sha256(actual.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("common transform host source differs from the original admitted release")
        for row in self.profile["automatic_checker_sources"]:
            if row["module"] is not None:
                actual = module_source_path(row["module"])
            else:
                root = module_source_path("merlin.verify.lean_checker").parent
                actual = root / "lean/Merlin/Foundation.lean"
            if hashlib.sha256(actual.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("automatic verifier host source differs from the original admitted release")

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
        import_roots = []
        for value in document["library_roots"]:
            root = original(value)
            matching = [
                target.parents[len(source.relative_to(root).parts) - 1]
                for source, target in zip(originals[1:], frozen[1:], strict=True)
                if source.is_relative_to(root)
            ]
            if not matching or len(set(matching)) != 1:
                raise ValueError("common proof import root has no exact frozen file closure")
            import_roots.append(matching[0])
        toolchain = LeanToolchain(
            executable=frozen[0],
            executable_sha256=executable["sha256"],
            library_roots=tuple(import_roots),
            files=tuple((path, row["sha256"]) for path, row in zip(frozen[1:], files, strict=True)),
            work_root=work_root,
            timeout_s=document.get("timeout_s", 120),
            memory_mb=document.get("memory_mb", 1024),
            dependency_revision=document.get("dependency_revision", ""),
        )
        toolchain.verify()
        return toolchain

    def apply(self, request: dict, *, work_root: Path, timeout_s: int | None = None) -> dict:
        """Run trusted transforms and independently check each actual IR change."""
        from merlin.transforms.certificates import ir_text, parse_ir
        from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
        from merlin.verify.lean_checker import accepted_pipeline_receipt, verify_pipeline_change

        if set(request) != {"schema", "source_ir", "options"} or request["schema"] != SCHEMA:
            raise ValueError("common transform request cannot select semantics, coverage or proof results")
        if len(json.dumps(request).encode()) > MAX_REQUEST_BYTES:
            raise ValueError("common transform input exceeds the fixed byte budget")
        source, options = request["source_ir"], request["options"]
        if not isinstance(source, str) or not source.strip() or not isinstance(options, dict):
            raise ValueError("common transform requires actual structured IR and explicit parameters")
        if set(options) - {field.name for field in fields(PipelineOptions)}:
            raise ValueError("common transform contains unsupported transformation parameters")
        index_bits = self.profile.get("host_index_bits")
        if type(index_bits) is not int:
            raise ValueError("common transform requires the independently reviewed host ABI index width")
        if "index_bits" in options and options["index_bits"] != index_bits:
            raise ValueError("common transform cannot change the frozen host ABI index width")
        selected = dict(options)
        selected["index_bits"] = index_bits
        for name in ("normalize_shapes", "simplify_integers", "reuse_buffers"):
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
        self._verify()
        toolchain = self._toolchain(work_root)
        if timeout_s is not None:
            if type(timeout_s) is not int or timeout_s < 1:
                raise ValueError("common transform host budget must be positive")
        budget = min(timeout_s, toolchain.timeout_s) if timeout_s is not None else toolchain.timeout_s
        deadline = time.monotonic() + budget
        receipts = []

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
            return receipt

        result = apply_common_pipeline(parse_ir(source), PipelineOptions(**selected), verifier=verify)
        if time.monotonic() > deadline:
            raise TimeoutError("common transformation exceeded its original host work budget")
        self._verify()
        document = {
            "status": "verified" if receipts else "unchanged",
            "scope": "host-issued common transformation preservation" if receipts else "structurally unchanged program",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "transformed_ir": ir_text(result.module),
            "receipts": receipts,
            "compiler_tool_profile_sha256": self.profile["sha256"],
            "hardware_conformance": "not_established",
            "numerical_execution": "not_established",
        }
        if len(json.dumps(document).encode()) > 2 * MAX_REQUEST_BYTES:
            raise ValueError("common transformation output exceeds its inherited byte budget")
        return document
