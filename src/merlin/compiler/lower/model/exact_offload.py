"""Exact captured-model selection for the whole-model device rewrite.

This is a placement *identity*, not a claim that a diagnostic outline has passed
Phase 1. It binds one immutable model, its selected operation IDs, the actual
interface bytes, one OOT compiler tree, and a reviewed Phase 0 release checked by
the installed host verifier. The rewrite and build recheck these identities; a changed
preparation pass or package must not turn a selection for one operation into a
shape-based choice of another. The release verifier belongs to the optional
experiment owner, so core does not read experiment paths. This is a trusted-host
evidence gate, not isolation against arbitrary Python running in that host process.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_bytes, sha256_text
from merlin.common.tree_hash import hash_tree
from merlin.development.targetgen.contract.model_kernel_outline import SCHEMA as OUTLINE_SCHEMA
from merlin.development.targetgen.contract.model_kernel_outline import outline_integer_matmuls


@dataclass(frozen=True)
class SelectedKernel:
    operation_id: str
    interface_mlir: str
    interface_sha256: str


@dataclass(frozen=True)
class ReleaseBinding:
    """Data selected by the trusted host; no caller-supplied verifier code."""

    seal_path: Path
    descriptor: Path
    application: str
    review_digest: str


@dataclass(frozen=True)
class ExactOffloadSelection:
    target: str
    model_sha256: str
    package_sha256: str
    transport: str
    abi_sha256: str
    kernels: tuple[SelectedKernel, ...]
    software_spec_sha256: str
    capability_contract_sha256: str
    certification_sha256: tuple[str, ...] = ()
    release_binding: ReleaseBinding | None = None

    @classmethod
    def from_outline(
        cls,
        outline: Mapping,
        *,
        model: bytes,
        target: str,
        software_spec: bytes,
        capability_contract: bytes,
        package_dir: str | Path,
        operation_ids: tuple[str, ...],
        interface_root: str | Path | None = None,
    ) -> ExactOffloadSelection:
        """Bind explicit IDs from one outline; never promote unknown support.

        Re-derivation from the exact source bytes is intentional. Changing a
        diagnostic JSON status cannot admit an operation, and the returned
        selection still cannot execute until :meth:`certify` runs independently.
        """
        regenerated = outline_integer_matmuls(
            model, target=target, software_spec=software_spec, capability_contract=capability_contract
        )
        digest = outline.get("model_sha256")
        if (
            outline.get("schema") != OUTLINE_SCHEMA
            or outline.get("target") != target
            or not is_sha256(digest)
            or digest != regenerated["model_sha256"]
            or outline.get("software_spec_sha256") != regenerated["software_spec_sha256"]
            or outline.get("capability_contract_sha256") != regenerated["capability_contract_sha256"]
        ):
            raise ValueError("outline, exact model, software spec, or capability bytes disagree")
        if not operation_ids or len(set(operation_ids)) != len(operation_ids):
            raise ValueError("exact selection needs distinct, explicitly named operation IDs")
        actual_candidates = []
        for row in outline.get("candidates", ()):
            candidate = dict(row)
            if "interface_file" in candidate:
                name = candidate.pop("interface_file")
                if interface_root is None or not isinstance(name, str) or Path(name).name != name:
                    raise ValueError("materialized outline needs a safe interface_root")
                path = Path(interface_root) / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"outline interface {name!r} is missing or a symlink")
                candidate["interface_mlir"] = path.read_text(encoding="utf-8")
            actual_candidates.append(candidate)
        if actual_candidates != regenerated["candidates"]:
            raise ValueError("outline candidates differ from deterministic re-derivation")
        by_id = {row["operation_id"]: row for row in regenerated["candidates"]}
        if len(by_id) != len(regenerated["candidates"]):
            raise ValueError("outline has duplicate candidate operation IDs")
        kernels = []
        for operation_id in operation_ids:
            candidate = by_id.get(operation_id)
            if candidate is None or not operation_id.startswith(f"mlir:{digest}:"):
                raise ValueError(f"{operation_id!r} is not a candidate of this exact model")
            if candidate.get("software_admission", {}).get("status") != "admitted":
                raise ValueError(f"{operation_id!r} has no reviewed SW admission")
            interface = candidate.get("interface_mlir")
            interface_sha = candidate.get("interface_sha256")
            if (
                not isinstance(interface, str)
                or not is_sha256(interface_sha)
                or sha256_text(interface) != interface_sha
            ):
                raise ValueError(f"{operation_id!r} has changed or unreadable interface bytes")
            kernels.append(SelectedKernel(operation_id, interface, interface_sha))
        package_sha256 = _package_sha256(Path(package_dir))
        transport, abi_sha256 = _backend_identity(target)
        return cls(
            str(target),
            str(digest),
            package_sha256,
            transport,
            abi_sha256,
            tuple(kernels),
            sha256_bytes(software_spec),
            sha256_bytes(capability_contract),
        )

    def certify(
        self, package_dir: str | Path, *, runs_root: str | Path, simulator: str, timeout: int
    ) -> ExactOffloadSelection:
        """Run the selected *exact* interfaces through the existing OOT numerical oracle.

        A pass flag copied into JSON is never accepted here. This method invokes
        the evaluator with accelerator trace required, retains its run artifacts,
        and refuses any skipped/unavailable oracle or source-tree mutation.
        """
        if not simulator or type(timeout) is not int or timeout <= 0:
            raise ValueError("certification needs an explicit simulator and positive timeout")
        from merlin.development.targetgen import package_runtime as oot_runner

        self.check_release()
        self.check_package(package_dir)
        self.check_backend_contract()
        root = Path(runs_root)
        root.mkdir(parents=True, exist_ok=True)
        receipts = []
        for index, kernel in enumerate(self.kernels):
            with tempfile.TemporaryDirectory(prefix="merlin_exact_iface_", dir=root) as td:
                interface = Path(td) / "selected.interface.mlir"
                interface.write_text(kernel.interface_mlir, encoding="utf-8")
                result = oot_runner.certify(
                    str(package_dir),
                    interface,
                    runs_root=str(root),
                    run_id=f"exact_{self.model_sha256[:12]}_{index}",
                    simulator=simulator,
                    target=self.target,
                    timeout=timeout,
                    require_accelerator_trace=True,
                )
            self.check_package(package_dir)
            self.check_backend_contract()
            trace = result.get("trace_check") or {}
            if (
                result.get("status") != "pass"
                or (result.get("oracle") or {}).get("result") != "pass"
                or trace.get("status") != "pass"
                or trace.get("drives_accelerator") is not True
            ):
                raise ValueError(f"selected {kernel.operation_id} did not pass a running accelerator oracle")
            receipts.append(sha256_text(json.dumps(result, sort_keys=True, default=str)))
        return ExactOffloadSelection(
            self.target,
            self.model_sha256,
            self.package_sha256,
            self.transport,
            self.abi_sha256,
            self.kernels,
            self.software_spec_sha256,
            self.capability_contract_sha256,
            tuple(receipts),
            self.release_binding,
        )

    @property
    def certified(self) -> bool:
        return (
            bool(self.kernels)
            and len(self.certification_sha256) == len(self.kernels)
            and all(is_sha256(receipt) for receipt in self.certification_sha256)
        )

    def check_package(self, package_dir: str | Path) -> None:
        observed = _package_sha256(Path(package_dir))
        if observed != self.package_sha256:
            raise ValueError("selected OOT compiler package tree changed after placement")

    def check_backend_contract(self) -> None:
        if _backend_identity(self.target) != (self.transport, self.abi_sha256):
            raise ValueError("selected transport or pointer ABI changed after exact placement")

    def check_release(self) -> None:
        binding = self.release_binding
        if (
            type(binding) is not ReleaseBinding
            or not is_sha256(binding.review_digest)
            or not isinstance(binding.seal_path, Path)
            or not isinstance(binding.descriptor, Path)
            or not isinstance(binding.application, str)
            or not binding.application
        ):
            raise ValueError("exact offload needs a verified reviewed Phase 0 release")
        if _release_verifier()(binding, self) != binding.review_digest:
            raise ValueError("exact offload release verifier did not return the selected review identity")

    @property
    def by_operation_id(self) -> dict[str, SelectedKernel]:
        return {kernel.operation_id: kernel for kernel in self.kernels}


def _release_verifier():
    """Resolve the installed host owner, never executable code carried by a selection.

    This is a trusted-process API, not a sandbox against arbitrary Python running
    in the host process. An absent or ambiguous owner refuses exact offload.
    """
    providers = tuple(entry_points(group="merlin.exact_offload_release"))
    if len(providers) != 1 or providers[0].name != "reviewed_phase0":
        raise ValueError("exact offload requires one installed reviewed Phase 0 release verifier")
    return providers[0].load()


def _backend_identity(target: str) -> tuple[str, str]:
    from merlin.compiler.lower.model.device_build import objects_buildable
    from merlin.compiler.lower.model.device_shim import kernel_abi_for
    from merlin.development.targetgen.target_experiment import load_capability_manifest
    from merlin.target.system.derive import link_for

    unavailable = objects_buildable(target)
    if unavailable:
        raise ValueError(f"selected target has no linkable whole-model path: {unavailable}")
    manifest = load_capability_manifest(target)
    link = link_for(target, manifest.endpoint_kind)
    abi = kernel_abi_for(target)
    if abi is None or not link.command_transport:
        raise ValueError("selected target has no derived command transport or kernel pointer ABI")
    return str(link.command_transport), sha256_text(repr(abi))


def _package_sha256(root: Path) -> str:
    generated = {"build", "__pycache__", ".git"}
    if (
        root.is_symlink()
        or not root.is_dir()
        or any(
            path.is_symlink() and not generated.intersection(path.relative_to(root).parts) for path in root.rglob("*")
        )
    ):
        raise ValueError("selected OOT compiler package must be a regular, symlink-free tree")
    package = hash_tree(root)
    if not package["present"] or not package["n_files"] or not is_sha256(package["sha256"]):
        raise ValueError("selected OOT compiler package has no readable source tree")
    return str(package["sha256"])
