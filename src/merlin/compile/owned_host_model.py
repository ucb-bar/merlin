"""Compile original model IR with a fixed host provider and a separate device package.

The service is selected by the trusted caller. Its per-model interface accepts
no host compiler, submitted whole-program LLVM, or submitted host arithmetic.
It uses the ordinary source-to-device routing and saved-model build, retaining
those APIs' capture, linked-ELF, and execution requirements. This is a build
service, not evidence that an experiment has passed its independent gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.compile.model_execution_inputs import strict_tree_sha256


@dataclass(frozen=True)
class OwnedHostModelCompiler:
    host_package: Path
    host_package_identity: dict[str, Any]
    target: str
    board_catalog: Path
    board: str
    dts: Path

    @classmethod
    def select(cls, *, host_package, target, board_catalog, board, dts):
        """Bind the trusted host provider once, outside candidate invocation."""
        package = Path(host_package)
        if package.is_symlink():
            raise ValueError("owned host provider must not be a symlink")
        package = package.resolve(strict=True)
        return cls(package, strict_tree_sha256(package), target, Path(board_catalog), board, Path(dts))

    def compile(
        self,
        *,
        capture,
        candidate_package,
        datapath,
        linked_elf_admission,
        output,
        arena_mb,
        timeout_s=300,
        run="none",
        rtl_facts=None,
        reference_file=None,
        tolerance=None,
        math_archive_symbols=None,
        readback="full",
        materialized_inputs=None,
        group_profile=False,
    ):
        """Build original source; the candidate supplies closed device groups only.

        The caller must independently select the accelerator contract and linked
        ELF admission. The resulting ordinary receipt retains the actual source,
        provider, device objects, and complete-output execution evidence.
        """
        from merlin.compile.baremetal_model import compile_saved_model
        from merlin.compile.route_before_build import plan_before_build
        from merlin.llvmlower.device_offload import BY_GROUP

        if linked_elf_admission is None:
            raise ValueError("an independently selected linked-ELF admission is required")
        candidate = Path(candidate_package)
        if candidate.is_symlink():
            raise ValueError("candidate device package must not be a symlink")
        candidate = candidate.resolve(strict=True)
        if (
            candidate == self.host_package
            or candidate.is_relative_to(self.host_package)
            or self.host_package.is_relative_to(candidate)
        ):
            raise ValueError("candidate device package must be separate from the owned host provider")
        if strict_tree_sha256(self.host_package) != self.host_package_identity:
            raise ValueError("selected owned host provider changed")
        source = Path(capture)
        routed = plan_before_build(
            self.target,
            (source / "model.mlir").read_text(encoding="utf-8"),
            datapath=datapath,
            device_package=str(candidate),
            capture=source / "model.mlir",
            granularity=BY_GROUP,
            linked_elf_admission=linked_elf_admission,
        )
        device = routed.get("device_routing")
        if device is None or routed.get("offload") is None:
            raise ValueError(f"no buildable source-bound device route: {routed.get('device_routing_why')}")
        receipt = compile_saved_model(
            capture=source,
            package=self.host_package,
            board_catalog=self.board_catalog,
            board=self.board,
            dts=self.dts,
            output=output,
            target=self.target,
            run=run,
            arena_mb=arena_mb,
            timeout_s=timeout_s,
            device=device,
            rtl_facts=rtl_facts,
            reference_file=reference_file,
            tolerance=tolerance,
            math_archive_symbols=math_archive_symbols,
            readback=readback,
            materialized_inputs=materialized_inputs,
            respect_captured_quantization_scope=True,
            group_profile=group_profile,
        )
        if strict_tree_sha256(self.host_package) != self.host_package_identity:
            raise ValueError("selected owned host provider changed during compilation")
        return receipt
