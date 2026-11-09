"""Byte-bound optional provider callbacks for an outlined arithmetic dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_file


def _identifier(value):
    return isinstance(value, str) and value.isascii() and value.isidentifier()


@dataclass(frozen=True)
class ReadonlyInput:
    """Require one whole dense input from the program's sealed frozen blob.

    This is not a promise inferred from pointer stability, parameter names or
    C ``const``. The builder checks the actual immutable member and its source
    ownership before exposing the existing MERLIN_WEIGHT argument kind.
    """

    index: int
    dtype: str
    shape: tuple[int, ...]
    bytes: int
    sha256: str
    alignment: int = 1

    def __post_init__(self):
        if (
            type(self.index) is not int
            or self.index < 0
            or not isinstance(self.dtype, str)
            or not self.dtype
            or not isinstance(self.shape, tuple)
            or any(type(d) is not int or d <= 0 for d in self.shape)
            or type(self.bytes) is not int
            or not 0 < self.bytes < 1 << 63
            or not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256)
            or type(self.alignment) is not int
            or not 0 < self.alignment < 1 << 63
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError("complete dense readonly input type/span/identity required")

    def record(self):
        return {
            "index": self.index,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "alignment": self.alignment,
        }


@dataclass(frozen=True)
class PreparedStagedKernel:
    """A provider's explicitly selected, already compiled staged implementation.

    No provider is inferred from a target name or tensor shape. The source key
    binds the complete canonical generic kernel IR. Declared state includes ALL
    provider metadata/scratch; the renderer reserves it in the whole-chain arena
    until complete quiescence. C static assertions enforce the type/alignment
    budget. Independent target numerical/context/completion gates remain needed.
    """

    source_ir_sha256: str
    header: Path
    state_type: str
    state_bytes: int
    state_alignment: int
    prepare: str
    compute: str
    finish: str
    worker_slots: tuple[int, ...]
    objects: tuple[Path, ...]
    input_sha256: Mapping[str, str]
    provenance: Mapping[str, Any]
    readonly_inputs: tuple[ReadonlyInput, ...] = ()

    def verify(self, *, worker_count: int) -> None:
        if (
            not isinstance(self.source_ir_sha256, str)
            or len(self.source_ir_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_ir_sha256)
            or any(not _identifier(s) for s in (self.state_type, self.prepare, self.compute, self.finish))
            or type(self.state_bytes) is not int
            or not 0 < self.state_bytes < 1 << 63
            or type(self.state_alignment) is not int
            or not 0 < self.state_alignment < 1 << 63
            or self.state_alignment & (self.state_alignment - 1)
            or not isinstance(self.worker_slots, tuple)
            or not self.worker_slots
            or len(set(self.worker_slots)) != len(self.worker_slots)
            or any(type(s) is not int or not 0 <= s < worker_count for s in self.worker_slots)
            or not isinstance(self.objects, tuple)
            or not self.objects
            or not isinstance(self.provenance, Mapping)
            or not self.provenance
            or not isinstance(self.readonly_inputs, tuple)
            or any(not isinstance(r, ReadonlyInput) for r in self.readonly_inputs)
            or len({r.index for r in self.readonly_inputs}) != len(self.readonly_inputs)
        ):
            raise ValueError("complete typed/bounded prepared staged kernel contract required")
        required = [Path(self.header), *map(Path, self.objects)]
        for path in required:
            if not path.is_absolute() or not path.is_file() or str(path) not in self.input_sha256:
                raise ValueError("prepared header/objects require absolute byte-bound existing files")
        for path, digest in self.input_sha256.items():
            if not Path(path).is_absolute() or not Path(path).is_file() or sha256_file(path) != digest:
                raise ValueError("prepared staged kernel input changed: " + str(path))

    def record(self) -> dict:
        return {
            "schema": "merlin.prepared_staged_kernel.v1",
            "source_ir_sha256": self.source_ir_sha256,
            "header": str(self.header),
            "state_type": self.state_type,
            "state_bytes": self.state_bytes,
            "state_alignment": self.state_alignment,
            "prepare": self.prepare,
            "compute": self.compute,
            "finish": self.finish,
            "worker_slots": list(self.worker_slots),
            "objects": [str(p) for p in self.objects],
            "input_sha256": dict(self.input_sha256),
            "provenance": dict(self.provenance),
            **({"readonly_inputs": [r.record() for r in self.readonly_inputs]} if self.readonly_inputs else {}),
        }


def readonly_bindings(plan, index: int, kernel: PreparedStagedKernel) -> dict[int, dict]:
    """Bind explicit requirements to immutable members; no alias inference."""
    if not kernel.readonly_inputs:
        return {}
    frozen = plan.frozen_inputs
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("schema") != "merlin.frozen_inputs.v1"
        or frozen.get("original_program_preserved") is not True
        or not isinstance(frozen.get("roots"), list)
        or frozen.get("source_ir_sha256") != plan.source_ir_sha256
        or not isinstance(frozen.get("readonly_buffers"), Mapping)
        or not isinstance(frozen.get("external_input_source_indices"), list)
        or any(type(i) is not int or i < 0 for i in frozen["external_input_source_indices"])
        or type(frozen.get("published_blob_bytes")) is not int
        or not 0 < frozen["published_blob_bytes"] < 1 << 63
        or not isinstance(frozen.get("published_blob_sha256"), str)
        or len(frozen["published_blob_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in frozen["published_blob_sha256"])
    ):
        raise ValueError("readonly selection requires the actual sealed FrozenInput program")
    roots = {r.get("arg_index"): r for r in frozen["roots"] if isinstance(r, Mapping)}
    if len(roots) != len(frozen["roots"]) or any(
        type(i) is not int or i < 0 or i in frozen["external_input_source_indices"] for i in roots
    ):
        raise ValueError("readonly selection requires removed immutable source argument ownership")
    step = plan.dispatches[index]
    specs = {b.id: b for b in plan.buffers}
    bindings = {}
    for requirement in kernel.readonly_inputs:
        if requirement.index >= len(step["inputs"]):
            raise ValueError("readonly selection names a non-input operand")
        bid = step["inputs"][requirement.index]
        spec = specs[bid]
        member = frozen["readonly_buffers"].get(bid)
        if (
            not isinstance(member, Mapping)
            or spec.workspace_offset is not None
            or spec.arg_index in frozen["external_input_source_indices"]
            or (
                spec.arg_index is not None
                and (spec.arg_index not in roots or roots[spec.arg_index].get("buffer") != bid)
            )
            or tuple(spec.shape) != requirement.shape
            or spec.dtype != requirement.dtype
            or spec.bytes != requirement.bytes
            or member.get("bytes") != requirement.bytes
            or member.get("sha256") != requirement.sha256
            or type(member.get("offset")) is not int
            or member["offset"] < 0
            or member["offset"] % requirement.alignment
            or plan.alignment_bytes < requirement.alignment
            or member["offset"] + requirement.bytes > frozen["published_blob_bytes"]
        ):
            raise ValueError("readonly input differs from the exact immutable type/span/content binding")
        if bid in plan.program.results or any(bid in s["outputs"] for s in plan.dispatches):
            raise ValueError("readonly input has a runtime arithmetic/output writer")
        views = [v for s in plan.dispatches for v in s["views_before"]] + list(plan.trailing_views)
        if any(v.get("output") == bid for v in views):
            raise ValueError("readonly input has a runtime view/transport rebinding")
        bindings[requirement.index] = {
            **requirement.record(),
            "buffer": bid,
            "blob_offset": member["offset"],
            "blob_sha256": frozen["published_blob_sha256"],
            "source_ir_sha256": plan.source_ir_sha256,
            "lifetime": "immutable program blob retained through all callbacks and complete joins",
            "runtime_argument_removed": True,
        }
    return bindings


def reserve_states(plan, kernels: Mapping[int, PreparedStagedKernel], *, budget_bytes: int):
    """Reserve every opaque provider state in the same explicit lifetime arena."""
    if any(type(index) is not int or not 0 <= index < len(plan.dispatches) for index in kernels):
        raise ValueError("prepared kernel selection names an unknown arithmetic dispatch")
    offset, alignment, layout = plan.workspace_bytes, plan.alignment_bytes, {}
    for index, kernel in sorted(kernels.items()):
        if not isinstance(kernel, PreparedStagedKernel):
            raise ValueError("explicit prepared staged kernel records required")
        kernel.verify(worker_count=plan.worker_count)
        alignment = max(alignment, kernel.state_alignment)
        offset = (offset + kernel.state_alignment - 1) // kernel.state_alignment * kernel.state_alignment
        layout[index] = {"offset": offset, "bytes": kernel.state_bytes, "alignment": kernel.state_alignment}
        offset += kernel.state_bytes
    total = (offset + alignment - 1) // alignment * alignment
    if total > budget_bytes or total >= 1 << 63:
        raise ValueError(f"whole-chain tensor/provider state needs {total} bytes, budget is {budget_bytes}")
    return replace(plan, workspace_bytes=total, alignment_bytes=alignment), layout


def verify_selection(plan, kernels, layout):
    """Check source identity and every reserved interval at the rendering seam."""
    import hashlib

    from merlin.llvmlower.deployable_ir import checked_generic_text
    from merlin.llvmlower.deployable_outline import canonical_kernel

    if set(kernels) != set(layout):
        raise ValueError("prepared selection and reserved state layout differ")
    ranges = [
        (b.workspace_offset, b.workspace_offset + b.bytes) for b in plan.buffers if b.workspace_offset is not None
    ]
    for index, kernel in sorted(kernels.items()):
        if type(index) is not int or not 0 <= index < len(plan.dispatches):
            raise ValueError("prepared kernel selection names an unknown dispatch")
        kernel.verify(worker_count=plan.worker_count)
        actual = checked_generic_text(canonical_kernel(plan.outlined.module, plan.dispatches[index]["symbol"]))
        if hashlib.sha256(actual.encode()).hexdigest() != kernel.source_ir_sha256:
            raise ValueError("prepared staged kernel source identity differs from selected dispatch")
        readonly_bindings(plan, index, kernel)
        slot = layout[index]
        offset = slot.get("offset")
        if (
            type(offset) is not int
            or offset < 0
            or offset % kernel.state_alignment
            or slot.get("bytes") != kernel.state_bytes
            or slot.get("alignment") != kernel.state_alignment
            or plan.alignment_bytes < kernel.state_alignment
            or offset + kernel.state_bytes > plan.workspace_bytes
        ):
            raise ValueError("prepared state violates its checked arena interval/alignment")
        end = offset + kernel.state_bytes
        if any(offset < b and a < end for a, b in ranges):
            raise ValueError("prepared state overlaps another live tensor/provider allocation")
        ranges.append((offset, end))
