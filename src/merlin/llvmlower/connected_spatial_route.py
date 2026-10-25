"""Explicit ordinary host ownership for one current-source spatial pair.

The original caller selects the source pair, exact staged plan and separately
qualified borrowing/workspace premises. No emitted command buffer can select
this route. Numerical, target and final-image admission remain ordinary gates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan, SpatialProducerPlan

    from .fresh_tensor_writer import FreshTensorWriterContract, PrivateWorkspaceContract


@dataclass(frozen=True)
class ConnectedSpatialRoute:
    source_pair: tuple[str, str]
    plan: SpatialProducerPlan | PackedSpatialProducerPlan
    writer: FreshTensorWriterContract
    workspace: PrivateWorkspaceContract
    pointer_alignments: tuple[int, ...]
    helper_symbol: str
    stack_budget_bytes: int | None = None

    def record(self):
        from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan, SpatialProducerPlan

        from .fresh_tensor_writer import FreshTensorWriterContract, PrivateWorkspaceContract
        from .spatial_scratch_abi import _symbol

        if (
            type(self.source_pair) is not tuple
            or len(self.source_pair) != 2
            or any(type(s) is not str for s in self.source_pair)
            or self.source_pair[0] == self.source_pair[1]
            or type(self.plan) not in (SpatialProducerPlan, PackedSpatialProducerPlan)
            or type(self.writer) is not FreshTensorWriterContract
            or type(self.workspace) is not PrivateWorkspaceContract
            or type(self.pointer_alignments) is not tuple
        ):
            raise ValueError("connected spatial route requires original typed source, plan and ownership premises")
        self.plan.__post_init__()
        if self.stack_budget_bytes is not None and (
            type(self.stack_budget_bytes) is not int or not 0 < self.stack_budget_bytes < 1 << 63
        ):
            raise ValueError("private stack storage requires an explicit positive bounded byte budget")
        _symbol(self.helper_symbol)
        # Return a value snapshot, never live nested dictionaries from a caller.
        return json.loads(json.dumps(asdict(self), sort_keys=True, allow_nan=False))


def freeze_route(device):
    selected = getattr(device, "connected_spatial", None)
    if selected is None:
        return None
    if type(selected) is not ConnectedSpatialRoute:
        raise ValueError("connected spatial route requires its original typed selection")
    if getattr(device, "granularity", None) != "group" or any(
        getattr(device, name, None) is not None
        for name in (
            "exact_selection",
            "catalog_manifest",
            "catalog_object",
            "catalog_builder",
            "offload_selection",
            "post_offload_transform",
        )
    ):
        raise ValueError("connected spatial ownership requires a stated group route without opaque ABI rewrites")
    return selected, ConnectedSpatialRoute.record(selected)


def require_route(device, frozen):
    selected = getattr(device, "connected_spatial", None)
    if frozen is None:
        if selected is not None:
            raise ValueError("connected spatial selection appeared after original preparation")
        return None
    original, record = frozen
    if selected is not original or freeze_route(device)[1] != record:
        raise ValueError("connected spatial original selection changed")
    return selected


def require_host(host, selected):
    from merlin.xdsl_dialects._common import text

    from .connected_spatial_host import ConnectedSpatialHost, _check_source, _digest

    if type(host) is not ConnectedSpatialHost or (
        host.source_pair != selected.source_pair
        or host.source_entry_sha256 != _digest(host.source_entry)
        or selected.plan.identity != host.scratch.plan_sha256
        or hashlib.sha256(text(host.scratch.module, generic=True).encode()).hexdigest()
        != host.scratch.selected_source_sha256
        or host.scratch.pointer_alignments != selected.pointer_alignments
        or host.scratch.receipt["pool"].get("stack_budget_bytes") != selected.stack_budget_bytes
    ):
        raise ValueError("connected spatial original host preparation changed")
    _check_source(selected.plan, host.source_entry)


def restore_group_rewrite(sidecar):
    """Restore ordinary early group preparation data; it grants no selection."""
    from .device_offload import DeviceRewrite, Routed, build_arguments

    args = build_arguments(sidecar, expected_granularity="group")
    fields = {k: v for k, v in sidecar.items() if k in DeviceRewrite.__dataclass_fields__}
    sequence_fields = ("parallel", "reduction", "dtypes", "tensor_types", "source_node_ids")
    fields["routed"] = tuple(
        Routed(**{k: tuple(v) if k in sequence_fields else v for k, v in row.items()}) for row in sidecar["routed"]
    )
    fields["signatures"] = args["signatures"]
    fields["arg_access"] = {k: tuple(v) for k, v in sidecar["arg_access"].items()}
    # Blob storage remains with the normal early preparation owner. Rewriting
    # this JSON inventory must not rewrite its separately retained blobs.
    return DeviceRewrite(**fields)


def prepare_host(device, selected, prepared, rewrite, directory):
    """Prepare the actual rewritten host before its ordinary sidecar is frozen."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects._common import text

    from .connected_spatial import group_connected_spatial
    from .connected_spatial_host import prepare_connected_spatial_host
    from .device_build import kernel_symbol
    from .device_shim import kernel_abi_for

    module = parse_mlir_file(prepared)
    grouped = group_connected_spatial(module, rewrite)
    matches = [r for r in grouped.receipt["groups"] if r["components"] == list(selected.source_pair)]
    if len(matches) != 1 or matches[0]["symbol"] != selected.writer.symbol:
        raise ValueError("connected route lost its original current-source pair")
    symbol = matches[0]["symbol"]
    abi = kernel_abi_for(device.device)
    if abi is None or abi.version != 2:
        raise ValueError("connected spatial route requires the selected ordinary logical kernel ABI")
    index = sorted(grouped.rewrite.signatures).index(symbol)
    host = prepare_connected_spatial_host(
        module,
        rewrite,
        selected.source_pair,
        selected.plan,
        selected.writer,
        selected.workspace,
        kernel_symbol=kernel_symbol(abi.symbol, index),
        pointer_alignments=selected.pointer_alignments,
        stack_budget_bytes=selected.stack_budget_bytes,
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "host.mlir"
    path.write_text(text(host.scratch.module, generic=True) + "\n")
    (directory / "receipt.json").write_text(json.dumps(host.receipt, sort_keys=True, indent=2) + "\n")
    return path, host


def select_emission(artifact, selected, host, *, entry_symbol):
    """Keep the checked producer artifact intact; derive one literal closed helper."""
    from xdsl.parser import Parser

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from merlin.xdsl_dialects._common import text

    from .connected_spatial_host import select_closed_spatial_device_helper

    require_host(host, selected)
    module = Parser(make_llvm_context(), artifact).parse_module()
    result, receipt = select_closed_spatial_device_helper(
        module,
        host,
        selected.plan,
        helper_symbol=selected.helper_symbol,
        entry_symbol=entry_symbol,
    )
    receipt = {
        **receipt,
        "checked_producer_sha256": hashlib.sha256(artifact.encode()).hexdigest(),
        "selected_artifact_sha256": hashlib.sha256(text(result, generic=True).encode()).hexdigest(),
    }
    return text(result, generic=True), receipt


def with_private_bridge(unit, selected, host, *, symbol, kernel):
    """Append only the original host's descriptor transport to ordinary shims."""
    from .spatial_scratch_abi import _bridge

    require_host(host, selected)
    if symbol != selected.writer.symbol or symbol in unit.symbols:
        raise ValueError("private spatial bridge has ambiguous original source membership")
    bridge = _bridge(selected.plan, selected.writer, selected.workspace, kernel, selected.pointer_alignments)
    if bridge != host.scratch.bridge_source:
        raise ValueError("private spatial bridge differs from the original host-bound kernel")
    return replace(
        unit, text=unit.text + "\n" + bridge, symbols=(*unit.symbols, symbol), kernels=(*unit.kernels, kernel)
    )
