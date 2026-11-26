"""Necessary single-invocation bounds before target region compilation.

These are lower bounds, not a proof of complete device or linked host fit.
Target compilation and deployment placement remain authoritative. A failed
host, runtime, or parameter bound selects bench staging instead of refusing.
"""

from math import prod

from merlin.compiler.lower.c.runtime import DT_BYTES
from merlin.compiler.semantic.target_binding import NativeCompilationError


def host_region_bounds(outlined, plan):
    placements = {item.id: item.placement for item in plan.selected}
    rows = []
    for key, region in outlined.regions:
        if placements[key] != "host":
            continue
        # Host emitters index dense source arrays directly. Even one such
        # array exceeding the entire aperture cannot have a legal placement.
        tensors = [(prod(node.type.shape) * DT_BYTES[node.type.dtype], node.id)
                   for node in region.nodes if node.effect in {"input", "constant"} or node.id in region.outputs]
        size, operand = max(tensors, default=(0, None))
        rows.append(dict(region=key, source_nodes=[node.id for node in region.nodes if node.effect == "pure"],
                         operand=operand, minimum_contiguous_bytes=size))
    return rows


def _host_overflow(row, capacity):
    return (f"application region {row['region']} (source: {', '.join(row['source_nodes'])}): "
            f"dense host operand {row['operand']} requires {row['minimum_contiguous_bytes']} bytes; "
            f"declared data window has {capacity} bytes")


def check_host_region_bounds(outlined, plan, capacity):
    """Refuse an oversized host region that no staging split rule admits."""
    from .staging import leading_row_counts, output_columns

    if type(capacity) is not int or capacity <= 0:
        raise ValueError("host region capacity needs a positive declared byte limit")
    rows = host_region_bounds(outlined, plan)
    regions = dict(outlined.regions)
    for row in rows:
        region = regions[row["region"]]
        if (row["minimum_contiguous_bytes"] > capacity and output_columns(region) is None
                and not leading_row_counts(region, capacity)):
            raise NativeCompilationError("resource_limit", _host_overflow(row, capacity) + "; no split rule admits it")
    return rows



def check_runtime_region_bounds(outlined, plan, capacity, parameters=None):
    """Reject oversized runtime arrays that have no onchip staging binding.

    Immutable parameters retain the existing provider refill path. Packed
    residency inventories are not simultaneous working-set lower bounds.
    The current application C ABI reads runtime arrays as complete pointers.
    """
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("runtime region capacity needs a positive declared byte limit")
    immutable = set(parameters.node_ids) if parameters is not None else set()
    placements = {item.id: item.placement for item in plan.selected}
    for key, region in outlined.regions:
        # Deployment treats every non-host placement as a device call.
        if placements[key] == "host":
            continue
        for node in region.nodes:
            if node.effect != "input" or node.id in immutable:
                continue
            size = prod(node.type.shape) * DT_BYTES[node.type.dtype]
            if size > capacity:
                sources = ", ".join(n.id for n in region.nodes if n.effect == "pure")
                raise NativeCompilationError(
                    "resource_limit",
                    f"application region {key} (source: {sources}): runtime operand {node.id} "
                    f"requires {size} contiguous bytes; declared data window has {capacity} bytes; "
                    "one invocation reads runtime arrays as complete pointers",
                )


def check_parameter_paging_bound(outlined, parameters, capacity):
    """Refuse multi-region parameter payloads that placement cannot page.

    Placement holds the complete immutable payload in the window, or pages it
    only for one operand in one device call. Either way, a payload exceeding
    the window in an application of several regions has no legal placement.
    """
    if parameters is None or parameters.payload_bytes <= capacity or len(outlined.regions) == 1:
        return
    immutable = set(parameters.node_ids)
    subject = "application parameters"
    large = next(((key, region, node) for key, region in outlined.regions for node in region.nodes
                  if node.id in immutable and prod(node.type.shape) * DT_BYTES[node.type.dtype] > capacity), None)
    if large is not None:
        key, region, node = large
        sources = ", ".join(n.id for n in region.nodes if n.effect == "pure")
        size = prod(node.type.shape) * DT_BYTES[node.type.dtype]
        subject = f"application region {key} (source: {sources}): immutable parameter {node.id} requires {size} bytes"
    raise NativeCompilationError(
        "resource_limit",
        f"{subject}; parameter payload has {parameters.payload_bytes} bytes; declared data window has {capacity} bytes; "
        f"parameter refill pages one operand in one device call, application has {len(outlined.regions)} regions",
    )


def staging_reason(outlined, plan, capacity, parameters=None):
    """Name the first necessary single-invocation bound the window cannot meet, or None.

    A named reason selects bench staging; it does not prove that staging fits.
    """
    over = [row for row in check_host_region_bounds(outlined, plan, capacity)
            if row["minimum_contiguous_bytes"] > capacity]
    if over:
        return _host_overflow(over[0], capacity)
    try:
        check_runtime_region_bounds(outlined, plan, capacity, parameters)
        check_parameter_paging_bound(outlined, parameters, capacity)
    except NativeCompilationError as error:
        return error.reason
    return None
