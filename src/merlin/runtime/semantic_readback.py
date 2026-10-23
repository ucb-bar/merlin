"""Strict optional semantic frames accompanying lossless output bytes."""

from __future__ import annotations


def parse_semantic_readback(console: str, output_count: int):
    from merlin.runtime.backends.spike_model import SpikeModelError

    metadata, aliases, pre = [], [], {}
    seen_done = False
    present = False
    for line in console.splitlines():
        parts = line.split()
        if parts == ["DONE"]:
            seen_done = True
        if not parts or parts[0] not in ("OUT_META", "OUT_ALIAS", "PRE_BYTES"):
            continue
        present = True
        try:
            if seen_done:
                raise ValueError("semantic frame after DONE")
            values = [int(v) for v in parts[1:]]
            if parts[0] == "OUT_META":
                index, offset, grad, rank, *axes = values
                if (
                    index != len(metadata)
                    or offset < 0
                    or grad not in (-1, 0, 1)
                    or rank < 0
                    or len(axes) != 2 * rank
                    or any(v < 0 for v in axes)
                ):
                    raise ValueError("invalid metadata frame")
                metadata.append(
                    dict(
                        storage_offset=offset,
                        requires_grad=None if grad == -1 else bool(grad),
                        shape=axes[::2],
                        stride=axes[1::2],
                        **({"imposed_by_harness": ["requires_grad"]} if grad == -1 else {}),
                    )
                )
            elif parts[0] == "OUT_ALIAS":
                index, *targets = values
                if (
                    index != len(aliases)
                    or len(set(targets)) != len(targets)
                    or any(v < 0 or v >= output_count for v in targets)
                ):
                    raise ValueError("invalid alias frame")
                aliases.append(targets)
            else:
                index, count, *data = values
                if index < 0 or str(index) in pre or count < 0 or len(data) != count:
                    raise ValueError("invalid snapshot frame")
                pre[str(index)] = bytes(data).hex()
        except (ValueError, OverflowError) as exc:
            raise SpikeModelError(f"malformed {parts[0]} semantic frame") from exc
    if not present:
        return None
    if len(metadata) != output_count or len(aliases) != output_count:
        raise SpikeModelError("semantic frame count differs from result count")
    return dict(metadata=metadata, aliases=aliases, pre_bytes=pre)
