"""One bounded DMA arena and physical DRAM contract for full-model bundles."""

from chipyard.arena import DEFAULT_ARENA, window


def placement(metadata):
    value = metadata.get("placement", dict(arena_base=DEFAULT_ARENA, dram_base=0x80000000, dram_bytes=64 << 30))
    if (not isinstance(value, dict) or set(value) != {"arena_base", "dram_base", "dram_bytes"}
            or any(type(number) is not int for number in value.values())):
        raise ValueError("invalid full-model placement")
    arena, base, size = (value[name] for name in ("arena_base", "dram_base", "dram_bytes"))
    if base < 0 or size <= 0 or base + size > 1 << 64 or not base <= arena < base + size:
        raise ValueError("full-model arena outside physical DRAM")
    window(arena, 1)
    return dict(value)


def arena_extent(metadata, size):
    value = placement(metadata)
    window(value["arena_base"], size)
    if value["arena_base"] + size > value["dram_base"] + value["dram_bytes"]:
        raise ValueError("full-model arena extent outside physical DRAM")
    return value
