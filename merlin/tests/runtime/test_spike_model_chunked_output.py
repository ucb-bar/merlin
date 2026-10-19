"""Large model outputs remain complete and ordered on the Spike console."""

import numpy as np
import pytest

from merlin.runtime.backends.spike_model import SpikeModelError, parse_console


def _console(count: int = 4097) -> str:
    words = np.arange(count, dtype=np.uint32) + 0x3F800000
    lines = [f"OUT_BEGIN f32 {count}"]
    for start in range(0, count, 1024):
        chunk = words[start : start + 1024]
        lines.append(f"OUT_CHUNK f32 {start} {len(chunk)} " + " ".join(map(str, chunk)))
    return "\n".join([*lines, "DONE", ""])


def test_chunked_full_output_preserves_every_word() -> None:
    result = parse_console(_console())
    assert result["output_dtype"] == "f32"
    assert result["raw_output_bits"].size == 4097
    assert result["raw_output_bits"][0] == 0x3F800000
    assert result["raw_output_bits"][-1] == 0x3F801000


def test_chunked_integer_and_boolean_storage() -> None:
    integer_chunks = [
        f"OUT_CHUNK i64 {start} {min(1024, 4097 - start)} "
        + " ".join("4294967295 4294967295" for _ in range(min(1024, 4097 - start)))
        for start in range(0, 4097, 1024)
    ]
    integer = parse_console("\n".join(["OUT_BEGIN i64 4097", *integer_chunks, "DONE", ""]))
    assert integer["outputs"].dtype == np.dtype("int64")
    assert integer["outputs"].tolist() == [-1] * 4097

    boolean_chunks = [
        f"OUT_CHUNK i1 {start} {min(1024, 4097 - start)} "
        + " ".join("1" if i % 2 else "0" for i in range(start, min(start + 1024, 4097)))
        for start in range(0, 4097, 1024)
    ]
    boolean = parse_console("\n".join(["OUT_BEGIN i1 4097", *boolean_chunks, "DONE", ""]))
    assert boolean["outputs"].dtype == np.dtype("bool")
    assert boolean["outputs"].tolist()[:3] == [False, True, False]


@pytest.mark.parametrize(
    "change",
    [
        lambda lines: lines.pop(2),
        lambda lines: lines.insert(2, lines[1]),
        lambda lines: lines.__setitem__(2, lines[2].replace("OUT_CHUNK f32", "OUT_CHUNK i64", 1)),
        lambda lines: lines.__setitem__(0, "OUT_BEGIN f32 4098"),
        lambda lines: lines.insert(1, "OUT 1 0"),
        lambda lines: lines.append(lines[1]),
    ],
)
def test_chunked_output_rejects_partial_or_ambiguous_stream(change) -> None:
    lines = _console().splitlines()
    change(lines)
    with pytest.raises(SpikeModelError):
        parse_console("\n".join(lines) + "\n")


def test_legacy_prefix_remains_bounded() -> None:
    assert parse_console("OUT 1 1065353216\nDONE\n")["outputs"].tolist() == [1.0]
    with pytest.raises(SpikeModelError):
        parse_console("OUT 4097\nDONE\n")


def test_large_complete_output_requires_explicit_independent_count() -> None:
    count = 1_048_577
    console = _console(count)
    with pytest.raises(SpikeModelError, match="output type or count"):
        parse_console(console)
    result = parse_console(console, expected_output_elements=count)
    assert result["raw_output_bits"].size == count
    assert np.array_equal(result["raw_output_bits"], np.arange(count, dtype=np.uint32) + 0x3F800000)


@pytest.mark.parametrize("expected", [1, 4096, 4098, 5120000])
def test_explicit_shape_never_accepts_prefix_or_other_tensor(expected) -> None:
    with pytest.raises(SpikeModelError):
        parse_console(_console(), expected_output_elements=expected)
    with pytest.raises(SpikeModelError):
        parse_console("OUT 2 0 0\nDONE\n", expected_output_elements=expected)


@pytest.mark.parametrize("expected", [True, 0, -1, 2**31, 1.5])
def test_invalid_expected_output_budget_refuses(expected) -> None:
    with pytest.raises(SpikeModelError, match="expected output count"):
        parse_console(_console(), expected_output_elements=expected)
