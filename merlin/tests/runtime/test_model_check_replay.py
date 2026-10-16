"""The replay image's digest and console verdicts are judged exactly, and a missing line is not a pass."""

from __future__ import annotations

import numpy as np

from merlin.model_check import replay as RP

M64 = 2**64


def _weighted(data: bytes) -> tuple[int, int]:
    """The digest's definition, word by word: weights ``2i+1`` and ``(i * GOLDEN) | 1``."""
    data = data + bytes([RP.POISON]) * ((-len(data)) % 8)
    a = b = 0
    for i in range(len(data) // 8):
        w = int.from_bytes(data[8 * i : 8 * i + 8], "little")
        a = (a + w * (2 * i + 1)) % M64
        b = (b + w * (((i * RP.GOLDEN) % M64) | 1)) % M64
    return a, b


def _c_digest(data: bytes) -> tuple[int, int]:
    """The C loop of the replay image, transcribed: pairs of words, adds only, then two products."""
    data = data + bytes([RP.POISON]) * ((-len(data)) % 8)
    w = [int.from_bytes(data[8 * i : 8 * i + 8], "little") for i in range(len(data) // 8)]
    s = prefix = even = 0
    i = 0
    while i + 1 < len(w):
        even = (even + w[i]) % M64
        s = (s + w[i]) % M64
        prefix = (prefix + s) % M64
        s = (s + w[i + 1]) % M64
        prefix = (prefix + s) % M64
        i += 2
    if i < len(w):
        even, s = (even + w[i]) % M64, (s + w[i]) % M64
        prefix = (prefix + s) % M64
    t = (len(w) * s - prefix) % M64
    return (2 * t + s) % M64, (t * RP.GOLDEN + even) % M64


def test_digest_matches_the_device_loop_including_a_poisoned_tail():
    for n in (1, 7, 8, 13, 16, 24, 64, 1000):
        data = np.random.default_rng(n).integers(0, 256, size=n, dtype=np.uint8).tobytes()
        assert RP.digest(data) == _c_digest(data) == _weighted(data)


def test_console_records_carry_each_calls_cycles():
    plan = [{"case": 0, "symbol": "k0", "label": "a", "outputs": [{"role": "out_0", "digest": (1, 2)}]}]
    console = "[UART] noise\nMCCYC 0 0xfa3e\nMCCASE 0 0 0x1 0x2\nMCDONE\n"
    seen, cycles = RP.console_records(console)
    assert seen == {(0, 0): (1, 2)} and cycles == {0: 0xFA3E}
    assert RP.judge_console(console, plan)[0]["cycles"] == 0xFA3E


def _case(symbol: str, n: int) -> RP.ReplayCase:
    return RP.ReplayCase(
        symbol=symbol,
        label=f"{symbol}#{n}",
        roles=("input_0", "out_0"),
        operands=(np.full(n, n, dtype=np.int8), np.zeros(n, dtype=np.int32)),
        expected=(np.arange(n, dtype=np.int32),),
    )


def test_board_image_gives_every_call_its_own_loader_poisoned_slots(tmp_path, monkeypatch):
    from types import SimpleNamespace

    buffers = [
        SimpleNamespace(role="input_0", kind="input", dtype="i8", shape=(10,)),
        SimpleNamespace(role="out_0", kind="output", dtype="i32", shape=(10,)),
    ]
    monkeypatch.setattr(RP, "_kernel_order", lambda _iface, _roles: (buffers, {"input_0": 0, "out_0": 1}))
    cases = [_case("k", 10), _case("k", 10)]
    main_c, data_s, plan = RP._write_image_sources(  # noqa: SLF001
        tmp_path, cases, {"k": "kernel_k"}, {"k": ""}, preload_outputs=True
    )
    offsets = [p["outputs"][0]["offset"] for p in plan]
    assert offsets == [0, 64]  # disjoint slots, not one reused buffer
    text, data = main_c.read_text(), data_s.read_text()
    assert "mc_poison(mc_out" not in text and "mc_results" in text
    assert f".fill 128, 1, {RP.POISON}" in data
    assert text.count("mc_cyc(") == 3  # the definition and one per call
    main_c, _, plan = RP._write_image_sources(tmp_path, cases, {"k": "kernel_k"}, {"k": ""})  # noqa: SLF001
    assert [p["outputs"][0]["offset"] for p in plan] == [0, 0]
    assert main_c.read_text().count("mc_poison(mc_out") == 2


def test_digest_sees_a_moved_word():
    data = np.arange(32, dtype=np.int32)
    swapped = data.copy()
    swapped[[0, 2]] = swapped[[2, 0]]
    assert RP.digest(data.tobytes()) != RP.digest(swapped.tobytes())


def test_judge_requires_every_reported_digest_to_match():
    plan = [
        {"case": 0, "symbol": "k0", "label": "a", "outputs": [{"role": "out_0", "digest": (1, 2)}]},
        {"case": 1, "symbol": "k1", "label": "b", "outputs": [{"role": "out_0", "digest": (3, 4)}]},
        {"case": 2, "symbol": "k1", "label": "c", "outputs": [{"role": "out_0", "digest": (5, 6)}]},
    ]
    console = "MCCASE 0 0 1 2\nMCCASE 1 0 3 5\nMCDONE\n"
    verdicts = RP.judge_console(console, plan)
    assert [v["exact"] for v in verdicts] == [True, False, False]
    assert verdicts[2]["outputs"][0]["reported"] is False
