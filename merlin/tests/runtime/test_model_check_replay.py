"""The replay image's digest and console verdicts are judged exactly, and a missing line is not a pass."""

from __future__ import annotations

import numpy as np

from merlin.model_check import replay as RP


def _c_digest(data: bytes) -> tuple[int, int]:
    """The C loop of the replay image, transcribed: whole words, tail poisoned, two weighted sums."""
    data = data + bytes([RP.POISON]) * ((-len(data)) % 8)
    a = b = 0
    for i in range(len(data) // 8):
        w = int.from_bytes(data[8 * i : 8 * i + 8], "little")
        a = (a + w * (2 * i + 1)) % 2**64
        b = (b + w * (((i * RP.GOLDEN) % 2**64) | 1)) % 2**64
    return a, b


def test_digest_matches_the_device_loop_including_a_poisoned_tail():
    for n in (1, 7, 8, 13, 64):
        data = np.random.default_rng(n).integers(0, 256, size=n, dtype=np.uint8).tobytes()
        assert RP.digest(data) == _c_digest(data)


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
