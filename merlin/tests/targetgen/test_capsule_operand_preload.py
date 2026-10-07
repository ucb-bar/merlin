"""A float capsule's operands must actually REACH the device.

``canonical_input_raws`` used to return bytes only when ``golden.yaml`` recorded ``fp8_raw_hex``. Only the
fp8 palette was ever written that way, so every bf16/fp16/f32 capsule handed the program oracle NOTHING:
its kernel ran on empty DRAM, produced zeros because almost every op maps 0 to 0, and the harness reported
``output_never_written`` -- blaming the agent for a store it had in fact emitted. On atlas the split was
exact: 12/12 capsules with recorded raws passed, 13/13 without them failed.

These pin the repaired contract: operands are supplied for any dtype the shared float table can encode,
they are byte-exact against the golden's own recorded values, and a capsule whose values do NOT sit on its
dtype's grid supplies nothing rather than a quantized approximation the golden never saw.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.runtime import fp8_formats as ff
from merlin.targetgen import capsule_common, capsule_golden

CAPSULES = repo_root() / "merlin/contract/capsules"
ATLAS = CAPSULES / "atlas"


def _capsule(name):
    """A tracked capsule by name, from any target's corpus: the preload contract is dtype-driven, so a
    bf16 donor serves whichever target's corpus holds it."""
    hits = [p.parent for p in CAPSULES.rglob("capsule.yaml") if p.parent.name == name]
    if not hits:
        pytest.skip(f"capsule {name} not present")
    cd = hits[0]
    return capsule_common.load_capsule(cd), cd


def _decode_leaf(cap, name, raw):
    dtype = capsule_golden._leaf_dtype(cap, name)
    width = ff.storage_bits(dtype) // 8
    return ff._decode(np.frombuffer(raw, dtype=f"<u{width}").astype(np.uint32), dtype)


def _codes(name: str, count: int, bits: int) -> np.ndarray:
    """Deterministic finite device codes for one leaf: exponent field kept below all-ones, so no
    inf/NaN pattern is drawn and every code decodes to an ordinary value of its format."""
    rng = np.random.default_rng(sum(name.encode()))
    codes = rng.integers(0, 1 << bits, size=count, dtype=np.uint32)
    return codes & ~np.uint32(1 << (bits - 2))


def _recorded(name, tmp_path, *, raw_hex: bool = False):
    """``name``'s real capsule beside a golden that records its operands, the way the golden generator
    records them: decoded values on each leaf's own grid (and, for the fp8 palette, the raw bytes).

    The golden is an untracked answer key, so a fresh checkout has none; the preload contract under
    test is about how operands are READ from it, which a recorded fixture states exactly.
    """
    cap, cd = _capsule(name)
    work = tmp_path / name
    work.mkdir()
    (work / "capsule.yaml").write_text((cd / "capsule.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    inputs = {}
    for leaf in cap["inputs"]:
        dtype, shape = leaf["dtype"], list(leaf["shape"])
        bits = ff.storage_bits(dtype)
        codes = _codes(leaf["name"], int(np.prod(shape)), bits)
        spec = {"shape": shape, "decoded": [float(v) for v in ff._decode(codes, dtype)]}
        if raw_hex:
            spec["fp8_raw_hex"] = [f"0x{int(c):02x}" for c in codes]
        inputs[leaf["name"]] = spec
    (work / "golden.yaml").write_text(
        yaml.safe_dump({"golden_source": "host_torch_eager", "oracle_provenance": {"inputs": inputs}}),
        encoding="utf-8",
    )
    return capsule_common.load_capsule(work), work


def test_a_bf16_capsule_supplies_device_operands(tmp_path):
    """The regression that cost atlas 13 capsules: this returned {} and nothing was preloaded."""
    cap, cd = _recorded("RP18_resadd_bf16_pt", tmp_path)
    raws = capsule_golden.canonical_input_raws(cap, cd)
    assert set(raws) == {"A", "B"}, "a two-operand bf16 capsule must supply BOTH operands"
    for name, raw in raws.items():
        assert len(raw) == 16 * 16 * 2, f"{name}: bf16 16x16 is 512 bytes"


def test_supplied_operands_decode_back_to_the_goldens_own_values(tmp_path):
    """Preloading anything other than the operands the golden used grades against the wrong reference."""
    for name in ("RP18_resadd_bf16_pt", "MF1_softmax_bf16_pt", "MF3_silu_bf16_pt"):
        cap, cd = _recorded(name, tmp_path)
        raws = capsule_golden.canonical_input_raws(cap, cd)
        vals = capsule_golden.canonical_input_values(cap, cd)
        assert raws, f"{name}: no operands supplied"
        for leaf, raw in raws.items():
            want = np.asarray(vals[leaf]["values"], dtype=np.float32).ravel()
            assert np.array_equal(_decode_leaf(cap, leaf, raw), want), f"{name}/{leaf} drifted"


def test_every_atlas_capsule_with_representable_float_operands_supplies_them():
    """The corpus-wide invariant. Stated as an implication so it is falsifiable rather than a tautology:
    if the recorded values round-trip exactly through the leaf's own dtype, the bytes MUST be supplied."""
    missing = []
    for p in sorted(ATLAS.rglob("capsule.yaml")):
        cd = p.parent
        cap = capsule_common.load_capsule(cd)
        raws = capsule_golden.canonical_input_raws(cap, cd)
        for leaf, spec in capsule_golden.canonical_input_values(cap, cd).items():
            dtype = capsule_golden._leaf_dtype(cap, leaf)
            if not dtype or leaf in raws:
                continue
            try:
                enc = ff.encode_bytes(spec["values"], dtype)
            except (KeyError, ValueError):
                continue  # dtype outside the float table / sub-byte
            want = np.asarray(spec["values"], dtype=np.float32).ravel()
            if np.array_equal(_decode_leaf(cap, leaf, enc), want):
                missing.append(f"{cd.name}/{leaf}")
    assert not missing, f"exactly-representable operands not supplied to the device: {missing}"


def test_a_lossy_reencoding_is_refused_rather_than_quantized(tmp_path):
    """A golden that stored pre-quantization floats for a narrow format must yield NO preload: handing the
    device quantized operands would grade the kernel against operands the golden never saw."""
    _, donor = _capsule("RP18_resadd_bf16_pt")  # a real, schema-valid capsule to vary from
    spec = yaml.safe_load((donor / "capsule.yaml").read_text())
    spec["name"] = "SYN_offgrid"
    spec["inputs"] = [{"name": "X", "role": "input", "shape": [1, 2], "dtype": "fp8_e4m3"}]
    spec["operation"] = {"op": "add", "attributes": {"out": "Y0", "arg_order": ["X"]}}
    spec.pop("interface_mlir", None)
    spec.pop("linalg_mlir", None)
    spec.pop("pytorch_ref", None)
    (tmp_path / "capsule.yaml").write_text(yaml.safe_dump(spec))
    off_grid = [0.1234567, 0.7654321]  # not on the e4m3 grid (3 mantissa bits)
    (tmp_path / "golden.yaml").write_text(
        yaml.safe_dump(
            {
                "golden_source": "host_torch_eager",
                "oracle_provenance": {"inputs": {"X": {"shape": [1, 2], "decoded": off_grid}}},
                "outputs": {"Y0": [[0.0, 0.0]]},
            }
        )
    )
    cap = capsule_common.load_capsule(tmp_path)
    assert capsule_golden.canonical_input_raws(cap, tmp_path) == {}
    # and the guard is not vacuous: on-grid values through the SAME path are supplied
    on_grid = [float(v) for v in ff._decode(np.array([0x30, 0x38], dtype=np.uint32), "fp8_e4m3")]
    (tmp_path / "golden.yaml").write_text(
        yaml.safe_dump(
            {
                "golden_source": "host_torch_eager",
                "oracle_provenance": {"inputs": {"X": {"shape": [1, 2], "decoded": on_grid}}},
                "outputs": {"Y0": [[0.0, 0.0]]},
            }
        )
    )
    assert set(capsule_golden.canonical_input_raws(cap, tmp_path)) == {"X"}


def test_recorded_device_bytes_win_over_reencoding(tmp_path):
    """fp8 capsules record the EXACT palette bytes; those must never be replaced by a re-encoding."""
    cap, cd = _recorded("AT2_single_tile_matmul", tmp_path, raw_hex=True)
    gy = yaml.safe_load((cd / "golden.yaml").read_text())
    # Make the two sources disagree, so the assertion below can tell which one was read: the decoded
    # values now re-encode to different bytes than the recorded palette.
    for spec in gy["oracle_provenance"]["inputs"].values():
        spec["decoded"] = [0.0] * len(spec["decoded"])
    (cd / "golden.yaml").write_text(yaml.safe_dump(gy), encoding="utf-8")
    recorded = {
        n
        for n, s in (gy["oracle_provenance"]["inputs"] or {}).items()
        if isinstance(s, dict) and (s.get("fp8_raw_hex") or s.get("raw_hex"))
    }
    assert recorded, "fixture no longer records raw hex; pick another fp8 capsule"
    raws = capsule_golden.canonical_input_raws(cap, cd)
    for n in recorded:
        spec = gy["oracle_provenance"]["inputs"][n]
        want = bytes(int(x, 16) & 0xFF for x in (spec.get("fp8_raw_hex") or spec.get("raw_hex")))
        assert raws[n] == want, f"{n}: recorded device bytes were overwritten"
