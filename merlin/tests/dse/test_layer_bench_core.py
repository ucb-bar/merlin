"""Per-layer bench core: key identity, console parsing, and the receipt cache."""

import json

import pytest

from merlin.perf.layer_bench import (
    ConsoleError,
    LayerKey,
    ReceiptCache,
    ReceiptError,
    parse_engine_finish,
    parse_layer_records,
)

SHA = "a" * 64


def _key(**over):
    fields = dict(
        target="t",
        design_pin="pin",
        engine_sha256=SHA,
        group_signature="conv:1x56x56x64",
        contract_digest="b" * 64,
        schedule_digest="c" * 64,
        emitter_digest="d" * 64,
        harness_version="v1",
        protocol="warm_then_measured",
    )
    fields.update(over)
    return LayerKey(**fields)


def test_every_field_changes_the_digest():
    base = _key().digest()
    for name, value in (
        ("design_pin", "other"),
        ("engine_sha256", "e" * 64),
        ("group_signature", "conv:1x28x28x128"),
        ("contract_digest", "f" * 64),
        ("schedule_digest", "0" * 64),
        ("emitter_digest", "1" * 64),
        ("harness_version", "v2"),
        ("protocol", "cold_single"),
    ):
        assert _key(**{name: value}).digest() != base, name


def test_key_validates_its_fields():
    with pytest.raises(ValueError):
        _key(engine_sha256="short")
    with pytest.raises(ValueError):
        _key(protocol="warm")
    with pytest.raises(ValueError):
        _key(group_signature="")


def test_records_parse_and_malformed_ones_raise():
    console = "boot\nLB_RECORD k1 cycles=527409 rs_active=51000\nnoise LB_RECORD x\nLB_RECORD k2 cycles=12\n"
    recs = parse_layer_records(console)
    assert [(r.label, r.cycles, r.fields) for r in recs] == [("k1", 527409, {"rs_active": 51000}), ("k2", 12, {})]
    for bad in (
        "LB_RECORD k1",
        "LB_RECORD k1 rs=3",
        "LB_RECORD k1 cycles=abc",
        "LB_RECORD k1 cycles=1 cycles=2",
        "LB_RECORD k1 cycles=-5",
    ):
        with pytest.raises(ConsoleError):
            parse_layer_records(bad)


def test_engine_finish_line():
    err = (
        "[tsi-probe] calls=500000 in_reset=14 done=0\n"
        "[gsim-emu] FINISHED: cycles=601947 wall=42.71s (14095 cyc/s) done=1 exit_code=0\n"
    )
    fin = parse_engine_finish(err)
    assert (fin.cycles, fin.wall_seconds, fin.done, fin.exit_code) == (601947, 42.71, True, 0)
    assert parse_engine_finish("[gsim-emu] 8000000 cycles, 543.1s, 14730 cyc/s\n") is None
    capped = parse_engine_finish("[gsim-emu] FINISHED: cycles=8000000 wall=543.12s (14730 cyc/s) done=0 exit_code=0")
    assert capped.done is False
    with pytest.raises(ConsoleError):
        parse_engine_finish("[gsim-emu] FINISHED: cycles=oops wall=1s done=1 exit_code=0")


def test_cache_round_trip_and_tamper_detection(tmp_path):
    cache = ReceiptCache(tmp_path)
    key = _key()
    assert cache.get(key) is None
    stored = cache.put(key, {"elf_sha256": SHA, "records": [{"label": "k1", "cycles": 527409}]})
    assert cache.get(key) == stored
    # Tamper with the stored cycles: the self-hash must catch it.
    path = cache.path_for(key)
    doc = json.loads(path.read_text())
    doc["records"][0]["cycles"] = 1
    path.write_text(json.dumps(doc))
    with pytest.raises(ReceiptError):
        cache.get(key)


def _elf64_with_loads(path, memsz):
    """A minimal ELF64 LE file whose program headers are PT_LOAD with the given p_memsz values."""
    import struct

    phoff, phentsize = 64, 56
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x20, phoff)
    struct.pack_into("<HH", header, 0x36, phentsize, len(memsz))
    body = bytearray()
    for size in memsz:
        ph = bytearray(phentsize)
        struct.pack_into("<I", ph, 0, 1)  # PT_LOAD
        struct.pack_into("<Q", ph, 0x28, size)  # p_memsz
        body += ph
    path.write_bytes(bytes(header) + bytes(body))


def test_loaded_bytes_counts_bss_zero_fill(tmp_path):
    from merlin.perf.layer_bench import BuildError, loaded_bytes

    elf = tmp_path / "x.elf"
    _elf64_with_loads(elf, [26_000, 4 << 20])  # code + a 4 MiB .bss, as in the E0 failure
    assert loaded_bytes(elf) == 26_000 + (4 << 20)
    (tmp_path / "bad.elf").write_bytes(b"not an elf at all")
    with pytest.raises(BuildError):
        loaded_bytes(tmp_path / "bad.elf")


def test_cache_refuses_a_receipt_filed_under_the_wrong_key(tmp_path):
    cache = ReceiptCache(tmp_path)
    a, b = _key(), _key(harness_version="v2")
    cache.put(a, {"elf_sha256": SHA})
    cache.path_for(b).parent.mkdir(parents=True, exist_ok=True)
    cache.path_for(b).write_text(cache.path_for(a).read_text())
    with pytest.raises(ReceiptError):
        cache.get(b)
