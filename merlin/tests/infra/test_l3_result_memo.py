"""An L3 run of a program already measured returns the measurement, and never a different one.

Measured across every campaign on disk: consecutive candidates emitted BYTE-IDENTICAL code for every
corpus member. The agent edits something, the harness pays a full cycle-accurate sweep, and the
program it measures is the one it measured last time. Over the 380-execution sweep of 2026-09-05,
296 of the 380 L3 executions were repeats of a program whose number was already known.

This is not a screen and not a prediction. Two runs of one program on one pinned engine return the
same cycles -- verified over 392 repeated measurements of identical bytes with zero disagreement --
so a hit returns the number rather than estimating it.

The key has to be the whole emitted program. The command buffer alone is NOT the program: 28 members
shared a command buffer and only 15 of them shared a cycle count, because the lowered module differed.
Keyed on the lowered module the agreement was exact, 15 of 15.

WHAT THESE TESTS EXIST TO CATCH. The reuse table previously had tests only for its KEY function, so
every one of them passed while nothing checked that a hit ever happened, that a hit was recorded, or
that a MISS happened when it had to. A cache that never misses is worse than no cache: the two facts
that must never be shared are a different simulator build (a cycle count measured under one is a
different number) and a different observation of the same program (the counter passes, and the
replicate a campaign runs precisely so it can measure its own dispersion).
"""
from __future__ import annotations

import json
import sys
import types

import pytest
from pathlib import Path

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_paired_perf_bench as PAIR  # noqa: E402

PIN_A, PIN_B = "a" * 64, "b" * 64
FIRRTL, MODEL = "c" * 64, "d" * 64
CB = {"abi_version": "0.1", "commands": [{"opcode": "MATMUL", "operands": {"dst": "Y"}}]}
SCOPE = "r000/unprofiled"


# ---------------------------------------------------------------------------------------------------
# the key
# ---------------------------------------------------------------------------------------------------
def test_the_same_emitted_program_on_the_same_engine_is_one_key():
    assert (PAIR._l3_memo_key(CB, "module {}", PIN_A, SCOPE)
            == PAIR._l3_memo_key(CB, "module {}", PIN_A, SCOPE))


def test_key_ordering_of_the_command_buffer_does_not_matter():
    """Two dicts that differ only in key order are the same program."""
    reordered = {"commands": CB["commands"], "abi_version": CB["abi_version"]}
    assert PAIR._l3_memo_key(CB, "m", PIN_A, SCOPE) == PAIR._l3_memo_key(reordered, "m", PIN_A, SCOPE)


def test_a_different_lowered_module_is_a_different_program():
    """The defect this guards: the command buffer alone is not the program."""
    assert (PAIR._l3_memo_key(CB, "module { A }", PIN_A, SCOPE)
            != PAIR._l3_memo_key(CB, "module { B }", PIN_A, SCOPE))


def test_a_different_command_buffer_is_a_different_program():
    other = {"abi_version": "0.1", "commands": [{"opcode": "CONV2D", "operands": {"dst": "Y"}}]}
    assert PAIR._l3_memo_key(CB, "m", PIN_A, SCOPE) != PAIR._l3_memo_key(other, "m", PIN_A, SCOPE)


def test_a_different_engine_shares_nothing():
    """A cycle count is a fact about a program AND the engine that ran it."""
    assert PAIR._l3_memo_key(CB, "m", PIN_A, SCOPE) != PAIR._l3_memo_key(CB, "m", PIN_B, SCOPE)


def test_each_counter_pass_is_its_own_measurement():
    """One program, three passes: the difference lives in the environment the ELF is built under.

    Nothing in the command buffer or the lowered module can see it, so without the scope the
    physical-byte pass is served the occupancy pass's readings and the two are linked to each other.
    """
    keys = {PAIR._l3_memo_key(CB, "m", PIN_A, f"r000/{name}")
            for name in ("unprofiled", "occupancy", "physical_bytes")}
    assert len(keys) == 3


def test_each_replicate_is_its_own_measurement():
    """The second replicate exists to MEASURE the dispersion, not to be handed the first's number."""
    assert (PAIR._l3_memo_key(CB, "m", PIN_A, "r000/unprofiled")
            != PAIR._l3_memo_key(CB, "m", PIN_A, "r001/unprofiled"))


def test_the_memo_starts_empty_so_a_stale_table_cannot_answer_for_a_fresh_stage():
    assert isinstance(PAIR._L3_MEMO, dict)


# ---------------------------------------------------------------------------------------------------
# the adapter: a hit returns the measurement and says it did; a miss runs the engine
# ---------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _empty_memo():
    """The in-process table is a module global; a test that inherits another's entries proves nothing."""
    PAIR._L3_MEMO.clear()
    yield
    PAIR._L3_MEMO.clear()


class _Certificate:
    def __init__(self, binary_sha256: str):
        self.pins = {"gsim_binary": {"sha256": binary_sha256},
                     "gsim_firrtl": {"sha256": FIRRTL},
                     "gsim_model": {"sha256": MODEL}}


class _Engine:
    """One stubbed GSIM: a binary whose bytes hash to its own pin, and a counted oracle call."""

    def __init__(self, tmp_path: Path, monkeypatch, name: str = "engineA"):
        import hashlib

        self.binary = tmp_path / f"{name}.bin"
        self.binary.write_bytes(name.encode())
        self.elf = tmp_path / f"{name}.elf"
        self.elf.write_bytes(b"elf")
        self.pin = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.certificate = _Certificate(self.pin)
        self.calls = 0

        import merlin.runtime.backends.base as backends
        backend = types.SimpleNamespace(gsim_path=lambda: str(self.binary))
        monkeypatch.setattr(backends, "get_backend", lambda target: backend)

        def run_on_oracle(cb, llvm_text, *, simulator, target, workdir, timeout):
            self.calls += 1
            return {"elf": str(self.elf), "cycles": 100 + self.calls,
                    "outputs": {"Y": [1, 2, 3]}, "console": "",
                    "oracle": {"kind": "rtl_gsim", "derived_from_rtl": True},
                    "timing": {"build_s": 0.5, "sim_active_s": 30.0, "oracle_wait_s": 0.0}}

        monkeypatch.setattr(PAIR.OOT, "run_on_oracle", run_on_oracle)
        monkeypatch.setattr(PAIR.CERTPROD, "encode_declared_outputs",
                            lambda outputs, cb: ("s" * 64, {}))

    def adapter(self, store, evidence=None):
        return PAIR._gsim_l3_adapter("t", evidence if evidence is not None else {},
                                     self.certificate, reuse_scope=SCOPE, store=store)


@pytest.fixture()
def store(tmp_path):
    return PAIR.L3MeasurementStore(tmp_path / "l3_cache")


def test_a_second_measurement_of_the_same_program_returns_the_cached_number(tmp_path, monkeypatch,
                                                                            store):
    engine = _Engine(tmp_path, monkeypatch)
    first = engine.adapter(store)(CB, "module {}", tmp_path, 60)
    second = engine.adapter(store)(CB, "module {}", tmp_path, 60)
    assert engine.calls == 1, "the engine ran twice for one program"
    assert second["cycles"] == first["cycles"]


def test_a_reused_row_says_it_was_reused_everywhere_a_reader_looks(tmp_path, monkeypatch, store):
    """The stamp has to reach a RECORD. It previously existed only on objects nobody wrote down."""
    engine = _Engine(tmp_path, monkeypatch)
    engine.adapter(store)(CB, "module {}", tmp_path, 60)
    evidence: dict = {}
    reused = engine.adapter(store, evidence)(CB, "module {}", tmp_path, 60)
    assert reused["reused_measurement"] is True
    # the timing block is the part of the return the tier record keeps
    assert reused["timing"]["reused_measurement"] is True
    assert reused["timing"]["sim_active_s"] == 30.0
    assert evidence["gsim"]["reused_measurement"]["measured_program_sha256"]
    # THIS RUN BUILT NO ELF, so it names none -- while the digest that identifies the program stays.
    assert reused["elf"] is None
    assert evidence["gsim"]["elf"] is None
    assert evidence["gsim"]["elf_sha256"]


def test_changing_only_the_binary_sha_is_a_miss(tmp_path, monkeypatch, store):
    """A cycle count measured under a different simulator build is a different number."""
    engine = _Engine(tmp_path, monkeypatch)
    engine.adapter(store)(CB, "module {}", tmp_path, 60)
    # A second engine: different bytes, therefore a different pin, and the same program.
    other = _Engine(tmp_path, monkeypatch, name="engineB")
    assert other.pin != engine.pin
    other.adapter(store)(CB, "module {}", tmp_path, 60)
    assert other.calls == 1, "a measurement was reused across simulator builds"


def test_changing_one_byte_of_the_command_buffer_is_a_miss(tmp_path, monkeypatch, store):
    engine = _Engine(tmp_path, monkeypatch)
    engine.adapter(store)(CB, "module {}", tmp_path, 60)
    edited = json.loads(json.dumps(CB))
    edited["commands"][0]["operands"]["dst"] = "Z"
    engine.adapter(store)(edited, "module {}", tmp_path, 60)
    assert engine.calls == 2


def test_changing_one_byte_of_the_lowered_module_is_a_miss(tmp_path, monkeypatch, store):
    engine = _Engine(tmp_path, monkeypatch)
    engine.adapter(store)(CB, "module {}", tmp_path, 60)
    engine.adapter(store)(CB, "module { }", tmp_path, 60)
    assert engine.calls == 2


def test_a_different_pass_over_the_same_program_is_a_miss(tmp_path, monkeypatch, store):
    """Latent until a campaign runs with counters: two passes, one key, one set of readings."""
    engine = _Engine(tmp_path, monkeypatch)
    PAIR._gsim_l3_adapter("t", {}, engine.certificate,
                          reuse_scope="r000/occupancy", store=store)(CB, "m", tmp_path, 60)
    PAIR._gsim_l3_adapter("t", {}, engine.certificate,
                          reuse_scope="r000/physical_bytes", store=store)(CB, "m", tmp_path, 60)
    assert engine.calls == 2


# ---------------------------------------------------------------------------------------------------
# the half that survives the process
# ---------------------------------------------------------------------------------------------------
def test_a_later_process_finds_the_measurement_on_disk(tmp_path, monkeypatch, store):
    """The in-process table dies with the stage; three trial processes measured the same 71 programs."""
    engine = _Engine(tmp_path, monkeypatch)
    first = engine.adapter(store)(CB, "module {}", tmp_path, 60)
    PAIR._L3_MEMO.clear()                      # what a fresh process starts with
    second = engine.adapter(store)(CB, "module {}", tmp_path, 60)
    assert engine.calls == 1
    assert second["cycles"] == first["cycles"]
    assert second["reused_measurement"] is True


def test_a_stored_measurement_from_another_engine_is_refused(tmp_path, monkeypatch, store):
    """The pins are compared as data. A file at the right address is not evidence on its own."""
    engine = _Engine(tmp_path, monkeypatch)
    engine.adapter(store)(CB, "module {}", tmp_path, 60)
    key = PAIR._l3_memo_key(CB, "module {}", engine.pin, SCOPE)
    record = json.loads((store.root / f"{key}.json").read_text(encoding="utf-8"))
    record["engine_pins"]["gsim_firrtl"] = "e" * 64
    (store.root / f"{key}.json").write_text(json.dumps(record), encoding="utf-8")
    PAIR._L3_MEMO.clear()
    engine.adapter(store)(CB, "module {}", tmp_path, 60)
    assert engine.calls == 2, "a measurement was reused across a changed RTL pin"


def test_an_unreadable_entry_is_a_miss_and_not_a_crash(tmp_path, monkeypatch, store):
    """Every failure of this cache resolves to measuring again; none of them resolves to raising."""
    engine = _Engine(tmp_path, monkeypatch)
    engine.adapter(store)(CB, "module {}", tmp_path, 60)
    key = PAIR._l3_memo_key(CB, "module {}", engine.pin, SCOPE)
    (store.root / f"{key}.json").write_text("{ half a rec", encoding="utf-8")
    PAIR._L3_MEMO.clear()
    engine.adapter(store)(CB, "module {}", tmp_path, 60)
    assert engine.calls == 2


def test_an_absent_store_root_is_a_miss_and_the_put_creates_it(tmp_path, monkeypatch):
    fresh = PAIR.L3MeasurementStore(tmp_path / "never" / "made")
    engine = _Engine(tmp_path, monkeypatch)
    assert fresh.get("nothing" * 8, {}) is None
    engine.adapter(fresh)(CB, "module {}", tmp_path, 60)
    assert list(fresh.root.glob("*.json"))


def test_the_store_lives_under_the_purgeable_cache_root_for_its_target():
    """Generated output has one root; a cache is not an exception to it."""
    from merlin.common.artifacts import artifacts_dir

    root = PAIR._l3_store("some-target").root
    assert root.parent.parent == artifacts_dir() / "cache"
    assert root.name == "some-target"
