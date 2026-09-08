"""The Muon GSIM tier grades a declared result page, not console silence."""
from __future__ import annotations

from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from merlin.runtime.backends.base import get_backend
from merlin.targetgen.isa_model import IsaModel


BACKEND = get_backend("muon")
H = BACKEND.muon_harness
MO = BACKEND.muon_oracles
MU = BACKEND.muon
RP = BACKEND.muon_result_page

MODEL = IsaModel(target="synthetic", runtime_abi={
    "special_csrs": {"mhartid": 0xF14},
    "apertures": {"console_mmio": 0xFF080000},
})


def _arg(name: str, count: int, dtype: str = "f32") -> object:
    return H.TensorArg(name, 1, count, [0.0] * count, dtype)


def test_result_page_harness_streams_through_fixed_acknowledged_mailbox() -> None:
    harness = H.build_external_kernel_main(
        [_arg("X", 65)], [_arg("Y", 65)], kernel_symbol="kernel", model=MODEL,
        result_page=True,
    )

    assert "merlin_result_mailbox[32]" in harness.source
    assert 'section(".data.merlin_result")' in harness.source
    assert "merlin_result_status" in harness.source
    assert "_base<65u;_base+=32u" in harness.source
    assert "merlin_result_status[1]=_count" in harness.source
    assert "merlin_result_status[0]=(0x4d525231u^_merlin_sequence)" in harness.source
    assert "merlin_result_status[2]!=(0x4d524131u^_merlin_sequence)" in harness.source
    release = harness.source.index('__asm__ volatile("fence rw,rw" ::: "memory");')
    publish = harness.source.index("merlin_result_status[0]=(0x4d525231u^_merlin_sequence)")
    observe = harness.source.index("merlin_result_status[2]!=(0x4d524131u^_merlin_sequence)")
    acquire = harness.source.index('__asm__ volatile("fence r,rw" ::: "memory");', observe)
    assert release < publish < observe < acquire
    assert 'for(;;)__asm__ volatile("nop" ::: "memory");' in harness.source
    assert "volatile uint32_t _out_Y[65]" in harness.source
    assert harness.results == [{"name": "Y", "elements": 65, "dtype": "f32"}]


def test_sequence_tokens_exclude_stale_ready_and_ack_interleavings() -> None:
    """A token from either adjacent transaction cannot authorize this one.

    Release/acquire ordering then makes the changing publication word the
    happens-before edge for the count/mailbox payload and for mailbox reuse.
    """
    for sequence in range(1, 257):
        ready = RP.ready_token(sequence)
        ack = RP.ack_token(sequence)
        assert ready != RP.ready_token(sequence - 1)
        assert ready != RP.ready_token(sequence + 1)
        assert ack != RP.ack_token(sequence - 1)
        assert ack != RP.ack_token(sequence + 1)


def test_legacy_harness_is_unchanged_when_numeric_mailbox_is_disabled() -> None:
    harness = H.build_external_kernel_main(
        [_arg("X", 2)], [_arg("Y", 2)], kernel_symbol="kernel", model=MODEL,
        result_page=False,
    )

    assert "merlin_result_status" not in harness.source
    assert "merlin_result_mailbox" not in harness.source
    assert "_merlin_sequence" not in harness.source
    assert 'for(;;)__asm__ volatile("nop"' not in harness.source
    assert '_ps("OUT Y 1 2")' in harness.source
    assert '_ps("DONE\\n")' in harness.source
    assert harness.results is None


def test_manifest_exposes_only_mailbox_address_not_output_buffers(monkeypatch) -> None:
    seen = []

    def addresses(_elf, names):
        seen.extend(names)
        return {RP.STATUS_SYMBOL: 0x4000, RP.MAILBOX_SYMBOL: 0x4040}

    monkeypatch.setattr(RP, "symbol_addresses", addresses)
    manifest = RP.manifest_from_elf("unused.elf", [_arg("Y", 256)], soc_offset=0x110000000)

    assert seen == [RP.STATUS_SYMBOL, RP.MAILBOX_SYMBOL]
    assert manifest["schema"] == "merlin.muon-result-mailbox.v2"
    assert manifest["mailbox"] == {
        "symbol": RP.MAILBOX_SYMBOL,
        "muon_address": 0x4040,
        "soc_address": 0x110004040,
        "words": 32,
    }
    assert manifest["outputs"] == [{"name": "Y", "elements": 256, "dtype": "f32"}]
    assert "soc_address" not in manifest["outputs"][0]


def test_carrier_is_generated_from_declared_layout_and_policy() -> None:
    manifest = {
        "status": {"soc_address": 0x110004000},
        "mailbox": {"soc_address": 0x110004040, "words": 32},
        "outputs": [
            {"name": "Y", "elements": 2, "dtype": "f32"},
            {"name": "Z", "elements": 1, "dtype": "f32"},
        ],
    }
    source = RP.render_carrier(
        manifest,
        {"Y": [1.0, -2.0], "Z": [0.5]},
        {"compare": "tolerance_float", "atol": 0.03125, "rtol": 0.015625},
    )

    assert "0x110004000ULL" in source
    assert "0x110004040ULL" in source
    assert "0x110004080ULL" not in source
    assert "merlin_numeric_pass" in source and "merlin_numeric_fail" in source
    assert "ordered_f32" in source
    assert "STATUS[0] != MERLIN_RESULT_READY(sequence)" in source
    assert "uint32_t count = STATUS[1]" in source
    assert "STATUS[2] = MERLIN_RESULT_ACK(sequence)" in source
    observe = source.index("STATUS[0] != MERLIN_RESULT_READY(sequence)")
    acquire = source.index('fence r,rw', observe)
    mailbox_read = source.index("uint32_t got = MAILBOX[i]")
    release_ack = source.index('fence rw,rw', mailbox_read)
    ack = source.index("STATUS[2] = MERLIN_RESULT_ACK(sequence)", mailbox_read)
    assert observe < acquire < mailbox_read < release_ack < ack


def test_private_expected_change_changes_only_the_trusted_carrier() -> None:
    manifest = {
        "status": {"soc_address": 0x110004000},
        "mailbox": {"soc_address": 0x110004040, "words": 32},
        "outputs": [{"name": "Y", "elements": 1, "dtype": "f32"}],
    }
    policy = {"compare": "tolerance_float", "atol": 0.03125, "rtol": 0.015625}

    positive = RP.render_carrier(manifest, {"Y": [1.0]}, policy)
    negative = RP.render_carrier(manifest, {"Y": [101.0]}, policy)

    assert positive != negative
    assert "MERLIN_MAILBOX_WORDS 32u" in positive
    assert "MERLIN_MAILBOX_WORDS 32u" in negative


def test_inline_source_large_numeric_output_uses_private_static_storage() -> None:
    source = H.build_program(
        "void kernel(const void *x, void *y) {}", [_arg("X", 1)], [_arg("Y", 2048)],
        kernel_symbol="kernel", model=MODEL, result_page=True,
    )

    declaration = "static volatile uint32_t _out_Y[2048];"
    assert declaration in source
    assert source.index(declaration) < source.index("int main(void)")
    assert "volatile uint32_t _out_Y[2048];" not in source[source.index("int main(void)"):]


def _artifact_verifier():
    root = Path(__file__).resolve().parents[3]
    path = root / "out/artifacts/capsule-bench/radiance/l3_result_page_v2_20260908/verify.py"
    module = ModuleType("radiance_result_artifact_verify")
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def test_publication_verifier_rejects_unlisted_and_hidden_forbidden_files(tmp_path) -> None:
    verifier = _artifact_verifier()
    safe = tmp_path / "receipt.json"
    safe.write_text("{}\n", encoding="utf-8")
    (tmp_path / "SHA256SUMS").write_text(
        f"{verifier.digest(safe)}  receipt.json\n", encoding="utf-8")
    assert verifier.publication_failures(tmp_path) == []

    extra = tmp_path / "unlisted.txt"
    extra.write_text("not indexed\n", encoding="utf-8")
    assert any("unlisted" in failure for failure in verifier.publication_failures(tmp_path))

    extra.unlink()
    private = tmp_path / "nested/result_carrier.c"
    private.parent.mkdir()
    private.write_text("private answer\n", encoding="utf-8")
    failures = verifier.publication_failures(tmp_path)
    assert any("forbidden" in failure and "result_carrier.c" in failure for failure in failures)


def test_outcome_requires_the_final_pc_to_reach_a_retained_symbol() -> None:
    symbols = {"merlin_numeric_pass": 0x80000086, "merlin_numeric_fail": 0x800000C6}
    assert RP.outcome_from_console(
        "[gsim-probe final] rocket_pc=0x80000086\n[gsim-emu] FINISHED: cycles=120000\n",
        symbols,
    ) == "pass"
    assert RP.outcome_from_console(
        "[gsim-probe final] rocket_pc=0x800000c6\n",
        symbols,
    ) == "fail"
    assert RP.outcome_from_console("[gsim-emu] FINISHED: cycles=120000\n", symbols) is None


@pytest.mark.parametrize(("final_pc", "status"), [
    (0x80000086, "pass"),
    (0x800000C6, "fail"),
])
def test_gsim_adapter_prefers_numeric_pc_witness_over_cycle_cap(
        final_pc, status, monkeypatch, tmp_path) -> None:
    cb = {
        "target": "synthetic",
        "_oracle_expected_outputs": {"Y": [1.0]},
        "_oracle_numeric_policy": {"compare": "tolerance_float", "atol": 0.01},
    }
    monkeypatch.setenv("MERLIN_MUON_GSIM_MAXCYCLES", "120000")
    monkeypatch.setattr(MO, "gsim_status", lambda target: (True, "stub"))
    from merlin.targetgen import gsim_emulator as GE
    monkeypatch.setattr(GE, "emulator_path", lambda *a, **k: tmp_path / "emu")
    monkeypatch.setattr(MU, "is_mlir_artifact", lambda src: True)
    monkeypatch.setattr(MU, "compile_mlir_forkfree", lambda *a, **k: tmp_path / "kernel.elf")
    monkeypatch.setattr(H, "args_from_cb", lambda cb: ([], [_arg("Y", 1)]))
    monkeypatch.setattr(RP, "manifest_from_elf", lambda *a, **k: {
        "status": {"soc_address": 0x110004000},
        "mailbox": {"soc_address": 0x110004040, "words": 32},
        "outputs": [{"name": "Y", "elements": 1, "dtype": "f32"}],
    })
    monkeypatch.setattr(RP, "render_carrier", lambda *a, **k: "int main(void){return 0;}\n")
    monkeypatch.setattr(MU, "fuse_soc_elf", lambda *a, **k: tmp_path / "kernel.soc.elf")
    monkeypatch.setattr(RP, "symbol_addresses", lambda *a, **k: {
        "merlin_numeric_pass": 0x80000086, "merlin_numeric_fail": 0x800000C6})
    monkeypatch.setattr(MO, "flops_from_cb", lambda cb: 0)
    monkeypatch.setattr(
        __import__("subprocess"), "run",
        lambda *a, **k: SimpleNamespace(returncode=0),
    )
    console = (f"[gsim-probe final] rocket_pc=0x{final_pc:x}\n"
               "[gsim-emu] FINISHED: cycles=120000\n")
    monkeypatch.setattr(MU, "_read_console", lambda log: (console, len(console), False))

    result = MO.gsim_muon_adapter("synthetic")(cb, "llvm.func @kernel()", tmp_path, 60)

    assert result["numeric_verdict"]["status"] == status
    assert result["numeric_verdict"]["elements_checked"] == 1
    assert not result.get("completion_only")
