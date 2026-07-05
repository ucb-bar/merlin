"""Board-free unit tests for the Buddy (buddy-mlir) baseline arm (merlin.baselines.buddy).

These never touch a board or require buddy-mlir to be built: they exercise bundle resolution
(including the legacy fp32-LLM dir fallback), the RVV-audit symbol->region mapping, the object
audit's libc-ignore behaviour, and the not_built/not_run honesty contract when the toolchain or
board is absent — so the gate stays green regardless of buddy build state.
"""
from __future__ import annotations

from merlin.baselines import buddy


# --- bundle resolution (legacy fp32-LLM fallback) ---------------------------------------------

def test_resolve_bundle_convention_path():
    # bitvla follows the <model>_fp32_consistent convention.
    b = buddy.resolve_bundle("bitvla", "fp32")
    assert b.model == "bitvla" and b.variant == "fp32"
    assert b.root.name == "bitvla_fp32_consistent"


def test_resolve_bundle_legacy_fp32_llm_fallback(tmp_path, monkeypatch):
    # tiny_llama/small_llama fp32 captures predate the convention: verify the legacy-dir fallback
    # is used ONLY when the convention dir is absent and the legacy dir exists.
    import merlin.baselines.bundle as _bundle
    import merlin.common.artifacts as _art

    monkeypatch.setattr(_art, "recaptures_dir", lambda: tmp_path)
    monkeypatch.setattr(_bundle, "recaptures_dir", lambda: tmp_path)
    # no convention dir; create the legacy tiny_consistent with a model.mlir
    legacy = tmp_path / "tiny_consistent"
    legacy.mkdir()
    (legacy / "model.mlir").write_text("module {}")
    b = buddy.resolve_bundle("tiny_llama", "fp32")
    assert b.root.name == "tiny_consistent"
    assert b.mlir.is_file()


def test_resolve_bundle_no_legacy_for_non_llm(tmp_path, monkeypatch):
    import merlin.baselines.bundle as _bundle
    import merlin.common.artifacts as _art

    monkeypatch.setattr(_art, "recaptures_dir", lambda: tmp_path)
    monkeypatch.setattr(_bundle, "recaptures_dir", lambda: tmp_path)
    # bitvla has no legacy alias -> resolve returns the (missing) convention path, mlir absent.
    b = buddy.resolve_bundle("bitvla", "fp32")
    assert b.root.name == "bitvla_fp32_consistent"
    assert not b.mlir.is_file()


# --- symbol -> region mapping -----------------------------------------------------------------

def test_region_of_symbol():
    assert buddy._region_of_symbol("forward_matmul_3") == "gemm"
    assert buddy._region_of_symbol("softmax_kernel") == "attention"
    assert buddy._region_of_symbol("rmsnorm_0") == "norm"
    assert buddy._region_of_symbol("elementwise_add") == "elementwise"
    assert buddy._region_of_symbol("some_glue") == "other"


# --- object audit ignores libc/harness symbols ------------------------------------------------

def test_audit_object_ignores_plumbing(tmp_path, monkeypatch):
    # feed a synthetic objdump text: one scalar model kernel + one scalar libc symbol. Only the
    # model kernel should surface as a labeled ScalarFallback (plumbing is ignored).
    disasm = (
        "0000000000010000 <forward_matmul_0>:\n"
        "   10000:\t00b50533          \tadd\ta0,a0,a1\n"
        "   10004:\t02c58533          \tmul\ta0,a1,a2\n"
        "   10008:\t00008067          \tret\n"
        "0000000000010100 <memcpy>:\n"
        "   10100:\t00b50533          \tadd\ta0,a0,a1\n"
        "   10104:\t00008067          \tret\n"
    )

    import merlin.baselines.rvv_audit as ra

    monkeypatch.setattr(ra, "audit_binary", lambda p, **k: ra.classify_disasm(disasm))
    cov, fallbacks, by_symbol = buddy.audit_object(tmp_path / "fake.o")
    syms = {f.symbol for f in fallbacks}
    assert "forward_matmul_0" in syms
    assert "memcpy" not in syms                       # plumbing ignored
    assert fallbacks[0].region == "gemm"              # matmul -> gemm bucket
    assert cov == 0.0                                 # all scalar -> 0% RVV


# --- honesty contract when toolchain / board absent -------------------------------------------

def test_run_model_missing_bundle_is_not_built(monkeypatch):
    # An absent capture bundle -> not_built with an explicit reason, never a fabricated pass.
    import merlin.baselines.bundle as _bundle

    fake = _bundle.CaptureBundle(model="bitvla", variant="fp32", root=_bundle.Path("/nonexistent"))
    monkeypatch.setattr(buddy, "resolve_bundle", lambda m, v="fp32": fake)
    r = buddy.run_model("bitvla", "fp32", write=False)
    assert r.status() == "not_built"
    assert "missing" in r.gap_reason
    r.validate()


def test_run_model_toolchain_absent_is_not_built(monkeypatch):
    # bundle present but buddy tools not built -> not_built with the toolchain reason.
    monkeypatch.setattr(buddy, "buddy_available", lambda: False)
    r = buddy.run_model("bitvla", "fp32", write=False)
    # bitvla fp32 bundle exists on disk in this repo; if it doesn't, the missing-bundle path is
    # also a valid not_built — either way it must be not_built with a reason.
    assert r.status() == "not_built"
    assert r.gap_reason
    r.validate()


def test_default_models_llm_subset_first():
    # The LLM subset must lead the corpus (harness shakedown ordering).
    assert buddy.DEFAULT_MODELS[:2] == ("tiny_llama", "small_llama")
    assert set(buddy.DEFAULT_MODELS) == {
        "tiny_llama", "small_llama", "bitvla", "rdt2", "rdt", "openvla",
        "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla"}
