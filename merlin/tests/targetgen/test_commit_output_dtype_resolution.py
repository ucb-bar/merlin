"""A capsule's committed dtype must survive generation.

`_resolve_output_dtype` used to read only the epilogue: an entry's own `output_dtype` was discarded and
the commit got the accumulator width. Nothing failed at generation time — a well-formed capsule came
out, just not the one the entry described — so the defect was invisible until the capsule reached a
backend.

It surfaced on 2026-09-05. Commit `2f06e353` regenerated the three hand-authored gemmini pooling
capsules from their profile entries for the first time; every one of them turned from `output_dtype:
i8` into `i32`, and `SY_epilogue_maxpool` (synthesized, so it declares no dtype at all) had been that
way from the start. Gemmini's native max-pool runs in the store DMA at the input width, so
`_native_pool_spec` refuses anything but an i8 store — the four capsules became uncompilable and 16
tests went red across `test_gemmini_native_pooling` and `test_rtl_checks`.

Two resolution rules, and the pooling one is the load-bearing half: an explicit declaration wins, and a
`maxpool` epilogue otherwise resolves to the target's OPERAND dtype. Honoring the declaration alone
would have fixed only the three hand-authored entries — the synthesized one has no author to declare
anything, and the axis that builds it knows nothing about a store path.
"""
from __future__ import annotations

import pytest


def _binding(target: str = "gemmini"):
    """The real derived binding for a target, via the generator's own seam.

    Built the way `generate_target` builds it rather than hand-constructed, so the operand/accumulator
    widths these tests compare are the ones the corpus is actually generated from.
    """
    import importlib.util

    from merlin.common.paths import merlin_dir
    from merlin.targetgen import corpus_spec as CS

    gen_path = merlin_dir() / "contract" / "capsules" / "generate_corpus.py"
    if not gen_path.is_file():
        pytest.skip("the corpus generator is not in this checkout")
    spec = importlib.util.spec_from_file_location("_gen_corpus_for_test", gen_path)
    gen = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(gen)
        descriptor = gen._descriptor_for(target)
        gen._ensure_contract_on_path(descriptor)
        te = gen.load_target_experiment(descriptor)
        return CS.derive_binding(te, gen.load_profile(target).get("datapath", {}))
    except Exception as exc:                      # no descriptor / no manifest in this checkout
        pytest.skip(f"cannot derive a binding for {target!r}: {exc}")


def test_an_entry_declaration_wins_over_the_accumulator_width():
    from merlin.targetgen.corpus_spec import _resolve_output_dtype

    b = _binding()
    assert _resolve_output_dtype(b, [], {}) == b.accum_dtype, "no declaration: the accumulator width"
    assert _resolve_output_dtype(b, [], {"output_dtype": "i8"}) == "i8", (
        "an entry that declares its committed dtype had the declaration silently discarded")


def test_an_unknown_declared_dtype_fails_generation_rather_than_falling_back():
    """A typo must not quietly become the accumulator width — that is the original defect's shape."""
    from merlin.targetgen.corpus_spec import _resolve_output_dtype

    with pytest.raises(KeyError):
        _resolve_output_dtype(_binding(), [], {"output_dtype": "i9"})


def test_a_maxpool_epilogue_commits_at_the_operand_width_not_the_accumulator_width():
    """Derived from the target's own descriptor, so the synthesized pooling capsule gets it too."""
    from merlin.targetgen.corpus_spec import _resolve_output_dtype

    b = _binding()
    got = _resolve_output_dtype(b, ["maxpool"], {})
    assert got == b.operand_dtype, (
        f"a fused max-pool commits through the store path at the operand width, got {got!r}")
    assert got != b.accum_dtype, "this target would not have exposed the defect; pick another"


@pytest.mark.parametrize("name", [
    "GP0_matmul_maxpool_i8",
    "GP1_matmul_maxpool_tail_i8",
    "GP2_conv2d_maxpool_i8",
    "SY_epilogue_maxpool",
])
def test_the_shipped_pooling_capsules_declare_the_narrow_store(name):
    """The end state, asserted on the tracked bytes a backend actually reads."""
    from merlin.common.paths import merlin_dir

    path = merlin_dir() / "contract" / "capsules" / "layers" / name / "capsule.interface.mlir"
    if not path.is_file():
        pytest.skip(f"{name} is not in this checkout")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if "merlin_iface.commit" in ln or "merlin_iface.conv2d" in ln]
    assert lines, "no committing op found; this test would be vacuous"
    for line in lines:
        assert 'output_dtype = "i8"' in line, (
            f"{name} commits at a width the native pooling store cannot write: {line.strip()}")
