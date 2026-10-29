"""Stock scheduling stays explicit, composes with O3 and respects caller opt-outs."""

import pytest

from merlin.llvmlower.impr_features import apply_cflags
from merlin.llvmlower.stock_host_optimization import FEATURE as O3
from merlin.llvmlower.stock_host_post_ra_scheduling import FEATURE, edit_cflags


def test_explicit_composition_preserves_numeric_and_backend_options():
    flags = ["--target=ordinary", "-O2", "-ffp-contract=off", "-mllvm", "-verify-machineinstrs"]
    original = flags.copy()
    assert apply_cflags(flags, frozenset()) == flags
    result = apply_cflags(flags, frozenset({FEATURE, O3}))
    assert result == [flags[0], "-O3", *flags[2:], "-mllvm", "-enable-post-misched"]
    assert flags == original
    assert edit_cflags(result) == result


@pytest.mark.parametrize("option", [
    ["-mllvm", "-enable-post-misched"],
    ["-mllvm", "--enable-post-misched=true"],
    ["-mllvm=-enable-post-misched=1"],
])
def test_existing_explicit_enable_is_preserved(option):
    assert edit_cflags(option) == option


@pytest.mark.parametrize("option", [
    ["-mllvm", "-enable-post-misched=false"],
    ["-mllvm=-enable-post-misched=0"],
    ["-mllvm", "-enable-post-misched=unknown"],
    ["-enable-post-misched"],
])
def test_caller_opt_out_or_malformed_selection_refuses(option):
    with pytest.raises(ValueError, match="conflicting backend option"):
        edit_cflags(option)


def test_ordinary_model_plan_changes_only_model_object(monkeypatch):
    from merlin.runtime.backends import spike_model

    def observe(compiler, flags):
        return dict(schema="merlin.selected-index-lowering.v1", compiler_requested=str(compiler),
                    compiler_resolved="/compiler", compiler_sha256="a" * 64, cross_flags=flags,
                    index_bits=64, data_layout="e-p:64:64")

    monkeypatch.setattr(spike_model.target_data_layout, "observe_index_width", observe)
    control = spike_model.selected_model_compiler_plan(
        backend="scalar", cflags_override=None, vlen=None, features=frozenset({O3}))
    candidate = spike_model.selected_model_compiler_plan(
        backend="scalar", cflags_override=None, vlen=None, features=frozenset({O3, FEATURE}))
    assert candidate["gcc_cflags"] == control["gcc_cflags"]
    assert candidate["clang_cflags"] == control["clang_cflags"]
    assert candidate["model_cflags"] == edit_cflags(control["model_cflags"])
