"""Phase-2-only performance-scale captures: declared, disjoint, and kept out of Phase 1."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import form_perf as FP
from merlin_experiments.phase0.requirements import performance_scale_selection


def _files(tmp_path, *names):
    out = {}
    for name in names:
        path = tmp_path / name / "model.mlir"
        path.parent.mkdir()
        path.write_text(f"// {name}\n")
        out[name] = path
    return out


def test_no_declared_roster_selects_nothing_and_refuses_an_undeclared_capture(tmp_path):
    captures = _files(tmp_path, "cnn")
    assert performance_scale_selection({"applications": ["cnn"]}, captures, None) == {}
    extra = _files(tmp_path, "cnn_scale")
    with pytest.raises(ValueError, match="performance_applications"):
        performance_scale_selection({"applications": ["cnn"]}, captures, extra)


def test_the_declared_roster_must_be_selected_exactly_and_disjointly(tmp_path):
    captures = _files(tmp_path, "cnn")
    scale = _files(tmp_path, "cnn_scale", "decoder_scale")
    spec = {"applications": ["cnn"], "performance_applications": ["cnn_scale", "decoder_scale"]}
    assert sorted(performance_scale_selection(spec, captures, scale)) == ["cnn_scale", "decoder_scale"]
    with pytest.raises(ValueError, match="missing=\\['decoder_scale'\\]"):
        performance_scale_selection(spec, captures, {"cnn_scale": scale["cnn_scale"]})
    with pytest.raises(ValueError, match="also iteration applications"):
        performance_scale_selection(
            {"applications": ["cnn"], "performance_applications": ["cnn"]}, captures, {"cnn": captures["cnn"]}
        )
    with pytest.raises(ValueError, match="disjoint"):
        performance_scale_selection(
            {"applications": ["cnn"], "performance_applications": ["cnn_scale"]},
            captures,
            {"cnn_scale": captures["cnn"]},
        )


def _stub_members(monkeypatch, members_by_text):
    from merlin.common import mlir_query as mq
    from merlin.perf import derived_bound

    monkeypatch.setattr(
        derived_bound,
        "machine_from_facts",
        lambda *_a, **_k: SimpleNamespace(to_dict=lambda: {"array_rows": 16, "array_cols": 16}),
    )
    monkeypatch.setattr(mq, "parse", lambda path: open(path).read().strip())
    monkeypatch.setattr(
        FP,
        "application_members",
        lambda target, module, binding, **_k: {"members": members_by_text[module], "groups": 1, "unstated": {}},
    )


def _member(m, k, n, cycles):
    key = {"placement": "device", "op": "matmul", "activation_source": "intermediate"}
    return {
        "group": 1,
        "placement": "systolic",
        "key": key,
        "entry": {"op": "matmul", "M": m, "K": k, "N": n, "epilogue": []},
        "price": {"macs": m * k * n, "predicted_cycles": cycles},
    }


def test_scale_captures_join_only_the_form_scope_and_carry_their_role(monkeypatch, tmp_path):
    captures = _files(tmp_path, "cnn")
    scale = _files(tmp_path, "cnn_scale")
    _stub_members(
        monkeypatch,
        {"// cnn": [_member(32, 72, 8, 40)], "// cnn_scale": [_member(2048, 72, 16, 90000)]},
    )
    scope = FP.derive_form_scope(
        "synthetic",
        captures,
        SimpleNamespace(compare="exact_int"),
        iteration_roster=["cnn"],
        held_out=["heldout_net"],
        performance_scale=scale,
        performance_scale_roster=["cnn_scale"],
    )
    assert scope["workload_role"] == "iteration_and_performance_scale"
    assert scope["performance_scale_roster"] == ["cnn_scale"]
    assert scope["applications"]["cnn"]["workload_role"] == "iteration"
    assert scope["applications"]["cnn_scale"]["workload_role"] == "performance_scale"
    (row,) = scope["classes"]
    representative = row["members"][row["representative"]]
    assert representative["application"] == "cnn_scale", "the costliest group of the class is the scale one"
    assert {m["application"] for m in row["members"]} == {"cnn", "cnn_scale"}


def test_a_held_out_or_relabelled_scale_capture_is_refused(monkeypatch, tmp_path):
    captures = _files(tmp_path, "cnn")
    held = _files(tmp_path, "heldout_net_scaled")
    with pytest.raises(ValueError, match="held-out"):
        FP.derive_form_scope(
            "synthetic",
            captures,
            SimpleNamespace(),
            iteration_roster=["cnn"],
            held_out=["heldout_net"],
            performance_scale=held,
            performance_scale_roster=["heldout_net_scaled"],
        )
    with pytest.raises(ValueError, match="declared performance roster"):
        FP.derive_form_scope(
            "synthetic",
            captures,
            SimpleNamespace(),
            iteration_roster=["cnn"],
            held_out=[],
            performance_scale=held,
            performance_scale_roster=[],
        )
    with pytest.raises(ValueError, match="same capture"):
        FP.derive_form_scope(
            "synthetic",
            captures,
            SimpleNamespace(),
            iteration_roster=["cnn"],
            held_out=[],
            performance_scale={"cnn_scale": captures["cnn"]},
            performance_scale_roster=["cnn_scale"],
        )
