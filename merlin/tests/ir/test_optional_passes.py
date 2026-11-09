"""The optional lowering passes are listed in one registry, and selecting one flips exactly its switch.

``merlin.llvmlower.optional_passes`` names each optional transform with what it changes, whether it is
exact and its default, and maps it to the switch that already existed (a lowering feature, a ``MERLIN_*``
variable, an integer-datapath pass, or a capture-variant key). What this pins:

* every entry names a switch that is real: a registered lowering feature, an environment variable the
  lowering reads, a pass of the integer datapath, or the capture key the capture reads;
* an empty selection leaves every switch at its default, so an unselected build is the old build;
* ``--pass`` / ``--no-pass`` / the builder's ``lowering_passes`` reach the switch, and are refused for an
  unknown pass, a contradictory pair, or a capture-time pass at compile time.
"""

from __future__ import annotations

import json
import os

import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower import fusion_guard
from merlin.llvmlower import optional_passes as OP


@pytest.fixture(autouse=True)
def _no_ambient_selection(monkeypatch):
    monkeypatch.delenv(OP.ENV, raising=False)


def test_every_entry_is_complete_and_names_one_real_switch():
    from merlin.llvmlower import (
        impr_features,
        pipeline,  # noqa: F401 -- registers the runner-gated features
        quant_passes,
    )

    names = [e.name for e in OP.entries()]
    assert len(names) == len(set(names))
    llvmlower = repo_root() / "src" / "merlin" / "llvmlower"
    sources = "".join(p.read_text(encoding="utf-8") for p in llvmlower.glob("*.py"))
    for e in OP.entries():
        assert e.summary and e.changes and e.default
        assert e.exactness in (OP.EXACT, OP.NUMERICS_CHANGING), e
        assert e.stage in OP.STAGES, e
        bindings = [b for b in (e.feature, e.switch, e.quant_pass, e.capture_key) if b]
        assert len(bindings) == 1, e
        if e.feature:
            impr_features.get(e.feature)  # raises for an unregistered feature
        if e.switch:
            assert e.switch.startswith("MERLIN_") and f'"{e.switch}"' in sources, e
        if e.quant_pass:
            assert e.quant_pass in quant_passes.known(), e
        if e.capture_key:
            capture = (repo_root() / "src" / "merlin" / "targetgen" / "capsule_source.py").read_text(encoding="utf-8")
            assert f'"{e.capture_key}"' in capture, e


def test_selection_spelling_round_trips_and_refuses_what_it_cannot_mean():
    s = OP.Selection.parse("exact-math-inline, -fusion-guard,+sink-deallocs")
    assert s.enable == {"exact-math-inline", "sink-deallocs"} and s.disable == {"fusion-guard"}
    assert OP.Selection.parse(s.spell()) == s
    assert not OP.Selection.parse(None) and not OP.Selection.parse("")
    with pytest.raises(OP.PassSelectionError, match="unknown pass"):
        OP.Selection.parse("no-such-pass")
    with pytest.raises(OP.PassSelectionError, match="at once"):
        OP.Selection.of(["fusion-guard"], ["fusion-guard"])
    with pytest.raises(OP.PassSelectionError, match="capture variant key"):
        OP.Selection.of(["integer-nonlinear"])
    later = OP.Selection.of(disable=["sink-deallocs"])
    assert s.merged(later).disable == {"fusion-guard", "sink-deallocs"}
    assert "sink-deallocs" not in s.merged(later).enable


def test_an_empty_selection_changes_no_switch(monkeypatch):
    feats = frozenset({"lower_exact_math_inline"})
    assert OP.selected_features(feats) == feats
    assert OP.selected_features(None) is None
    assert OP.selected_quant_passes(["softmax_int", "gelu_int"]) == ["softmax_int", "gelu_int"]
    assert fusion_guard.enabled() is True
    monkeypatch.setenv("MERLIN_FUSION_GUARD", "0")
    assert fusion_guard.enabled() is False  # the variable still works on its own, for A/B


def test_a_selection_reaches_each_kind_of_switch(monkeypatch):
    from merlin.llvmlower import pipeline

    monkeypatch.setenv(OP.ENV, "roundeven-intrinsic,-exact-math-inline,-fusion-guard,sink-deallocs,-int-gelu")
    assert OP.selected_features({"lower_exact_math_inline"}) == {"lower_roundeven_to_intrinsic"}
    assert fusion_guard.enabled() is False
    assert pipeline._sink_deallocs() is True
    assert OP.selected_quant_passes(["softmax_int", "gelu_int"]) == ["softmax_int"]
    monkeypatch.setenv(OP.ENV, "fusion-guard")
    monkeypatch.setenv("MERLIN_FUSION_GUARD", "0")
    assert fusion_guard.enabled() is True  # an explicit selection wins over the A/B variable


def test_applied_scopes_the_selection_without_mutating_environment(monkeypatch):
    monkeypatch.setenv(OP.ENV, "fusion-guard")
    with OP.applied(OP.Selection.of(["sink-deallocs"])) as outer:
        assert os.environ[OP.ENV] == "fusion-guard"
        assert OP.active() == outer and outer.enable == {"sink-deallocs", "fusion-guard"}
        with OP.applied(OP.Selection.of(disable=["sink-deallocs", "fusion-guard"])) as inner:
            assert inner.disable == {"sink-deallocs", "fusion-guard"} and not inner.enable
            assert OP.active() == inner
        assert OP.active() == outer
    assert OP.active().enable == {"fusion-guard"}
    with pytest.raises(RuntimeError), OP.applied(OP.Selection.of(["sink-deallocs"])):
        raise RuntimeError("a failed build")
    assert os.environ[OP.ENV] == "fusion-guard" and OP.active().enable == {"fusion-guard"}


def test_merlin_compile_lists_and_selects_passes(capsys, monkeypatch):
    from merlin import compile_cli

    assert compile_cli.main(["--list-passes"]) == 0
    table = capsys.readouterr().out
    for e in OP.entries():
        assert e.name in table
    assert compile_cli.main(["--list-passes", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in listed] == [e.name for e in OP.entries()]
    assert all(row["exactness"] in (OP.EXACT, OP.NUMERICS_CHANGING) for row in listed)
    for bad in (["--pass", "no-such-pass"], ["--pass", "integer-nonlinear"]):
        with pytest.raises(SystemExit) as excinfo:
            compile_cli.main(["--workload", "w", *bad])
        assert excinfo.value.code == 2

    seen = {}

    def workflow(a, run):
        seen["passes"] = OP.active().spell()
        return {"status": "compiled"}

    monkeypatch.setattr(compile_cli, "_workflow", workflow)
    assert compile_cli.main(["--workload", "w", "--run", "none", "--pass", "sink-deallocs", "--json"]) == 0
    assert seen["passes"] == "sink-deallocs"
    assert json.loads(capsys.readouterr().out)["lowering_passes"] == "sink-deallocs"
    assert OP.ENV not in os.environ


def test_the_whole_model_builder_takes_lowering_passes_from_its_build_options(monkeypatch, tmp_path):
    from merlin.perf import whole_model_builder as B

    seen = {}

    def build(package_dir, **options):
        seen["passes"] = OP.active().spell()
        seen["options"] = options
        return {"notes": {}}

    monkeypatch.setattr(B, "_build", build)
    common = dict(target="t", out_dir=tmp_path, model_capsule="c", machine="m", header="h")
    record = B.build("pkg", lowering_passes=["sink-deallocs", "-fusion-guard"], **common)
    assert seen["passes"] == "sink-deallocs,-fusion-guard"
    assert "lowering_passes" not in seen["options"]
    assert record["notes"]["lowering_passes"] == "sink-deallocs,-fusion-guard"
    record = B.build("pkg", **common)
    assert seen["passes"] == "" and "lowering_passes" not in record["notes"]
    with pytest.raises(OP.PassSelectionError):
        B.build("pkg", lowering_passes=["no-such-pass"], **common)


def test_concurrent_builds_keep_opposite_choices_and_child_environments_private(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from merlin.llvmlower import codegen_env

    monkeypatch.setenv(OP.ENV, "exact-math-inline")
    rendezvous = Barrier(2)

    def build(enabled):
        selection = OP.Selection.parse("fusion-guard" if enabled else "-fusion-guard")
        with OP.applied(selection):
            rendezvous.wait(timeout=10)
            assert fusion_guard.enabled() is enabled
            child = OP.child_environment()
            assert OP.active(child).state("fusion-guard") is enabled
            assert OP.active(child).state("exact-math-inline") is True
            assert os.environ[OP.ENV] == "exact-math-inline"
            assert codegen_env.snapshot()[OP.ENV] == child[OP.ENV]
            assert codegen_env.cache_digest() == codegen_env.cache_digest(child)
            digest = codegen_env.cache_digest()
            rendezvous.wait(timeout=10)
        assert OP.active().state("fusion-guard") is None
        return digest

    with ThreadPoolExecutor(max_workers=2) as pool:
        on, off = pool.submit(build, True), pool.submit(build, False)
        assert on.result(timeout=20) != off.result(timeout=20)


def test_child_environment_preserves_explicit_defaults_and_empty_scope():
    base = {OP.ENV: "sink-deallocs", "OTHER_SETTING": "retained"}
    assert OP.child_environment(base) == base
    with OP.applied(OP.Selection()):
        assert OP.active(base).enable == {"sink-deallocs"}
        assert OP.child_environment(base) == {"OTHER_SETTING": "retained"}
    assert base[OP.ENV] == "sink-deallocs"


@pytest.mark.parametrize("route", ["ordinary", "retained", "callback"])
def test_native_runner_exports_selection_to_real_child(route, monkeypatch, tmp_path):
    import contextlib
    import sys

    from merlin.llvmlower import llvm_dialect_product as product
    from merlin.llvmlower import source_stage_transport as transport

    monkeypatch.setattr(product, "shared_compute_pool", lambda **_: contextlib.nullcontext())
    script = tmp_path / "child.py"
    script.write_text("import os\nprint(os.environ.get('MERLIN_PASSES', '<unset>'))\n")
    command = [sys.executable, str(script)]
    with OP.applied(OP.Selection.parse("sink-deallocs,-fusion-guard")):
        if route == "retained":
            result = product.SerialLLVMDialectRetention.prepare(tmp_path).run(
                command, source=script, runner=script, dependencies=(), timeout=10
            )
        elif route == "callback":
            # A failing native child must still see its invocation settings. It deliberately
            # exits before the handshake, without claiming a successful scalar callback.
            script.write_text(script.read_text() + "raise SystemExit(3)\n")
            result = transport.run_command(
                command,
                directory=tmp_path,
                callback=lambda *_: pytest.fail("unexpected callback"),
                max_source_bytes=1000,
                max_response_bytes=1000,
                timeout=10,
            )
        else:
            result = product.run_translation(
                command,
                timeout=10,
                audit=None,
                retention=None,
                source=script,
                runner=script,
                dependencies=(),
                scalar_stage=None,
                scalar_carrier=None,
            )
        assert result.stdout.strip() == "sink-deallocs,-fusion-guard"
    assert OP.ENV not in os.environ


def test_device_compiler_worker_carries_selection(tmp_path):
    from merlin.llvmlower.device_build_jobs import ArtifactCompiler

    artifact = tmp_path / "input.mlir"
    artifact.write_text("module {}")
    with ArtifactCompiler(2) as compiler, OP.applied(OP.Selection.parse("-fusion-guard")):
        future = compiler.submit(artifact, "input-identity", lambda: OP.active())
        assert future.result(timeout=10).state("fusion-guard") is False
