"""Board-free tests for the ExecuTorch arm's EXPORT-PATH SELECTION (which model, which recipe).

No board, no ET venv, no torch: everything under test is the selection logic that decides what the
AOT export subprocess is asked to do, plus the two reconciliation rules that decide whether a model
reaches the exporter at all.

Why these three concerns and not the exporter itself:

  * ``capture_locations`` / ``loader_env`` — the export subprocess inherits the environment the
    model2MLIR loader needs to FIND its weights. A gemma-class model whose checkpoint sits in a
    non-default HF cache failed with ``OSError: gated repo … 401`` — reading as "unexportable" when
    the weights were on disk the whole time. The rule must keep the per-host LOCATIONS and drop the
    smoke FIDELITY KNOBS in the same file (a 2-layer capture.toml setting against a 26-layer bundle
    would export a different model than the golden).
  * ``reconcile_input_arity`` — a stateful controller's forward takes recurrent state the capture
    bundle does not store, so the npz alone under-feeds it. Padding from the loader's own example is
    allowed; dropping captured tensors is not.
  * the int8 recipe — ``pt2e_qd8`` (dynamic per-row activation quant, per-channel weights) is the
    only recipe that mirrors merlin's own int8 datapath. ``weight_only`` const-folds to an fp32 GEMM.
    A cell must record which one ran, and the two must never be silently interchangeable.
"""
from __future__ import annotations

import importlib.util

import pytest

from merlin.baselines import executorch as et
from merlin.common.paths import merlin_dir


def _load_et_export_helper():
    """Import the export helper as a plain module (its torch imports are all inside ``main``)."""
    p = merlin_dir() / "python" / "merlin" / "baselines" / "_et_export.py"
    spec = importlib.util.spec_from_file_location("_et_export_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _workload(root, model: str, capture_toml: str):
    d = root / "workloads" / model
    d.mkdir(parents=True, exist_ok=True)
    (d / "capture.toml").write_text(capture_toml)
    return d


# ------------------------------------------------------------------- loader environment derivation

def test_capture_locations_keeps_paths_and_drops_fidelity_knobs(tmp_path, monkeypatch):
    """A capture.toml [env] mixes per-host LOCATIONS with SMOKE knobs; only the former may replay."""
    cache = tmp_path / "hf_cache"
    cache.mkdir()
    _workload(tmp_path, "toy", f'''
venv = "/nowhere/.venv"

[env]
HF_HOME = "{cache}"
M2M_TOY_LAYERS = "2"
M2M_TOY_SESSION = "e2e"
''')
    monkeypatch.setenv("MERLIN_MODEL2MLIR", str(tmp_path))

    env = et.capture_locations("toy")
    assert env == {"HF_HOME": str(cache)}, (
        "only the entry naming an existing directory is a location; a layer count and a session "
        "mode are fidelity knobs whose capture.toml value is the SMOKE setting")


def test_capture_locations_drops_a_path_that_is_not_present(tmp_path, monkeypatch):
    """A stale path in someone else's capture.toml must not be exported as a real location."""
    _workload(tmp_path, "toy", '[env]\nHF_HOME = "/definitely/not/here"\n')
    monkeypatch.setenv("MERLIN_MODEL2MLIR", str(tmp_path))
    assert et.capture_locations("toy") == {}


@pytest.mark.parametrize("body", ["", "venv = \"/x\"\n", "[env]\nX = 3\n", "!! not toml !!"])
def test_capture_locations_is_empty_when_there_is_nothing_to_read(tmp_path, monkeypatch, body):
    """No workload, no [env], a non-string value, or unparseable TOML -> no environment, no crash."""
    _workload(tmp_path, "toy", body)
    monkeypatch.setenv("MERLIN_MODEL2MLIR", str(tmp_path))
    assert et.capture_locations("toy") == {}
    assert et.capture_locations("no_such_model") == {}


def test_loader_env_lets_the_curated_full_fidelity_knob_win(tmp_path, monkeypatch):
    """capture.toml supplies locations; bundle.full_env supplies (and overrides) fidelity knobs."""
    cache = tmp_path / "hf_cache"
    cache.mkdir()
    other = tmp_path / "other_cache"
    other.mkdir()
    _workload(tmp_path, "toy", f'[env]\nHF_HOME = "{cache}"\nM2M_TOY_LAYERS = "2"\n')
    monkeypatch.setenv("MERLIN_MODEL2MLIR", str(tmp_path))
    monkeypatch.setattr(et._bundle, "full_env",
                        lambda m: {"M2M_TOY_LAYERS": "26", "HF_HOME": str(other)})

    assert et.loader_env("toy") == {"HF_HOME": str(other), "M2M_TOY_LAYERS": "26"}


def test_loader_env_of_a_real_workload_carries_no_fidelity_knob():
    """Guard the split on the REAL workloads: every value replayed must be an existing directory."""
    for model in et.DEFAULT_MODELS:
        for key, value in et.capture_locations(model).items():
            from pathlib import Path
            assert Path(value).is_dir(), f"{model}: {key}={value!r} is not a location"


# --------------------------------------------------------------------------- input-arity reconcile

def test_reconcile_input_arity_pads_from_the_loader_example():
    """A bundle that stores only the VARIED tensors is padded with the loader's initial state."""
    helper = _load_et_export_helper()
    captured, keys, note = helper.reconcile_input_arity(
        ("depth", "desvel", "quat"), ["in0", "in1", "in2"],
        ("d", "v", "q", _Shaped((3, 128)), _Shaped((3, 128))))

    assert captured == ("depth", "desvel", "quat", _Shaped((3, 128)), _Shaped((3, 128)))
    assert keys == ["in0", "in1", "in2", "loader_init0", "loader_init1"]
    assert "3 captured + 2 loader-initial" in note


def test_reconcile_input_arity_is_a_no_op_when_the_arities_match():
    helper = _load_et_export_helper()
    captured, keys, note = helper.reconcile_input_arity(("x",), ["in0"], ("x0",))
    assert (captured, keys, note) == (("x",), ["in0"], "")


def test_reconcile_input_arity_fails_closed_on_too_many_captured_tensors():
    """More captured tensors than the forward accepts is an ABI disagreement, never a silent drop."""
    helper = _load_et_export_helper()
    with pytest.raises(RuntimeError) as e:
        helper.reconcile_input_arity(("a", "b"), ["in0", "in1"], ("a0",))
    assert "input-arity mismatch" in str(e.value)
    assert "refusing to guess" in str(e.value)


class _Shaped:
    """Minimal stand-in for a tensor: ``reconcile_input_arity`` only reads ``.shape``."""

    def __init__(self, shape):
        self.shape = shape

    def __eq__(self, other):
        return isinstance(other, _Shaped) and self.shape == other.shape

    def __repr__(self):
        return f"_Shaped{self.shape}"


# ------------------------------------------------------------------------------- int8 recipe choice

def test_qd8_and_weight_only_are_not_interchangeable():
    """The two int8 recipes are different arithmetics; asking for both must raise, not pick one."""
    with pytest.raises(ValueError) as e:
        et.run_model("small_llama", "int8", qd8=True, int8_whole_model=True,
                     run_board=False, write=False)
    assert "two different int8 recipes" in str(e.value)


def test_a_missing_bundle_names_the_gap_and_never_runs_the_board(tmp_path, monkeypatch):
    """The recipe is recorded on the result even when the run stops before exporting."""
    monkeypatch.setattr(et.artifacts, "recaptures_dir", lambda: tmp_path)

    res = et.run_model("small_llama", "int8", qd8=True, run_board=False, write=False)

    assert res.quant_recipe == "pt2e_qd8", "the cell must say WHICH int8 arithmetic it asked for"
    assert not res.ran and res.gap_reason, "a gap must carry a reason (validate() enforces this)"
    res.validate()


# ------------------------------------------------------------------- the failure a gap reports

def test_failure_summary_leads_with_the_exception_not_a_stack_frame():
    """Every consumer truncates a long gap_reason from the FRONT, so the error must come first."""
    out = et._failure_summary(
        'Traceback (most recent call last):\n'
        '  File "/x/_program.py", line 1328, in to_edge_transform_and_lower\n'
        '    edge_manager = _gen(...)\n'
        '                   ^^^^^^^^\n'
        'RuntimeError: PT2E int8 quantization (qd8) failed: indices must be long\n')

    assert out.startswith("RuntimeError: PT2E int8 quantization (qd8) failed:")
    assert "_program.py" not in out and "^^^" not in out


def test_failure_summary_keeps_preceding_context_after_the_exception():
    out = et._failure_summary("first line\nsecond line\nException: the real problem\n")
    assert out.startswith("Exception: the real problem")
    assert "second line" in out


def test_failure_summary_never_returns_nothing():
    """An empty or all-scaffold output still has to say something a reader can act on."""
    assert et._failure_summary("") == "(no output)"
    assert et._failure_summary('Traceback (most recent call last):\n  File "/x", line 1\n')


# ---------------------------------------------------------------------------- the model roster

def test_the_default_roster_is_not_a_single_architecture():
    """The int8 bar is a MAJORITY of a DIVERSE set, so the roster must span more than one family."""
    for model in ("small_llama", "spectformer", "lstmnetvit", "gemma2_2b"):
        assert model in et.DEFAULT_MODELS, f"{model} is not first-class in the ExecuTorch arm"


def test_every_rostered_model_has_a_torch_loader():
    """ExecuTorch ingests torch: a model with no model2MLIR loader can never reach the exporter."""
    root = et._bundle.model2mlir_root()
    if not (root / "workloads").is_dir():
        pytest.skip(f"model2MLIR checkout absent at {root} (set MERLIN_M2M_DIR)")
    missing = [m for m in et.DEFAULT_MODELS
               if not et._bundle.resolve(m, "int8").torch_loader.is_file()]
    assert not missing, f"no model2MLIR torch loader for {missing}"
