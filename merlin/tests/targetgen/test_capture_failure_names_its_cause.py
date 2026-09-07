"""A failed capture must report what failed, not whatever its subprocess printed last.

`stderr[-1500:]` was the excerpt, and a python subprocess interleaves warnings with tracebacks -- so a
capture whose warnings arrive AFTER its traceback reported the warnings. Measured on the lstmnetvit
roster model: the failure was attributed to

    UserWarning: The tensor attributes self.net.lstm._flat_weights[0], ... are not part of a single
    contiguous chunk of memory

which is not an error at all. Three roster models fail on gemmini and this one's stated reason sent a
reader to a benign PyTorch contiguity notice instead of the actual fault.

The two failure conditions were also collapsed into one message. A non-zero return code means the
worker died; a missing meta.json after rc=0 means it exited cleanly and produced no capture. Those
license different next steps.
"""
from __future__ import annotations

from pathlib import Path

from merlin.targetgen.capsule_source import _stderr_cause


def test_the_last_traceback_wins_over_trailing_warnings():
    """The exact shape of the defect: a real error followed by a wall of warnings."""
    warnings = "UserWarning: flat_weights are not part of a single contiguous chunk\n" * 60
    err = ("Traceback (most recent call last):\n"
           '  File "loader.py", line 3\n'
           "ValueError: the actual fault\n")
    got = _stderr_cause(err + warnings)
    assert "last traceback" in got
    assert "ValueError: the actual fault" in got
    assert got.index("Traceback") < got.index("UserWarning") if "UserWarning" in got else True


def test_the_latest_traceback_wins_when_there_are_several():
    first = "Traceback (most recent call last):\nRuntimeError: an earlier retry\n"
    second = "Traceback (most recent call last):\nRuntimeError: the final failure\n"
    got = _stderr_cause(first + second)
    assert "the final failure" in got
    assert "an earlier retry" not in got


def test_with_no_traceback_the_tail_is_used_and_says_so():
    """A worker can fail without a python traceback, and the excerpt must not pretend otherwise."""
    got = _stderr_cause("ld: cannot find -lfoo\n" * 40)
    assert "no traceback found" in got
    assert "cannot find -lfoo" in got


def test_silence_is_reported_as_silence():
    """An empty excerpt reads as 'no reason given'; say that the worker wrote nothing."""
    assert "wrote nothing" in _stderr_cause("")
    assert "wrote nothing" in _stderr_cause("   \n  ")


def test_the_excerpt_is_bounded():
    got = _stderr_cause("Traceback (most recent call last):\n" + ("x" * 50_000), limit=500)
    assert len(got) < 1_000


def test_the_two_failure_conditions_are_distinguished_in_the_message():
    """A worker that died and one that produced nothing are different problems."""
    from merlin.common.paths import repo_root
    src = (repo_root() / "merlin" / "python" / "merlin" / "targetgen"
           / "capsule_source.py").read_text(encoding="utf-8")
    assert "worker exited non-zero" in src
    assert "wrote no meta.json" in src


def test_a_worker_verdict_is_reported_from_meta_not_from_stderr(tmp_path, monkeypatch):
    """rc=3 is a VERDICT, not a crash, and the verdict is in meta.json.

    The worker returns 3 for "ran fine, but the program is not clean" and writes meta.json with
    `opaque` and `opaque_detail` BEFORE doing so. The caller's non-zero-rc branch fired first and
    dumped the stderr tail -- for a torch export, a wall of warnings -- while the block just below
    already had the useful message.

    Measured on the lstmnetvit roster model at W8A8: the real answer is opaque=44 from un-inlined
    torchao choose_qparams_affine / quantize_affine calls, and what got reported was an LSTM
    `_flat_weights` contiguity notice. The same model captures CLEANLY at plain int8, so the scheme is
    the fact that matters and it is now named in the message.

    The fake worker writes meta.json into the directory it is HANDED, because `_run` substitutes a
    content-addressed cache slot for the caller's workdir -- pre-writing the file next to the caller's
    path tests nothing (it was this test's first shape, and it passed the old code).
    """
    import json
    import subprocess

    from merlin.targetgen import capsule_source as CS

    payload = {"ok": True, "opaque": 44, "scheme": "int8_dyn_act_int8_weight",
               "opaque_detail": {"torchao_quantize_affine_default_1": 6,
                                 "torchao_choose_qparams_affine_default_1": 6},
               "input_abi": []}

    class _Proc:
        returncode = 3
        stdout = '__M2M_CAPTURE__ {"ok": true, "opaque": 44}\n'
        stderr = "UserWarning: _flat_weights are not part of a single contiguous chunk\n" * 40

    def _fake_run(cmd, **_kw):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    src = CS.PytorchRefSource()
    monkeypatch.setattr(type(src), "available", lambda self: True)

    loader = tmp_path / "loader.py"
    loader.write_text("# loader\n", encoding="utf-8")
    try:
        src.capture_loader(loader, "i8", workdir=tmp_path / "cap",
                           scheme="int8_dyn_act_int8_weight")
    except CS.M2MUnavailable as exc:
        msg = str(exc)
    else:
        raise AssertionError("a non-clean program must be refused")

    assert "opaque=44" in msg, msg
    assert "int8_dyn_act_int8_weight" in msg, "the scheme distinguishes the clean run from this one"
    assert "torchao_quantize_affine_default_1" in msg, "name what to go and inline"
    assert "_flat_weights" not in msg, "the stderr warnings must not be the reported cause"


def test_a_semantically_normalizable_capture_reaches_the_shared_zero_opaque_gate(
        tmp_path, monkeypatch):
    """The real failure seam: the worker returns rc=3 for two TorchAO affine placeholders.

    Their semantics are recoverable from the module's own types and quantization provenance.  The
    shared capture boundary must apply that canonical normalization before judging cleanliness,
    preserve the raw census, and return the normalized artifact a backend will actually consume.
    """
    import json
    import subprocess

    from merlin.targetgen import capsule_source as CS

    raw = '''builtin.module attributes {prov.quantization = "int8_dyn_act_int8_weight"} {
  func.func private @torchao_choose_qparams_affine_default(tensor<1x32xf32>) -> tensor<1xf32>
  func.func private @torchao_quantize_affine_default(tensor<1x32xf32>, tensor<1xf32>) -> tensor<1x32xi8>
  func.func @forward(%x: tensor<1x32xf32>) -> tensor<1x32xi8> {
    %s = func.call @torchao_choose_qparams_affine_default(%x) : (tensor<1x32xf32>) -> tensor<1xf32>
    %q = func.call @torchao_quantize_affine_default(%x, %s) : (tensor<1x32xf32>, tensor<1xf32>) -> tensor<1x32xi8>
    func.return %q : tensor<1x32xi8>
  }
}'''
    payload = {
        "ok": True,
        "opaque": 2,
        "opaque_detail": {
            "torchao_choose_qparams_affine_default": 1,
            "torchao_quantize_affine_default": 1,
        },
        "scheme": "int8_dyn_act_int8_weight",
        "linalg_ops": 0,
        "input_abi": [{"shape": [1, 32], "dtype": "f32"}],
        "loader_provenance_status": "declared",
    }

    class _Proc:
        returncode = 3
        stdout = '__M2M_CAPTURE__ {"ok": true, "opaque": 2}\n'
        stderr = ""

    def _fake_run(cmd, **_kw):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "linalg.mlir").write_text(raw, encoding="utf-8")
        (out / "inputs.json").write_text(json.dumps([[[0.0] * 32]]), encoding="utf-8")
        (out / "golden.json").write_text(json.dumps([[0] * 32]), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    src = CS.PytorchRefSource()
    monkeypatch.setattr(type(src), "available", lambda self: True)
    monkeypatch.setattr(type(src), "_cache_slot", lambda *args, **kwargs: None)
    loader = tmp_path / "loader.py"
    loader.write_text("# loader\n", encoding="utf-8")

    art = src.capture_loader(loader, "i8", workdir=tmp_path / "cap",
                             scheme="int8_dyn_act_int8_weight")

    assert art.meta["opaque"] == 0
    receipt = art.meta["capture_normalization"]
    assert receipt["raw_opaque_detail"] == payload["opaque_detail"]
    assert receipt["remaining_opaque_detail"] == {}
    assert receipt["normalizers"] == [{
        "identity": "merlin.frontends.capture_normalization.torchao_affine/v1",
        "rewrites": 2,
    }]
    assert "func.call @torchao" not in art.linalg_mlir
    assert "linalg.generic" in art.linalg_mlir


def test_capture_normalization_refuses_a_stale_worker_census():
    """A normalization cannot launder capture metadata that disagrees with the actual program."""
    from merlin.frontends.capture_normalization import (
        CaptureNormalizationError,
        normalize_capture_mlir,
    )

    raw = '''builtin.module {
  func.func private @unknown(tensor<4xf32>) -> tensor<4xf32>
  func.func @forward(%x: tensor<4xf32>) -> tensor<4xf32> {
    %y = func.call @unknown(%x) : (tensor<4xf32>) -> tensor<4xf32>
    func.return %y : tensor<4xf32>
  }
}'''
    try:
        normalize_capture_mlir(raw, reported_opaque={"different": 1})
    except CaptureNormalizationError as exc:
        message = str(exc)
    else:
        raise AssertionError("a stale opaque census must be refused")
    assert "disagrees" in message and "unknown" in message


def test_a_worker_that_produced_no_diagnosis_still_reports_stderr(tmp_path, monkeypatch):
    """The stderr path must survive for a worker that died before writing meta.json."""
    import subprocess

    from merlin.targetgen import capsule_source as CS

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "Traceback (most recent call last):\nImportError: no module named lerobot\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    src = CS.PytorchRefSource()
    monkeypatch.setattr(type(src), "available", lambda self: True)
    loader = tmp_path / "loader.py"
    loader.write_text("# loader\n", encoding="utf-8")
    try:
        src.capture_loader(loader, "i8", workdir=tmp_path / "cap2")
    except CS.M2MUnavailable as exc:
        msg = str(exc)
    else:
        raise AssertionError("a worker that wrote no meta must be refused")
    assert "no module named lerobot" in msg
    assert "worker exited non-zero" in msg
