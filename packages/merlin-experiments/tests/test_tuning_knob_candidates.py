"""Policy switches already in an author's submission are named as tuning-knob candidates.

The fixture tree mirrors the archived attempt's shape: an option read with a string default passed as a
keyword into a function of another module that compares it with its literal alternatives (the readout
retirement policy that fenced after every tile), a parameter-only switch, a declared knob and a copy
of the shipped library that is not the author's code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin_experiments.phase1.feedback import knob_lint as KL


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _submission(root: Path) -> None:
    _write(
        root / "backend" / "dense.py",
        (
            "from .retire import emit_with_retirement\n"
            "def lower(kernel, attrs):\n"
            "    out = emit_with_retirement(kernel, policy=attrs.get('readout_retirement_policy', 'store_epochs'))\n"
            "    order = attrs.get('tile_order', 'm_first')\n"
            "    if order == 'n_first':\n"
            "        kernel = kernel[::-1]\n"
            "    name = attrs.get('label', 'kernel')\n"
            "    return out, name\n"
        ),
    )
    _write(
        root / "backend" / "retire.py",
        (
            "def emit_with_retirement(kernel, *, policy='product'):\n"
            "    if policy not in ('product', 'store_epochs'):\n"
            "        raise ValueError(policy)\n"
            "    return kernel\n"
            "def pack(rows, layout_mode='packed'):\n"
            "    return rows if layout_mode == 'plain' else sorted(rows)\n"
        ),
    )
    _write(
        root / "compiler_library" / "merlin" / "lib.py",
        "def f(x, cache_mode='on'):\n    return x if cache_mode == 'off' else x\n",
    )
    _write(
        root / "TUNING_KNOBS.json",
        json.dumps({"schema": "merlin.tuning_knobs.v1", "knobs": {"tile_order": {"values": ["m_first", "n_first"]}}}),
    )


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
    return h.hexdigest()


def test_knob_lint_finds_policy_switches_without_touching_the_submission(tmp_path):
    root = tmp_path / "submission"
    _submission(root)
    before = _digest(root)
    found = KL.scan(root)
    assert _digest(root) == before
    by_name = {entry["name"]: entry for entry in found["candidates"]}
    retire = by_name["readout_retirement_policy"]
    assert retire["kind"] == "option_read" and retire["default"] == "store_epochs"
    assert retire["observed_values"] == ["product", "store_epochs"]
    assert retire["locations"] == ["backend/dense.py:3"] and retire["compared_in"] == ["emit_with_retirement(policy)"]
    assert by_name["pack(layout_mode)"]["observed_values"] == ["packed", "plain"]
    assert "label" not in by_name  # one literal value is not a switch
    assert not any("cache_mode" in name for name in by_name)  # the shipped library copy is not the author's
    assert found["already_declared"] == ["tile_order"]
    assert "TUNING_KNOBS.json" in found["how_to_declare"] and found["advisory_only"] is True


def test_knob_candidates_attach_only_to_submission_responses(tmp_path):
    _submission(tmp_path / "submission")
    document = {"performance_brief": {"largest_headroom": []}, "selfcheck_submission": "submission"}
    KL.attach(document, tmp_path)
    assert document["performance_brief"]["tuning_knob_candidates"]["candidates"][0]["name"] == (
        "readout_retirement_policy"
    )
    variant = {"performance_brief": {}, "variant": {"official": False}}
    assert "tuning_knob_candidates" not in KL.attach(variant, tmp_path)["performance_brief"]
    assert KL.attach({"performance_brief": "broken"}, tmp_path) == {"performance_brief": "broken"}
