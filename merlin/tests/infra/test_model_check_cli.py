"""The model_check CLI keeps its contract: argument surface, report shape, status/exit mapping, agent view."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.common.schemas import validate
from merlin.model_check import cli
from merlin.model_check import report as R
from merlin.model_check.models import ModelsFileError, load


def _models_file() -> Path:
    return repo_root() / "examples" / "gemmini" / "numerics" / "models.yaml"


def test_models_file_declares_three_chained_models():
    models = load(_models_file())
    assert set(models.models) == {"resnet50", "smolvla", "tiny_llama"}
    assert [p.name for p in models.model("smolvla").programs] == ["prefix_encode", "flow_denoise", "action_decode"]
    assert models.model("tiny_llama").chain["kind"] == "causal"
    with pytest.raises(ModelsFileError):
        models.model("unknown")


@pytest.mark.parametrize(
    "argv",
    [
        ["check", "--submission", "S", "--model", "m", "--split", "visible", "--out", "o.json"],
        [
            "replay-kernel",
            "--submission",
            "S",
            "--model",
            "m",
            "--program",
            "p",
            "--all",
            "--split",
            "all",
            "--engine",
            "gsim",
            "--out",
            "o.json",
        ],
        ["census", "--submission", "S", "--model", "m", "--out", "o.json"],
        ["binding", "--out", "o.json"],
        ["bundle-verify", "--out", "o.json"],
    ],
)
def test_contract_argument_surface_parses(argv):
    args = cli.build_parser().parse_args(argv)
    assert args.command == argv[0]


def test_unknown_model_is_a_usage_error_with_a_report(tmp_path):
    sub = tmp_path / "submission"
    sub.mkdir()
    (sub / "manifest.yaml").write_text("x: 1\n")
    out = tmp_path / "r.json"
    code = cli.main(
        ["census", "--submission", str(sub), "--models", str(_models_file()), "--model", "nope", "--out", str(out)]
    )
    doc = json.loads(out.read_text())
    assert code == R.EXIT_USAGE
    assert doc["status"] == "error"
    assert validate(doc, R.SCHEMA_NAME) == []
    assert doc["submission"]["tree_sha256"]


def test_settle_requires_every_gate_and_no_partial():
    rep = R.new_report("check", model="m", split="visible", submission={})
    assert R.settle(rep)["status"] == "error"  # nothing judged
    rep["gates"] = {"build": {"status": "pass"}, "model_numerics": {"status": "pass", "partial": True}}
    assert R.settle(rep)["status"] == "fail"
    rep["gates"]["model_numerics"]["partial"] = False
    assert R.settle(rep) == rep and rep["status"] == "pass" and rep["exit_code"] == R.EXIT_PASS


def test_agent_view_drops_sealed_values():
    rep = R.new_report("check", model="m", split="all", submission={})
    rep["programs"] = [{"runs": [{"input": "v1", "split": "visible"}, {"input": "s1", "split": "sealed"}]}]
    rep["model_metrics"] = {"per_input": [{"input": "s1", "split": "sealed", "kl": 0.2}], "worst": {"kl": 0.2}}
    rep["failures"] = [{"kind": "gate", "split": "sealed", "input": "s1", "detail": "kl 0.2"}]
    view = R.agent_view(rep)
    assert [r["input"] for r in view["programs"][0]["runs"]] == ["v1"]
    assert view["model_metrics"]["per_input"] == []
    assert "input" not in view["failures"][0] and "detail" not in view["failures"][0]
    assert view["agent_view"] is True
