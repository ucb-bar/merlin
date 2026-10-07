"""Smoke tests for the design-pressure and dse CLIs."""

import yaml

from merlin.design_pressure import cli as dp_cli
from merlin.dse import cli as dse_cli

#: The two mined rules the design-pressure synthesis consults. The real file is a kernel-mining product
#: (``out/artifacts/kernel-index/policy_rules.yaml``) that a clean clone does not have, and this smoke
#: test is about the CLI writing its artifacts, not about the mined corpus, so it supplies the rules.
_POLICY_RULES = [
    {
        "policy": "packed_rhs_policy",
        "evidence": ["openblas_rvv_gemm", "xnnpack_rvv_gemm"],
        "when": {"rhs_reuse_count": ">= 2", "rhs_mutable": "false"},
        "actions": ["preserve_packed_rhs_layout", "hoist_pack", "consider_resident_packed_tensor"],
    },
    {
        "policy": "accumulator_commit_policy",
        "evidence": ["xnnpack_rvv_gemm"],
        "when": {"op": "gemm|matmul|conv", "has_epilogue": "true", "accumulator_live_across_epilogue": "true"},
        "actions": ["keep_accumulator_resident", "fuse_epilogue_before_commit", "single_commit_store"],
    },
]


def test_design_pressure_cli_writes_artifacts(tmp_path, monkeypatch):
    out_root = tmp_path / "out_root"
    rules = out_root / "artifacts" / "kernel-index" / "policy_rules.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text(yaml.safe_dump(_POLICY_RULES))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(out_root))
    out = tmp_path / "dp"
    rc = dp_cli.main(["--workload", "vla_action_chunk_decode", "--H", "8", "--out", str(out)])
    assert rc == 0
    assert (out / "design_pressure.json").is_file()
    assert (out / "candidate_contracts.yaml").is_file()


def test_dse_cli_no_experiment(tmp_path):
    rc = dse_cli.main(["--workload", "vla_action_chunk_decode", "--no-experiment", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "resident_packed_tensor" / "dse_result.yaml").is_file()
