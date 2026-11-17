"""The optional public source cases agree with the selected model's arithmetic.

This exercises numerical operations, not AtlasTile execution or physical timing.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("atlas")


def test_public_floating_cases_match_selected_model_numerics(tmp_path):
    if os.environ.get("MERLIN_TEST_MODEL_NUMERICS") != "1":
        pytest.skip("explicitly select the model environment for numerical comparison")
    torch = pytest.importorskip("torch")
    pytest.importorskip("npu_model")
    from merlin_experiments.phase0.floating_suite import materialize
    from npu_model.configs.numerics import RtlNumerics

    from merlin.targetgen.golden_store import load_golden

    selected = yaml.safe_load((repo_root() / "examples/atlas/phase0/floating-tests.yaml").read_bytes())
    root = tmp_path / "public-sources"
    manifest = materialize(selected, root)
    dtypes = {"f8E4M3FN": torch.float8_e4m3fn, "bf16": torch.bfloat16}
    for member in manifest["members"]:
        directory = root / member["path"]
        capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        golden = load_golden(directory)
        env = {
            row["name"]: torch.tensor(
                golden["oracle_provenance"]["inputs"][row["name"]]["decoded"], dtype=dtypes[row["dtype"]]
            ).reshape(row["shape"])
            for row in capsule["inputs"]
        }
        for node in capsule["component_program"]["nodes"]:
            args = [env[name] for name in node["actual_inputs"]]
            op = node["op"]
            if op == "matmul":
                result = RtlNumerics.systolic_matmul(*args, torch.zeros(node["shape"], dtype=torch.bfloat16))
            elif op in {"add", "update"}:
                result = RtlNumerics.add(*args)
            elif op == "relu":
                result = RtlNumerics.unary("relu", args[0])
            elif op == "convert":
                # Scale encoding is selected target data; these source cases use unit scaling.
                result = (
                    RtlNumerics.to_fp8(args[0], 127)
                    if node["dtype"] == "f8E4M3FN"
                    else RtlNumerics.from_fp8(args[0], 127)
                )
            elif op == "transpose":
                result = args[0].T.contiguous()
            else:
                result = args[0].clone()
            env[node["name"]] = result
        for row in capsule["component_program"]["outputs"]:
            expected = np.asarray(golden["outputs"][row["name"]], dtype=np.float32)
            actual = env[row["actual_value"]].float().numpy()
            assert np.array_equal(actual, expected), (Path(member["path"]).name, row["name"])
