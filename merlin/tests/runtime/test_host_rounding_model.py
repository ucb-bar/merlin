"""The selected helper survives the ordinary model build recipe and final link."""

from pathlib import Path

import numpy as np
import pytest


def test_normal_model_build_links_selected_rounding_and_completes_recipe(tmp_path):
    from merlin.common.digest import sha256_file
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.impr_features import FUSE_AFTER_GENERALIZE_NAME
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe
    from merlin.runtime import host_rounding
    from merlin.runtime.backends import spike, spike_model

    if not spike.available() or not toolchain.m2m_python().is_file() or not toolchain.clang().is_file():
        pytest.skip("bare-metal and upstream toolchain unavailable")
    bundle, work = tmp_path / "bundle", tmp_path / "build"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("""module {
      func.func @forward(%values: tensor<8xf32>) -> tensor<8xf32> {
        %empty = tensor.empty() : tensor<8xf32>
        %result = linalg.generic {
          indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]
        } ins(%values : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %rounded = math.roundeven %value : f32
          linalg.yield %rounded : f32
        } -> tensor<8xf32>
        return %result : tensor<8xf32>
      }
    }""")
    (bundle / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    values = np.asarray([-0.0, 0.5, -0.5, 1.5, 2.5, -1.5, 8388607.5, -8388608.0], np.float32)
    np.savez(bundle / "inputs.npz", in0=values)
    built = spike_model.build(
        bundle,
        work,
        backend="scalar",
        arena_mb=1,
        host_roundeven=True,
        features=frozenset({FUSE_AFTER_GENERALIZE_NAME}),
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
    )
    recipe_path = work / "compilation_recipe.json"
    recipe = verify_completed_recipe(
        recipe_path, executable=Path(built["elf"]), expected_recipe_sha256=sha256_file(recipe_path)
    )
    assert recipe["status"] == "completed"
    assert recipe["preparation"]["host_roundeven_implementation"]["sha256"] == sha256_file(Path(host_rounding.__file__))
    assert any(Path(row["path"]).name == "host_roundeven.o" for row in recipe["commands"][-1]["inputs"])
    assert any("-DMERLIN_EXTERNAL_ROUNDEVEN_F32" in row["argv"] for row in recipe["commands"])
    fusion_pin = recipe["preparation"]["host_elementwise_fusion"]
    import json
    fusion = json.loads(Path(fusion_pin["path"]).read_bytes())
    assert fusion["effective_pipeline"] == built["index_lowering"]["effective_pipeline"]
    assert fusion["rewritten_work"] == "unmeasured"
    result = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"])
    assert result["metrics"]["memref_rank_mismatch"] == 0
    np.testing.assert_array_equal(result["outputs"].view(np.uint32), np.rint(values).view(np.uint32))
