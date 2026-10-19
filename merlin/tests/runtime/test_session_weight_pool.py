"""Different program-local tables retain their own immutable byte spans."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_session_projection import fixture

from merlin.llvmlower import session_bundle
from merlin.llvmlower.session_weight_pool import materialize_program_weight_pool, plan_program_weight_pool


def test_distinct_captured_weight_tables_reach_the_correct_compiled_stage(tmp_path):
    root, _ = fixture(tmp_path)
    weights = root / "stages/decode/capture/weights.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(original[:-20] + np.array([11, 13, 15, 17, 19], np.float32).tobytes())
    generated = tmp_path / "generated"
    manifest = session_bundle.generate(root, generated, materialize_weights=False, pack_program_weights=True)
    session = session_bundle.load(root)
    with pytest.raises(ValueError, match="requires its selected program weight pool"):
        session_bundle.materialize_shared_weights(session, manifest, tmp_path / "wrong_route")
    result = materialize_program_weight_pool(session, manifest, tmp_path / "pool")
    raw = Path(result["path"]).read_bytes()
    spans = result["program_weight_pool"]["programs"]
    assert spans[0]["sha256"] != spans[1]["sha256"]
    assert spans[1]["offset"] >= spans[0]["bytes"]
    assert spans[1]["offset"] % result["program_weight_pool"]["alignment"] == 0
    for span in spans:
        assert hashlib.sha256(raw[span["offset"]:span["offset"] + span["bytes"]]).hexdigest() == span["sha256"]
    assert raw[spans[0]["bytes"]:spans[1]["offset"]] == bytes(spans[1]["offset"] - spans[0]["bytes"])
    if not shutil.which("cc"):
        pytest.skip("host C compiler unavailable for generated caller execution")
    # Compile the actual generated dispatch with only the selected run paths
    # live. Each entry observes its own entire blob, not merely its first word.
    values = ",".join(map(str, raw))
    source = tmp_path / "check.c"
    checks = []
    for index, span in enumerate(spans):
        segment = ",".join(map(str, raw[span["offset"]:span["offset"] + span["bytes"]]))
        checks.append(f'''int merlin_stage_{index}_run(const void *w,long s,int v) {{
          static const unsigned char expected[]={{{segment}}};
          return w != pool+{span['offset']} || memcmp(w,expected,sizeof(expected)) || s!=7 || v!=3;
        }}''')
    source.write_text(f'''#include <string.h>
      #include "merlin_session.h"
      static const unsigned char pool[]={{{values}}};
      {chr(10).join(checks)}
      int main(void) {{
        return merlin_session_run_step(0,pool,7,3)
          || merlin_session_run_step(1,pool,7,3)
          || merlin_session_run_step(2,pool,7,3)!=-1;
      }}''')
    binary = tmp_path / "check"
    subprocess.run(["cc", "-O2", "-ffunction-sections", "-Wl,--gc-sections", "-I", str(generated),
                    str(generated / "merlin_session.c"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_equal_projected_blobs_share_storage_and_recheck_every_projection(tmp_path):
    root, projections = fixture(tmp_path)
    manifest = session_bundle.generate(root, tmp_path / "generated", materialize_weights=False,
                                       entry_projections=projections, pack_program_weights=True)
    session = session_bundle.load(root)
    result = materialize_program_weight_pool(session, manifest, tmp_path / "pool", entry_projections=projections)
    assert result["bytes"] == 20
    assert [r["offset"] for r in result["program_weight_pool"]["programs"]] == [0, 0]
    source = projections["decode"].source
    source.write_text(source.read_text().replace("arith.addf", "arith.mulf"))
    with pytest.raises(ValueError, match="source or proof changed"):
        materialize_program_weight_pool(session, manifest, tmp_path / "refused", entry_projections=projections)
    assert not (tmp_path / "refused/weights.bin").exists()


@pytest.mark.parametrize("mutation", ["offset", "capture", "storage", "missing_projection"])
def test_changed_plan_capture_or_selected_storage_refuses(tmp_path, mutation):
    root, projections = fixture(tmp_path)
    manifest = session_bundle.generate(root, tmp_path / "generated", materialize_weights=True,
                                       entry_projections=projections, pack_program_weights=True)
    sources = {row["name"]: Path(row["generated"]) / "weights.bin" for row in manifest["programs"]}
    if mutation == "offset":
        manifest["program_weight_pool"]["programs"][1]["offset"] += 1
    elif mutation == "capture":
        path = root / "stages/decode/capture/weights.safetensors"
        path.write_bytes(path.read_bytes()[:-4] + np.array([100], np.float32).tobytes())
    elif mutation == "storage":
        sources["decode"].write_bytes(bytes(20))
    else:
        projections.pop("decode")
    with pytest.raises(ValueError):
        materialize_program_weight_pool(session_bundle.load(root), manifest, tmp_path / "pool",
                                        entry_projections=projections, weights_sources=sources)
    assert not (tmp_path / "pool/weights.bin").exists()


def test_zero_length_and_repeated_blobs_have_no_extra_storage():
    values = [b"", b"abc", b"different", b"abc"]
    rows = [{"name": str(i), "weights_bytes": len(value), "weights_sha256": hashlib.sha256(value).hexdigest()}
            for i, value in enumerate(values)]
    plan = plan_program_weight_pool(rows)
    assert plan["programs"][0]["offset"] == 0
    assert plan["programs"][1]["offset"] == plan["programs"][3]["offset"] == 0
    assert plan["bytes"] == plan["alignment"] + len(values[2])
    assert json.loads(json.dumps(plan)) == plan
