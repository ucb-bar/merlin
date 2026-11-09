"""Application publication replay and real C output; synthetic device transport only."""

import json
import shutil
import subprocess
from hashlib import sha256
from types import SimpleNamespace

import pytest
from semantic_application_fixtures import plan_for

from merlin.compiler.build.application import (
    CompiledCRegion,
    compile_semantic_application,
    verify_semantic_application,
)
from merlin.compiler.lower.c.layout import emit_layout_c
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.search import SearchLimits


class TransportAdapter:
    """Synthetic native bytes test packaging; C performs an actual byte copy."""

    def __init__(self):
        self.planned_sources = []

    def plan(self, source):
        self.planned_sources.append(source.digest())
        return plan_for(source, ((0,),))

    def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
        path = segments / name
        path.mkdir()
        payload = b"transport-fixture-not-a-device-program"
        (path / "program.bin").write_bytes(payload)
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "engine": "merlin_native",
                    "artifact_kind": "program",
                    "request_digest": region.digest(),
                    "target_identity": region.target_identity,
                    "binary_sha256": sha256(payload).hexdigest(),
                }
            )
        )
        return self.export_region(region, alternative, name, segments, target_source)

    def export_region(self, region, alternative, name, segments, target_source):
        description, files = emit_layout_c(region, name)
        return CompiledCRegion(
            name,
            name + ".h",
            tuple(description["inputs"]),
            tuple(description["outputs"]),
            {key: value.encode() for key, value in files.items()},
            (),
            ("segments/" + name,),
            description,
        )


def test_application_replays_exact_artifacts_and_executes_generated_c(tmp_path):
    scalar = TensorType((4,), "f32", "raw-bit-pattern")
    source = KernelRequest(
        (SemanticNode("x", "input", (), scalar, effect="input"), SemanticNode("y", "reshape", ("x",), scalar)),
        ("y", "x"),
        ("external",) * 2,
        (("x", "external"),),
        "synthetic",
        lowering_policy="diagnostic",
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text("{}")
    snapshot = SimpleNamespace(root=snapshot)
    adapter = TransportAdapter()
    output = tmp_path / "application"
    args = dict(adapter=adapter, snapshot=snapshot, target_source=tmp_path)
    manifest = compile_semantic_application(
        source, destination=output, limits=SearchLimits(), name="model", capacity=1024, **args
    )
    # Compilation and independent replay each select once. Materialization
    # must use that choice rather than invoking the provider planner again.
    assert adapter.planned_sources == [source.digest()] * 2
    assert manifest["artifact_kind"] == "program_set"
    assert manifest["qualification_scope"] == "diagnostic_only"
    assert verify_semantic_application(source, artifact=output, **args)["execution"] == "not_run"
    assert adapter.planned_sources == [source.digest()] * 3
    # Re-sealing a different selected implementation cannot replace the
    # source-replayed choice, even when all buffer boundaries are identical.
    plan_path = output / "selected-plan.json"
    selected_bytes = plan_path.read_bytes()
    selected = json.loads(selected_bytes)
    selected["selected"][0]["implementation"] = "different-implementation"
    plan_path.write_text(json.dumps(selected))
    manifest["stage_artifacts"]["selected-plan.json"] = sha256(plan_path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="selected plan differs from source replay"):
        verify_semantic_application(source, artifact=output, **args)
    plan_path.write_bytes(selected_bytes)
    manifest["stage_artifacts"]["selected-plan.json"] = sha256(selected_bytes).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    # Actually compile/run the generated C; this does not execute the transport fixture bytes.
    compiler = shutil.which("cc")
    assert compiler, "this conformance test requires a C compiler"
    root = output / "handoff"
    main = root / "main.c"
    main.write_text("""#include "model_combined.h"
#include <assert.h>
#include <stdint.h>
#include <string.h>
int main(void) {
  uint32_t x[]={0,0x80000000u,0x7fc01234u,0x7f800000u}, y[4], z[4];
  const float *inputs[]={(const float *)x}; float *outputs[]={(float *)y,(float *)z};
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES];
  model_driver_state state={0}; void *items[]={0}; model_driver_contexts contexts={items,1};
  assert(model_invoke(&state,workspace,sizeof workspace,inputs,outputs,&contexts)==0);
  assert(!memcmp(x,y,sizeof x) && !memcmp(x,z,sizeof x)); return 0;
}""")
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-Wall",
            "-Werror",
            str(root / "model_combined.c"),
            str(main),
            "-o",
            str(tmp_path / "model"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(tmp_path / "model")], check=True)
    # Files introduced by the consumer are outside the immutable compiler artifact.
    main.unlink()
    cpath = output / "c-package/combined/model_combined.c"
    cpath.write_bytes(cpath.read_bytes() + b"\n/* changed */\n")
    with pytest.raises(ValueError, match="changed|differs"):
        verify_semantic_application(source, artifact=output, **args)
    # Resealing a changed file cannot defeat independent source-to-C replay.
    manifest["stage_artifacts"][cpath.relative_to(output).as_posix()] = sha256(cpath.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="replayed source"):
        verify_semantic_application(source, artifact=output, **args)
