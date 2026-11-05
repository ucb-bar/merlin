"""Actual original47 projection backfill, using the frozen numerical engine."""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
MERLIN_ROOT = HERE.parents[2]
ATLAS_ROOT = Path(os.environ["ATLAS_ROOT"]).absolute()
ROOT = Path(os.environ["TINY_WORK_ROOT"]).absolute()
ATLAS_HELPERS = ATLAS_ROOT / "tools/tiny_full"
MERLIN_HELPERS = MERLIN_ROOT / "examples/atlas/comparisons"
MODEL_ROOT = Path(os.environ["ATLAS_NPU_MODEL_ROOT"]).absolute()
spec = importlib.util.spec_from_file_location("d1_v2_support", ATLAS_HELPERS / "d1-v2-support.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
w = s.load("d1-stream-worker")
old_closure = w.closure


def closure(worker):
    return dict(sorted({**old_closure(worker), **s.adapter_sources()}.items()))


w.closure = closure


def run(root, record_path, output, freeze):
    output, root = s.safe(output), s.safe(root)
    if not output.is_relative_to(ROOT / "d1-v2-stream-original") or output.is_relative_to(root):
        raise ValueError("output must be in D/d1-v2-stream-original outside capture")
    if output.exists():
        raise ValueError("fresh output required; audit an existing receipt instead")
    worker = w.verify_freeze(freeze)
    inputs = w.Inputs(root, record_path)
    prepared, _ = w.prepare(inputs, worker)
    before = w.source_state()
    executed = worker.model.execute(
        prepared["source"],
        prepared["initial"],
        prepared["output_offset"],
        prepared["output_bytes"],
        max_instructions=20_000_000,
        clip_exponent=inputs.record["output_scale"] if inputs.record["role"] == 5 else None,
    )
    execution = executed.metadata()
    clipping = execution["observed"].get("fp8_output")
    w.check_result(
        inputs,
        worker,
        prepared,
        executed.arena,
        executed.output,
        execution,
        clipping,
        b"".join(worker.model.CLIP_TILES),
    )
    after = w.source_state()
    if before != after or before != freeze:
        raise ValueError("actual original47 source bookends differ")
    inputs.verify()
    record = {key: inputs.record[key] for key in w.RECORD_KEYS if key in inputs.record}
    record.update(
        schema="atlas.d1.stream_original_execution.v1",
        model_commit=w.PIN,
        numerics="rtl",
        accelerated=False,
        native_dispatch_enabled=False,
        execution=execution,
        output_quantization=clipping,
        model_sources_before=before["model"],
        model_sources_after=after["model"],
        harness_sources_before=before["harness"],
        harness_sources_after=after["harness"],
        implementation_sources=[
            dict(path=p, sha256=d, bytes=s.safe(p).stat().st_size) for p, d in before["harness"].items()
        ],
        method="executed_original47_on_speculative_capture_inputs",
        speculative_capture_input_numerical_evidence=True,
        final_full_source_capture_replay_closure_required=True,
        overall_acceptance=False,
        source_capture_root=str(inputs.root),
        source_capture_record=str(inputs.path),
        source_capture_record_sha256=s.sha(inputs.raw),
        source_capture_record_stat=inputs.stat,
        capture_payload_sha256=inputs.pins,
        exact=True,
        full_arena_exact=True,
        completed_unix=time.time(),
    )
    w.save(output, record)
    return w.audit(root, record_path, output, freeze)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "record", "out", "freeze"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    freeze = json.loads(s.safe(args.freeze).read_bytes())
    result = (w.audit if args.audit_only else run)(args.capture, args.record, args.out, freeze)
    print(
        json.dumps(dict(status="PASS", record=args.out, instructions=result["execution"]["instructions"])), flush=True
    )
