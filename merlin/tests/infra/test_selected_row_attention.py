"""Complete neutral attention and carried cache around selected row scheduling.

Only the row pass is optimized. Ordered host contractions/cache updates remain
identical in both arms; this does not qualify target attention instructions.
"""

import ctypes
import json
import math
import shutil
import subprocess
import time

import numpy as np
from test_row_softmax import native_program
from test_selected_row_contracts import inputs, schedule

from merlin.compile.scheduling.row_softmax import CausalRowMask


def attention_source(result, opts, *, bad_reset=False):
    source = native_program(result, opts, bad_reset=bad_reset)
    # A neutral, explicitly ordered host producer/consumer and cache owner.
    # No target instruction or hardware cost is inferred from this fixture.
    return source + r'''
void attend(double *q,double *kn,double *vn,double *keys,double *values,int64_t *position,
            double *output,double *probabilities) {
 int64_t p=*position;
 if(p<0 || p>=17) abort();
 for(int h=0;h<2;h++) {
  for(int k=0;k<3;k++) keys[(h*17+p)*3+k]=kn[h*3+k];
  for(int d=0;d<5;d++) values[(h*17+p)*5+d]=vn[h*5+d];
 }
 double scores[2*17];
 for(int h=0;h<2;h++) for(int c=0;c<17;c++) {
  double total=0;
  for(int k=0;k<3;k++) total=total+q[h*3+k]*keys[(h*17+c)*3+k];
  scores[h*17+c]=total;
 }
 invoke(scores,probabilities);
 for(int h=0;h<2;h++) for(int d=0;d<5;d++) {
  double total=0;
  for(int c=0;c<17;c++) total=total+probabilities[h*17+c]*values[(h*17+c)*5+d];
  output[h*5+d]=total;
 }
 *position=p+1;
}
'''


def build(tmp_path, source, name):
    from merlin.llvmlower import toolchain
    path, library_path = tmp_path / (name + ".c"), tmp_path / (name + ".so")
    path.write_text(source)
    start = time.perf_counter()
    subprocess.run([str(toolchain.clang()), "-O2", "-ffp-contract=off", "-shared", "-fPIC",
                    str(path), "-lm", "-o", str(library_path)], check=True, capture_output=True)
    compile_seconds = time.perf_counter() - start
    library = ctypes.CDLL(str(library_path))
    library.attend.argtypes = [ctypes.c_void_p] * 8
    for field in ("executed_commands", "requested_reads", "requested_writes"):
        getattr(library, field).restype = ctypes.c_uint64
    metrics = dict(native_compile_seconds=compile_seconds, native_library_bytes=library_path.stat().st_size,
                   c_source_bytes=len(source.encode()))
    if shutil.which("size"):
        sections = subprocess.run(["size", "-A", "-d", str(library_path)], check=True,
                                  capture_output=True, text=True).stdout.splitlines()
        metrics["native_text_bytes"] = sum(int(line.split()[1]) for line in sections
                                           if line.split() and line.split()[0] == ".text")
    return library, metrics


def reference(q, kn, vn, keys, values, position):
    keys[:, position] = kn
    values[:, position] = vn
    probabilities, outputs = [], []
    for h in range(2):
        scores = []
        for c in range(17):
            score = 0.0
            for k in range(3):
                score += float(q[h, k]) * float(keys[h, c, k])
            scores.append(score if c <= position else -math.inf)
        exps = [math.exp(score - max(scores)) for score in scores]
        total = exps[0]
        for value in exps[1:]:
            total += value
        probs = [value / total for value in exps]
        probabilities.append(probs)
        row = []
        for d in range(5):
            total = 0.0
            for c in range(17):
                total += probs[c] * float(values[h, c, d])
            row.append(total)
        outputs.append(row)
    return np.array(probabilities), np.array(outputs)


def guarded(shape):
    storage = np.zeros(math.prod(shape) + 4, dtype=np.float64)
    storage[:2] = storage[-2:] = -491
    return storage, storage[2:-2].reshape(shape)


def test_selected_rows_preserve_complete_attention_cache_and_positions(tmp_path, record_property):
    state = []
    for _arm in range(2):
        key_guard, keys = guarded((2, 17, 3))
        value_guard, values = guarded((2, 17, 5))
        positions = np.array([-71, 0, -83], dtype=np.int64)
        state.append((key_guard, keys, value_guard, values, positions))
    reference_keys, reference_values = np.zeros((2, 17, 3)), np.zeros((2, 17, 5))
    carried = np.zeros((2, 5))
    rng = np.random.default_rng(991)
    records = []
    for position in range(3):
        directory = tmp_path / str(position)
        directory.mkdir()
        model = inputs(directory, dtype="f64", separate=True, columns=17, rows=2,
                       mask=CausalRowMask(1, position))
        q = rng.normal(size=(2, 3)) + carried[:, :3] * 0.125
        kn, vn = rng.normal(size=(2, 3)), rng.normal(size=(2, 5))
        unchanged = [array.copy() for array in (q, kn, vn)]
        expected_probs, expected = reference(q, kn, vn, reference_keys, reference_values, position)
        source_versions = None
        for arm, chunk in enumerate((1, None)):
            start = time.perf_counter()
            result = schedule(model, chunk_elements=chunk)
            plan_seconds = time.perf_counter() - start
            versions = result.report["pipeline"]["library_sources_sha256"]
            if source_versions is None:
                source_versions = versions
            assert versions == source_versions
            library, metrics = build(directory, attention_source(result, model[1]), "arm" + str(arm))
            key_guard, keys, value_guard, values, positions = state[arm]
            out_guard, output = guarded((2, 5))
            prob_guard, probabilities = guarded((2, 17))
            start = time.perf_counter()
            library.attend(q.ctypes.data, kn.ctypes.data, vn.ctypes.data, keys.ctypes.data, values.ctypes.data,
                           positions[1:].ctypes.data, output.ctypes.data, probabilities.ctypes.data)
            wall = time.perf_counter() - start
            for actual, wanted in ((output, expected), (probabilities, expected_probs),
                                   (keys, reference_keys), (values, reference_values)):
                np.testing.assert_array_equal(actual.view("u8"), wanted.view("u8"))
            for actual, wanted in zip((q, kn, vn), unchanged):
                np.testing.assert_array_equal(actual, wanted)
            for guard in (key_guard, value_guard, out_guard, prob_guard):
                assert np.all(guard[:2] == -491) and np.all(guard[-2:] == -491)
            np.testing.assert_array_equal(positions, [-71, position + 1, -83])
            counts = dict(executed_commands=library.executed_commands(),
                          requested_read_bytes=library.requested_reads(),
                          requested_write_bytes=library.requested_writes())
            assert all(result.report[key] == value for key, value in counts.items())
            records.append(dict(position=position, arm="chunk_one" if arm == 0 else "default_chunk",
                                row_plan_seconds=plan_seconds, whole_graph_host_call_seconds=wall,
                                **metrics, **counts))
        if position == 1:
            # A qualified label and successful plan cannot authorize a machine
            # that resets a denominator lane at each chunk boundary.
            bad, _metrics = build(directory, attention_source(result, model[1], bad_reset=True), "bad_reset")
            wrong = np.zeros((2, 5))
            probs = np.zeros((2, 17))
            keys, values = reference_keys.copy(), reference_values.copy()
            pos = np.array([position], dtype=np.int64)
            bad.attend(q.ctypes.data, kn.ctypes.data, vn.ctypes.data, keys.ctypes.data, values.ctypes.data,
                       pos.ctypes.data, wrong.ctypes.data, probs.ctypes.data)
            assert not np.array_equal(wrong.view("u8"), expected.view("u8"))
        carried[:] = expected
    for baseline, optimized in zip(records[::2], records[1::2]):
        assert optimized["executed_commands"] < baseline["executed_commands"]
        assert optimized["requested_read_bytes"] < baseline["requested_read_bytes"]
    record_property("selected_row_attention_pair", json.dumps(dict(
        measurements=records, shared_library_sources=source_versions,
        selected_contracts=result.report["pipeline"]["selected_contracts"],
        host_work="identical ordered QK/PV and mutable KV/position updates in both arms",
        execution_model="native_host_only; no accelerator executed",
        observed_counters_scope="row primitive commands and requested payloads only",
        numerical_agreement="complete output, probabilities, KV and positions exact for three invocations",
        accelerator_timing="unmeasured", compile_peak_memory="unmeasured",
        scope="complete neutral attention graph; no real target/model or attention instruction qualification")))
