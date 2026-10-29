"""Ordinary recurrent model compilation repairs explicit device pressure.

A same-hardware, manually saved/reloaded baseline has the same commands as the
shared repair. Neutral native/Spike execution establishes mechanism scope only;
there is no accelerator latency or optimization speedup claim.
"""

import json

import numpy as np
import pytest
from test_pointwise_source_provider import Provider
from test_pointwise_source_whole_model import capture as original_capture
from test_pointwise_source_whole_model import model_source as original_model_source
from test_pointwise_source_whole_model import test_ordinary_recurrent_model_runs_source_bound_fusion as run_pair
from test_selected_storage_contracts import STORAGE_EXECUTOR

from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets


def increment(name, value, m, n):
    return f"""
 %{name}_init=tensor.empty():tensor<{m}x{n}xi8>
 %{name}=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}}
 ins(%{value}:tensor<{m}x{n}xi8>) outs(%{name}_init:tensor<{m}x{n}xi8>) {{
 ^bb0(%x:i8,%unused:i8): %{name}_one=arith.constant 1:i8
 %result=arith.addi %x,%{name}_one:i8
 linalg.yield %result:i8
 }}->tensor<{m}x{n}xi8>
 """


def model_source(m, k, n):
    from test_dense_source import contraction

    source = original_model_source(m, k, n)
    for before, after, columns, weight, result in (
        ("a", "adjusted_a", k, "w0", "first"),
        ("activation", "adjusted_activation", n, "w1", "second"),
    ):
        output_columns = n if result == "first" else k
        original = contraction((m, columns, output_columns), ("i8", "i8", "i32"), lhs=before, rhs=weight, output=result)
        replacement = increment(after, before, m, columns) + contraction(
            (m, columns, output_columns), ("i8", "i8", "i32"), lhs=after, rhs=weight, output=result
        )
        assert source.count(original) == 1
        source = source.replace(original, replacement)
    return source


def wrapped_increment(array):
    return ((array.astype(np.int64) + 129) % 256 - 128).astype("i1")


def reference(w0, w1, b0, b1, data, initial, position):
    state, history, calls = initial.copy(), [], []
    for _ in range(3):
        a = ((data.astype(np.int64) + state.astype(np.int64) + 128) % 256 - 128).astype("i1")
        calls.append((0, a.copy()))
        first = (wrapped_increment(a).astype(np.int64) @ w0.astype(np.int64)).astype("<i4")
        wide = first.astype(np.int64) + b0.astype(np.int64)
        activation = np.clip((wide + (1 << 31)) % (1 << 32) - (1 << 31), -128, 127).astype("i1")
        calls.append((1, activation.copy()))
        second = (wrapped_increment(activation).astype(np.int64) @ w1.astype(np.int64)).astype("<i4")
        wide = second.astype(np.int64) + b1.astype(np.int64)
        state = np.clip((wide + (1 << 31)) % (1 << 32) - (1 << 31), -128, 127).astype("i1")
        history.append(state.copy())
    return (np.stack(history), state, np.array(position + 3, "<i4"), data.copy(), w0.copy()), calls


def capture(directory, shape):
    original_capture(directory, shape)
    with np.load(directory / "inputs.npz") as inputs:
        data, initial = inputs["in0"], inputs["in1"]
    # Read this test's retained public tensors, rather than deriving an answer
    # from either compiler or the neutral instruction implementations.
    from safetensors.numpy import load_file

    weights = load_file(directory / "weights.safetensors")
    expected, calls = reference(**weights, data=data, initial=initial, position=7)
    np.save(directory / "golden.npy", expected[0])
    (directory / "model.mlir").write_text(model_source(*shape))
    return expected, calls


class SpillProvider(Provider):
    capability_shape = (1, 47)
    input_dtypes = ("i1",)
    expect_command_reduction = False
    executor_source = """
#include <stdint.h>
static void *neutral_move(void *dst,const void *src,unsigned long n) {
 unsigned char *d=dst;const unsigned char *s=src;
 if((uintptr_t)d<(uintptr_t)s)for(unsigned long i=0;i<n;i++)d[i]=s[i];
 else for(unsigned long i=n;i>0;i--)d[i-1]=s[i-1];return dst;
}
static void neutral_set(void *dst,int value,unsigned long n) {
 unsigned char *d=dst;for(unsigned long i=0;i<n;i++)d[i]=(unsigned char)value;
}
""" + STORAGE_EXECUTOR.replace("memmove(", "neutral_move(").replace("memset(", "neutral_set(").replace(
        "int run(", "static __attribute__((noinline)) int run("
    )
    faulty_spill = False

    def case_options(self, directory, m, n):
        from test_spilled_pointwise_source import case

        assert self.separate
        return case(directory, m=m, n=n, width=16, register_slots=1, manual=not self.enabled)

    def schedule_budgets(self):
        return ScheduleBudgets(max_fusions=0, eliminate_copies=False)

    def expected_instructions(self):
        return ["copy", "increment", "copy", "copy", "increment", "copy", "copy", "increment", "copy", "copy", "copy"]

    @staticmethod
    def qualification_reference(array):
        return wrapped_increment(array)

    @staticmethod
    def packed_bytes(shape):
        return 3 * 2 * shape[0] * (shape[1] + shape[2])

    def _wrapper(self, symbol, m, n, image, separate):
        encoded, count = ",".join(str(v) for v in image), m * n
        return f"""
void {symbol}(const descriptor *a,descriptor *out) {{
 static const uint8_t image[] __attribute__((used,retain,aligned(1),section(".merlin_pointwise.{symbol}")))=
 {{{encoded}}};
 uint8_t first[56],second[56],input[{count + 8}],output[{count + 8}];
 uint64_t counts[5];
 if(a->sizes[0]!={m}||a->sizes[1]!={n}||out->sizes[0]!={m}||out->sizes[1]!={n})__builtin_trap();
 for(int i=0;i<56;i++){{first[i]=167;second[i]=199;}}
 for(int i=0;i<{count + 8};i++){{input[i]=71;output[i]=91;}}
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++){{
  input[i*{n}+j]=((const uint8_t*)a->aligned)[a->offset+i*a->strides[0]+j*a->strides[1]];
  totals[3]++;totals[4]++;
 }}
 if(run(image,sizeof(image),8,0,1,{int(self.faulty_spill)},first,second,input,output,48,{count + 8},counts))
  __builtin_trap();
 if(counts[3]>16||counts[4]>48)__builtin_trap();
 for(int i=16;i<56;i++)if(first[i]!=167)__builtin_trap();
 for(int i=48;i<56;i++)if(second[i]!=199)__builtin_trap();
 for(int i={count};i<{count + 8};i++)if(input[i]!=71||output[i]!=91)__builtin_trap();
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++){{
  if(input[i*{n}+j]!=((const uint8_t*)a->aligned)[a->offset+i*a->strides[0]+j*a->strides[1]])__builtin_trap();
  ((uint8_t*)out->aligned)[out->offset+i*out->strides[0]+j*out->strides[1]]=output[i*{n}+j];
  totals[3]++;totals[4]++;
 }}
 for(int i=0;i<3;i++)totals[i]+=counts[i];totals[5]++;
 pointwise_log();
}}
"""


@pytest.mark.parametrize("shape", [(1, 47, 47), (2, 23, 17), (1, 47, 46), (2, 23, 18)])
def test_ordinary_recurrent_model_repairs_device_spills(tmp_path, record_property, shape):
    run_pair(
        tmp_path,
        record_property,
        shape,
        True,
        provider_class=SpillProvider,
        optimization="spills",
        capture_builder=capture,
    )
    observations = json.loads((tmp_path / "pair.json").read_text())
    assert all(row["native_qualification"]["counters"][0] == 66 for row in observations)
    assert observations[0]["encoded_program_bytes"] == observations[1]["encoded_program_bytes"] == 594
    for result in observations[1]["source_conformance"]:
        assert result["actual_commands"] == 11
    for arm in ("baseline_provider", "optimized_provider"):
        for path in (tmp_path / arm).glob("*/scheduling.json"):
            schedule = json.loads(path.read_text())["schedule"]
            assert schedule["repair_rounds"] == (1 if arm == "optimized_provider" else 0)
    record_property(
        "spill_model_scope", "same hardware manual save/reload versus shared pressure repair; no speedup claim"
    )


def test_wrong_native_spill_implementation_fails_independent_model_qualification(tmp_path, record_property):
    class FaultySpillProvider(SpillProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.faulty_spill = self.enabled

    with pytest.raises(AssertionError, match="Arrays are not equal"):
        run_pair(
            tmp_path,
            record_property,
            (1, 47, 46),
            True,
            provider_class=FaultySpillProvider,
            optimization="spills",
            capture_builder=capture,
        )
    assert (tmp_path / "baseline/model.elf").is_file()
    assert not (tmp_path / "optimized_provider/native_qualification.json").exists()
    assert not (tmp_path / "optimized/model.elf").exists()
