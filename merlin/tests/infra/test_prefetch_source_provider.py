"""Neutral unary provider for ordinary source-bound input-copy prefetch tests.

The fixed capability release precedes model intake. Its independently reviewed
byte interpreter permits one pending transfer, reads it at its encoded token
wait, and safely executes byte-disjoint synchronous operations meanwhile. This
is neutral software qualification, not accelerator support or a timing model.
"""

import hashlib
import json
from dataclasses import replace

import numpy as np
from test_pointwise_async_source import ASYNC_EXECUTOR, command
from test_pointwise_async_source import case as async_case
from test_pointwise_source_provider import Provider
from test_private_copy_elimination import SPACES

from merlin.compile.scheduling.pointwise_source import PointwiseBindings
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
from merlin.compile.scheduling.temporary_storage import TemporaryObject
from merlin.sched.ir import Kernel, TensorArg

QUALIFICATION = "reviewed-neutral-token-copy-integer-concurrent-envelope-v1"
WIDTH = 16
MAX_ELEMENTS = 48
WORD_FORMAT = "<BBBQQQ"

# Wait's sole selected precondition is a matching token. Its size field has no
# declared memory effect and must not impose an unreported extra restriction.
assert ASYNC_EXECUTOR.count("if(!pending || n!=pn || token!=pt) return 2;") == 1
EXECUTOR = (
    ASYNC_EXECUTOR.replace("if(!pending || n!=pn || token!=pt) return 2;", "if(!pending || token!=pt) return 2;")
    .replace("memset(counts,0,5*sizeof(*counts));", "memset(counts,0,7*sizeof(*counts));")
    .replace(
        "counts[0]++;", "counts[0]++; if(op==0 || op==2) counts[5]++; if(pending && (op==0 || op==1)) counts[6]++;"
    )
)


def case(directory, m, n, *, separate):
    """Map any admitted source shape using an unchanged original release."""
    if not 0 < m * n <= MAX_ELEMENTS:
        return None

    def declare(facts, _software, storage):
        facts.update(prefetch_inflight=1, prefetch_queue=1)
        storage["prefetch"] = dict(
            schema="merlin.async_prefetch.v1",
            max_in_flight={"fact": ["facts", "prefetch_inflight"]},
            queue_capacity={"fact": ["facts", "prefetch_queue"]},
            independence="memory_disjoint",
            pairs=[dict(launch_instruction="async_copy", crossed_instruction=name) for name in ("copy", "increment")],
        )

    # Freeze resources from this release, not from source m,n or a capture ID.
    options, _word = async_case(directory, m=1, n=47, width=WIDTH, separate=separate, mutate=declare)
    calls, objects = [], []
    for index, offset in enumerate(range(0, m * n, WIDTH)):
        size, address, bank = min(WIDTH, m * n - offset), 80 + 32 * index, index % 2
        view = bank if separate else 1
        objects.append(TemporaryObject("fragment_" + str(index), SPACES[bank], address, size))
        calls.extend(
            (
                command("async_copy", offset, address, size, 2, bank, token=index + 1),
                command("wait", 0, 0, 0, token=index + 1),
                command("increment", address, address, size, view, view),
                command("copy", address, offset, size, view, 3),
            )
        )
    options.update(
        kernel=Kernel(
            "input_prefetch",
            (TensorArg("input", (m, n), "i8", "read"), TensorArg("output", (m, n), "i8", "write")),
            tuple(calls),
        ),
        objects=tuple(objects),
        bindings=PointwiseBindings(
            (AddressWindow("input", 0, m * n),), AddressWindow("output", 0, m * n), ("little", "little"), True
        ),
    )
    return options


class PrefetchProvider(Provider):
    capability_shape = (1, 47)
    input_dtypes = ("i1",)
    observation_count = 10
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
""" + EXECUTOR.replace("memmove(", "neutral_move(").replace("memset(", "neutral_set(").replace(
        "int run(", "static __attribute__((noinline)) int run("
    )

    def case_options(self, directory, m, n):
        return case(directory, m, n, separate=self.separate)

    def selection(self, trace):
        selected = super().selection(trace)
        (self.work / "original_concurrency_qualification.json").write_text(
            json.dumps(
                dict(
                    identity=QUALIFICATION,
                    executor_source_sha256=hashlib.sha256(self.executor_source.encode()).hexdigest(),
                    facts_sha256=hashlib.sha256(selected.selected_contracts.facts_bytes).hexdigest(),
                    software_sha256=hashlib.sha256(selected.selected_contracts.software_bytes).hexdigest(),
                    envelope=dict(
                        max_pending=1,
                        token_bits=6,
                        private_capacity_bytes=192,
                        external_capacity_bytes=78,
                        permitted_launch="async_copy",
                        crossed=("copy", "increment"),
                        input_selector=2,
                        private_selectors=(0, 1),
                        output_selector=3,
                        waits="exact named token; size operand unused",
                        independence="distinct physical destination bank and byte-disjoint source/destination",
                    ),
                    scope="original reviewed neutral C operational model; "
                    "native numerical trials are additional evidence, "
                    "not exhaustive hardware qualification or a timing model",
                ),
                indent=2,
            )
        )
        return replace(selected, prefetch_qualification_identity=QUALIFICATION)

    def lower(self, source):
        if len(source.shape) != 2 or len(source.graph.input_bits) != 1 or not 0 < np.prod(source.shape) <= MAX_ELEMENTS:
            return None
        return super().lower(source)

    def schedule_budgets(self):
        return ScheduleBudgets(max_fusions=0, eliminate_copies=False, prefetch_distance=2 if self.enabled else 0)

    @staticmethod
    def qualification_reference(array):
        return ((array.astype(np.int64) + 129) % 256 - 128).astype("i1")

    @staticmethod
    def packed_bytes(shape):
        return 3 * 2 * int(np.prod(shape))

    def _wrapper(self, symbol, m, n, image, separate):
        encoded, count = ",".join(str(value) for value in image), m * n
        return f"""
void {symbol}(const descriptor *a,descriptor *out) {{
 static const uint8_t image[] __attribute__((used,retain,aligned(1),section(".merlin_pointwise.{symbol}")))=
 {{{encoded}}};
 uint8_t first[200],second[200],input[86],output[86];uint64_t counts[7];
 if(a->sizes[0]!={m}||a->sizes[1]!={n}||out->sizes[0]!={m}||out->sizes[1]!={n})__builtin_trap();
 for(int i=0;i<200;i++){{first[i]=99;second[i]=119;}}
 for(int i=0;i<86;i++){{input[i]=71;output[i]=91;}}
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++){{
  input[i*{n}+j]=((const uint8_t*)a->aligned)[a->offset+i*a->strides[0]+j*a->strides[1]];
  totals[3]++;totals[4]++;
 }}
 if(run(image,sizeof(image),8,0,{int(separate)},0,first,second,input,output,192,78,counts))__builtin_trap();
 for(int i=0;i<8;i++)if(first[i]!=99)__builtin_trap();
 for(int i=0;i<16;i++)if(second[i]!=119)__builtin_trap();
 for(int i=192;i<200;i++)if(first[i]!=99||second[i]!=119)__builtin_trap();
 for(int i={count};i<86;i++)if(input[i]!=71||output[i]!=91)__builtin_trap();
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++){{
  if(input[i*{n}+j]!=((const uint8_t*)a->aligned)[a->offset+i*a->strides[0]+j*a->strides[1]])__builtin_trap();
  ((uint8_t*)out->aligned)[out->offset+i*out->strides[0]+j*out->strides[1]]=output[i*{n}+j];
  totals[3]++;totals[4]++;
 }}
 for(int i=0;i<3;i++)totals[i]+=counts[i];totals[5]++;
 for(int i=0;i<2;i++)if(counts[3+i]>totals[6+i])totals[6+i]=counts[3+i];
 totals[8]+=counts[5];totals[9]+=counts[6];pointwise_log();
}}
"""


def test_release_resources_do_not_depend_on_mapped_source_shape(tmp_path):
    snapshots = []
    for m, n in ((1, 1), (1, 47), (2, 23), (1, 48)):
        directory = tmp_path / f"{m}_{n}"
        directory.mkdir()
        options = case(directory, m, n, separate=True)
        selected = options["selected_contracts"]
        snapshots.append((selected.facts_bytes, selected.software_bytes, options["limits"]))
        assert options["kernel"].args[0].shape == (m, n)
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    assert case(tmp_path / "unsupported", 1, 49, separate=True) is None


def test_neutral_provider_executes_actual_prefetched_words(tmp_path):
    from test_pointwise_source import source_program
    from test_spilled_pointwise_source import source_text

    from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract
    from merlin.llvmlower.source_pointwise import SourcePointwiseBinding, discover_pointwise

    observations = []
    for separate in (False, True):
        arms = []
        for enabled in (False, True):
            work = tmp_path / f"{separate}_{enabled}"
            data = np.arange(47, dtype=np.int64).reshape(1, 47) * 17 + 251
            trace = []
            for _ in range(3):
                current = ((data + 128) % 256 - 128).astype("i1")
                trace.append((0, current))
                data = current.astype(np.int64) + 1
            provider = PrefetchProvider(work, enabled=enabled, separate=separate)
            selection = provider.selection(trace)
            module, _ = source_program(source_text())
            regions, declined = discover_pointwise(module, selection, selection.selected_contracts.revalidate)
            assert len(regions) == 1 and not declined
            writer = FreshTensorWriterContract("neutral_source", 1, (1,), "neutral_prefetch", 32)
            provider.prepare((SourcePointwiseBinding(regions[0].compilation, writer),), work)
            native = json.loads((work / "native_qualification.json").read_text())
            assert native["counters"][:6] == [36, 423, 423, 282, 282, 3]
            assert native["counters"][8:] == [18, 12 if separate and enabled else 0]
            actual = regions[0].compilation
            assert len(actual.images()[0]) == 12 * 27
            arms.append(
                dict(
                    enabled=enabled,
                    counters=native["counters"],
                    provider_source_sha256=native["source_sha256"],
                    facts_sha256=hashlib.sha256(selection.selected_contracts.facts_bytes).hexdigest(),
                    software_sha256=hashlib.sha256(selection.selected_contracts.software_bytes).hexdigest(),
                    library_sources=dict(actual.library_sources),
                    program_sha256=provider.records[0]["program_sha256"],
                    report=actual.compiled.report["storage"]["prefetch"],
                )
            )
        assert (arms[0]["program_sha256"] != arms[1]["program_sha256"]) is separate
        observations.append(dict(separate=separate, arms=arms))
    (tmp_path / "provider_prefetch.json").write_text(json.dumps(observations, indent=2))
