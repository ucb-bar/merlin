"""Test-only byte-machine mapping and observed storage/copy counters.

The fixed local bank windows remain 192 bytes with eight guard bytes in both arms. Observed
address high-water is an access footprint, not allocated bytes or hardware RAM.
"""

import struct

from test_private_copy_elimination import SPACES, command
from test_selected_storage_contracts import inputs
from test_spilled_pointwise_model import SpillProvider
from test_spilled_pointwise_source import INCREMENT

from merlin.compile.scheduling.pointwise_source import PointwiseBindings
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
from merlin.compile.scheduling.temporary_storage import TemporaryObject
from merlin.sched.ir import Kernel, TensorArg


class StorageProvider(SpillProvider):
    capability_shape = (1, 1)
    observation_count = 9
    optimization = "storage"
    corrupt_address = False
    executor_source = SpillProvider.executor_source.replace(
        "neutral_set(counts,0,5*sizeof(*counts));", "neutral_set(counts,0,6*sizeof(*counts));"
    ).replace("counts[0]++;", "counts[0]++; if(op==0) counts[5]++;")

    def case_options(self, directory, m, n):
        def declare(facts, software, storage):
            facts["byte_bits"] = 8
            software["scheduling"]["fusion"] = dict(
                schema="merlin.physical_fusion_contract.v1",
                elementwise={
                    "increment": dict(
                        scalar_ir=INCREMENT.decode(),
                        elements=[{"argument": "size"}],
                        element_bits=[{"fact": ["facts", "byte_bits"]}] * 2,
                        stages=["increment"],
                        byte_orders=["little"] * 2,
                        readout=None,
                        execution="contiguous_read_then_write",
                    )
                },
                readouts={},
            )

        selected, _, _, adapter, codec, limits, _, _ = inputs(
            directory, width=7, separate=self.separate, mutate=declare
        )
        calls, objects = [], []
        count = m * n
        assert count <= 19
        for chunk, offset in enumerate(range(0, count, 7)):
            size = min(7, count - offset)
            address = 80 + 32 * chunk
            bank = chunk % 2 if self.optimization == "storage" else 0
            view = bank if self.separate else 1
            objects.append(TemporaryObject("value_" + str(chunk), SPACES[bank], address, size))
            calls.extend(
                (
                    command("copy", offset, address, size, 2, bank),
                    command("increment", address, address, size, view, view),
                )
            )
            if self.optimization == "copies":
                copied = address + 16
                objects.append(TemporaryObject("copy_" + str(chunk), SPACES[1], copied, size))
                calls.append(command("copy", address, copied, size, view, 1))
                address, view = copied, 1
            calls.append(command("copy", address, offset, size, view, 3))
        return dict(
            selected_contracts=selected,
            adapter=adapter,
            encoding=codec,
            limits=limits,
            kernel=Kernel(
                "increment_chunks",
                (TensorArg("input", (m, n), "i8", "read"), TensorArg("output", (m, n), "i8", "write")),
                tuple(calls),
            ),
            objects=tuple(objects),
            private_ownership=True,
            bindings=PointwiseBindings(
                (AddressWindow("input", 0, count),), AddressWindow("output", 0, count), ("little", "little"), True
            ),
            fusion_qualification_identity="neutral-native-byte-increment-v1",
        )

    def schedule_budgets(self):
        return ScheduleBudgets(
            max_fusions=0,
            max_spills=0,
            eliminate_configuration=False,
            optimize_storage=self.enabled if self.optimization == "storage" else True,
            eliminate_copies=self.enabled if self.optimization == "copies" else False,
        )

    def _wrapper(self, symbol, m, n, image, separate):
        if self.corrupt_address and self.enabled:
            word = struct.Struct("<BBBQQQ")
            changed = bytearray(image)
            for offset in range(0, len(changed), word.size):
                op, read, write, source, destination, size = word.unpack_from(changed, offset)
                if op == 0 and write == 3:
                    word.pack_into(changed, offset, op, read, write, source + 1, destination, size)
                    break
            assert bytes(changed) != image
            (self.work / (symbol + "_wrong_address.bin")).write_bytes(changed)
            image = bytes(changed)
        encoded, count = ",".join(str(v) for v in image), m * n
        return f"""
void {symbol}(const descriptor *a,descriptor *out) {{
 static const uint8_t image[] __attribute__((used,retain,aligned(1),section(".merlin_pointwise.{symbol}")))=
 {{{encoded}}};
 uint8_t first[200],second[200],input[50],output[50];uint64_t counts[6];
 if(a->sizes[0]!={m}||a->sizes[1]!={n}||out->sizes[0]!={m}||out->sizes[1]!={n})__builtin_trap();
 for(int i=0;i<200;i++){{first[i]=99;second[i]=119;}}
 for(int i=0;i<50;i++){{input[i]=71;output[i]=91;}}
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++){{
  input[i*{n}+j]=((const uint8_t*)a->aligned)[a->offset+i*a->strides[0]+j*a->strides[1]];
  totals[3]++;totals[4]++;
 }}
 if(run(image,sizeof(image),8,0,{int(separate)},0,first,second,input,output,192,42,counts))__builtin_trap();
 for(int i=0;i<8;i++)if(first[i]!=99)__builtin_trap();
 for(int i=0;i<16;i++)if(second[i]!=119)__builtin_trap();
 for(int i=32;i<40;i++)if(first[i]!=99)__builtin_trap();
 for(int i=64;i<72;i++)if(first[i]!=99)__builtin_trap();
 for(int i=192;i<200;i++)if(first[i]!=99||second[i]!=119)__builtin_trap();
 for(int i={count};i<50;i++)if(input[i]!=71||output[i]!=91)__builtin_trap();
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++){{
  if(input[i*{n}+j]!=((const uint8_t*)a->aligned)[a->offset+i*a->strides[0]+j*a->strides[1]])__builtin_trap();
  ((uint8_t*)out->aligned)[out->offset+i*out->strides[0]+j*out->strides[1]]=output[i*{n}+j];
  totals[3]++;totals[4]++;
 }}
 for(int i=0;i<3;i++)totals[i]+=counts[i];totals[5]++;
 for(int i=0;i<2;i++)if(counts[3+i]>totals[6+i])totals[6+i]=counts[3+i];
 totals[8]+=counts[5];pointwise_log();
}}
"""


class CopyProvider(StorageProvider):
    optimization = "copies"
