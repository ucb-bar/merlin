"""Original neutral byte-machine premises for physical multi-output execution."""

import hashlib
import json
import struct
from dataclasses import replace

import numpy as np
import yaml
from test_multi_output_source import scalar_sources, wrap
from test_shared_lhs_provider import SharedLHSProvider

from merlin.compile.scheduling.fusion_semantics import ElementwiseSemantics, scalar_graph
from merlin.compile.scheduling.program_limits import (
    AddressWindow,
    DecodedInstruction,
    EncodedAccess,
    ProgramDecoder,
    ProgramLimits,
)
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, ScheduleEncoding
from merlin.compile.scheduling.temporary_storage import StorageAdapter, StorageEffects, TemporaryObject
from merlin.sched.ir import Call, Const, Kernel, TensorArg, evaluate

SPACES = ("private0", "private1", "input0", "input1", "input2", "input3", "output0", "output1")
OPS = ("produce", "add", "multiply")
KEYS = ("n", "a", "ao", "b", "bo", "c", "co", "dst", "out", "width")
WORD = struct.Struct("<B" + "I" * len(KEYS))
QUALIFICATION = "original-neutral-fixed-integer-byte-machine-v1"
COUNTERS = (
    ("commands", "decoded neutral arithmetic instructions completed"),
    ("requested_read_bytes", "sum of actual decoded primitive operand byte extents, including private reads"),
    ("requested_write_bytes", "sum of actual decoded primitive result byte extents, including private writes"),
    ("producer_elements", "logical elements completed by actual produce instructions"),
    ("descriptor_pack_bytes", "bytes copied from actual input descriptors into bridge packed operands"),
    ("descriptor_publish_bytes", "bytes copied from bridge results into actual output descriptors"),
    ("bridge_calls", "completed actual descriptor bridge invocations"),
    ("reserved_zero", "reserved protocol slot; no measurement or resource estimate"),
)


def command(op, **arguments):
    return Call(op, tuple((key, Const(arguments.get(key, 0))) for key in KEYS))


def parameters(call):
    return {key: evaluate(call.arg(key), {}) for key in KEYS}


def effects(call):
    if call.instr not in OPS:
        return None
    p = parameters(call)
    if p["width"] not in (1, 2) or not 0 < p["n"] <= 256:
        return None
    inputs = (("a", "ao"), ("b", "bo"), ("c", "co")) if call.instr == "produce" else (("a", "ao"), ("b", "bo"))
    reads = [EncodedAccess(SPACES[p[selector]], p[offset], p["n"] * p["width"]) for selector, offset in inputs]
    output = EncodedAccess(SPACES[p["dst"]], p["out"], p["n"] * p["width"], True)
    semantic = ElementwiseSemantics(
        scalar_sources(p["width"])[OPS.index(call.instr)], p["n"], (call.instr,), ("little",) * (len(reads) + 1)
    )
    # All nonrelocated behavior, including the actual external selectors, is
    # represented. Extents and addresses are decoded separately as accesses.
    return StorageEffects(
        repr((call.instr, p["width"], p["a"], p["b"], p["c"], p["dst"])),
        (*reads, output),
        "synchronous",
        elementwise=semantic,
    )


def relocate(call, accesses):
    offsets = ("ao", "bo", "co") if call.instr == "produce" else ("ao", "bo")
    changed = dict(zip(offsets, (a.start for a in accesses[:-1]), strict=True), out=accesses[-1].start)
    return replace(call, args=tuple((key, Const(changed.get(key, evaluate(value, {})))) for key, value in call.args))


def original_model(directory, case):
    """One fixed hardware/software envelope, selected before source discovery."""
    from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
    from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction

    width = case.width
    facts = {
        "facts": {
            "target": "neutral",
            "word_bits": 8 * WORD.size,
            "opcode_bits": 8,
            "argument_bits": 32,
            "offsets": [0] + [8 + 32 * i for i in range(len(KEYS))],
            "codes": list(range(3)),
            "selectors": list(range(len(SPACES))),
            "capacity": 512,
            "zero": 0,
            "alignment": width,
            "element_bits": 8 * width,
        }
    }

    def ref(*path):
        return {"fact": ["facts", *path]}

    def argument(name):
        return {"argument": name}

    def selector(name):
        return {
            "argument": name,
            "choices": [{"value": ref("selectors", i), "resource": space} for i, space in enumerate(SPACES)],
        }

    declarations, instructions, elementwise = {}, [], {}
    for code, op in enumerate(OPS):
        inputs = (("a", "ao"), ("b", "bo"), ("c", "co")) if op == "produce" else (("a", "ao"), ("b", "bo"))
        accesses = [
            dict(
                space=selector(space),
                start=argument(offset),
                size=dict(elements=[argument("n")], element_bits=ref("element_bits")),
                writes=False,
            )
            for space, offset in inputs
        ]
        accesses.append(
            dict(
                space=selector("dst"),
                start=argument("out"),
                size=dict(elements=[argument("n")], element_bits=ref("element_bits")),
                writes=True,
            )
        )
        declarations[op] = dict(accesses=accesses, completion="synchronous", token=None, waits=[], exact_copy=False)
        instructions.append(
            dict(
                name=op,
                operands=[
                    dict(name=key, type=dict(kind="integer", bits=ref("argument_bits"), signed=False), unit="integer")
                    for key in KEYS
                ],
                results=[],
                immediates=[],
                encoding=[
                    dict(
                        offset=ref("offsets", i),
                        width=ref("opcode_bits" if i == 0 else "argument_bits"),
                        **(dict(fixed=ref("codes", code)) if i == 0 else dict(argument=key)),
                    )
                    for i, key in enumerate(("opcode", *KEYS))
                ],
                semantics=dict(primitive="neutral." + op, bindings={key: key for key in KEYS}),
                effects=dict(reads=list(SPACES), writes=list(SPACES)),
            )
        )
        elementwise[op] = dict(
            scalar_ir=scalar_sources(width)[code].decode(),
            elements=[argument("n")],
            element_bits=[ref("element_bits")] * len(accesses),
            stages=[op],
            byte_orders=["little"] * len(accesses),
            readout=None,
            execution="contiguous_read_then_write",
        )
    storage = dict(
        schema="merlin.temporary_storage_contract.v1",
        banks={
            space: dict(
                backing="common" if case.shared else space,
                start=ref("zero"),
                size=ref("capacity"),
                alignment=ref("alignment"),
                ownership="invocation_private",
            )
            for space in SPACES[:2]
        },
        external={
            space: dict(backing=space, start=ref("zero"), size=ref("capacity"), ownership="runtime_managed")
            for space in SPACES[2:]
        },
        reservations=[],
        instructions=declarations,
    )
    software = dict(
        schema="merlin.software_spec.v1",
        target="neutral",
        status="reviewed",
        isa=dict(
            schema="merlin.isa_spec.v1",
            namespace="neutral_multioutput",
            facts_sha256="0" * 64,
            word_bits=ref("word_bits"),
            resources=[dict(name=s, kind="memory", unit="bytes", capacity=ref("capacity")) for s in SPACES],
            instructions=instructions,
        ),
        scheduling=dict(
            schema="merlin.schedule_effects.v1",
            numerical_policy=f"integer-modular-{8 * width}",
            instruction_byteorder="little",
            configuration={},
            storage=storage,
            fusion=dict(schema="merlin.physical_fusion_contract.v1", elementwise=elementwise, readouts={}),
        ),
    )
    directory.mkdir()
    fact_path, software_path = directory / "facts.json", directory / "software.yaml"
    fact_path.write_text(json.dumps(facts))
    software["isa"]["facts_sha256"] = hashlib.sha256(fact_path.read_bytes()).hexdigest()
    software_path.write_text(yaml.safe_dump(software))
    selected = read_scheduling_contracts(fact_path, software_path, target="neutral")
    adapter = StorageAdapter(effects, relocate, selected.hardware_identity, selected.numerical_policy, QUALIFICATION)

    def encode(kernel):
        return b"".join(
            encode_instruction(selected.isa, c.instr, parameters(c)).to_bytes(WORD.size, "little") for c in kernel.body
        )

    def decode(image):
        rows = []
        for offset in range(0, len(image), WORD.size):
            name, values = decode_instruction(
                selected.isa, int.from_bytes(image[offset : offset + WORD.size], "little")
            )
            rows.append(command(name, **values))
        return tuple(rows)

    def instructions_from_bytes(image):
        return tuple(
            DecodedInstruction(
                i * WORD.size, WORD.size, effects(c).accesses, completion="synchronous", relocatable=True
            )
            for i, c in enumerate(decode(image))
        )

    codec = ScheduleEncoding(
        encode,
        decode,
        ProgramDecoder(instructions_from_bytes, selected.hardware_identity, QUALIFICATION),
        selected.numerical_policy,
        QUALIFICATION,
    )
    limits = ProgramLimits(16 * WORD.size, 16, 1, tuple(AddressWindow(s, 0, 512) for s in SPACES), ())
    return selected, adapter, codec, limits


def check_actual_trace(rows, arguments, kernel, case):
    """One physical-byte map, including exact produced generations and values."""
    memory, cursor = {}, 0

    def backing(space):
        return "common" if case.shared and space < 2 else SPACES[space]

    for arg, array in zip(kernel.args, arguments, strict=True):
        if arg.access == "read":
            space = SPACES.index(arg.name)
            for offset, value in enumerate(array.tobytes()):
                memory[backing(space), offset] = (value, ("input", arg.name, offset))
    totals = dict(commands=0, requested_read_bytes=0, requested_write_bytes=0, producer_elements=0)
    producer_generations, consumed_generations = [], []
    for row in rows:
        opcode, *values = row[: 1 + len(KEYS)]
        p = dict(zip(KEYS, values, strict=True))
        op, count, width = OPS[opcode], p["n"], p["width"]
        assert 0 < count <= 256 and width == case.width
        inputs = (("a", "ao"), ("b", "bo"), ("c", "co")) if op == "produce" else (("a", "ao"), ("b", "bo"))
        raw, generations = [], []
        for selector, offset in inputs:
            data = [memory[backing(p[selector]), p[offset] + i] for i in range(count * width)]
            raw.append(bytes(value for value, _ in data))
            generations.append(tuple(origin for _, origin in data))
        result = bytearray()
        for i in range(count):
            a, b = (int.from_bytes(data[i * width : (i + 1) * width], "little", signed=True) for data in raw[:2])
            value = a + b if op == "add" else a * b
            if op == "produce":
                c = int.from_bytes(raw[2][i * width : (i + 1) * width], "little", signed=True)
                value = wrap(value, 8 * width) + c
            result.extend((wrap(value, 8 * width) % (1 << (8 * width))).to_bytes(width, "little"))
        assert bytes(row[1 + len(KEYS) :]) == result
        origin = ("produced", cursor, op)
        if op == "produce":
            producer_generations.append(origin)
            totals["producer_elements"] += count
        elif p["a"] < 2:
            consumed_generations.append(tuple(set(generations[0])))
        for offset, value in enumerate(result):
            memory[backing(p["dst"]), p["out"] + offset] = (value, origin)
        cursor += 1
        totals["commands"] += 1
        totals["requested_read_bytes"] += len(inputs) * count * width
        totals["requested_write_bytes"] += count * width
    if len(kernel.args) == 6:
        assert len(rows) == 3 and len(producer_generations) == 1
        assert consumed_generations == [(producer_generations[0],)] * 2
    return totals


def machine_source(kernel, image, case, symbol, bridge):
    """Actual encoded instructions drive every address and arithmetic operation."""
    m, n = case.shape
    inputs = sum(arg.access == "read" for arg in kernel.args)
    outputs, width = len(kernel.args) - inputs, case.width
    descriptors = ",".join(
        f"const descriptor *d{i}" if i < inputs else f"descriptor *d{i}" for i in range(len(kernel.args))
    )
    assignments = "\n".join(f"spaces[{SPACES.index(arg.name)}]=packed[{i}];" for i, arg in enumerate(kernel.args))
    encoded = ",".join(str(v) for v in image)
    return f"""#include <stdint.h>
#include <stddef.h>
typedef struct {{void *allocated,*aligned;int64_t offset,sizes[2],strides[2];}} descriptor;
static uint64_t counts[8],nr;
static int64_t trace[64][{1 + len(KEYS) + 512}];
uint64_t {symbol}_observe_count(int i){{return i>=0&&i<8?counts[i]:0;}}
void {symbol}_observe_reset(void){{for(int i=0;i<8;i++)counts[i]=0;nr=0;}}
uint64_t {symbol}_trace_size(void){{return nr;}}
int64_t {symbol}_trace(int r,int c){{
 if(r<0||(uint64_t)r>=nr||c<0||c>={1 + len(KEYS) + 512})__builtin_trap();return trace[r][c];}}
static uint32_t word(const uint8_t*p){{
 return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}}
static uint32_t read(const uint8_t*p,int w){{return p[0]|(w==2?(uint32_t)p[1]<<8:0);}}
void {bridge}({descriptors}){{
 static const uint8_t image[] __attribute__((used,retain,aligned(1),
 section(".merlin_multioutput.{bridge}")))={{{encoded}}};
 const descriptor *descriptors[]={{{",".join("d" + str(i) for i in range(len(kernel.args)))}}};
 uint8_t packed[{len(kernel.args)}][512],private0[544],private1[544],allowed[512];
 for(int i=0;i<512;i++)((volatile uint8_t*)allowed)[i]=0;
 uint8_t *spaces[8]={{private0+16,{"private0+16" if case.shared else "private1+16"},0,0,0,0,0,0}};
 for(int i=0;i<544;i++)private0[i]=private1[i]=173;
 for(int d=0;d<{len(kernel.args)};d++){{
  const descriptor *p=descriptors[d];
  if(!p||!p->aligned||p->offset<0||p->offset>(1<<20)||p->sizes[0]!={m}||p->sizes[1]!={n}||
     p->strides[0]<=0||p->strides[1]<=0||p->strides[0]>(1<<20)||p->strides[1]>(1<<20))__builtin_trap();
  for(int i=0;i<512;i++)packed[d][i]=193;
  if(d<{inputs})for(int i=0;i<{m};i++)for(int j=0;j<{n};j++)for(int b=0;b<{width};b++){{
   packed[d][{width}*(i*{n}+j)+b]=((const uint8_t*)p->aligned)[{width}*(p->offset+i*p->strides[0]+j*p->strides[1])+b];
   counts[4]++;
  }}
 }}
 {assignments}
 for(size_t pc=0;pc<sizeof(image);pc+={WORD.size}){{
  if(nr>=64||pc+{WORD.size}>sizeof(image))__builtin_trap();
  uint32_t p[{len(KEYS)}];int op=image[pc];
  if(op>2)__builtin_trap();
  trace[nr][0]=op;
  for(int i=0;i<{len(KEYS)};i++)trace[nr][i+1]=p[i]=word(image+pc+1+4*i);
  uint32_t count=p[0],a=p[1],ao=p[2],b=p[3],bo=p[4],c=p[5],co=p[6],dst=p[7],out=p[8],w=p[9];
  if(!count||count>256||w!={width}||a>=8||b>=8||c>=8||dst>=8||!spaces[a]||!spaces[b]||!spaces[dst]||
     (op==0&&!spaces[c])||ao>512-count*w||bo>512-count*w||out>512-count*w||
     (op==0&&co>512-count*w))__builtin_trap();
  if(dst>=2&&dst<6)__builtin_trap();
  for(uint32_t base=0;base<count;base+=16){{
   uint32_t end=base+16<count?base+16:count;
   for(uint32_t i=base;i<end;i++){{
    uint32_t x=read(spaces[a]+ao+i*w,w),y=read(spaces[b]+bo+i*w,w);
    uint32_t value=op==1?x+y:x*y;
    if(op==0)value+=read(spaces[c]+co+i*w,w);
    for(uint32_t byte=0;byte<w;byte++){{
     uint8_t v=(uint8_t)(value>>(8*byte));spaces[dst][out+i*w+byte]=v;
     trace[nr][{1 + len(KEYS)}+i*w+byte]=v;
     if(dst<2)allowed[out+i*w+byte]=1;
    }}
   }}
  }}
  counts[0]++;counts[1]+=(op==0?3:2)*count*w;counts[2]+=count*w;
  if(op==0)counts[3]+=count;
  nr++;
 }}
 for(int i=0;i<544;i++){{
  if((i<16||i>=528||!allowed[i-16])&&private0[i]!=173)__builtin_trap();
  if(private1[i]!=173)__builtin_trap();
 }}
 for(int d={inputs};d<{inputs + outputs};d++)for(int i=0;i<{m};i++)for(int j=0;j<{n};j++)for(int b=0;b<{width};b++){{
  descriptor const*p=descriptors[d];
  ((uint8_t*)p->aligned)[{width}*(p->offset+i*p->strides[0]+j*p->strides[1])+b]=packed[d][{width}*(i*{n}+j)+b];counts[5]++;
 }}
 counts[6]++;
}}
"""


class MultiOutputProvider(SharedLHSProvider):
    """The same original primitive implementation serves all three boundaries."""

    def __init__(self, work, model, trace, case, *, grouped):
        self.work, self.model, self.trace, self.case, self.grouped = work, model, trace, case, grouped
        self.records, self.results = [], []
        work.mkdir()

    def lower(self, source):
        from merlin.compile.scheduling.pointwise_source import PointwiseBindings
        from merlin.llvmlower.source_pointwise import PointwiseProgram

        graphs = [scalar_graph(text, 32, 65536) for text in scalar_sources(self.case.width)]
        if source.graph not in graphs or tuple(source.shape) != self.case.shape:
            return None
        opcode = OPS[graphs.index(source.graph)]
        count, width = int(np.prod(source.shape)), self.case.width
        inputs = 3 if opcode == "produce" else 2
        args = tuple(TensorArg(f"input{i}", source.shape, f"i{8 * width}", "read") for i in range(inputs))
        args += (TensorArg("output0", source.shape, f"i{8 * width}", "write"),)
        call = command(opcode, n=count, a=2, b=3, c=4 if inputs == 3 else 0, dst=6, width=width)
        bindings = PointwiseBindings(
            tuple(AddressWindow(f"input{i}", 0, count * width) for i in range(inputs)),
            AddressWindow("output0", 0, count * width),
            ("little",) * (inputs + 1),
            True,
        )
        return PointwiseProgram(Kernel("original_" + opcode, args, (call,)), (), bindings)

    def lower_multioutput(self, source):
        from merlin.compile.scheduling.multioutput_fusion import MultiOutputBindings
        from merlin.llvmlower.source_multioutput import MultiOutputProgram

        count, width = int(np.prod(source.shape)), self.case.width
        assert tuple(source.shape) == self.case.shape
        assert source.input_indices == ((0, 1, 2), (-1, 3), (-1, 3))
        args = tuple(TensorArg(f"input{i}", source.shape, f"i{8 * width}", "read") for i in range(4))
        args += tuple(TensorArg(f"output{i}", source.shape, f"i{8 * width}", "write") for i in range(2))
        body = self.group_calls()
        bindings = MultiOutputBindings(
            tuple(AddressWindow(f"input{i}", 0, count * width) for i in range(4)),
            tuple(AddressWindow(f"output{i}", 0, count * width) for i in range(2)),
            ("little",) * 6,
            True,
        )
        return MultiOutputProgram(
            Kernel("retained_producer", args, body),
            (TemporaryObject("p", "private0", 1024, count * width, width),),
            bindings,
        )

    def group_calls(self):
        count, width = int(np.prod(self.case.shape)), self.case.width
        other = 1 if self.case.shared else 0
        return (
            command("produce", n=count, a=2, b=3, c=4, dst=0, out=1024, width=width),
            command("add", n=count, a=0, ao=1024, b=5, dst=6, width=width),
            command("multiply", n=count, a=other, ao=1024, b=5, dst=7, width=width),
        )

    def selection(self):
        from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
        from merlin.llvmlower.source_pointwise import SourcePointwiseLowering

        selected, adapter, codec, limits = self.model
        config = SourcePointwiseLowering(
            selected_contracts=selected,
            adapter=adapter,
            encoding=codec,
            limits=limits,
            execution_domain="host",
            allocation_alignment=32,
            lower=self.lower,
            prepare=self.prepare,
            build=self.build,
            private_ownership=True,
            external_buffers_disjoint=True,
            scalar_qualification_identity=QUALIFICATION,
            borrowed_effects=BorrowedBufferEffects(QUALIFICATION, True, True, True),
            budgets=ScheduleBudgets(max_fusions=0, prefetch_distance=0),
            max_source_elements=4096,
        )
        if not self.grouped:
            return config
        from merlin.compile.scheduling.multioutput_fusion import MultiOutputCompositionLaw
        from merlin.llvmlower.source_multioutput import SourceMultiOutputLowering

        p, e0, e1 = self.group_calls()
        law = MultiOutputCompositionLaw(
            adapter.configuration_identity,
            adapter.numerical_policy_identity,
            adapter.qualification_identity,
            QUALIFICATION,
            producer=(p.instr, effects(p).semantic_key),
            epilogues=((e0.instr, effects(e0).semantic_key), (e1.instr, effects(e1).semantic_key)),
            complete_projected_effects=True,
            synchronous_private_composition=True,
            suppressed_intermediate_transfers_no_faults=True,
            suppressed_intermediate_transfers_unobserved=True,
        )
        return replace(config, multi_output=SourceMultiOutputLowering(self.lower_multioutput, law))

    def prepare(self, bindings, directory):
        from merlin.common.digest import sha256_file

        self.bindings = tuple(bindings)
        assert len(bindings) == (1 if self.grouped else 3)
        for index, binding in enumerate(bindings):
            result = binding.compilation
            result.revalidate()
            kernel = result.compiled.kernel
            image = b"".join(result.compiled.program.segments)
            actual = self.model[2].decode_calls(image)
            assert actual == kernel.body
            path = self.work / f"program{index}.bin"
            path.write_bytes(image)
            symbol = binding.writer.symbol + "_primitive"
            bridge = "_mlir_ciface_" + binding.writer.borrowed_symbol
            source = self.work / f"provider{index}.c"
            source.write_text(machine_source(kernel, image, self.case, symbol, bridge))
            self.records.append(
                dict(
                    result=result,
                    kernel=kernel,
                    symbol=symbol,
                    bridge=bridge,
                    source=source,
                    source_sha256=sha256_file(source),
                    image=path,
                    image_sha256=sha256_file(path),
                )
            )
            self.results.append(result)
        self.qualify()

    def operands(self, item, epoch):
        trace = self.trace[epoch]
        if self.grouped:
            return trace[:4], trace[5:]
        opcode = item["kernel"].body[0].instr
        return (
            (trace[:3], (trace[4],))
            if opcode == "produce"
            else ((trace[4], trace[3]), (trace[5 if opcode == "add" else 6],))
        )

    def qualify(self):
        import ctypes
        import shutil

        from test_pointwise_source_provider import Descriptor

        from merlin.common import invocation_record as invocation
        from merlin.common.digest import sha256_file
        from merlin.llvmlower import toolchain

        sources = tuple(item["source"] for item in self.records)
        compiler, image = toolchain.clang().resolve(), self.work / "qualification.so"
        argv = [str(compiler), "-O2", "-fno-builtin", "-shared", "-fPIC", *map(str, sources), "-o", str(image)]
        invocation.run(
            argv,
            directory=self.work,
            stage="neutral-multioutput-qualification",
            inputs=sources,
            outputs=(image,),
            capture_output=True,
            check=True,
        )
        # Select the actual product by bytes even if a prior Python dlopen
        # retained an older image under a reused test-directory pathname.
        loaded = self.work / f"qualification-{sha256_file(image)}.so"
        shutil.copyfile(image, loaded)
        assert sha256_file(loaded) == sha256_file(image)
        native = ctypes.CDLL(str(loaded))
        observed = []
        for index, item in enumerate(self.records):
            symbol = item["symbol"]
            size, query = getattr(native, symbol + "_trace_size"), getattr(native, symbol + "_trace")
            size.restype, query.restype = ctypes.c_uint64, ctypes.c_int64
            query.argtypes = [ctypes.c_int, ctypes.c_int]
            function = getattr(native, item["bridge"])
            function.argtypes = [ctypes.POINTER(Descriptor)] * len(item["kernel"].args)
            for epoch in range(3):
                operands, expected = self.operands(item, epoch)
                buffers, originals, descriptors = [], [], []
                for array in (*operands, *(np.full_like(a, 71) for a in expected)):
                    m, n = array.shape
                    owner = np.full((m + 2, 2 * n + 5), 71, array.dtype)
                    owner[1 : m + 1, 1 : 2 * n + 1 : 2] = array
                    buffers.append(owner)
                    originals.append(owner.copy())
                    descriptors.append(
                        Descriptor(owner.ctypes.data, owner.ctypes.data, 2 * n + 6, (m, n), (2 * n + 5, 2))
                    )
                start = size()
                function(*(ctypes.byref(d) for d in descriptors))
                rows = [
                    [query(r, c) for c in range(1 + len(KEYS) + self.case.width * int(np.prod(self.case.shape)))]
                    for r in range(start, size())
                ]
                effects_record = check_actual_trace(rows, (*operands, *expected), item["kernel"], self.case)
                for i, want in enumerate(expected, len(operands)):
                    m, n = want.shape
                    originals[i][1 : m + 1, 1 : 2 * n + 1 : 2] = want
                for actual, want in zip(buffers, originals, strict=True):
                    np.testing.assert_array_equal(actual, want)
                raw = self.work / f"qualification-{index}-{epoch}.npz"
                np.savez(raw, **{f"owner{i}": a for i, a in enumerate(buffers)})
                observed.append(
                    dict(
                        binding=index,
                        epoch=epoch,
                        actual_rows=rows,
                        effects=effects_record,
                        raw_buffers=dict(path=str(raw), sha256=sha256_file(raw)),
                    )
                )
        self.witness = self.work / "qualification.json"
        self.witness.write_text(
            json.dumps(
                dict(
                    scope=(
                        "neutral actual C/encoded instructions/ranked descriptors; "
                        "original effects laws separately declared"
                    ),
                    compiler=dict(path=str(compiler), sha256=sha256_file(compiler)),
                    sources={str(p): sha256_file(p) for p in sources},
                    image=dict(path=str(image), sha256=sha256_file(image)),
                    loaded_image=dict(path=str(loaded), sha256=sha256_file(loaded)),
                    actual_argv=argv,
                    observations=observed,
                ),
                indent=2,
            )
        )
        for receipt in self.work.glob("invocations/*/invocation.json"):
            invocation.verify(receipt)
