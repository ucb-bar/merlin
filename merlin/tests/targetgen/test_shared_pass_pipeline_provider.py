"""Neutral byte-packet machine with real multi-pending, token-only completion."""

import hashlib
import io
import json
import struct
from dataclasses import replace

import test_copy_pipeline as leaf
from xdsl.context import Context
from xdsl.dialects import arith, builtin
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin.compile.scheduling.program_limits import DecodedInstruction, EncodedAccess, ProgramDecoder, ProgramLimits
from merlin.compile.scheduling.schedule_pipeline import ScheduleEncoding
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.sched.ir import Const, evaluate
from merlin.targetgen.isa_dialect_gen import build_dialect, decode_instruction, encode_instruction

KEYS = ("address", "width", "host", "token", "view")
NAMES = ("setup", "fetch", "await", "affine", "store", "descriptor")


def project(call):
    effect = leaf.project(call)
    # Address relocation preserves the operation, numeric width and token identity.
    key = repr((call.instr, tuple(evaluate(call.arg(k), {}) for k in KEYS[1:])))
    return replace(effect, semantic_key=key) if effect is not None else None


def rewrite(call, accesses):
    if not accesses:
        return call
    local = next(a for a in accesses if a.space in ("bytes", "words"))
    return replace(call, args=tuple((k, Const(local.start) if k == "address" else v) for k, v in call.args))


def fixture(directory, *, width=7, panels=7, slots=2, alias=True, bits=64, big=False):
    kernel, objects, banks, _, options = leaf.fixture(width=width, panels=panels, slots=slots, alias=alias)
    word = struct.Struct((">" if big else "<") + "B" + ("Q" if bits == 64 else "I") * len(KEYS))
    fields = (8, *(bits for _ in KEYS))
    byte_offsets = (0, *(1 + i * bits // 8 for i in range(len(KEYS))))
    offsets = [word.size * 8 - b * 8 - n if big else b * 8 for b, n in zip(byte_offsets, fields, strict=True)]
    facts = dict(
        facts=dict(
            target="neutral",
            word_bits=word.size * 8,
            widths=fields,
            offsets=offsets,
            opcodes=list(range(len(NAMES))),
            capacities=[b.capacity for b in banks],
            external=[a.nbytes for a in kernel.args],
            inflight=slots,
            queue=slots,
            starts=[0, 0],
            alignment=1,
            view_codes=[0, 1],
            reservation_start=0,
            reservation_size=8,
        )
    )

    def ref(*parts):
        return {"fact": ["facts", *parts]}

    def arg(name):
        return {"argument": name}

    def space():
        return dict(
            argument="view",
            choices=[dict(value=ref("view_codes", i), resource=name) for i, name in enumerate(("bytes", "words"))],
        )

    instructions, effects = [], {}
    for code, name in enumerate(NAMES):
        accesses = []
        if name == "fetch":
            accesses = [
                dict(space={"resource": "input"}, start=arg("host"), size=arg("width"), writes=False),
                dict(space=space(), start=arg("address"), size=arg("width"), writes=True),
            ]
        elif name == "affine":
            accesses = [dict(space=space(), start=arg("address"), size=arg("width"), writes=w) for w in (False, True)]
        elif name == "store":
            accesses = [
                dict(space=space(), start=arg("address"), size=arg("width"), writes=False),
                dict(space={"resource": "output"}, start=arg("host"), size=arg("width"), writes=True),
            ]
        effects[name] = dict(
            accesses=accesses,
            completion="asynchronous" if name == "fetch" else "wait" if name == "await" else "synchronous",
            token=arg("token") if name == "fetch" else None,
            waits=[arg("token")] if name == "await" else [],
            exact_copy=name in ("fetch", "store"),
        )
        instructions.append(
            dict(
                name=name,
                operands=[
                    dict(name=k, type=dict(kind="integer", signed=False, bits=ref("widths", i + 1)), unit="integer")
                    for i, k in enumerate(KEYS)
                ],
                results=[],
                immediates=[],
                encoding=[
                    dict(
                        offset=ref("offsets", i),
                        width=ref("widths", i),
                        **(dict(fixed=ref("opcodes", code)) if i == 0 else dict(argument=KEYS[i - 1])),
                    )
                    for i in range(6)
                ],
                semantics=dict(primitive="neutral." + name, bindings={k: k for k in KEYS}),
                effects=dict(
                    reads=(["input"] if name == "fetch" else ["bytes", "words"] if name in ("affine", "store") else []),
                    writes=(
                        ["bytes", "words"] if name in ("fetch", "affine") else ["output"] if name == "store" else []
                    ),
                ),
            )
        )
    resources = [
        dict(name=b.name, kind="memory", unit="bytes", capacity=ref("capacities", i)) for i, b in enumerate(banks)
    ]
    resources += [
        dict(name=a.name, kind="memory", unit="bytes", capacity=ref("external", i)) for i, a in enumerate(kernel.args)
    ]
    storage = dict(
        schema="merlin.temporary_storage_contract.v1",
        banks={
            b.name: dict(
                backing=b.backing,
                start=ref("starts", i),
                size=ref("capacities", i),
                alignment=ref("alignment"),
                ownership="invocation_private",
            )
            for i, b in enumerate(banks)
        },
        external={
            a.name: dict(
                backing=a.name + "_memory", start=ref("starts", i), size=ref("external", i), ownership="runtime_managed"
            )
            for i, a in enumerate(kernel.args)
        },
        reservations=[
            dict(resource=r.storage, start=ref("reservation_start"), size=ref("reservation_size"), purpose="reserved")
            for r in options["reservations"]
        ],
        instructions=effects,
    )
    software = dict(
        schema="merlin.software_spec.v1",
        target="neutral",
        status="reviewed",
        isa=dict(
            schema="merlin.isa_spec.v1",
            namespace="pipeline_test",
            facts_sha256="0" * 64,
            word_bits=ref("word_bits"),
            resources=resources,
            instructions=instructions,
        ),
        scheduling=dict(
            schema="merlin.schedule_effects.v1",
            numerical_policy="exact-i8",
            instruction_byteorder="big" if big else "little",
            configuration={},
            storage=storage,
        ),
    )
    fp, sp = directory / "facts.json", directory / "software.json"
    fp.write_text(json.dumps(facts))
    software["isa"]["facts_sha256"] = hashlib.sha256(fp.read_bytes()).hexdigest()
    sp.write_text(json.dumps(software))
    selected = read_scheduling_contracts(fp, sp, target="neutral")
    adapter = replace(leaf.ADAPTER, project=project, rewrite=rewrite, configuration_identity=selected.hardware_identity)
    declaration = dict(
        schema="merlin.copy_pipeline.v1",
        crossing_pairs=[["fetch", n] for n in ("await", "affine", "store")],
        concurrent_pairs=[["fetch", "fetch"]],
        max_in_flight=ref("inflight"),
        queue_capacity=ref("queue"),
        independence="disjoint_memory_preserves_state_and_issue_order",
        wait_semantics="token_completion_only_preserves_state",
    )
    admission = leaf.resolve_copy_pipeline_admission(
        declaration, facts, adapter=adapter, qualification_identity="independent-neutral-delayed-copy-v1"
    )

    def encode(kernel):
        return b"".join(
            encode_instruction(selected.isa, c.instr, {k: evaluate(v, {}) for k, v in c.args}).to_bytes(
                word.size, selected.instruction_byteorder
            )
            for c in kernel.body
        )

    def decode(image):
        calls = []
        if len(image) % word.size:
            raise ValueError("partial actual command")
        for offset in range(0, len(image), word.size):
            name, v = decode_instruction(
                selected.isa, int.from_bytes(image[offset : offset + word.size], selected.instruction_byteorder)
            )
            if v["view"] not in (0, 1):
                raise ValueError("unknown actual physical view")
            calls.append(leaf.op(name, **v))
        return tuple(calls)

    def decode_program(image):
        pending, seen, result = set(), set(), []
        for i, c in enumerate(decode(image)):
            v = {k: evaluate(x, {}) for k, x in c.args}
            a, n, h, t = (v[k] for k in KEYS[:4])
            local = ("bytes", "words")[v["view"]]
            access, completion = (), "synchronous"
            if c.instr == "fetch":
                if t in seen:
                    raise ValueError("actual duplicate token")
                seen.add(t)
                pending.add(t)
                access = (EncodedAccess("input", h, n), EncodedAccess(local, a, n, True))
                completion = "asynchronous"
            elif c.instr == "await":
                if t not in pending:
                    raise ValueError("actual wait without pending launch")
                pending.remove(t)
                completion = "synchronous" if pending else "drain"
            elif c.instr == "affine":
                access = (EncodedAccess(local, a, n), EncodedAccess(local, a, n, True))
            elif c.instr == "store":
                access = (EncodedAccess(local, a, n), EncodedAccess("output", h, n, True))
            result.append(DecodedInstruction(i * word.size, word.size, access, completion=completion, relocatable=True))
        if pending:
            raise ValueError("actual return before completion")
        return tuple(result)

    codec = ScheduleEncoding(
        encode,
        decode,
        ProgramDecoder(decode_program, adapter.configuration_identity, "independent-neutral-token-decoder"),
        "exact-i8",
        "neutral-encoded-delayed-copy",
    )
    limits = ProgramLimits(
        word.size * len(kernel.body),
        len(kernel.body),
        1,
        (*options["external_windows"], *(leaf.AddressWindow(b.name, 0, b.capacity) for b in banks)),
        (),
    )
    return selected, kernel, objects, adapter, codec, limits, admission, options, banks, word


def target_text(kernel, selected):
    from merlin.llvmlower.target_schedule import instruction_module

    module = instruction_module(replace(kernel, name="target_program", args=(), attrs=()), selected)
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def read_target(text, model):
    selected, original = model[:2]
    context = Context()
    for d in (builtin.Builtin, arith.Arith, build_dialect(selected.isa)):
        context.load_dialect(d)
    module = Parser(context, text).parse_module()
    module.verify()
    values = {}
    calls = []
    for op in module.body.block.ops:
        if isinstance(op, arith.ConstantOp):
            assert isinstance(op.value, builtin.IntegerAttr)
            values[op.result] = op.value.value.data
        else:
            prefix = selected.isa.namespace + "."
            assert op.name.startswith(prefix) and not op.results and not op.regions and len(op.operands) == len(KEYS)
            calls.append(
                leaf.op(op.name[len(prefix) :], **dict(zip(KEYS, (values[v] for v in op.operands), strict=True)))
            )
    selected.revalidate()
    return replace(original, body=tuple(calls))


def image_from_target(text, model):
    return model[4].encode(read_target(text, model))


# Unlike the historical single-pending C fixture, this original neutral machine
# has an independent bounded pending set. Fetch never publishes destination bytes;
# only its matching wait copies bytes. Pending read/write intervals forbid hazards.
RUNTIME = r"""
#include <stdint.h>
#include <string.h>
struct Pending {int active,view;uint64_t a,n,h,token;};
static uint64_t field(const unsigned char*p,int bits,int big){
 uint64_t v=0;for(int i=0;i<bits/8;i++)v|=(uint64_t)p[i]<<(8*(big?bits/8-1-i:i));return v;
}
static int overlap(unsigned char*x,uint64_t a,uint64_t n,unsigned char*y,uint64_t b,uint64_t m){
 return x==y&&a<b+m&&b<a+n;
}
int execute(const unsigned char*code,uint64_t bytes,unsigned char**memory,uint64_t*caps,uint64_t*counts,int fault){
 const int bits=PACKET_BITS,big=PACKET_BIG,word=1+5*(PACKET_BITS/8),queue=QUEUE_BOUND;
 struct Pending pending[QUEUE_BOUND];unsigned char seen[256]={0};
 memset(pending,0,sizeof(pending));memset(counts,0,6*sizeof(*counts));
 int active=0,configured=0;
 for(uint64_t pc=0;pc<bytes;pc+=word){
  if(pc+word>bytes)return 1;int op=code[pc];uint64_t v[5];for(int j=0;j<5;j++)v[j]=field(code+pc+1+j*bits/8,bits,big);
  uint64_t a=v[0],n=v[1],h=v[2],t=v[3],view=v[4];if(op>5||view>1)return 2;counts[0]++;
  if(op==0||op==5){if(active)return 3;configured=1;continue;}
  if(op==2){int slot=-1;for(int j=0;j<queue;j++)if(pending[j].active&&pending[j].token==t)slot=j;if(slot<0)return 4;
   struct Pending*p=&pending[slot];if(fault!=1)memcpy(memory[p->view]+p->a,memory[2]+p->h,p->n);
   counts[1]+=p->n;counts[2]+=p->n;p->active=0;active--;counts[5]++;continue;}
  if(!n||a>caps[view]||n>caps[view]-a)return 5;
  if((op==1||op==4)&&(h>caps[op==1?2:3]||n>caps[op==1?2:3]-h))return 6;
  for(int j=0;j<queue;j++)if(pending[j].active){struct Pending*p=&pending[j];
   if(overlap(memory[view],a,n,memory[p->view],p->a,p->n))return 7;
   if(op==4&&overlap(memory[3],h,n,memory[2],p->h,p->n))return 8;
  }
  if(op==1){if(t>=256||seen[t]||active==queue)return 9;seen[t]=1;int slot=0;while(pending[slot].active)slot++;
   pending[slot]=(struct Pending){1,(int)view,a,n,h,t};active++;
   if((uint64_t)active>counts[3])counts[3]=active;counts[4]++;}
  else if(op==3){if(!configured)return 10;
   for(uint64_t j=0;j<n;j++)memory[view][a+j]=(unsigned char)(3*memory[view][a+j]+7);counts[1]+=n;counts[2]+=n;}
  else if(op==4){memcpy(memory[3]+h,memory[view]+a,n);counts[1]+=n;counts[2]+=n;}
  else return 11;
 }
 return active?12:0;
}
"""


def runtime(model):
    word = model[-1]
    bits = (word.size - 1) * 8 // 5
    return (
        f"#define PACKET_BITS {bits}\n"
        f"#define PACKET_BIG {int(model[0].instruction_byteorder == 'big')}\n"
        f"#define QUEUE_BOUND {model[6].queue_capacity}\n" + RUNTIME
    )


def verify_artifact(selected_text, artifact, model):
    """Independently join actual function operand DAG and actual encoded calls."""
    from xdsl.dialects import llvm

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    expected = image_from_target(selected_text, model)
    module = Parser(make_llvm_context(), artifact).parse_module()
    module.verify()
    globals_ = [op for op in module.body.block.ops if isinstance(op, llvm.GlobalOp)]
    definitions = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp) and op.body.blocks]
    assert len(globals_) == 1 and len(definitions) == 1, "unrecognized emitted globals or function bodies"
    global_ = globals_[0]
    assert global_.sym_name.data == "fixture_commands"
    assert bytes(x & 255 for x in global_.value.get_values()) == expected, (
        "actual emitted call operands differ from selected IR"
    )
    definition = definitions[0]
    assert definition.sym_name.data == "fixture_kernel" and len(definition.body.blocks) == 1
    block = definition.body.block
    values = dict(zip(block.args, ("spaces", "caps", "counts", "fault"), strict=True))
    calls = []
    for op in block.ops:
        if isinstance(op, llvm.AddressOfOp):
            values[op.result] = ("global", op.global_name.root_reference.data)
        elif isinstance(op, llvm.ConstantOp):
            values[op.result] = op.value.value.data
        elif isinstance(op, llvm.CallOp):
            assert op.callee is not None
            calls.append((op.callee.root_reference.data, tuple(values[v] for v in op.operands)))
            assert len(op.results) == 1
            values[op.results[0]] = "status"
        else:
            assert isinstance(op, llvm.ReturnOp) and [values[v] for v in op.operands] == ["status"]
    assert calls == [
        (
            "execute",
            (
                ("global", "fixture_commands"),
                len(expected),
                "spaces",
                "caps",
                "counts",
                "fault",
            ),
        )
    ]
    return expected
