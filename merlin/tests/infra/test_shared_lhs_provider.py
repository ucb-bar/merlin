"""Neutral byte-addressed execution of actual grouped dense kernel operands.

Only test callers select this original implementation. Its explicit synchronous
no-retention premises are not facts about a physical accelerator.
"""

from merlin.sched.codegen import emit_c_function
from merlin.sched.ir import Ptr, instances
from merlin.sched.isa import InstrDef, InstructionSet, Operand

CTYPES = {"i8": "int8_t", "i16": "int16_t", "i32": "int32_t"}


def machine_source(kernel, problems, storages, symbol, bridge, *, index_bits):
    """Emit C from the actual Kernel, with pointer-driven operand identities."""
    members = len(problems)
    assert members in (1, 2) and len(storages) == members
    expected = ("lhs", "rhs", "output") if members == 1 else ("lhs", "rhs0", "rhs1", "output0", "output1")
    assert tuple(a.name for a in kernel.args) == expected
    lhs = storages[0].lhs
    assert all(s.lhs == lhs for s in storages)
    views = (lhs, *(s.rhs for s in storages), *(s.accumulator for s in storages))
    sizes = {
        b: max((v.base + v.capacity) * v.unit_bytes for v in views if v.backing == b)
        for b in {v.backing for v in views}
    }
    names = {b: f"bank{i}" for i, b in enumerate(sorted(sizes))}
    declarations = "\n".join(f"static uint8_t {names[b]}[{n + 32}];" for b, n in sizes.items())

    def array(values):
        return "{" + ",".join(map(str, values)) + "}"

    rhs = [s.rhs for s in storages]
    acc = [s.accumulator for s in storages]
    preamble = f"""
#include <stdint.h>
#include <stddef.h>
{declarations}
static uint8_t *const rhs_bank[{members}]={{{",".join(names[v.backing] + "+16" for v in rhs)}}};
static uint8_t *const acc_bank[{members}]={{{",".join(names[v.backing] + "+16" for v in acc)}}};
static const int64_t rhs_base[]={array(v.base for v in rhs)},rhs_end[]={array(v.base + v.capacity for v in rhs)};
static const int64_t acc_base[]={array(v.base for v in acc)},acc_end[]={array(v.base + v.capacity for v in acc)};
static const int64_t rhs_unit[]={array(v.unit_bytes for v in rhs)},acc_unit[]={array(v.unit_bytes for v in acc)};
static const int64_t columns[]={array(p.n for p in problems)};
static const int64_t byte_width[]={array(p.rhs_bytes for p in problems)};
static uint8_t *volatile external[{len(expected)}];
static const uint64_t external_sizes[]={array(a.nbytes for a in kernel.args)};
static uint64_t counts[8],trace_used;
static int64_t trace_rows[8192][12];
static int64_t pending_member=-1,weight_row,acc_row,continuation,active_columns;
static void trace(int op,int member,int n,const int64_t *values){{
 if(trace_used>=8192||n<0||n>9)__builtin_trap();
 trace_rows[trace_used][0]=op;trace_rows[trace_used][1]=member;trace_rows[trace_used][2]=n;
 for(int i=0;i<n;i++)trace_rows[trace_used][3+i]=values[i];trace_used++;
}}
static void put(uint8_t *p,uint32_t value,int width){{for(int i=0;i<width;i++)p[i]=(uint8_t)(value>>(8*i));}}
static int32_t get(const uint8_t *p,int width){{
 uint32_t v=0;for(int i=0;i<width;i++)v|=(uint32_t)p[i]<<(8*i);
 if(width==1)return (int8_t)v;if(width==2)return (int16_t)v;return (int32_t)v;
}}
static int identify(const uint8_t *p,int first,int count,int64_t *offset){{
 uintptr_t address=(uintptr_t)p;int found=-1;
 for(int i=first;i<first+count;i++)if(address>=(uintptr_t)external[i]&&
 address<(uintptr_t)external[i]+external_sizes[i]){{if(found>=0)__builtin_trap();found=i;}}
 if(found<0)__builtin_trap();*offset=(int64_t)(address-(uintptr_t)external[found]);return found-first;
}}
static int weight_member(int64_t row){{
 int found=-1;for(int i=0;i<{members};i++)if(row>=rhs_base[i]&&row<rhs_end[i]){{
 if(found>=0)__builtin_trap();found=i;}}if(found<0)__builtin_trap();return found;
}}
static void do_lhs_load(int64_t a,int64_t b,int64_t r,int64_t c,int64_t rows,int64_t cols,
 int64_t dst,int64_t step,uint8_t *src){{
 int64_t offset;identify(src,0,1,&offset);
 trace(0,-1,9,(int64_t[]){{a,b,r,c,rows,cols,dst,step,offset}});
 counts[0]++;counts[1]+=rows*cols*{problems[0].lhs_bytes};
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)
 put({names[lhs.backing]}+16+(dst+i)*{lhs.unit_bytes}+j*{problems[0].lhs_bytes},
 (uint32_t)get(src+(i*step*{problems[0].k}+j)*{problems[0].lhs_bytes},{problems[0].lhs_bytes}),{problems[0].lhs_bytes});
}}
static void do_weight_load(int64_t a,int64_t b,int64_t r,int64_t c,int64_t rows,int64_t cols,
 int64_t dst,int64_t step,uint8_t *src){{
 int64_t offset;int member=identify(src,1,{members},&offset);
 trace(1,member,9,(int64_t[]){{a,b,r,c,rows,cols,dst,step,offset}});
 if(member!=weight_member(dst))__builtin_trap();counts[0]++;counts[2]+=rows*cols*byte_width[member];
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)
 put(rhs_bank[member]+(dst+i)*rhs_unit[member]+j*byte_width[member],
 (uint32_t)get(src+(i*step*columns[member]+j)*byte_width[member],byte_width[member]),byte_width[member]);
}}
static void do_preload(int64_t w,int64_t k,int64_t n,int64_t acc,int64_t cont,
 int64_t rows,int64_t cols,int64_t depth){{
 int member=weight_member(w);if(pending_member!=-1||acc<acc_base[member]||acc>=acc_end[member])__builtin_trap();
 trace(2,member,8,(int64_t[]){{w,k,n,acc,cont,rows,cols,depth}});
 counts[0]++;pending_member=member;weight_row=w;acc_row=acc;continuation=cont;active_columns=cols;
}}
static void do_compute(int64_t left,int64_t m,int64_t k,int64_t rows,int64_t depth,int64_t member){{
 if(member<0||member!=pending_member)__builtin_trap();
 trace(3,member,5,(int64_t[]){{left,m,k,rows,depth}});counts[0]++;
 for(int i=0;i<rows;i++)for(int j=0;j<active_columns;j++){{
 uint8_t *target=acc_bank[member]+(acc_row+i)*acc_unit[member]+j*4;
 uint32_t value=continuation?(uint32_t)get(target,4):0;
 for(int t=0;t<depth;t++)value+=(uint32_t)get(
 {names[lhs.backing]}+16+(left+i)*{lhs.unit_bytes}+t*{problems[0].lhs_bytes},{problems[0].lhs_bytes})*
 (uint32_t)get(rhs_bank[member]+(weight_row+t)*rhs_unit[member]+j*byte_width[member],byte_width[member]);
 put(target,value,4);}}pending_member=-1;
}}
static void do_store(int64_t r,int64_t c,int64_t rows,int64_t cols,int64_t acc,uint8_t *out){{
 int64_t offset;int member=identify(out,1+{members},{members},&offset);
 if(acc<acc_base[member]||acc>=acc_end[member])__builtin_trap();
 trace(4,member,6,(int64_t[]){{r,c,rows,cols,acc,offset}});counts[0]++;counts[3]+=rows*cols*4;
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)
 put(out+(i*columns[member]+j)*4,(uint32_t)get(acc_bank[member]+(acc+i)*acc_unit[member]+j*4,4),4);
}}
uint64_t {symbol}_observe_count(int i){{if(i<0||i>=8)__builtin_trap();return counts[i];}}
uint64_t {symbol}_trace_size(void){{return trace_used;}}
int64_t {symbol}_trace(int r,int c){{if(r<0||r>=trace_used||c<0||c>=12)__builtin_trap();return trace_rows[r][c];}}
void {symbol}_observe_reset(void){{for(int i=0;i<8;i++)counts[i]=0;trace_used=0;}}
"""
    definitions = {}
    for call, _ in instances(kernel):
        if call.instr in definitions:
            continue
        pointers = [(key, value) for key, value in call.args if isinstance(value, Ptr)]
        assert len(pointers) == int(call.instr in ("lhs_load", "weight_load", "store"))
        definitions[call.instr] = InstrDef(
            call.instr,
            tuple(Operand(key, "ptr" if isinstance(value, Ptr) else "int") for key, value in call.args),
            render_c=lambda args, name=call.instr: f"do_{name}({', '.join(args)});",
        )
    iset = InstructionSet("neutral-group-byte-memory", definitions, CTYPES, "/* synchronous completed call */")
    implementation = emit_c_function(kernel, iset, symbol=symbol + "_implementation", index_bits=index_bits)
    start = implementation.index("{") + 1
    implementation = (
        implementation[:start]
        + "\n"
        + "".join(f"external[{i}]=(uint8_t*){name};\n" for i, name in enumerate(expected))
        + implementation[start:]
    )
    wrapper = _bridge(kernel, storages, sizes, names, symbol, bridge)
    return preamble + implementation + wrapper


def _bridge(kernel, storages, sizes, banks, symbol, bridge):
    """Pack actual descriptors, retaining complete guards and readonly bytes."""
    pieces = ["typedef struct {void *allocated,*aligned;int64_t offset,sizes[2],strides[2];} descriptor;\n"]
    parameters = ",".join(f"const descriptor *d{i}" for i in range(len(kernel.args)))
    pieces.append(f"void {bridge}({parameters}){{\n")
    for backing, size in sizes.items():
        pieces.append(f"for(int i=0;i<{size + 32};i++){banks[backing]}[i]=167;counts[7]+={size + 32};\n")
    for i, argument in enumerate(kernel.args):
        rows, cols = argument.shape
        ctype = CTYPES[argument.dtype]
        width = argument.nbytes // rows // cols
        pieces.append(f"""
 {ctype} storage{i}[{argument.nbytes // width + 128 // width}] __attribute__((aligned(64)));
 uint8_t *scratch{i}=(uint8_t*)storage{i};
 if(!d{i}->aligned||d{i}->offset<0||d{i}->sizes[0]!={rows}||d{i}->sizes[1]!={cols}||
 d{i}->strides[0]<=0||d{i}->strides[1]<=0)__builtin_trap();
 for(int b=0;b<sizeof(storage{i});b++)scratch{i}[b]=167;
""")
        if argument.access == "read":
            pieces.append(f"""
 for(int r=0;r<{rows};r++)for(int c=0;c<{cols};c++)for(int b=0;b<{width};b++){{
 scratch{i}[64+(r*{cols}+c)*{width}+b]=((uint8_t*)d{i}->aligned)[(d{i}->offset+r*d{i}->strides[0]+c*d{i}->strides[1])*{width}+b];
 counts[4]++;counts[5]++;}}
""")
    parameters = ",".join(f"({CTYPES[a.dtype]}*)(scratch{i}+64)" for i, a in enumerate(kernel.args))
    pieces.append(f"{symbol}_implementation({parameters});\nif(pending_member!=-1)__builtin_trap();\n")
    for i, argument in enumerate(kernel.args):
        rows, cols = argument.shape
        width = argument.nbytes // rows // cols
        pieces.append(
            f"for(int b=0;b<sizeof(storage{i});b++)if((b<64||b>={argument.nbytes + 64})"
            f"&&scratch{i}[b]!=167)__builtin_trap();\n"
        )
        if argument.access == "read":
            pieces.append(f"""
 for(int r=0;r<{rows};r++)for(int c=0;c<{cols};c++)for(int b=0;b<{width};b++)
 if(scratch{i}[64+(r*{cols}+c)*{width}+b]!=((uint8_t*)d{i}->aligned)[(d{i}->offset+r*d{i}->strides[0]+c*d{i}->strides[1])*{width}+b])__builtin_trap();
""")
        else:
            assert argument.access == "write"
            pieces.append(f"""
 for(int r=0;r<{rows};r++)for(int c=0;c<{cols};c++)for(int b=0;b<{width};b++){{
 ((uint8_t*)d{i}->aligned)[(d{i}->offset+r*d{i}->strides[0]+c*d{i}->strides[1])*{width}+b]=scratch{i}[64+(r*{cols}+c)*{width}+b];
 counts[4]++;counts[5]++;}}
""")
    views = (storages[0].lhs, *(s.rhs for s in storages), *(s.accumulator for s in storages))
    for backing, size in sizes.items():
        allowed = "||".join(
            f"(i>={16 + v.base * v.unit_bytes}&&i<{16 + (v.base + v.capacity) * v.unit_bytes})"
            for v in views
            if v.backing == backing
        )
        pieces.append(f"for(int i=0;i<{size + 32};i++)if(!({allowed})&&{banks[backing]}[i]!=167)__builtin_trap();\n")
    pieces.append(f"for(int i=0;i<{len(kernel.args)};i++)external[i]=0;counts[6]++;}}\n")
    return "".join(pieces)


def check_actual_trace(records, problems, storages):
    """Invert runtime rows in one persistent physical-byte provenance map.

    Member checks never reset memory. An intervening RHS/accumulator overwrite
    of a retained LHS is therefore observable even for aliasing physical views.
    """
    memory, reductions, outputs, lhs_reads = {}, set(), set(), {}
    pending = None
    counts = [0] * 4
    names = (
        ("a", "b", "r", "c", "rows", "cols", "dst", "step", "pointer"),
        ("a", "b", "r", "c", "rows", "cols", "dst", "step", "pointer"),
        ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth"),
        ("lhs", "m", "k", "rows", "depth"),
        ("r", "c", "rows", "cols", "acc", "pointer"),
    )

    def extent(start, size, tile):
        assert 0 <= start < size and start % tile == 0, "invalid logical tile origin"
        return min(tile, size - start)

    def physical(view, row, rows, columns, width):
        assert view.base <= row and row + rows <= view.base + view.capacity
        assert row % view.alignment == 0 and 0 < columns * width <= view.unit_bytes
        return [
            (i, j, b, (view.backing, (row + i) * view.unit_bytes + j * width + b))
            for i in range(rows)
            for j in range(columns)
            for b in range(width)
        ]

    for record in records:
        op, member, length, *values = record
        assert 0 <= op < 5 and length == len(values) == len(names[op])
        assert all(type(v) is int and v >= 0 for v in values)
        assert member == -1 if op == 0 else 0 <= member < len(problems)
        assert pending is None or op == 3, "preload is not adjacent to actual compute"
        problem, storage = problems[max(0, member)], storages[max(0, member)]
        a = dict(zip(names[op], values, strict=True))
        counts[0] += 1
        if op in (0, 1):
            left = op == 0
            shape = (problem.m, problem.k) if left else (problem.k, problem.n)
            tile = (problem.tile_m, problem.tile_k) if left else (problem.tile_k, problem.tile_n)
            width, view = (problem.lhs_bytes, storage.lhs) if left else (problem.rhs_bytes, storage.rhs)
            assert (a["r"], a["c"]) == (a["a"] * tile[0], a["b"] * tile[1])
            assert a["rows"] == extent(a["r"], shape[0], tile[0])
            assert a["cols"] == extent(a["c"], shape[1], tile[1]) and a["step"] == 1
            assert a["pointer"] == (a["r"] * shape[1] + a["c"]) * width, "wrong actual external byte pointer"
            for i, j, b, address in physical(view, a["dst"], a["rows"], a["cols"], width):
                source = ("lhs" if left else "rhs", member, a["r"] + i, a["c"] + j, b)
                memory[address] = source
                if left:
                    lhs_reads[source[2:]] = lhs_reads.get(source[2:], 0) + 1
            counts[1 if left else 2] += a["rows"] * a["cols"] * width
        elif op == 2:
            assert a["continue"] in (0, 1)
            pending = member, a
        elif op == 3:
            assert pending is not None and pending[0] == member
            p = pending[1]
            assert (a["k"], a["rows"], a["depth"]) == (p["k"], p["rows"], p["depth"])
            assert a["rows"] == extent(a["m"] * problem.tile_m, problem.m, problem.tile_m)
            assert a["depth"] == extent(a["k"] * problem.tile_k, problem.k, problem.tile_k)
            assert p["cols"] == extent(p["n"] * problem.tile_n, problem.n, problem.tile_n)
            assert p["continue"] == int(a["k"] != 0)
            key = member, a["m"], p["n"], a["k"]
            assert key not in reductions
            reductions.add(key)
            for i, t, b, address in physical(storage.lhs, a["lhs"], a["rows"], a["depth"], problem.lhs_bytes):
                assert memory.get(address) == (
                    "lhs",
                    -1,
                    a["m"] * problem.tile_m + i,
                    a["k"] * problem.tile_k + t,
                    b,
                ), "retained LHS overwritten or wrong source"
            for t, j, b, address in physical(storage.rhs, p["weight"], a["depth"], p["cols"], problem.rhs_bytes):
                assert memory.get(address) == (
                    "rhs",
                    member,
                    a["k"] * problem.tile_k + t,
                    p["n"] * problem.tile_n + j,
                    b,
                ), "wrong member RHS provenance"
            for i, j, b, address in physical(storage.accumulator, p["acc"], a["rows"], p["cols"], 4):
                owner = ("acc", member, a["m"] * problem.tile_m + i, p["n"] * problem.tile_n + j)
                if p["continue"]:
                    assert memory.get(address) == (*owner, a["k"] * problem.tile_k, b)
                memory[address] = (*owner, a["k"] * problem.tile_k + a["depth"], b)
            pending = None
        else:
            assert a["rows"] == extent(a["r"], problem.m, problem.tile_m)
            assert a["cols"] == extent(a["c"], problem.n, problem.tile_n)
            assert a["pointer"] == (a["r"] * problem.n + a["c"]) * 4
            for i, j, b, address in physical(storage.accumulator, a["acc"], a["rows"], a["cols"], 4):
                item = member, a["r"] + i, a["c"] + j, b
                assert memory.get(address) == ("acc", *item[:3], problem.k, b), "incomplete ordered reduction"
                assert item not in outputs
                outputs.add(item)
            counts[3] += a["rows"] * a["cols"] * 4
    assert pending is None
    assert len(outputs) == sum(p.m * p.n * 4 for p in problems)
    assert len(reductions) == sum(p.tiles[0] * p.tiles[1] * p.tiles[2] for p in problems)
    return {
        "commands": counts[0],
        "primitive_lhs_input_payload_bytes": counts[1],
        "primitive_rhs_input_payload_bytes": counts[2],
        "primitive_output_payload_bytes": counts[3],
        "complete_ordered_reductions": len(reductions),
        "lhs_coordinate_read_counts": [[*coordinate, count] for coordinate, count in sorted(lhs_reads.items())],
        "scope": "actual native primitive byte effects; no physical bus or timing claim",
    }


def test_runtime_trace_keeps_physical_provenance_across_members():
    from dataclasses import replace

    import pytest

    from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
    from merlin.transforms.dense_planning import DenseProblem

    problem = DenseProblem(2, 2, 2, 2, 2, 2, 1, 1, 4)
    left = DenseMemoryView("lhs", "left", 2, 2, 2)
    storage = tuple(
        DenseLoopStorage(
            left,
            DenseMemoryView(f"rhs{i}", "right", 2, i * 2, 2),
            DenseMemoryView(f"acc{i}", "sums", 8, i * 2, 2),
        )
        for i in range(2)
    )
    records = [[0, -1, 9, 0, 0, 0, 0, 2, 2, 2, 1, 0]]
    for member in range(2):
        row = member * 2
        records.extend(
            (
                [1, member, 9, 0, 0, 0, 0, 2, 2, row, 1, 0],
                [2, member, 8, row, 0, 0, row, 0, 2, 2, 2],
                [3, member, 5, 2, 0, 0, 2, 2],
                [4, member, 6, 0, 0, 2, 2, row, 0],
            )
        )
    effects = check_actual_trace(records, (problem, problem), storage)
    assert effects["commands"] == 9
    assert effects["primitive_lhs_input_payload_bytes"] == 4
    assert effects["primitive_rhs_input_payload_bytes"] == 8
    assert effects["primitive_output_payload_bytes"] == 32
    assert all(row[-1] == 1 for row in effects["lhs_coordinate_read_counts"])
    alias = (storage[0], replace(storage[1], rhs=replace(storage[1].rhs, backing="left")))
    with pytest.raises(AssertionError, match="LHS overwritten"):
        check_actual_trace(records, (problem, problem), alias)
    wrong = [row[:] for row in records]
    wrong[5][-1] = 1
    with pytest.raises(AssertionError, match="external byte pointer"):
        check_actual_trace(wrong, (problem, problem), storage)


def original_model(directory, case):
    """Select one fixed fact/SW envelope before any captured source discovery."""
    from dataclasses import replace

    from test_selected_dense_contracts import inputs

    from merlin.compile.scheduling.compact_dense import DenseLoopStorage

    def declare(facts, software, dense):
        # These explicit test hardware capacities cover every reviewed window;
        # model discovery never derives or expands a physical capability.
        if case.shared:
            facts["capacities"] = {"local": 1568}
        else:
            facts["capacities"]["right"] = 27 * facts["row_bytes"]["rhs"]
            facts["capacities"]["sums"] = 25 * facts["row_bytes"]["accumulator"]
        dense["schema"] = "merlin.dense_schedule.v2"
        dense["external_pointer_arguments"] = {
            kind: {"argument": "buffer", "space": space}
            for kind, space in (("lhs_load", "input_lhs"), ("rhs_load", "input_rhs"), ("store", "output"))
        }
        dense["logical_pointer_abi"] = {
            "schema": "merlin.logical_pointer_abi.v1",
            "layout": "dense_row_major",
            "alignment_bytes": [1, 1, 1],
            "access": ["read_only", "read_only", "fully_written"],
            "bounds": "logical_elements",
            "completion": "synchronous",
        }
        facts["word_bits"] = 8 + 9 * 64
        facts["offsets"] = [8 + 64 * i for i in range(9)]
        facts["padding_offsets"] = [8 + 64 * i for i in range(10)]
        facts["padding_widths"] = [64 * (9 - i) for i in range(10)]
        for instruction in software["isa"]["instructions"]:
            name = instruction["name"]
            if name in ("lhs_load", "weight_load", "store"):
                space = {"lhs_load": "input_lhs", "weight_load": "input_rhs", "store": "output"}[name]
                instruction["operands"].append(
                    {
                        "name": "buffer",
                        "unit": "bytes",
                        "type": {
                            "kind": "address",
                            "signed": False,
                            "bits": {"fact": ["facts", "arg_bits"]},
                            "space": space,
                        },
                    }
                )
            if name == "compute":
                instruction["operands"].append(
                    {
                        "name": "member",
                        "unit": "integer",
                        "type": {"kind": "integer", "signed": True, "bits": {"fact": ["facts", "arg_bits"]}},
                    }
                )
            count = len(instruction["operands"])
            instruction["encoding"] = instruction["encoding"][:1] + [
                {
                    "offset": {"fact": ["facts", "offsets", i]},
                    "width": {"fact": ["facts", "arg_bits"]},
                    "argument": argument["name"],
                }
                for i, argument in enumerate(instruction["operands"])
            ]
            if count < 9:
                instruction["encoding"].append(
                    {
                        "offset": {"fact": ["facts", "padding_offsets", count]},
                        "width": {"fact": ["facts", "padding_widths", count]},
                        "fixed": {"fact": ["facts", "origin"]},
                    }
                )

    selected, _, storage, adapter = inputs(
        directory,
        tile=case.tile,
        widths=(case.width, case.width, 4),
        shape=case.shape[:3],
        shared=case.shared,
        mutate=declare,
    )
    lhs = replace(storage.lhs, name="lhs")
    storages = tuple(
        DenseLoopStorage(
            lhs,
            replace(storage.rhs, name=f"rhs{i}", base=(16 + i * 12) if case.shared else (3 + i * 12)),
            replace(storage.accumulator, name=f"acc{i}", base=(32 + i * 12) if case.shared else (1 + i * 12)),
        )
        for i in range(2)
    )
    adapter = replace(adapter, qualification_identity="neutral-reviewed-shared-lhs-byte-machine-v1")
    return selected, storages, adapter


class OriginalGroupBindings:
    """Original typed transport and inverse decoding for the fixed test ISA."""

    def __init__(self, problems, storages):
        self.problems, self.storages = problems, storages

    @staticmethod
    def single_encode(operation):
        from dataclasses import replace

        from test_compact_dense import encode

        from merlin.compile.scheduling.block_schedule import Compute
        from merlin.sched.ir import Const

        call = encode(operation)
        # Source-independent single-region pointer materialization belongs to
        # the selected v2 dense contract, which knows its actual source shape.
        return call if not isinstance(operation, Compute) else replace(call, args=(*call.args, ("member", Const(0))))

    @staticmethod
    def single_project(call):
        from test_compact_dense import project

        return project(call)

    def encode_member(self, member, operation):
        from dataclasses import replace

        from test_compact_dense import encode

        from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
        from merlin.sched.ir import Const, as_expr

        call = encode(operation)
        if isinstance(operation, Compute):
            return replace(call, args=(*call.args, ("member", Const(member))))
        if isinstance(operation, Preload):
            return call
        if isinstance(operation, Load):
            if operation.role == "lhs":
                assert member is None
                name, stride, width = "lhs", self.problems[0].k, self.problems[0].lhs_bytes
            else:
                assert member in (0, 1)
                name, stride, width = f"rhs{member}", self.problems[member].n, self.problems[member].rhs_bytes
        else:
            assert isinstance(operation, Store)
            name, stride, width = f"output{member}", self.problems[member].n, 4
        offset = (as_expr(operation.dram_row) * stride + as_expr(operation.dram_col)) * width
        return replace(call, args=(*call.args, ("buffer", Ptr(name, offset))))

    def project_member(self, call):
        from test_compact_dense import project

        from merlin.sched.ir import Const

        operation = project(call)
        values = dict(call.args)
        if call.instr == "lhs_load":
            assert isinstance(values["buffer"], Ptr) and values["buffer"].tensor == "lhs"
            member = None
        elif call.instr in ("weight_load", "store"):
            pointer = values["buffer"]
            assert isinstance(pointer, Ptr)
            prefix = "rhs" if call.instr == "weight_load" else "output"
            assert pointer.tensor in (prefix + "0", prefix + "1")
            member = int(pointer.tensor[-1])
        elif call.instr == "compute":
            assert isinstance(values["member"], Const) and values["member"].value in (0, 1)
            member = values["member"].value
        else:
            assert call.instr == "preload"
            row = values["acc"]
            # The original ISA exposes actual accumulator-row selectors. Its
            # closed member disambiguator is the constant term of that affine
            # selector; the verifier independently bounds the complete span.
            from merlin.sched.ir import evaluate

            candidates = []
            for i, storage in enumerate(self.storages):
                try:
                    value = evaluate(row, {})
                except ValueError:
                    # Loop variables affect rows inside a fixed selected slot.
                    from merlin.sched.ir.expr import free_vars

                    value = evaluate(row, {name: 0 for name in free_vars(row)})
                if storage.accumulator.base <= value < storage.accumulator.base + storage.accumulator.capacity:
                    candidates.append(i)
            assert len(candidates) == 1
            member = candidates[0]
        return member, operation


class SharedLHSProvider:
    """Original test-only source binding and retained neutral native objects."""

    def __init__(self, work, model, trace, *, grouped):
        from dataclasses import replace

        self.work, self.trace, self.grouped = work, trace, grouped
        self.selected, self.storages, adapter = model
        first = trace[0]
        self.problems = tuple(self.selected.dense.problem(*first[0].shape, weight.shape[1]) for weight in first[1:3])
        self.transport = OriginalGroupBindings(self.problems, self.storages)
        self.adapter = replace(adapter, encode=self.transport.single_encode, project=self.transport.single_project)
        self.records = []
        work.mkdir()

    def selection(self):
        from merlin.compile.scheduling.compact_dense_group_check import DenseLHSReuseLaw
        from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
        from merlin.llvmlower.source_dense import SourceDenseLowering
        from merlin.llvmlower.source_dense_group import SourceDenseGrouping

        # Premises concern the closed scalar C template above on admitted
        # bounded descriptors. Private diagnostics observe costs only; they
        # are not part of the captured model's logical state or output ABI.
        law = DenseLHSReuseLaw(
            self.adapter.configuration_identity,
            self.adapter.numerical_policy_identity,
            self.adapter.qualification_identity,
            "neutral-reviewed-shared-lhs-retention-and-load-omission-v1",
            lhs_unchanged_by_member_operations=True,
            independent_member_interleaving=True,
            suppressed_loads_no_faults=True,
            suppressed_loads_no_extra_observations=True,
        )
        group = SourceDenseGrouping(
            self.storages, self.transport.encode_member, self.transport.project_member, law, max_members=2
        )
        return SourceDenseLowering(
            selected_contracts=self.selected,
            storage=self.storages[0],
            adapter=self.adapter,
            execution_domain="host",
            allocation_alignment=64,
            static_budget=256,
            prepare=self.prepare,
            build=self.build,
            private_ownership=True,
            external_buffers_disjoint=True,
            max_source_elements=4096,
            verification_timeout_ms=20_000,
            borrowed_effects=BorrowedBufferEffects("neutral-reviewed-shared-lhs-descriptor-v1", True, True, True),
            lhs_grouping=group if self.grouped else None,
        )

    def prepare(self, bindings, directory):
        from merlin.common.digest import sha256_file

        self.bindings = tuple(bindings)
        assert len(bindings) == (1 if self.grouped else 2)
        for i, binding in enumerate(bindings):
            result = binding.compilation
            kernel = result.compiled.kernel
            problems = tuple(
                self.selected.dense.problem(*kernel.args[0].shape, a.shape[1])
                for a in kernel.args
                if a.access == "write"
            )
            storages = self.storages if self.grouped else (self.storages[0],)
            assert len(problems) == len(storages)
            source = self.work / f"provider{i}.c"
            symbol = binding.writer.symbol + "_primitive"
            bridge = "_mlir_ciface_" + binding.writer.borrowed_symbol
            source.write_text(
                machine_source(kernel, problems, storages, symbol, bridge, index_bits=result.compiled.index_bits)
            )
            self.records.append(
                {
                    "result": result,
                    "kernel": kernel,
                    "problems": problems,
                    "storages": storages,
                    "symbol": symbol,
                    "bridge": bridge,
                    "source": source,
                    "source_sha256": sha256_file(source),
                }
            )
        self.qualify()

    def qualify(self):
        import ctypes
        import json

        import numpy as np
        from test_pointwise_source_provider import Descriptor

        from merlin.common import invocation_record as invocation
        from merlin.common.digest import sha256_file
        from merlin.llvmlower import toolchain

        sources = tuple(item["source"] for item in self.records)
        library = self.work / "qualification.so"
        compiler = toolchain.clang().resolve()
        argv = [str(compiler), "-O2", "-fno-builtin", "-shared", "-fPIC", *map(str, sources), "-o", str(library)]
        invocation.run(
            argv,
            directory=self.work,
            stage="neutral-group-qualification",
            inputs=sources,
            outputs=(library,),
            capture_output=True,
            check=True,
        )
        image = ctypes.CDLL(str(library))
        observed = []
        for index, item in enumerate(self.records):
            symbol = item["symbol"]
            size = getattr(image, symbol + "_trace_size")
            size.restype = ctypes.c_uint64
            query = getattr(image, symbol + "_trace")
            query.argtypes = [ctypes.c_int, ctypes.c_int]
            query.restype = ctypes.c_int64
            function = getattr(image, item["bridge"])
            function.argtypes = [ctypes.POINTER(Descriptor)] * len(item["kernel"].args)
            for epoch, inputs in enumerate(self.trace):
                operands = inputs[:3] if self.grouped else (inputs[0], inputs[1 + index])
                expected = inputs[3:] if self.grouped else (inputs[3 + index],)
                buffers, originals, descriptors = [], [], []
                for array in (*operands, *[np.full_like(value, 71) for value in expected]):
                    rows, columns = array.shape
                    owner = np.full((rows + 2, columns * 2 + 5), 71, array.dtype)
                    owner[1 : rows + 1, 1 : 2 * columns + 1 : 2] = array
                    buffers.append(owner)
                    originals.append(owner.copy())
                    descriptors.append(
                        Descriptor(
                            owner.ctypes.data, owner.ctypes.data, 2 * columns + 6, (rows, columns), (2 * columns + 5, 2)
                        )
                    )
                start = size()
                function(*(ctypes.byref(d) for d in descriptors))
                rows = [[query(r, c) for c in range(3 + query(r, 2))] for r in range(start, size())]
                effects = check_actual_trace(rows, item["problems"], item["storages"])
                for actual, original in zip(buffers[: len(operands)], originals[: len(operands)], strict=True):
                    np.testing.assert_array_equal(actual, original)
                for offset, value in enumerate(expected, len(operands)):
                    originals[offset][1 : value.shape[0] + 1, 1 : 2 * value.shape[1] + 1 : 2] = value
                    np.testing.assert_array_equal(buffers[offset], originals[offset])
                raw = self.work / f"qualification-{index}-{epoch}.npz"
                np.savez(raw, **{f"owner{i}": a for i, a in enumerate(buffers)})
                observed.append(
                    {
                        "binding": index,
                        "epoch": epoch,
                        "effects": effects,
                        "raw_buffers": {"path": str(raw), "sha256": sha256_file(raw)},
                        "actual_trace": rows,
                    }
                )
        for receipt in self.work.glob("invocations/*/invocation.json"):
            invocation.verify(receipt)
        self.witness = self.work / "qualification.json"
        self.witness.write_text(
            json.dumps(
                {
                    "scope": (
                        "neutral primitive and descriptor functional checks; "
                        "original no-retention/completion premises separately declared"
                    ),
                    "compiler": {"path": str(compiler), "sha256": sha256_file(compiler)},
                    "sources": {str(p): sha256_file(p) for p in sources},
                    "image": {"path": str(library), "sha256": sha256_file(library)},
                    "observations": observed,
                },
                indent=2,
            )
        )

    def build(self, bindings, context):
        from dataclasses import replace
        from pathlib import Path

        from merlin.common.digest import sha256_file
        from merlin.runtime.host_provider import HostProviderBuild, HostProviderObject, read_host_function_abi

        assert tuple(bindings) == self.bindings
        context = replace(context, compiler_flags=(*context.compiler_flags, "-fno-builtin"))
        objects = []
        for item in self.records:
            item["result"].revalidate()
            source, symbol = item["source"], item["symbol"]
            assert sha256_file(source) == item["source_sha256"]
            llvm, obj = context.workdir / (symbol + ".ll"), context.workdir / (symbol + ".o")
            commands = []
            for flags, destination in ((("-S", "-emit-llvm"), llvm), (("-c",), obj)):
                argv = tuple(map(str, (context.compiler, *context.compiler_flags, *flags, source, "-o", destination)))
                context.compile(argv, inputs=(source,), output=destination)
                commands.append(argv)
            defined = (
                item["bridge"],
                *(symbol + suffix for suffix in ("_observe_count", "_observe_reset", "_trace_size", "_trace")),
            )
            objects.append(
                HostProviderObject(
                    obj,
                    sha256_file(obj),
                    llvm,
                    sha256_file(llvm),
                    tuple(read_host_function_abi(llvm, s) for s in defined),
                    (),
                    tuple((p, sha256_file(p)) for p in (source, context.compiler.resolve())),
                    tuple(commands),
                    Path.cwd(),
                )
            )
            item.update(llvm=llvm, object=obj)
        return HostProviderBuild(
            sha256_file(context.prepared_path),
            sha256_file(context.model_llvm_path),
            sha256_file(context.model_object_path),
            tuple(objects),
            sha256_file(self.witness),
            sha256_file(self.witness),
        )
