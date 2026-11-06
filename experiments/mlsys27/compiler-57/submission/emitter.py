"""LLVM dialect emission with bounded command-issue loops and internal storage."""
import json
import math
import struct
from rtl_isa import CUSTOM_OPCODE, FUNCT3, LEGAL_FUNCT


def storage_bytes(dtype):
    aliases={'int8':'i8','int16':'i16','int32':'i32','int64':'i64',
             'uint8':'u8','uint16':'u16','uint32':'u32','uint64':'u64'}
    dtype=aliases.get(dtype,dtype)
    return {'i1':1,'i8':1,'u8':1,'i16':2,'u16':2,'f16':2,'bf16':2,
            'i32':4,'u32':4,'f32':4,'i64':8,'u64':8,'f64':8}.get(dtype, 1)


def abi_order(cb):
    if cb.get('kernel_abi', {}).get('kind') == 'whole_program':
        return [a['tensor'] for a in cb['kernel_abi']['args']]
    produced = {c.get('operands', {}).get('dst') for c in cb['commands']}
    inputs = [n for n,t in cb.get('tensors', {}).items() if n not in produced and t.get('role') not in ('output','intermediate')]
    # The default caller uses publication order. `outputs` selects the public
    # results; its array order does not override the selected pointer ABI.
    selected = set(cb['outputs']) if 'outputs' in cb else None
    outputs = []
    published = ('COMMIT','BIAS_ADD','RESIDUAL_ADD','ATTENTION_QK',
                 'ATTENTION_PV','CONV2D','BATCHED_MATMUL','MOVEMENT')
    for command in cb['commands']:
        name = command.get('operands', {}).get('dst')
        if command['opcode'] in published and name in cb.get('tensors', {}):
            if selected is not None and name not in selected:
                continue
            if cb['tensors'][name].get('role') != 'intermediate' and name not in outputs:
                outputs.append(name)
    outputs += [name for name,tensor in cb.get('tensors', {}).items()
                if tensor.get('role') == 'output' and name not in outputs
                and (selected is None or name in selected)]
    return inputs + outputs


class Emitter:
    def __init__(self, cb):
        self.cb = cb
        self.tensors = {n:dict(t) for n,t in cb.setdefault('tensors', {}).items()}
        self.names = abi_order(cb)
        self.pointers = {}
        self.lines = []
        self.prologue = []
        self.globals = []
        self.declarations = {}
        self.serial = 0
        self.instruction_count = 0
        self.schedule_count = 0
        self.current_task = None
        self.owner_tags = []
        self.current_source = None
        self.current_region = None
        self.current_stage = None
        self.source_tags = []
        self._scale_words = {}
        for i,n in enumerate(self.names):
            v = self.fresh()
            self.prologue.append(f'    {v} = llvm.ptrtoint %arg{i} : !llvm.ptr to i64')
            self.pointers[n] = v
        for n,t in list(self.tensors.items()):
            if n not in self.pointers and t.get('role') == 'intermediate':
                self._storage(n, math.prod(t['shape'])*storage_bytes(t['dtype']))
        if not cb.get('declined'):
            self.fence()
            self.issue(7, 0, 0)
            self.issue(0, (0x3f800000 << 32) | (1 << 16) | 4, 1 << 48)
        self._contraction_started = True

    def fresh(self):
        self.serial += 1
        return f'%v{self.serial}'

    def declare_function(self,name,argtypes,resulttype):
        signature=(tuple(argtypes),resulttype)
        if name in self.declarations and self.declarations[name]!=signature:
            raise ValueError('inconsistent external host function signature for '+name)
        self.declarations[name]=signature

    def emit(self, text):
        self.lines.append('    ' + text)
        self.owner_tags.append(self.current_task)
        self.source_tags.append((self.current_source,self.current_region,self.current_stage))
        self.schedule_count += 1

    def const(self, x):
        if isinstance(x,str):
            return x
        v=self.fresh()
        x=int(x)&((1<<64)-1)
        if x >= 1<<63: x-=1<<64
        self.emit(f'{v} = llvm.mlir.constant({x} : i64) : i64')
        return v

    def binary(self, op, a, b):
        if isinstance(a,int) and isinstance(b,int):
            if op=='add': return a+b
            if op=='mul': return a*b
            if op=='sub': return a-b
            if op=='or': return a|b
            if op=='and': return a&b
            if op=='shl': return a<<b
            if op=='udiv': return a//b
            if op=='urem': return a%b
        aa,bb=self.const(a),self.const(b)
        v=self.fresh()
        self.emit(f'{v} = llvm.{op} {aa}, {bb} : i64')
        return v

    def add(self,a,b): return self.binary('add',a,b)
    def mul(self,a,b): return self.binary('mul',a,b)
    def sub(self,a,b): return self.binary('sub',a,b)
    def or_(self,a,b): return self.binary('or',a,b)
    def and_(self,a,b): return self.binary('and',a,b)
    def shl(self,a,b): return self.binary('shl',a,b)
    def udiv(self,a,b): return self.binary('udiv',a,b)
    def urem(self,a,b): return self.binary('urem',a,b)

    def cmp(self,predicate,a,b):
        a,b=self.const(a),self.const(b)
        v=self.fresh()
        self.emit(f'{v} = llvm.icmp "{predicate}" {a}, {b} : i64')
        return v

    def select(self,predicate,yes,no):
        yes,no=self.const(yes),self.const(no)
        v=self.fresh()
        self.emit(f'{v} = llvm.select {predicate}, {yes}, {no} : i1, i64')
        return v

    def ptr(self, name, byteoffset=0):
        if name not in self.pointers:
            t=self.tensors[name]
            self._storage(name, math.prod(t['shape'])*storage_bytes(t['dtype']))
        return self.add(self.pointers[name], byteoffset) if byteoffset != 0 else self.pointers[name]

    def _storage(self,name,size,data=None):
        sym='buffer_'+str(len(self.globals))
        if data is None:
            value=f'dense<0> : tensor<{max(size,1)}xi8>'
        else:
            value='"'+''.join('\\'+format(b,'02X') for b in data)+'"'
        self.globals.append(f'  llvm.mlir.global internal {"constant " if data is not None else ""}@{sym}({value}) {{alignment = 64 : i64}} : !llvm.array<{max(size,1)} x i8>')
        p=self.fresh(); self.prologue.append(f'    {p} = llvm.mlir.addressof @{sym} : !llvm.ptr')
        v=self.fresh(); self.prologue.append(f'    {v} = llvm.ptrtoint {p} : !llvm.ptr to i64')
        self.pointers[name]=v

    def global_data(self,name,values,dtype):
        if name in self.pointers: return name
        fmt={'i8':'b','u8':'B','i16':'h','i32':'i','i64':'q','f32':'f'}[dtype]
        raw=b''.join(struct.pack('<'+fmt,x) for x in values)
        self.tensors.setdefault(name,{'shape':[len(values)],'dtype':dtype,'role':'intermediate'})
        self._storage(name,len(raw),raw)
        return name

    def load_scale_word(self,name):
        if name not in self._scale_words:
            p,v,w=self.fresh(),self.fresh(),self.fresh()
            self.prologue.extend([
                f'    {p} = llvm.inttoptr {self.pointers[name]} : i64 to !llvm.ptr',
                f'    {v} = llvm.load {p} : !llvm.ptr -> i32',
                f'    {w} = llvm.zext {v} : i32 to i64'])
            self._scale_words[name]=w
        return self._scale_words[name]

    def issue(self,funct,rs1=0,rs2=0):
        if funct not in LEGAL_FUNCT:
            raise ValueError('instruction funct is absent from selected RTL decode facts')
        a,b=self.const(rs1),self.const(rs2)
        asm=f'.insn r {hex(CUSTOM_OPCODE)}, {FUNCT3}, {funct}, x0, $0, $1'
        self.emit(f'"llvm.inline_asm"({a}, {b}) <{{asm_string = {json.dumps(asm)}, constraints = "r,r,~{{memory}}", has_side_effects}}> : (i64, i64) -> ()')
        self.instruction_count += 1

    def fence(self):
        self.emit('"llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}> : () -> ()')

    def loop(self,start,end,step,callback):
        if isinstance(start,int) and isinstance(end,int) and end<=start: return
        # One iteration needs no control flow.
        if all(isinstance(x,int) for x in (start,end,step)) and end-start<=step:
            callback(start); return
        a,b,s=self.const(start),self.const(end),self.const(step)
        head='loop'+str(self.serial); body='body'+str(self.serial); tail='exit'+str(self.serial)
        iv=self.fresh()
        self.emit(f'llvm.br ^{head}({a} : i64)')
        self.lines.append(f'  ^{head}({iv}: i64):')
        cond=self.fresh(); self.emit(f'{cond} = llvm.icmp "slt" {iv}, {b} : i64')
        self.emit(f'llvm.cond_br {cond}, ^{body}, ^{tail}')
        self.lines.append(f'  ^{body}:')
        callback(iv)
        nxt=self.add(iv,s)
        self.emit(f'llvm.br ^{head}({nxt} : i64)')
        self.lines.append(f'  ^{tail}:')

    def render(self):
        self.fence()
        self.emit('llvm.return')
        args=', '.join(f'%arg{i}: !llvm.ptr' for i in range(len(self.names)))
        declarations='\n'.join(f'  llvm.func @{name}('+', '.join(types)+')'+('' if result=='void' else ' -> '+result) for name,(types,result) in self.declarations.items())
        text='module {\n'+'\n'.join(self.globals)+'\n'+declarations+f'\n  llvm.func @kernel_entry({args}) {{\n'+'\n'.join(self.prologue+self.lines)+'\n  }\n  llvm.func @kernel_complete() {\n    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}> : () -> ()\n    llvm.return\n  }\n}\n'
        task_owned=any(t is not None for t in self.owner_tags)
        if task_owned or any(t[0] is not None for t in self.source_tags):
            from io import StringIO
            from xdsl.context import Context
            from xdsl.dialects import builtin,llvm
            from xdsl.dialects.builtin import IntegerAttr,StringAttr
            from xdsl.parser import Parser
            from xdsl.printer import Printer
            context=Context();context.load_dialect(builtin.Builtin);context.load_dialect(llvm.LLVM)
            module=Parser(context,text).parse_module()
            fn=next(o for o in module.body.block.ops if o.name=='llvm.func' and o.sym_name.data=='kernel_entry')
            ops=[o for block in fn.body.blocks for o in block.ops]
            first=next((i for i,t in enumerate(self.owner_tags) if t is not None),0)
            tags=([-1]*len(self.prologue)+[t if t is not None else (-1 if i<first else -2) for i,t in enumerate(self.owner_tags)]) if task_owned else [None]*len(ops)
            sources=[(None,None,None)]*len(self.prologue)+self.source_tags
            if len(ops)!=len(tags):raise ValueError('LLVM ownership schedule lost an operation')
            for op,tag,(source,region,stage) in zip(ops,tags,sources):
                if tag is not None:op.attributes['merlin.global_task']=IntegerAttr(tag,64)
                if source is not None:op.attributes['merlin.source_op_index']=IntegerAttr(source,64)
                if region is not None:op.attributes['prov.region_id']=StringAttr(region)
                if stage is not None:op.attributes['prov.composition_stage']=StringAttr(stage)
            module.verify()
            stream=StringIO();Printer(stream=stream,print_generic_format=True).print_op(module)
            text=stream.getvalue()+'\n'
        return text
