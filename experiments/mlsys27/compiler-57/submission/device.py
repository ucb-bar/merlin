"""Agent-authored Gemmini instruction lowering and compact LLVM IR construction.

Device descriptors implement arithmetic; scalar instructions only calculate addresses,
loop bounds and instruction words. Layout fields follow the selected RTL bundle facts.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import struct
import subprocess

DIM = 16
ACC = 1 << 31
ACCUM = 1 << 30
FULL = 1 << 29
GARBAGE = (1 << 32) - 1


def fbits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


class Emitter:
    def __init__(self, args):
        self.args = list(args)
        self.pointers = {name: '%arg' + str(i) for i, name in enumerate(args)}
        self.lines = []
        self.allocations = []
        self.counter = 0
        self.block = 'entry'
        self.instruction_count = 0
        self.declarations = set()
        self.globals = []
        self.stack_bytes = 0
        self.shared_input_pointers = set()
        self._shared_a_frame_key = None

    def fresh(self):
        self.counter += 1
        return '%v' + str(self.counter)

    def label_name(self):
        self.counter += 1
        return 'b' + str(self.counter)

    def emit(self, line):
        self.lines.append('  ' + line)

    def value(self, instruction):
        result = self.fresh()
        self.emit(result + ' = ' + instruction)
        return result

    def op(self, op, a, b):
        if isinstance(a, int) and isinstance(b, int):
            if op == 'add': return a + b
            if op == 'sub': return a - b
            if op == 'mul': return a * b
            if op == 'udiv': return a // b
            if op == 'urem': return a % b
            if op == 'and': return a & b
            if op == 'or': return a | b
            if op == 'shl': return a << b
        return self.value(f'{op} i64 {a}, {b}')

    def add(self, a, b): return self.op('add', a, b)
    def sub(self, a, b): return self.op('sub', a, b)
    def mul(self, a, b): return self.op('mul', a, b)
    def udiv(self, a, b): return self.op('udiv', a, b)
    def urem(self, a, b): return self.op('urem', a, b)

    def icmp(self, pred, a, b, dtype='i64'):
        return self.value(f'icmp {pred} {dtype} {a}, {b}')

    def select(self, cond, yes, no, dtype='i64'):
        return self.value(f'select i1 {cond}, {dtype} {yes}, {dtype} {no}')

    def min(self, a, b):
        if isinstance(a, int) and isinstance(b, int): return min(a, b)
        return self.select(self.icmp('ult', a, b), a, b)

    def max(self, a, b):
        if isinstance(a, int) and isinstance(b, int): return max(a, b)
        return self.select(self.icmp('sgt', a, b), a, b)

    def gep(self, ptr, byteoffset):
        if byteoffset == 0: return ptr
        return self.value(f'getelementptr i8, ptr {ptr}, i64 {byteoffset}')

    def load(self, ptr, dtype='i8'):
        return self.value(f'load {dtype}, ptr {ptr}, align 1')

    def store(self, ptr, value, dtype='i8'):
        self.emit(f'store {dtype} {value}, ptr {ptr}, align 1')

    def alloca(self, nbytes):
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError('scratch storage requires a positive static byte extent')
        # Device temporaries belong to the module, including sub-tile values.
        # Every producer must initialize its window before use on each call;
        # the static initializer is not a substitute for device initialization.
        label = '@scratch_' + str(self.counter)
        self.counter += 1
        self.globals.append(f'{label} = internal global [{nbytes} x i8] zeroinitializer, align 64')
        return label

    def ptrint(self, ptr):
        if ptr in (None, 0): return 0
        return self.value(f'ptrtoint ptr {ptr} to i64')

    def fence(self):
        self.emit('call void asm sideeffect "fence", "~{memory}"()')

    def rocc(self, funct, rs1, rs2):
        self.emit(f'call void asm sideeffect ".insn r 0x7b, 3, {funct}, x0, $0, $1", "r,r,~{{memory}}"(i64 {rs1}, i64 {rs2})')
        self.instruction_count += 1

    def loop(self, count, callback):
        if isinstance(count, int) and count <= 0: return
        if count == 1:
            callback(0)
            return
        pre = self.block
        head, body, latch, end = [self.label_name() for _ in range(4)]
        iv, nxt = self.fresh(), self.fresh()
        self.emit(f'br label %{head}')
        self.lines.append(head + ':')
        self.block = head
        self.emit(f'{iv} = phi i64 [ 0, %{pre} ], [ {nxt}, %{latch} ]')
        cond = self.icmp('ult', iv, count)
        self.emit(f'br i1 {cond}, label %{body}, label %{end}')
        self.lines.append(body + ':')
        self.block = body
        callback(iv)
        self.emit(f'br label %{latch}')
        self.lines.append(latch + ':')
        self.block = latch
        self.emit(f'{nxt} = add i64 {iv}, 1')
        self.emit(f'br label %{head}')
        self.lines.append(end + ':')
        self.block = end

    def conditional(self, condition, yes, no=None):
        a, b, end = [self.label_name() for _ in range(3)]
        self.emit(f'br i1 {condition}, label %{a}, label %{b}')
        self.lines.append(a + ':')
        self.block = a
        yes()
        self.emit(f'br label %{end}')
        self.lines.append(b + ':')
        self.block = b
        if no is not None: no()
        self.emit(f'br label %{end}')
        self.lines.append(end + ':')
        self.block = end

    def llvm_ir(self):
        signature = ', '.join('ptr %arg' + str(i) for i in range(len(self.args)))
        return ('target triple = "riscv64-unknown-elf"\n' +
                '\n'.join(self.globals) + '\n' +
                f'define void @kernel_entry({signature}) {{\nentry:\n' +
                '\n'.join(self.allocations + self.lines) + '\n  ret void\n}\n'
                'define void @kernel_complete() {\nentry:\n  call void asm sideeffect "fence", "~{memory}"()\n  ret void\n}\n' + '\n'.join(sorted(self.declarations)) + '\n')

    def finish(self):
        install = os.environ.get('MERLIN_MLIR_INSTALL')
        candidates = ([str(Path(install) / 'bin/mlir-translate')] if install else []) + [
            shutil.which('mlir-translate'),
            str(Path(__file__).resolve().parent.parent / 'llvm-install/bin/mlir-translate'),
            '/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/mlir-translate',
        ]
        tool = next((p for p in candidates if p and Path(p).is_file()), None)
        if tool is None: raise RuntimeError('selected stock LLVM mlir-translate tool is unavailable')
        result = subprocess.run([tool, '--import-llvm', '--mlir-print-op-generic'], input=self.llvm_ir(), text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError('LLVM import failed: ' + result.stderr)
        lines=[]
        for line in result.stdout.splitlines():
            if '"llvm.inline_asm"' in line and line.rstrip().endswith('-> !llvm.void'):
                indent=line[:len(line)-len(line.lstrip())]
                if ' = ' in line:line=indent+line.split(' = ',1)[1]
                line=line.rsplit('-> !llvm.void',1)[0]+'-> ()'
            lines.append(line)
        return '\n'.join(lines)+'\n'


def descriptor(g, address, rows, cols):
    return g.op('or', g.op('or', address, g.op('shl', cols, 32)), g.op('shl', rows, 48))


def config_ld(g, stride, state=0, dtype='i8', scale=1.0, block_stride=16):
    # shrink=1 selects int8 loading into a full-width accumulator; int32 seeds
    # must use shrink=0. Scratchpad always stores the selected int8 format.
    shrink = 1 if dtype in ('i8', 'int8') else 0
    word = (fbits(scale) << 32) | (block_stride << 16) | (1 << 8) | (state << 3) | (shrink << 2) | 1
    g.rocc(0, word, stride)


def config_st_bits(g, stride, scale_bits, relu=False):
    word = g.op('or', stride, g.op('shl', scale_bits, 32))
    g.rocc(0, 2 | (4 if relu else 0), word)


def config_st(g, stride, scale=1.0, relu=False):
    config_st_bits(g, stride, fbits(scale), relu)


def config_ex(g, dataflow='WS'):
    if dataflow not in ('WS', 'OS'):
        raise ValueError('unknown systolic dataflow '+str(dataflow))
    # Unit A/C row strides and identity execute scale in the selected mode.
    g.rocc(0, (fbits(1.0) << 32) | (1 << 16) | (4 if dataflow == 'WS' else 0), 1 << 48)


def transfer(g, src, dst, rows, cols, src_stride, dst_stride,
             input_dtype='i8', output_dtype='i8', scale=1.0, relu=False):
    """One <=16x16 DMA tile, optionally widened or narrowed on the device.

    Both strides are byte strides. Counts and strides may be LLVM i64 values.
    """
    if input_dtype not in ('i8', 'int8', 'i32', 'int32'):
        raise NotImplementedError('device transfer input dtype ' + input_dtype)
    if output_dtype not in ('i8', 'int8', 'i32', 'int32'):
        raise NotImplementedError('device transfer output dtype ' + output_dtype)
    g.fence()
    to_acc = input_dtype in ('i32', 'int32') or output_dtype in ('i32', 'int32') or scale != 1.0 or relu
    local = ACC if to_acc else 0
    full = FULL if output_dtype in ('i32', 'int32') else 0
    config_ld(g, src_stride, dtype=input_dtype)
    config_st(g, dst_stride, scale, relu)
    g.rocc(2, g.ptrint(src), descriptor(g, local, rows, cols))
    g.fence()
    g.rocc(3, g.ptrint(dst), descriptor(g, local | full, rows, cols))
    g.fence()


def bias_stride(tensor, rows, cols):
    """Validate an i32 seed and return its byte stride between logical rows."""
    if tensor['dtype'] not in ('i32', 'int32'):
        raise ValueError('accumulator seed must have i32 storage')
    shape = list(tensor['shape'])
    if shape in ([cols], [1, cols]):
        return 0
    if shape == [rows, cols]:
        return cols * 4
    raise ValueError('accumulator seed must be a column vector or the full output tensor')


def matmul(g, a, w, y, m, k, n, attrs, output_dtype='i32', bias=None,
           bias_row_stride=0):
    """Tiled i8 @ i8 with exact int32 accumulation and hardware readout.

    K passes are bounded to DIM=16, safely below the selected 20-bit internal
    partial-sum bound even for -128 operands. Intermediate sums stay in int32.
    """
    stages = attrs.get('epilogue', [])
    supported = {'bias', 'bias_add', 'acc_scale', 'relu', 'requant'}
    if any(s not in supported for s in stages):
        raise NotImplementedError('matmul epilogue ' + str(stages))
    if output_dtype not in ('i8', 'int8', 'i32', 'int32'):
        raise NotImplementedError('matmul output dtype ' + str(output_dtype))
    if 'requant' in stages:
        raise NotImplementedError('integer requant requires exact readout lowering')
    if any(key in attrs for key in ('scale_numerator', 'scale_denominator', 'zero_point', 'quant_min', 'quant_max')):
        raise NotImplementedError('exact rational host readout is not hardware f32 readout')
    scale = attrs.get('acc_scale', 1.0) if 'acc_scale' in stages else 1.0
    if isinstance(scale, str): scale = float(scale)
    scales = attrs.get('acc_scale_by_column')
    relu = 'relu' in stages
    is_full = output_dtype in ('i32', 'int32')
    if is_full and (relu or scales is not None or float(scale) != 1.0):
        raise NotImplementedError('full i32 readout applies no scale or activation')
    has_negative = (scales is not None and any(float(v) < 0 for v in scales)) or float(scale) < 0
    if relu and has_negative and 'acc_scale' in stages and stages.index('relu') < stages.index('acc_scale'):
        raise NotImplementedError('negative scaling after full-width relu needs a separate exact device construction')
    if not all(isinstance(v, int) and v > 0 for v in (m, k, n)):
        raise NotImplementedError('matmul needs positive static logical extents')
    elem_size = 4 if is_full else 1
    dataflow = attrs.get('dataflow', 'WS')
    if dataflow not in ('WS', 'OS'):
        raise ValueError('unknown contraction dataflow '+str(dataflow))
    # Keep every instruction site's transfer sizes constant. Runtime induction
    # variables only affect DRAM addresses; full tiles and tails are separate
    # code paths, so static legality does not depend on interpreting loop PHIs.
    def tiles(total, body, start=0):
        full, tail = divmod(total, DIM)
        if full:
            g.loop(full, lambda i: body(g.add(start, g.mul(i, DIM)), DIM))
        if tail:
            body(start + full * DIM, tail)

    if scales is None:
        runs = [(0, n, float(scale))]
    else:
        if len(scales) != n:
            raise ValueError('per-column scale count must equal N')
        runs = []
        begin = 0
        for i in range(1, n + 1):
            if i == n or fbits(scales[i]) != fbits(scales[begin]):
                runs.append((begin, i - begin, float(scales[begin])))
                begin = i
    narrow_tmp = g.alloca(DIM * DIM) if relu and any(s < 0 for _, _, s in runs) else None
    g.fence()
    config_ex(g, dataflow)
    config_ld(g, k, state=0, dtype='i8')
    config_ld(g, n, state=1, dtype='i8')
    if bias is not None:
        config_ld(g, bias_row_stride, state=2, dtype='i32')

    # Only the backend's proved readonly, shared ABI pointers are eligible.
    # The frame has one padded-M row slab per K panel and never overlaps the
    # ordinary A/B working rows. A cold load is emitted into the kernel body,
    # so each invocation initializes its frame from its own runtime inputs.
    padded_m = ((m + DIM - 1)//DIM)*DIM
    panels = (k + DIM - 1)//DIM
    shared_a = a in g.shared_input_pointers and padded_m*panels <= 1024
    if shared_a:
        frame_key = (a,m,k)
        if g._shared_a_frame_key != frame_key:
            def load_panel(panel,kk,tk):
                def load_rows(row,tm):
                    source = g.gep(a,g.add(g.mul(row,k),kk))
                    address = g.add(1024,g.add(g.mul(panel,padded_m),row))
                    g.rocc(2,g.ptrint(source),descriptor(g,address,tm,tk))
                tiles(m,load_rows)
            full_panels,tail_panel = divmod(k,DIM)
            g.fence()
            if full_panels:
                g.loop(full_panels,lambda panel:load_panel(panel,g.mul(panel,DIM),DIM))
            if tail_panel:
                load_panel(full_panels,full_panels*DIM,tail_panel)
            g.fence()
            g._shared_a_frame_key = frame_key

    def m_tile(row, tm):
        def n_tile(col, tn, tile_scale):
            if bias is not None:
                g.fence()
                bp = g.gep(bias, g.add(g.mul(row, bias_row_stride), g.mul(col, 4)))
                g.rocc(14, g.ptrint(bp), descriptor(g, ACC, tm, tn))
                g.fence()

            def k_pass(kk, tk, accumulate, panel):
                g.fence()
                if shared_a:
                    a_address = g.add(1024,g.add(g.mul(panel,padded_m),row))
                else:
                    a_address = 0
                    ap = g.gep(a, g.add(g.mul(row, k), kk))
                wp = g.gep(w, g.add(g.mul(kk, n), col))
                if not shared_a:
                    g.rocc(2, g.ptrint(ap), descriptor(g, 0, tm, tk))
                g.rocc(1, g.ptrint(wp), descriptor(g, DIM, tk, tn))
                g.fence()
                acc_addr = ACC | (ACCUM if accumulate else 0)
                if dataflow == 'WS':
                    g.rocc(6, descriptor(g, DIM, tk, tn), descriptor(g, acc_addr, tm, tn))
                    g.rocc(4, descriptor(g, a_address, tm, tk), descriptor(g, GARBAGE, tm, tn))
                else:
                    # In OS the mesh retains C, not B. Start each bounded K
                    # pass from zero in the PE, then commit that pass into the
                    # full-width accumulator (adding after the first pass).
                    g.rocc(6, descriptor(g, GARBAGE, tm, tn), descriptor(g, acc_addr, tm, tn))
                    g.rocc(4, descriptor(g, a_address, tm, tk), descriptor(g, DIM, tk, tn))
                g.fence()

            first_k = min(DIM, k)
            k_pass(0, first_k, bias is not None, 0)
            remaining_full, remaining_tail = divmod(k - first_k, DIM)
            if remaining_full:
                if attrs.get('_static_k_passes', False) and remaining_full <= 1024:
                    for ki in range(remaining_full):
                        k_pass(first_k + ki * DIM, DIM, True, ki+1)
                else:
                    g.loop(remaining_full, lambda ki: k_pass(g.add(first_k, g.mul(ki, DIM)), DIM, True,g.add(ki,1) if shared_a else 0))
            if remaining_tail:
                k_pass(first_k + remaining_full * DIM, remaining_tail, True,remaining_full+1)
            yp = g.gep(y, g.mul(g.add(g.mul(row, n), col), elem_size))
            post_relu = relu and tile_scale < 0
            dest = narrow_tmp if post_relu else yp
            dest_stride = DIM if post_relu else n * elem_size
            config_st(g, dest_stride, tile_scale, relu and not post_relu)
            g.rocc(3, g.ptrint(dest), descriptor(g, ACC | (FULL if is_full else 0), tm, tn))
            g.fence()
            if post_relu:
                transfer(g, narrow_tmp, yp, tm, tn, DIM, n, 'i8', 'i8', relu=True)
                config_ld(g, k, state=0, dtype='i8')

        for col_start, width, run_scale in runs:
            tiles(width, lambda col, tn, run_scale=run_scale: n_tile(col, tn, run_scale), col_start)

    tiles(m, m_tile)
    g.fence()
