"""Affine DMA addressing and bounded command-loop unrolling for integer routes.

An address is a sum of static byte strides times surrounding loop indices.
Keep that representation until forming a typed GEP, rather than emitting scalar
integer multiplies and additions for every command's address. No payload value
can enter this address representation.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Offset:
    terms: tuple
    constant: int = 0


class AddressEmitter:
    def __init__(self, emitter):
        self.emitter = emitter
        self.expansion = 1

    def __getattr__(self, name):
        return getattr(self.emitter, name)

    @staticmethod
    def offset(value):
        if isinstance(value, Offset):
            return value
        if isinstance(value, int):
            return Offset((), value)
        raise ValueError('integer DMA address must derive from static strides and loop indices')

    def add(self, left, right):
        if isinstance(left, int) and isinstance(right, int):
            return left+right
        left, right = self.offset(left), self.offset(right)
        terms = dict(left.terms)
        for index, stride in right.terms:
            terms[index] = terms.get(index, 0)+stride
        return Offset(tuple((i,s) for i,s in terms.items() if s), left.constant+right.constant)

    def mul(self, left, right):
        if isinstance(left, int) and isinstance(right, int):
            return left*right
        if isinstance(left, int):
            left, right = right, left
        if not isinstance(right, int):
            raise ValueError('integer DMA address cannot multiply two runtime indices')
        left = self.offset(left)
        return Offset(tuple((i,s*right) for i,s in left.terms if s*right), left.constant*right)

    def ptr(self, name, byteoffset=0):
        if isinstance(byteoffset, int):
            return self.emitter.ptr(name, byteoffset)
        offset = self.offset(byteoffset)
        base = self.emitter.ptr(name, offset.constant)
        if not offset.terms:
            return base
        e = self.emitter
        pointer = e.fresh()
        e.emit(f'{pointer} = llvm.inttoptr {base} : i64 to !llvm.ptr')
        for index, stride in offset.terms:
            if stride <= 0:
                raise ValueError('integer DMA address requires positive static strides')
            target = e.fresh()
            e.emit(f'{target} = llvm.getelementptr {pointer}[{index}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<{stride} x i8>')
            pointer = target
        result = e.fresh()
        e.emit(f'{result} = llvm.ptrtoint {pointer} : !llvm.ptr to i64')
        return result

    def loop(self, start, end, step, callback):
        if not all(isinstance(v, int) for v in (start,end,step)):
            raise ValueError('integer command loops require static source extents')
        trips = max(0, (end-start+step-1)//step)
        if not trips:
            return
        factor = min(trips, 16, max(1,64//self.expansion))
        bulk = trips//factor*factor
        old = self.expansion
        self.expansion *= factor
        def group(index):
            base = Offset(((index,1),)) if isinstance(index,str) else index
            for i in range(factor):
                callback(self.add(base,i*step))
        try:
            self.emitter.loop(start,start+bulk*step,step*factor,group)
            for i in range(bulk,trips):
                callback(start+i*step)
        finally:
            self.expansion = old
