"""Canonical kernel inputs and independent NumPy references.

Generator/reference helpers are transcribed from the supplied kernel handoff
section 6. This module imports no TVM and never supplies answers to compilation.
"""
import hashlib
import math
import struct
import numpy as np

MASK64 = (1 << 64) - 1

def splitmix64(x):
    x = (int(x) + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64

def historical_i8_arena(nbytes, seed=0x5EED):
    seed_block = bytearray()
    for i in range(2048):
        w = splitmix64(seed + i) & 0x0F0F0F0F0F0F0F0F
        sign_bits = (w >> 3) & 0x0101010101010101
        seed_block += struct.pack('<Q', w | (sign_bits * 0xF0))
    spad = bytearray(16384 * 16)
    for row in range(0, 16384, 16):
        src = (row * 16 + (row // 1024) * 272) % (16384 - 256)
        spad[row*16:row*16+256] = seed_block[src:src+256]
    arena = bytearray(((nbytes + 255) // 256) * 256)
    for b in range(len(arena) // 256):
        row = (b * 16 + (b // 1024) * 48) % 16384
        arena[b*256:(b+1)*256] = spad[row*16:row*16+256]
    return arena[:nbytes]

def make_inputs(roster):
    # roster = [(name, shape), ...], INT8 inputs only, in historical order.
    placements, end = [], 0
    for name, shape in roster:
        count = math.prod(shape)
        placements.append((name, tuple(shape), end, count))
        end = ((end + count + 4095) // 4096) * 4096
    arena = historical_i8_arena(end)
    arrays = {}
    for name, shape, offset, count in placements:
        arrays[name] = np.frombuffer(arena, dtype=np.int8,
                                     count=count, offset=offset).reshape(shape).copy()
    return arrays

def make_bias(channels, seed):
    return np.array([(splitmix64(seed+i) & 2047) - 1024
                     for i in range(channels)], dtype='<i4')

def as_i32(values):
    # Defined modular INT32 semantics, without signed-overflow dependence.
    u = np.remainder(np.asarray(values, dtype=np.int64), 1 << 32).astype('<u4')
    return u.view('<i4')

def raw(a, w):
    return as_i32(a.astype(np.int64) @ w.astype(np.int64))

def q8(acc, scale):
    x = np.asarray(acc, dtype=np.float32) * np.float32(scale)
    return np.clip(np.rint(x), -128, 127).astype(np.int8)

def conv_reference(ifm, w, bias):
    batch, hi, wi, ci = ifm.shape
    ho, wo, co = hi - 2, wi - 2, w.shape[1]
    weights = w.reshape(3, 3, ci, co).astype(np.int64)
    acc = np.broadcast_to(bias.astype(np.int64), (batch, ho, wo, co)).copy()
    for r in range(3):
        for s in range(3):
            acc += ifm[:, r:r+ho, s:s+wo, :].astype(np.int64) @ weights[r, s]
    return q8(as_i32(acc), 1.0).reshape(batch*ho*wo, co)

def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order='C')).hexdigest()


def case_spec(case):
    gemms = dict(G01=(512,512,512), G02=(12544,256,64), G03=(12544,64,256),
                 G04=(3136,512,128), G05=(3136,128,512), G06=(784,256,1024),
                 T03=(16,5632,2048), T05=(16,2048,2048), T07=(1,2048,32000))
    if case in gemms:
        m,k,n = gemms[case]
        return dict(id=case, kind="matmuls", roster=[("W",(k,n)),("A0",(m,k))],
                    products=[("A0","W","Y0")], scale=1. if case.startswith("G") else None)
    if case in ("T01","T02"):
        k,n = (64,16) if case=="T01" else (16,64)
        return dict(id=case, kind="matmuls", scale=None if case=="T01" else 1/128,
                    roster=[(f"K{i}",(k,n)) for i in range(4)]+[(f"A{i}",(16,k)) for i in range(32)],
                    products=[(f"A{i}",f"K{i//8}",f"Y{i}") for i in range(32)])
    if case in ("T04","T06"):
        weights = [("Wg",5632,"Yg"),("Wu",5632,"Yu")] if case=="T04" else [
            ("Wq",2048,"Yq"),("Wk",256,"Yk"),("Wv",256,"Yv")]
        return dict(id=case, kind="matmuls", scale=1/128,
                    roster=[(name,(2048,n)) for name,n,_ in weights]+[("A0",(16,2048))],
                    products=[("A0",name,out) for name,_,out in weights])
    convs=dict(C01=(56,64,64,0x6000),C02=(28,128,128,0x5000),C03=(14,256,256,0x4000))
    if case in convs:
        h,ci,co,seed=convs[case]
        return dict(id=case, kind="conv", scale=1., bias_seed=seed,
                    roster=[("IFM",(4,h+2,h+2,ci)),("W",(9*ci,co))])
    raise ValueError("Unknown kernel case: "+case)


CASES = tuple([f"G{i:02d}" for i in range(1,7)]+[f"C{i:02d}" for i in range(1,4)]+
              [f"T{i:02d}" for i in range(1,8)])


def fixtures(spec):
    inputs=make_inputs(spec["roster"])
    if spec["kind"]=="conv":
        inputs["B"]=make_bias(inputs["W"].shape[1],spec["bias_seed"])
        return inputs, {"Y0":conv_reference(inputs["IFM"],inputs["W"],inputs["B"])}
    outputs={}
    for a,w,name in spec["products"]:
        value=raw(inputs[a],inputs[w])
        outputs[name]=value if spec["scale"] is None else q8(value,spec["scale"])
    return inputs,outputs
