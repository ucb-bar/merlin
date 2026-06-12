#!/usr/bin/env python3
"""Materialize torchao qinner int8 weights into a compiled-path-runnable bundle.

torchao weight-only int8 under torch>=2.8 (xr0, pi05) leaves the int8 inner tensors as
uninitialized `tensor.empty` ops tagged on their `quant_ext.dequantize_per_channel`
consumers (prov.quant_inner_w/_s = extra.npz `qinner::` keys). The numpy interpreter binds
them at eval; the compiled (spike/FireSim) path leaves them garbage (cos 0). This rewrites
a bundle so the compiled path works:

  1. Lift each qinner `tensor.empty` to a forward-function argument (replace its uses).
  2. Append the qinner int8/scale tensors to weights.safetensors + the manifest as `param`
     entries (in the lifted-arg order) — reusing c_runtime's existing param->weights-blob
     binding, so no c_runtime change is needed.

Produces a NEW bundle dir. Usage: fix_qinner.py <src_bundle> <dst_bundle>
"""
import json, struct, sys, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "merlin" / "python"))
from merlin.frontends.linalg_mlir import parse_mlir_file          # noqa: E402
from merlin.xdsl_dialects._common import text as to_text          # noqa: E402

VIEW = ("linalg.transpose", "tensor.collapse_shape", "tensor.expand_shape",
        "tensor.cast", "tensor.reshape", "linalg.copy")


def _opname(op):
    for a in ("op_name__", "op_name"):
        if a in op.attributes:
            return getattr(op.attributes[a], "data", None)
    return op.name


def _trace_empty(val):
    cur = val
    for _ in range(8):
        o = getattr(cur, "owner", None)
        if o is None or not hasattr(o, "name"):
            return None
        if o.name == "tensor.empty":
            return o
        if o.name in VIEW or _opname(o) in VIEW:
            cur = o.operands[0]
            continue
        return None
    return None


def lift(src_mlir):
    """Lift qinner empties to forward args. Returns (module, [qinner_key per new arg])."""
    mod = parse_mlir_file(src_mlir)
    fwd = next(op for op in mod.walk()
               if op.name == "func.func" and "kernel" not in op.properties["sym_name"].data)
    blk = fwd.body.blocks[0]
    deq = [op for op in mod.walk()
           if str(_opname(op)).endswith("dequantize_per_channel")]
    # collect (empty_op, key) in a stable order
    targets = []
    seen = set()
    for op in deq:
        ks = {k: getattr(op.attributes[k], "data", None)
              for k in op.attributes if "quant_inner" in k}
        for oi, suf in ((0, "prov.quant_inner_w"), (1, "prov.quant_inner_s")):
            key = ks.get(suf)
            if not key:
                continue
            e = _trace_empty(op.operands[oi])
            if e is not None and id(e) not in seen:
                seen.add(id(e))
                targets.append((e, key))
    keys = []
    for e, key in targets:
        res = e.results[0]
        arg = blk.insert_arg(res.type, len(blk.args))      # append a block arg of same type
        res.replace_by(arg)
        e.detach(); e.erase()
        keys.append(key)
    # rebuild the function_type with the appended input types
    from xdsl.dialects.builtin import FunctionType
    fwd.properties["function_type"] = FunctionType.from_lists(
        [a.type for a in blk.args], list(fwd.function_type.outputs.data))
    return mod, keys


def _sanitize(key):
    return "qinner_" + key.replace(".", "_").replace("::", "_")


_NP_TO_ST = {"int8": "I8", "uint8": "U8", "int32": "I32", "int64": "I64",
             "float16": "F16", "float32": "F32", "float64": "F64"}


def append_safetensors(src_st, dst_st, new_tensors):
    """Append {name: ndarray} to a safetensors file (8B LE hdr-len + JSON hdr + payload),
    writing dst_st. Returns the list of names in append order."""
    from merlin.llvmlower.weights_pack import load_safetensors_header
    header, payload_off = load_safetensors_header(src_st)
    data = Path(src_st).read_bytes()
    payload = bytearray(data[payload_off:])
    for name, arr in new_tensors:
        arr = np.ascontiguousarray(arr)
        begin = len(payload)
        payload += arr.tobytes()
        header[name] = {"dtype": _NP_TO_ST[str(arr.dtype)],
                        "shape": list(arr.shape),
                        "data_offsets": [begin, len(payload)]}
    hdr_json = json.dumps(header).encode("utf-8")
    with open(dst_st, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_json)))
        f.write(hdr_json)
        f.write(payload)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    extra = dict(np.load(src / "extra.npz"))
    mod, keys = lift(src / "model.mlir")
    print(f"lifted {len(keys)} qinner empties to forward args (order = appended)")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "model.mlir").write_text(to_text(mod))

    # carry over the non-weight bundle files unchanged
    for f in ("inputs.npz", "golden.npy", "extra.npz", "input_order.json"):
        if (src / f).is_file():
            shutil.copy2(src / f, dst / f)

    # extend weights.safetensors + manifest with the qinner tensors (lifted-arg order).
    # c_runtime binds forward arg i from manifest[i] -> safetensors payload offset, so the
    # appended args (indices n0..) must get param entries pointing at the new tensors.
    man = json.loads((src / "weights.safetensors.manifest.json").read_text())
    n0 = max(int(k) for k in man if k.isdigit()) + 1
    new_tensors = []
    for j, key in enumerate(keys):
        arr = np.ascontiguousarray(extra[f"qinner::{key}"])
        name = _sanitize(key)
        new_tensors.append((name, arr))
        man[str(n0 + j)] = {"kind": "param", "weight": name, "name": name,
                            "dtype": str(arr.dtype), "shape": list(arr.shape)}
    append_safetensors(src / "weights.safetensors", dst / "weights.safetensors", new_tensors)
    (dst / "weights.safetensors.manifest.json").write_text(json.dumps(man))
    print(f"extended safetensors: +{len(keys)} qinner params "
          f"(manifest now {len([k for k in man if k.isdigit()])} args); wrote {dst}")


if __name__ == "__main__":
    main()
