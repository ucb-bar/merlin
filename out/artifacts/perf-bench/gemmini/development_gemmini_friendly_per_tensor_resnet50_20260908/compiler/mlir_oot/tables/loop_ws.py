"""Pinned-header LOOP_WS descriptor expressions for the opt-in source route."""
import ast
import json
from functools import lru_cache
from pathlib import Path
from . import rtl_facts as F


@lru_cache(maxsize=1)
def contract():
    data = json.loads(Path(__file__).with_name("loop_ws_contract.json").read_text())
    if data["schema"] != "pinned_loop_ws_descriptor_v1":
        raise ValueError("unknown loop descriptor schema")
    geometry = data["geometry"]
    if (geometry["DIM"], geometry["BANK_NUM"]*geometry["BANK_ROWS"], geometry["ACC_ROWS"]) != (F.DIM,F.SPAD_ROWS,F.ACC_ROWS):
        raise ValueError("loop header geometry differs from compiler target facts")
    return data


def opcode(name):
    records = [row for row in contract()["records"] if row["name"] == name]
    if len(records) != 1:
        raise ValueError("unknown loop command")
    return records[0]["funct"]


def expression(source, values):
    node = ast.parse(source, mode="eval").body
    def visit(item):
        if isinstance(item, ast.Constant) and type(item.value) is int:
            return item.value
        if isinstance(item, ast.Name) and item.id in values:
            return values[item.id]
        if isinstance(item, ast.BinOp):
            a,b = visit(item.left), visit(item.right)
            if isinstance(item.op, ast.BitOr): return a | b
            if isinstance(item.op, ast.LShift): return a << b
        raise ValueError("unsupported pinned loop expression")
    def fields(item):
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.BitOr):
            return fields(item.left)+fields(item.right)
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.LShift):
            return [(visit(item.right),visit(item.left))]
        return [(0,visit(item))]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        packed = sorted(fields(node))
        for index,(shift,value) in enumerate(packed):
            end = packed[index+1][0] if index+1 < len(packed) else 64
            if not (0 <= shift < end <= 64 and 0 <= value < 1 << (end-shift)):
                raise ValueError("loop descriptor field overflow")
    result = visit(node)
    if not 0 <= result < 1 << 64:
        raise ValueError("loop register overflow")
    return result


def reduction_resident_block_shape(mt, nt, kt):
    """Return the largest legal output block whose full reduction stays resident.

    LOOP_WS can overlap two independent descriptors, but it has no DRAM-alias
    dependency tracking between them.  A K-split implemented by storing a partial C
    and reading it back as the next descriptor's D therefore needs separate
    serialization.  Prefer shrinking the independent M/N dimensions so K does not
    split at all.  Besides removing that dependency, this deletes partial-result
    traffic and follows the accelerator library's resident-reduction strategy.
    """
    capacity = contract()["capacity"]
    acc_tiles = capacity["max_acc_rows"]["rows"] // F.DIM
    spad_tiles = capacity["max_spad_rows"]["rows"] // F.DIM
    if min(mt,nt,kt,acc_tiles,spad_tiles) <= 0:
        raise ValueError("empty or unknown loop capacity")

    best = None
    best_score = None
    for bm in range(1, min(mt, acc_tiles) + 1):
        for bn in range(1, min(nt, acc_tiles) + 1):
            if bm * bn > acc_tiles or (bm + bn) * kt > spad_tiles:
                continue
            # First minimize independent output descriptors.  On an equal-area
            # choice prefer a balanced block, then wider N for contiguous readout.
            score = (bm * bn, min(bm, bn), bn, bm)
            if best_score is None or score > best_score:
                best, best_score = (bm, bn, kt), score
    return best


def block_shape(mt, nt, kt):
    """Choose a reduction-resident block, or a capacity-bounded split fallback."""
    resident = reduction_resident_block_shape(mt, nt, kt)
    if resident is not None:
        return resident

    capacity = contract()["capacity"]
    acc_tiles = capacity["max_acc_rows"]["rows"] // F.DIM
    spad_tiles = capacity["max_spad_rows"]["rows"] // F.DIM
    bn = min(nt, max(1, acc_tiles // min(mt,acc_tiles)))
    bm = min(mt, max(1, acc_tiles // bn))
    bk = min(kt, max(1, spad_tiles // (bm+bn)))
    if bm*bn > acc_tiles or (bm+bn)*bk > spad_tiles:
        raise ValueError("loop block exceeds header capacity partition")
    return bm,bn,bk


def loop_ws_static(*, rows, cols, depth, row_stride_a, row_stride_b, row_stride_c,
                   full_c=True, activation=0, accumulate=False):
    if min(rows,cols,depth,row_stride_a,row_stride_b,row_stride_c) <= 0:
        raise ValueError("LOOP_WS requires positive static geometry")
    tiles = [-(-n//F.DIM) for n in (rows,cols,depth)]
    values = dict(zip(("I","J","K"),tiles))
    values.update(dict(zip(("pad_I","pad_J","pad_K"),
        (tile*F.DIM-n for tile,n in zip(tiles,(rows,cols,depth))))))
    values.update(A_stride=row_stride_a,B_stride=row_stride_b,
        C_stride=row_stride_c,D_stride=row_stride_c if accumulate else 0,
        A_transpose=0,B_transpose=0,full_C=int(full_c),low_D=0,
        ex_accumulate=int(accumulate),act=activation,a_spad_id=0,b_spad_id=0,is_resadd=0)
    result = []
    for row in contract()["records"]:
        if row["name"] in {"k_LOOP_WS_CONFIG_ADDRS_AB","k_LOOP_WS_CONFIG_ADDRS_DC"}:
            continue
        result.append((row["funct"],expression(row["rs1"],values),expression(row["rs2"],values)))
    return result
