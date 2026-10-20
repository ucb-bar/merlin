"""V3 conservative source exclusions across checked public interface grammars.

Linalg retains its exact V1 canonicalizer. The restricted raw integer
merlin_iface value algebra is reified in standard named/generic linalg forms
and passed to that SAME canonicalizer. Images may conservatively overexclude;
they neither qualify numerical execution nor enable a compiler transformation.
"""

from __future__ import annotations

from xdsl.dialects import builtin

from merlin.common import mlir_query as Q
from merlin.verify.merlin_iface_semantics import parse_merlin_iface

from . import operator_typed_exclusion as T

SCHEMA = "merlin.phase0.checked_mixed_source_exclusion.v1"


def _string(attributes, name):
    value = attributes.get(name)
    if not isinstance(value, builtin.StringAttr) or not value.data:
        raise ValueError("raw interface exclusion needs its exact string declarations")
    return value.data


def _attrs(op, allowed):
    attributes = {k: v for k, v in op.attributes.items() if k != "op_name__"}
    if set(attributes) != set(allowed) or op.properties or op.regions:
        raise ValueError("raw interface exclusion encountered unowned attributes or regions")
    return attributes


def _tensor(value, dtype):
    typ = value.type
    if (not isinstance(typ, builtin.TensorType) or len(typ.get_shape()) != 2
            or any(n < 1 for n in typ.get_shape()) or str(typ.get_element_type()) != dtype
            or not isinstance(typ.encoding, builtin.NoneAttr)):
        raise ValueError("raw interface exclusion needs exact positive rank-two integer storage")
    return tuple(typ.get_shape())


def raw_interface_geometry(source):
    """Check complete SSA, declarations, value effects and resident ownership.

    This is the restricted algebra defined by merlin_iface_semantics: transparent
    packed RHS, one zero-initial signed-i8 product/i32 reduction, raw-i32 commit
    and eviction. There is no value evaluation, source-specific answer, target
    instruction qualification, readout conversion or accumulation continuation.
    """
    module = parse_merlin_iface(source.decode("utf-8"))
    module.verify()
    expected = {"merlin_iface.version", "merlin_iface.target", "merlin_iface.abi_version"}
    if (set(module.attributes) != expected or module.properties or len(module.body.blocks) != 1
            or module.body.block.args or _string(module.attributes, "merlin_iface.version") != "0.1"
            or _string(module.attributes, "merlin_iface.abi_version") != "0.1"):
        raise ValueError("raw interface exclusion has no exact original grammar binding")
    _string(module.attributes, "merlin_iface.target")
    operations = list(module.body.block.ops)
    names = [Q.op_name(op) for op in operations]
    if names != ["merlin_iface.tensor", "merlin_iface.tensor", "merlin_iface.resident_pack",
                 "merlin_iface.matmul", "merlin_iface.commit", "merlin_iface.evict"]:
        raise ValueError("raw interface exclusion requires the complete restricted single-contraction program")
    leaves, labels = {}, set()
    for op in operations[:2]:
        attributes = _attrs(op, {"name", "role"})
        if op.operands or len(op.results) != 1:
            raise ValueError("raw interface exclusion has an invalid source leaf")
        role, label = _string(attributes, "role"), _string(attributes, "name")
        if role not in {"input", "weight"} or label in labels:
            raise ValueError("raw interface exclusion source role or identity is ambiguous")
        labels.add(label)
        leaves[op.results[0]] = (role, _tensor(op.results[0], "i8"))
    pack, contraction, commit, evict = operations[2:]
    attributes = _attrs(pack, {"layout"})
    if (len(pack.operands) != 1 or len(pack.results) != 1
            or _string(attributes, "layout") != "packed_rhs" or pack.operands[0] not in leaves
            or leaves[pack.operands[0]][0] != "weight" or str(pack.results[0].type) != "!merlin_iface.resident"):
        raise ValueError("raw interface exclusion has no owned transparent RHS pack")
    _attrs(contraction, set())
    if (len(contraction.operands) != 2 or len(contraction.results) != 1
            or contraction.operands[0] not in leaves or leaves[contraction.operands[0]][0] != "input"
            or contraction.operands[1] is not pack.results[0]
            or str(contraction.results[0].type) != "!merlin_iface.acc<i32>"):
        raise ValueError("raw interface exclusion has no zero-initial signed-i8/i32 contraction")
    m, k = leaves[contraction.operands[0]][1]
    rhs_k, n = leaves[pack.operands[0]][1]
    if k != rhs_k:
        raise ValueError("raw interface exclusion contracted dimensions disagree")
    attributes = _attrs(commit, {"name", "epilogue", "output_dtype"})
    epilogue = attributes.get("epilogue")
    if (len(commit.operands) != 1 or len(commit.results) != 1
            or commit.operands[0] is not contraction.results[0]
            or not isinstance(epilogue, builtin.ArrayAttr) or epilogue.data
            or _string(attributes, "output_dtype") != "i32" or _string(attributes, "name") in labels
            or _tensor(commit.results[0], "i32") != (m, n)):
        raise ValueError("raw interface exclusion has no complete raw-i32 result ownership")
    _attrs(evict, set())
    if len(evict.operands) != 1 or evict.results or evict.operands[0] is not pack.results[0]:
        raise ValueError("raw interface exclusion evicts no owned resident RHS")
    return dict(M=m, N=n, K=k)


def _image(m, n, k, *, named):
    header = (f"module {{\n func.func @raw(%a: tensor<{m}x{k}xi8>, %b: tensor<{k}x{n}xi8>) "
              f"-> tensor<{m}x{n}xi32> {{\n"
              "%z = arith.constant 0 : i32\n"
              f"%e = tensor.empty() : tensor<{m}x{n}xi32>\n"
              f"%i = linalg.fill ins(%z : i32) outs(%e : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>\n")
    if named:
        def widen(name, rows, columns):
            return (
                f"%{name}e = tensor.empty() : tensor<{rows}x{columns}xi32>\n"
                f'%{name}w = linalg.generic {{indexing_maps = [affine_map<(i,j)->(i,j)>, '
                'affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]} '
                f'ins(%{name} : tensor<{rows}x{columns}xi8>) outs(%{name}e : tensor<{rows}x{columns}xi32>) {{\n'
                '^bb0(%v: i8, %unused: i32):\n %wide = arith.extsi %v : i8 to i32\n'
                f'linalg.yield %wide : i32\n }} -> tensor<{rows}x{columns}xi32>\n')

        # This equivalent form makes signed widening explicit before the
        # registered homogeneous named matmul. The retained xDSL named-op
        # builder cannot represent mixed i8 inputs/i32 accumulation correctly.
        contraction = (widen("a", m, k) + widen("b", k, n)
                       + f"%r = linalg.matmul ins(%aw, %bw : tensor<{m}x{k}xi32>, tensor<{k}x{n}xi32>) "
                       f"outs(%i : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>\n")
    else:
        contraction = (
            '%r = linalg.generic {indexing_maps = [affine_map<(m,n,k)->(m,k)>, '
            'affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>], '
            'iterator_types = ["parallel", "parallel", "reduction"]} '
            f'ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>) outs(%i : tensor<{m}x{n}xi32>) {{\n'
            '^bb0(%a0: i8, %b0: i8, %acc: i32):\n'
            '%a32 = arith.extsi %a0 : i8 to i32\n %b32 = arith.extsi %b0 : i8 to i32\n'
            '%p = arith.muli %a32, %b32 : i32\n %sum = arith.addi %acc, %p : i32\n'
            f'linalg.yield %sum : i32\n }} -> tensor<{m}x{n}xi32>\n')
    return (header + contraction + f"return %r : tensor<{m}x{n}xi32>\n }}\n}}").encode()


def signatures(source, *, max_bytes):
    """Explicit V3 entry: same typed signature hashes, checked grammar images."""
    if type(max_bytes) is not int or max_bytes < 1 or not isinstance(source, bytes) or len(source) > max_bytes:
        raise ValueError("mixed source exclusion exceeds the selected compact parse budget")
    if b"merlin_iface.version" not in source:
        return T.signatures(source, max_bytes=max_bytes)
    shape = raw_interface_geometry(source)
    result = set()
    for named in (False, True):
        result.update(T.signatures(_image(shape["M"], shape["N"], shape["K"], named=named), max_bytes=max_bytes))
    if len(result) != 2:
        raise ValueError("raw interface exclusion did not produce both checked canonical forms")
    return tuple(sorted(result))
