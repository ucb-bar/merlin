"""Native-compatible checked reinterpretation of ranked/unranked buffers."""

# Verification rules adapted from the LLVM Project's MemRefOps.cpp.
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import cache
from typing import ClassVar

from xdsl.dialects import builtin, memref
from xdsl.ir import Dialect
from xdsl.irdl import AttrSizedOperandSegments, IRDLOperation, irdl_op_definition, operand_def, prop_def, region_def, result_def, traits_def, var_operand_def
from xdsl.traits import IsTerminator, MemoryReadEffect, MemoryWriteEffect, NoMemoryEffect, is_side_effect_free
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ReinterpretCastOp(IRDLOperation):
    name = "memref.reinterpret_cast"
    DYNAMIC_INDEX: ClassVar[int] = -(2**63)
    source = operand_def(builtin.MemRefType | builtin.UnrankedMemRefType)
    offsets = var_operand_def(builtin.IndexType)
    sizes = var_operand_def(builtin.IndexType)
    strides = var_operand_def(builtin.IndexType)
    static_offsets = prop_def(builtin.DenseArrayBase.constr(builtin.i64))
    static_sizes = prop_def(builtin.DenseArrayBase.constr(builtin.i64))
    static_strides = prop_def(builtin.DenseArrayBase.constr(builtin.i64))
    result = result_def(builtin.MemRefType)
    irdl_options = (AttrSizedOperandSegments(as_property=True),)
    assembly_format = memref.ReinterpretCastOp.assembly_format
    custom_directives = memref.ReinterpretCastOp.custom_directives
    traits = traits_def(NoMemoryEffect())

    def verify_(self):
        source, result = self.source.type, self.result.type
        offsets, sizes, strides = (tuple(a.get_values()) for a in
                                    (self.static_offsets, self.static_sizes, self.static_strides))
        rank = len(result.get_shape())
        if len(offsets) != 1 or len(sizes) != rank or len(strides) != rank:
            raise VerifyException("reinterpretation needs one offset and one size/stride per result axis")
        for static, dynamic in ((offsets, self.offsets), (sizes, self.sizes), (strides, self.strides)):
            if sum(n == self.DYNAMIC_INDEX for n in static) != len(dynamic):
                raise VerifyException("reinterpretation dynamic operands do not match the static markers")
        if source.element_type != result.element_type or source.memory_space != result.memory_space:
            raise VerifyException("reinterpretation must retain element type and memory space")
        for actual, expected in zip(result.get_shape(), sizes, strict=True):
            if actual >= 0 and actual != expected:
                raise VerifyException("reinterpretation result sizes differ from their static attributes")
        layout_strides, layout_offset = result.get_strides(), result.get_offset()
        if layout_strides is None:
            raise VerifyException("reinterpretation result requires a strided layout")
        if layout_offset is not None and layout_offset != offsets[0]:
            raise VerifyException("reinterpretation result offset differs from its static attribute")
        for actual, expected in zip(layout_strides, strides, strict=True):
            if actual is not None and actual != expected:
                raise VerifyException("reinterpretation result strides differ from their static attributes")


@irdl_op_definition
class AtomicYieldOp(IRDLOperation):
    name = "memref.atomic_yield"
    value = operand_def()
    traits = traits_def(IsTerminator(), NoMemoryEffect())

    def verify_(self):
        if not isinstance(self.parent_op(), GenericAtomicRMWOp):
            raise VerifyException("atomic yield requires its standard generic atomic parent")


@irdl_op_definition
class GenericAtomicRMWOp(IRDLOperation):
    """Standard MemRefOps.td region atomic, absent from the pinned xDSL schema."""
    name = "memref.generic_atomic_rmw"
    memref = operand_def(builtin.MemRefType)
    indices = var_operand_def(builtin.IndexType)
    result = result_def()
    body = region_def("single_block")
    traits = traits_def(MemoryReadEffect(), MemoryWriteEffect())

    def verify_(self):
        element = self.memref.type.element_type
        block = self.body.block
        if (self.result.type != element or len(self.indices) != len(self.memref.type.get_shape())
                or len(block.args) != 1 or block.args[0].type != element
                or not isinstance(block.last_op, AtomicYieldOp) or block.last_op.value.type != element
                or any(not is_side_effect_free(op) for op in block.ops)):
            raise VerifyException("generic atomic requires matching element/index types and a pure typed atomic yield")


@cache
def memref_dialect():
    return Dialect("memref", [ReinterpretCastOp if op is memref.ReinterpretCastOp else op
                              for op in memref.MemRef.operations] + [GenericAtomicRMWOp, AtomicYieldOp], list(memref.MemRef.attributes))
