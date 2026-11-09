"""Checked standard bufferization bindings independent of physical layout."""

# Verification rules adapted from the LLVM Project's BufferizationOps.cpp.
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from xdsl.dialects import builtin
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, opt_prop_def, opt_result_def, result_def
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ToTensorOp(IRDLOperation):
    name = "bufferization.to_tensor"
    memref = operand_def(builtin.MemRefType | builtin.UnrankedMemRefType)
    tensor = result_def(builtin.TensorType | builtin.UnrankedTensorType)
    restrict = opt_prop_def(builtin.UnitAttr)
    writable = opt_prop_def(builtin.UnitAttr)
    assembly_format = "$memref (`restrict` $restrict^)? (`writable` $writable^)? attr-dict `:` type($memref) `to` type($tensor)"

    def verify_(self):
        source, result = self.memref.type, self.tensor.type
        if (isinstance(source, builtin.MemRefType) != isinstance(result, builtin.TensorType)
                or source.element_type != result.element_type):
            raise VerifyException("to_tensor requires matching buffer/tensor rank and element type")
        if isinstance(source, builtin.MemRefType) and source.get_shape() != result.get_shape():
            raise VerifyException("to_tensor requires identical buffer/tensor shapes")


@irdl_op_definition
class MaterializeInDestinationOp(IRDLOperation):
    """Standard MLIR shape/element relation; tensor types do not imply layout.

    Mirrors BufferizationOps.td's element-type trait and the operation verifier
    in upstream BufferizationOps.cpp. Access/alias promises remain explicit and
    independently require admission; this binding establishes no such proof.
    """

    name = "bufferization.materialize_in_destination"
    source = operand_def(builtin.TensorType | builtin.UnrankedTensorType)
    dest = operand_def(builtin.TensorType | builtin.UnrankedTensorType | builtin.MemRefType | builtin.UnrankedMemRefType)
    result = opt_result_def(builtin.TensorType | builtin.UnrankedTensorType)
    restrict = opt_prop_def(builtin.UnitAttr)
    writable = opt_prop_def(builtin.UnitAttr)
    assembly_format = "$source `in` (`restrict` $restrict^)? (`writable` $writable^)? $dest attr-dict `:` functional-type(operands, results)"

    def verify_(self):
        source, dest = self.source.type, self.dest.type
        memory = isinstance(dest, (builtin.MemRefType, builtin.UnrankedMemRefType))
        if memory and self.results:
            raise VerifyException("memref destination implies zero materialization results")
        if not memory and (len(self.results) != 1 or self.results[0].type != dest):
            raise VerifyException("tensor destination requires one result of its exact type")
        if self.restrict is not None and not memory:
            raise VerifyException("restrict is valid only for memref destinations")
        if (self.writable is not None) != memory:
            raise VerifyException("writable must be specified exactly for memref destinations")
        if source.element_type != dest.element_type:
            raise VerifyException("materialization source/destination element types must match")
        ranked_source = isinstance(source, builtin.TensorType)
        ranked_dest = isinstance(dest, (builtin.TensorType, builtin.MemRefType))
        if ranked_source != ranked_dest:
            raise VerifyException("materialization source/destination ranks are incompatible")
        if ranked_source:
            a, b = source.get_shape(), dest.get_shape()
            if len(a) != len(b) or any(x >= 0 and y >= 0 and x != y for x, y in zip(a, b, strict=True)):
                raise VerifyException("materialization source/destination shapes are incompatible")
