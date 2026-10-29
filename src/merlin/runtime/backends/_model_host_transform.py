"""Compatibility imports for the independent public host LLVM dispatcher."""

from ...llvmlower.host_transform_dispatch import HostLLVMTransformError as SpikeModelError
from ...llvmlower.host_transform_dispatch import _transform_host_ir as _transform_host_ir

__all__ = ["SpikeModelError", "_transform_host_ir"]
