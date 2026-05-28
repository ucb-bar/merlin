#ifndef IREE_RADIANCE_DIALECT_RADIANCE_IR_RADIANCEATTRS_TD_
#define IREE_RADIANCE_DIALECT_RADIANCE_IR_RADIANCEATTRS_TD_

include "RadianceDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

// Address-space enum (radiance.AddrSpace) and its dialect-attr wrapper
// (#radiance.global / #radiance.shared, parsed long-form as
// #radiance.addrspace<global>). The integer values stored in the enum
// (Global=1, Shared=3) are an internal convention; ConvertRadianceAddrSpaces
// translates them to the Muon ABI's LLVM-addrspace mapping
// (radiance-kernels/lib/include/shared_mem.h):
//   __global -> addrspace(0)   (DRAM / ext_gpu_mem)
//   __shared -> addrspace(1)   (cluster L0/L1 SMEM)
//
// We define the enum via I32EnumAttrBase + the dialect-attr wrapper via
// EnumAttr<Radiance_Dialect, ...> (SPIRV-style). This produces a real
// dialect attribute class — necessary so MemRefType's
// isSupportedMemorySpace verifier accepts it as a non-builtin attr.
def Radiance_AddrSpaceEnum : I32Enum<
    "AddrSpace", "Radiance memory address space",
    [
      I32EnumCase<"Global", 1, "global">,
      I32EnumCase<"Shared", 3, "shared">
    ]> {
  let cppNamespace = "::mlir::iree_compiler::Radiance";
}

def Radiance_AddrSpaceAttr : EnumAttr<
    Radiance_Dialect, Radiance_AddrSpaceEnum, "addrspace"> {
  let cppNamespace = "::mlir::iree_compiler::Radiance";
  let assemblyFormat = "`<` $value `>`";
}

#endif  // IREE_RADIANCE_DIALECT_RADIANCE_IR_RADIANCEATTRS_TD_
