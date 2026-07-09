module {
  irdl.dialect @merlin_iface {
    irdl.type @"!acc" 
    irdl.type @"!resident" 
    irdl.operation @commit {
      %0 = irdl.base @merlin_iface::@"!acc" 
      %1 = irdl.c_pred "(::llvm::isa<::mlir::RankedTensorType>($_self))" 
      %2 = irdl.all_of(%1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.any
      %6 = irdl.c_pred "(::llvm::isa<::mlir::ArrayAttr>($_self))" 
      %7 = irdl.c_pred "::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); })" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any
      irdl.operands(src: %0)
      irdl.results(result: %4)
      irdl.attributes {"name" = %5, "epilogue" = %8, "output_dtype" = %9}
    }
    irdl.operation @conv2d {
      %0 = irdl.c_pred "(::llvm::isa<::mlir::RankedTensorType>($_self))" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.base @merlin_iface::@"!resident" 
      %5 = irdl.c_pred "(::llvm::isa<::mlir::RankedTensorType>($_self))" 
      %6 = irdl.all_of(%5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "(::llvm::isa<::mlir::ArrayAttr>($_self))" 
      %10 = irdl.c_pred "::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))); })" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "(::llvm::isa<::mlir::ArrayAttr>($_self))" 
      %13 = irdl.c_pred "::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))); })" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "(::llvm::isa<::mlir::ArrayAttr>($_self))" 
      %16 = irdl.c_pred "::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))); })" 
      %17 = irdl.all_of(%15, %16) 
      %18 = irdl.c_pred "(::llvm::isa<::mlir::ArrayAttr>($_self))" 
      %19 = irdl.c_pred "::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))); })" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any
      %22 = irdl.c_pred "(::llvm::isa<::mlir::ArrayAttr>($_self))" 
      %23 = irdl.c_pred "::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); })" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.any
      %26 = irdl.any
      irdl.operands(ifm: %3, weight: %4)
      irdl.results(result: %8)
      irdl.attributes {"kernel" = %11, "stride" = %14, "padding" = %17, "dilation" = %20, "name" = %21, "epilogue" = %24, "output_dtype" = %25, "layout" = %26}
    }
    irdl.operation @evict {
      %0 = irdl.base @merlin_iface::@"!resident" 
      irdl.operands(handle: %0)
    }
    irdl.operation @matmul {
      %0 = irdl.c_pred "(::llvm::isa<::mlir::RankedTensorType>($_self))" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.base @merlin_iface::@"!resident" 
      %5 = irdl.base @merlin_iface::@"!acc" 
      irdl.operands(lhs: %3, rhs: %4)
      irdl.results(result: %5)
    }
    irdl.operation @movement {
      %0 = irdl.c_pred "(::llvm::isa<::mlir::RankedTensorType>($_self))" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "(::llvm::isa<::mlir::RankedTensorType>($_self))" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.any
      irdl.operands(src: %3)
      irdl.results(result: %7)
      irdl.attributes {"name" = %8}
    }
    irdl.operation @resident_pack {
      %0 = irdl.c_pred "(::llvm::isa<::mlir::RankedTensorType>($_self))" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.base @merlin_iface::@"!resident" 
      %5 = irdl.any
      irdl.operands(src: %3)
      irdl.results(result: %4)
      irdl.attributes {"layout" = %5}
    }
    irdl.operation @tensor {
      %0 = irdl.c_pred "(::llvm::isa<::mlir::RankedTensorType>($_self))" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any
      %5 = irdl.any
      irdl.results(result: %3)
      irdl.attributes {"name" = %4, "role" = %5}
    }
  }
}
