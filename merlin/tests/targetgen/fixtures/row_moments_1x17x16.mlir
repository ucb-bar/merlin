builtin.module {
  func.func @row_moments(%x: tensor<1x17x16xf32>) -> (tensor<1x17x1xf32>, tensor<1x17x1xf32>) {
    %0 = "tensor.empty"() : () -> tensor<1x17x1xf32>
    %1 = "tensor.empty"() : () -> tensor<1x17x1xf32>
    %2, %3 = "linalg.generic"(%0, %1) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 2>}> ({
    ^bb0(%4: f32, %5: f32):
      %6 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %7 = "linalg.index"() <{dim = 1 : i64}> : () -> index
      %8 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %9 = "arith.constant"() <{value = 0 : index}> : () -> index
      %10 = "arith.constant"() <{value = 2 : index}> : () -> index
      %11 = "arith.constant"() <{value = 1 : index}> : () -> index
      %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27 = "scf.for"(%9, %10, %11, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8) ({
      ^bb1(%28: index, %29: f32, %30: f32, %31: f32, %32: f32, %33: f32, %34: f32, %35: f32, %36: f32, %37: f32, %38: f32, %39: f32, %40: f32, %41: f32, %42: f32, %43: f32, %44: f32):
        %45 = "arith.constant"() <{value = 1 : index}> : () -> index
        %46 = "arith.addi"(%28, %45) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %47 = "arith.index_cast"(%46) : (index) -> i64
        %48 = "arith.sitofp"(%47) : (i64) -> f32
        %49 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
        %50 = "arith.divf"(%49, %48) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %51 = "arith.constant"() <{value = 8 : index}> : () -> index
        %52 = "arith.muli"(%28, %51) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %53 = "arith.constant"() <{value = 0 : index}> : () -> index
        %54 = "arith.addi"(%52, %53) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %55 = "tensor.extract"(%x, %6, %7, %54) : (tensor<1x17x16xf32>, index, index, index) -> f32
        %57 = "arith.subf"(%55, %29) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %58 = "math.fma"(%57, %50, %29) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %59 = "arith.subf"(%55, %58) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %60 = "math.fma"(%57, %59, %37) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %61 = "arith.constant"() <{value = 8 : index}> : () -> index
        %62 = "arith.muli"(%28, %61) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %63 = "arith.constant"() <{value = 1 : index}> : () -> index
        %64 = "arith.addi"(%62, %63) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %65 = "tensor.extract"(%x, %6, %7, %64) : (tensor<1x17x16xf32>, index, index, index) -> f32
        %66 = "arith.subf"(%65, %30) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %67 = "math.fma"(%66, %50, %30) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %68 = "arith.subf"(%65, %67) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %69 = "math.fma"(%66, %68, %38) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %70 = "arith.constant"() <{value = 8 : index}> : () -> index
        %71 = "arith.muli"(%28, %70) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %72 = "arith.constant"() <{value = 2 : index}> : () -> index
        %73 = "arith.addi"(%71, %72) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %74 = "tensor.extract"(%x, %6, %7, %73) : (tensor<1x17x16xf32>, index, index, index) -> f32
        %75 = "arith.subf"(%74, %31) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %76 = "math.fma"(%75, %50, %31) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %77 = "arith.subf"(%74, %76) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %78 = "math.fma"(%75, %77, %39) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %79 = "arith.constant"() <{value = 8 : index}> : () -> index
        %80 = "arith.muli"(%28, %79) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %81 = "arith.constant"() <{value = 3 : index}> : () -> index
        %82 = "arith.addi"(%80, %81) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %83 = "tensor.extract"(%x, %6, %7, %82) : (tensor<1x17x16xf32>, index, index, index) -> f32
        %84 = "arith.subf"(%83, %32) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %85 = "math.fma"(%84, %50, %32) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %86 = "arith.subf"(%83, %85) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %87 = "math.fma"(%84, %86, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %88 = "arith.constant"() <{value = 8 : index}> : () -> index
        %89 = "arith.muli"(%28, %88) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %90 = "arith.constant"() <{value = 4 : index}> : () -> index
        %91 = "arith.addi"(%89, %90) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %92 = "tensor.extract"(%x, %6, %7, %91) : (tensor<1x17x16xf32>, index, index, index) -> f32
        %93 = "arith.subf"(%92, %33) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %94 = "math.fma"(%93, %50, %33) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %95 = "arith.subf"(%92, %94) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %96 = "math.fma"(%93, %95, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %97 = "arith.constant"() <{value = 8 : index}> : () -> index
        %98 = "arith.muli"(%28, %97) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %99 = "arith.constant"() <{value = 5 : index}> : () -> index
        %100 = "arith.addi"(%98, %99) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %101 = "tensor.extract"(%x, %6, %7, %100) : (tensor<1x17x16xf32>, index, index, index) -> f32
        %102 = "arith.subf"(%101, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %103 = "math.fma"(%102, %50, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %104 = "arith.subf"(%101, %103) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %105 = "math.fma"(%102, %104, %42) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %106 = "arith.constant"() <{value = 8 : index}> : () -> index
        %107 = "arith.muli"(%28, %106) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %108 = "arith.constant"() <{value = 6 : index}> : () -> index
        %109 = "arith.addi"(%107, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %110 = "tensor.extract"(%x, %6, %7, %109) : (tensor<1x17x16xf32>, index, index, index) -> f32
        %111 = "arith.subf"(%110, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %112 = "math.fma"(%111, %50, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %113 = "arith.subf"(%110, %112) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %114 = "math.fma"(%111, %113, %43) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %115 = "arith.constant"() <{value = 8 : index}> : () -> index
        %116 = "arith.muli"(%28, %115) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %117 = "arith.constant"() <{value = 7 : index}> : () -> index
        %118 = "arith.addi"(%116, %117) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %119 = "tensor.extract"(%x, %6, %7, %118) : (tensor<1x17x16xf32>, index, index, index) -> f32
        %120 = "arith.subf"(%119, %36) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %121 = "math.fma"(%120, %50, %36) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %122 = "arith.subf"(%119, %121) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %123 = "math.fma"(%120, %122, %44) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        "scf.yield"(%58, %67, %76, %85, %94, %103, %112, %121, %60, %69, %78, %87, %96, %105, %114, %123) : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> ()
      }) : (index, index, index, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
      %124 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %125 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %126 = "arith.divf"(%124, %125) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %127 = "arith.subf"(%12, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %128 = "arith.mulf"(%126, %127) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %129 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %130 = "arith.mulf"(%127, %129) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %131 = "arith.addf"(%8, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %132 = "arith.addf"(%8, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %133 = "math.fma"(%130, %128, %132) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %134 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %135 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %136 = "arith.divf"(%134, %135) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %137 = "arith.subf"(%13, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %138 = "arith.mulf"(%136, %137) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %139 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %140 = "arith.mulf"(%137, %139) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %141 = "arith.addf"(%8, %138) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %142 = "arith.addf"(%8, %21) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %143 = "math.fma"(%140, %138, %142) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %144 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %145 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %146 = "arith.divf"(%144, %145) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %147 = "arith.subf"(%14, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %148 = "arith.mulf"(%146, %147) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %149 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %150 = "arith.mulf"(%147, %149) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %151 = "arith.addf"(%8, %148) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %152 = "arith.addf"(%8, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %153 = "math.fma"(%150, %148, %152) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %154 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %155 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %156 = "arith.divf"(%154, %155) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %157 = "arith.subf"(%15, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %158 = "arith.mulf"(%156, %157) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %159 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %160 = "arith.mulf"(%157, %159) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %161 = "arith.addf"(%8, %158) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %162 = "arith.addf"(%8, %23) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %163 = "math.fma"(%160, %158, %162) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %164 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %165 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %166 = "arith.divf"(%164, %165) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %167 = "arith.subf"(%16, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %168 = "arith.mulf"(%166, %167) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %169 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %170 = "arith.mulf"(%167, %169) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %171 = "arith.addf"(%8, %168) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %172 = "arith.addf"(%8, %24) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %173 = "math.fma"(%170, %168, %172) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %174 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %175 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %176 = "arith.divf"(%174, %175) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %177 = "arith.subf"(%17, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %178 = "arith.mulf"(%176, %177) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %179 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %180 = "arith.mulf"(%177, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %181 = "arith.addf"(%8, %178) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %182 = "arith.addf"(%8, %25) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %183 = "math.fma"(%180, %178, %182) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %184 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %185 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %186 = "arith.divf"(%184, %185) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %187 = "arith.subf"(%18, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %188 = "arith.mulf"(%186, %187) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %189 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %190 = "arith.mulf"(%187, %189) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %191 = "arith.addf"(%8, %188) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %192 = "arith.addf"(%8, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %193 = "math.fma"(%190, %188, %192) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %194 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %195 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %196 = "arith.divf"(%194, %195) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %197 = "arith.subf"(%19, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %198 = "arith.mulf"(%196, %197) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %199 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %200 = "arith.mulf"(%197, %199) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %201 = "arith.addf"(%8, %198) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %202 = "arith.addf"(%8, %27) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %203 = "math.fma"(%200, %198, %202) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %204 = "arith.constant"() <{value = 16 : index}> : () -> index
      %205 = "arith.constant"() <{value = 16 : index}> : () -> index
      %206 = "arith.constant"() <{value = 1 : index}> : () -> index
      %207, %208 = "scf.for"(%204, %205, %206, %8, %8) ({
      ^bb2(%209: index, %210: f32, %211: f32):
        %212 = "tensor.extract"(%x, %6, %7, %209) : (tensor<1x17x16xf32>, index, index, index) -> f32
        %213 = "arith.subf"(%212, %210) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %214 = "arith.constant"() <{value = 16 : index}> : () -> index
        %215 = "arith.subi"(%209, %214) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %216 = "arith.constant"() <{value = 1 : index}> : () -> index
        %217 = "arith.addi"(%215, %216) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %218 = "arith.index_cast"(%217) : (index) -> i64
        %219 = "arith.sitofp"(%218) : (i64) -> f32
        %220 = "arith.divf"(%213, %219) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %221 = "arith.addf"(%210, %220) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %222 = "arith.subf"(%212, %221) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %223 = "arith.mulf"(%213, %222) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %224 = "arith.addf"(%211, %223) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "scf.yield"(%221, %224) : (f32, f32) -> ()
      }) : (index, index, index, f32, f32) -> (f32, f32)
      %225 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %226 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %227 = "arith.divf"(%225, %226) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %228 = "arith.subf"(%131, %207) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %229 = "math.fma"(%227, %228, %207) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %230 = "arith.mulf"(%228, %228) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %231 = "arith.mulf"(%230, %227) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %232 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %233 = "math.fma"(%231, %232, %133) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %234 = "arith.addf"(%208, %233) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %235 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %236 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %237 = "arith.divf"(%235, %236) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %238 = "arith.subf"(%141, %229) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %239 = "math.fma"(%237, %238, %229) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %240 = "arith.mulf"(%238, %238) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %241 = "arith.mulf"(%240, %237) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %242 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %243 = "math.fma"(%241, %242, %143) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %244 = "arith.addf"(%234, %243) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %245 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %246 = "arith.constant"() <{value = 6.000000e+00 : f32}> : () -> f32
      %247 = "arith.divf"(%245, %246) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %248 = "arith.subf"(%151, %239) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %249 = "math.fma"(%247, %248, %239) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %250 = "arith.mulf"(%248, %248) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %251 = "arith.mulf"(%250, %247) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %252 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %253 = "math.fma"(%251, %252, %153) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %254 = "arith.addf"(%244, %253) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %255 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %256 = "arith.constant"() <{value = 8.000000e+00 : f32}> : () -> f32
      %257 = "arith.divf"(%255, %256) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %258 = "arith.subf"(%161, %249) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %259 = "math.fma"(%257, %258, %249) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %260 = "arith.mulf"(%258, %258) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %261 = "arith.mulf"(%260, %257) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %262 = "arith.constant"() <{value = 6.000000e+00 : f32}> : () -> f32
      %263 = "math.fma"(%261, %262, %163) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %264 = "arith.addf"(%254, %263) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %265 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %266 = "arith.constant"() <{value = 1.000000e+01 : f32}> : () -> f32
      %267 = "arith.divf"(%265, %266) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %268 = "arith.subf"(%171, %259) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %269 = "math.fma"(%267, %268, %259) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %270 = "arith.mulf"(%268, %268) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %271 = "arith.mulf"(%270, %267) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %272 = "arith.constant"() <{value = 8.000000e+00 : f32}> : () -> f32
      %273 = "math.fma"(%271, %272, %173) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %274 = "arith.addf"(%264, %273) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %275 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %276 = "arith.constant"() <{value = 1.200000e+01 : f32}> : () -> f32
      %277 = "arith.divf"(%275, %276) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %278 = "arith.subf"(%181, %269) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %279 = "math.fma"(%277, %278, %269) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %280 = "arith.mulf"(%278, %278) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %281 = "arith.mulf"(%280, %277) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %282 = "arith.constant"() <{value = 1.000000e+01 : f32}> : () -> f32
      %283 = "math.fma"(%281, %282, %183) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %284 = "arith.addf"(%274, %283) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %285 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %286 = "arith.constant"() <{value = 1.400000e+01 : f32}> : () -> f32
      %287 = "arith.divf"(%285, %286) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %288 = "arith.subf"(%191, %279) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %289 = "math.fma"(%287, %288, %279) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %290 = "arith.mulf"(%288, %288) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %291 = "arith.mulf"(%290, %287) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %292 = "arith.constant"() <{value = 1.200000e+01 : f32}> : () -> f32
      %293 = "math.fma"(%291, %292, %193) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %294 = "arith.addf"(%284, %293) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %295 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %296 = "arith.constant"() <{value = 1.600000e+01 : f32}> : () -> f32
      %297 = "arith.divf"(%295, %296) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %298 = "arith.subf"(%201, %289) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %299 = "math.fma"(%297, %298, %289) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %300 = "arith.mulf"(%298, %298) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %301 = "arith.mulf"(%300, %297) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %302 = "arith.constant"() <{value = 1.400000e+01 : f32}> : () -> f32
      %303 = "math.fma"(%301, %302, %203) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %304 = "arith.addf"(%294, %303) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %305 = "arith.constant"() <{value = 1.600000e+01 : f32}> : () -> f32
      %306 = "arith.divf"(%304, %305) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%299, %306) : (f32, f32) -> ()
    }) {prov.aten = "aten.native_layer_norm.default"} : (tensor<1x17x1xf32>, tensor<1x17x1xf32>) -> (tensor<1x17x1xf32>, tensor<1x17x1xf32>)
    func.return %2, %3 : tensor<1x17x1xf32>, tensor<1x17x1xf32>
  }
}
