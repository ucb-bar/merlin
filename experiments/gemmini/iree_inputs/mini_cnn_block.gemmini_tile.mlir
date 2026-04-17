module {
  func.func @mini_cnn_block(%arg0: memref<1x3x32x32xf32>, %arg1: memref<16x3x3x3xf32>, %arg2: memref<32x16x3x3xf32>, %arg3: memref<1x32x26x26xf32>) {
    %alloc = memref.alloc() : memref<1x16x30x30xf32>
    %alloc_0 = memref.alloc() : memref<1x32x26x26xf32>
    %alloc_1 = memref.alloc() : memref<1x32x32x3xf32>
    %alloc_2 = memref.alloc() : memref<27x16xf32>
    %alloc_3 = memref.alloc() : memref<16xi32>
    %alloc_4 = memref.alloc() : memref<900x16xf32>
    %c30_i64 = arith.constant 30 : i64
    %c3 = arith.constant 3 : index
    %c3_5 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_6 = arith.constant 1 : index
    scf.for %arg4 = %c0 to %c1 step %c1_6 {
      %c0_27 = arith.constant 0 : index
      %c3_28 = arith.constant 3 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c3_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c32_31 = arith.constant 32 : index
        %c1_32 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c32_31 step %c1_32 {
          %c0_33 = arith.constant 0 : index
          %c32_34 = arith.constant 32 : index
          %c1_35 = arith.constant 1 : index
          scf.for %arg7 = %c0_33 to %c32_34 step %c1_35 {
            %0 = memref.load %arg0[%arg4, %arg5, %arg6, %arg7] : memref<1x3x32x32xf32>
            memref.store %0, %alloc_1[%arg4, %arg6, %arg7, %arg5] : memref<1x32x32x3xf32>
          }
        }
      }
    }
    %c0_7 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1_8 = arith.constant 1 : index
    scf.for %arg4 = %c0_7 to %c16 step %c1_8 {
      %c0_27 = arith.constant 0 : index
      %c3_28 = arith.constant 3 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c3_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c3_31 = arith.constant 3 : index
        %c1_32 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c3_31 step %c1_32 {
          %c0_33 = arith.constant 0 : index
          %c3_34 = arith.constant 3 : index
          %c1_35 = arith.constant 1 : index
          scf.for %arg7 = %c0_33 to %c3_34 step %c1_35 {
            %0 = arith.muli %arg6, %c3 : index
            %1 = arith.muli %0, %c3_5 : index
            %2 = arith.muli %arg7, %c3_5 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg5 : index
            %5 = memref.load %arg1[%arg4, %arg5, %arg6, %arg7] : memref<16x3x3x3xf32>
            memref.store %5, %alloc_2[%4, %arg4] : memref<27x16xf32>
          }
        }
      }
    }
    %c3_i64 = arith.constant 3 : i64
    gemmini.tile_conv %alloc_1 %alloc_2 %alloc_3 %alloc_4 %c30_i64 %c30_i64 %c3_i64 : memref<1x32x32x3xf32> memref<27x16xf32> memref<16xi32> memref<900x16xf32> i64 i64 i64
    %c0_9 = arith.constant 0 : index
    %c1_10 = arith.constant 1 : index
    %c1_11 = arith.constant 1 : index
    scf.for %arg4 = %c0_9 to %c1_10 step %c1_11 {
      %c0_27 = arith.constant 0 : index
      %c16_28 = arith.constant 16 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c16_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c30 step %c1_31 {
          %c0_32 = arith.constant 0 : index
          %c30_33 = arith.constant 30 : index
          %c1_34 = arith.constant 1 : index
          scf.for %arg7 = %c0_32 to %c30_33 step %c1_34 {
            %c30_35 = arith.constant 30 : index
            %0 = arith.muli %arg4, %c30_35 : index
            %1 = arith.muli %0, %c30_35 : index
            %2 = arith.muli %arg6, %c30_35 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg7 : index
            %5 = memref.load %alloc_4[%4, %arg5] : memref<900x16xf32>
            memref.store %5, %alloc[%arg4, %arg5, %arg6, %arg7] : memref<1x16x30x30xf32>
          }
        }
      }
    }
    memref.dealloc %alloc_1 : memref<1x32x32x3xf32>
    memref.dealloc %alloc_2 : memref<27x16xf32>
    memref.dealloc %alloc_4 : memref<900x16xf32>
    memref.dealloc %alloc_3 : memref<16xi32>
    %alloc_12 = memref.alloc() : memref<1x30x30x16xf32>
    %alloc_13 = memref.alloc() : memref<144x32xf32>
    %alloc_14 = memref.alloc() : memref<32xi32>
    %alloc_15 = memref.alloc() : memref<676x32xf32>
    %c26_i64 = arith.constant 26 : i64
    %c3_16 = arith.constant 3 : index
    %c16_17 = arith.constant 16 : index
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    %c1_20 = arith.constant 1 : index
    scf.for %arg4 = %c0_18 to %c1_19 step %c1_20 {
      %c0_27 = arith.constant 0 : index
      %c16_28 = arith.constant 16 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c16_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c30 step %c1_31 {
          %c0_32 = arith.constant 0 : index
          %c30_33 = arith.constant 30 : index
          %c1_34 = arith.constant 1 : index
          scf.for %arg7 = %c0_32 to %c30_33 step %c1_34 {
            %0 = memref.load %alloc[%arg4, %arg5, %arg6, %arg7] : memref<1x16x30x30xf32>
            memref.store %0, %alloc_12[%arg4, %arg6, %arg7, %arg5] : memref<1x30x30x16xf32>
          }
        }
      }
    }
    %c0_21 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1_22 = arith.constant 1 : index
    scf.for %arg4 = %c0_21 to %c32 step %c1_22 {
      %c0_27 = arith.constant 0 : index
      %c16_28 = arith.constant 16 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c16_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c3_31 = arith.constant 3 : index
        %c1_32 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c3_31 step %c1_32 {
          %c0_33 = arith.constant 0 : index
          %c3_34 = arith.constant 3 : index
          %c1_35 = arith.constant 1 : index
          scf.for %arg7 = %c0_33 to %c3_34 step %c1_35 {
            %0 = arith.muli %arg6, %c3_16 : index
            %1 = arith.muli %0, %c16_17 : index
            %2 = arith.muli %arg7, %c16_17 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg5 : index
            %5 = memref.load %arg2[%arg4, %arg5, %arg6, %arg7] : memref<32x16x3x3xf32>
            memref.store %5, %alloc_13[%4, %arg4] : memref<144x32xf32>
          }
        }
      }
    }
    %c3_i64_23 = arith.constant 3 : i64
    gemmini.tile_conv %alloc_12 %alloc_13 %alloc_14 %alloc_15 %c26_i64 %c26_i64 %c3_i64_23 : memref<1x30x30x16xf32> memref<144x32xf32> memref<32xi32> memref<676x32xf32> i64 i64 i64
    %c0_24 = arith.constant 0 : index
    %c1_25 = arith.constant 1 : index
    %c1_26 = arith.constant 1 : index
    scf.for %arg4 = %c0_24 to %c1_25 step %c1_26 {
      %c0_27 = arith.constant 0 : index
      %c32_28 = arith.constant 32 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c32_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c26 = arith.constant 26 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c26 step %c1_31 {
          %c0_32 = arith.constant 0 : index
          %c26_33 = arith.constant 26 : index
          %c1_34 = arith.constant 1 : index
          scf.for %arg7 = %c0_32 to %c26_33 step %c1_34 {
            %c26_35 = arith.constant 26 : index
            %0 = arith.muli %arg4, %c26_35 : index
            %1 = arith.muli %0, %c26_35 : index
            %2 = arith.muli %arg6, %c26_35 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg7 : index
            %5 = memref.load %alloc_15[%4, %arg5] : memref<676x32xf32>
            memref.store %5, %alloc_0[%arg4, %arg5, %arg6, %arg7] : memref<1x32x26x26xf32>
          }
        }
      }
    }
    memref.dealloc %alloc_12 : memref<1x30x30x16xf32>
    memref.dealloc %alloc_13 : memref<144x32xf32>
    memref.dealloc %alloc_15 : memref<676x32xf32>
    memref.dealloc %alloc_14 : memref<32xi32>
    linalg.copy ins(%alloc_0 : memref<1x32x26x26xf32>) outs(%arg3 : memref<1x32x26x26xf32>)
    memref.dealloc %alloc : memref<1x16x30x30xf32>
    memref.dealloc %alloc_0 : memref<1x32x26x26xf32>
    return
  }
}

