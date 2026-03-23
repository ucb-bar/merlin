// Test that the cuda_tile backend compiles an external cubin into a CTL1
// .vmfb that runs on --device=cuda_tile.
//
// Compile:
//   iree-compile smoketest.mlir \
//     --iree-hal-target-backends=cuda_tile \
//     --iree-cuda-tile-sm-arch=sm_86 \
//     --iree-hal-executable-object-search-path=/scratch/ashvin/cuda_tile_test/ \
//     -o /tmp/cuda_tile_test.vmfb
//
// Run:
//   iree-run-module --device=cuda_tile \
//     --module=/tmp/cuda_tile_test.vmfb \
//     --function=matmul \
//     --input=128x128xf32=1 \
//     --input=128x128xf32=1

// cuda_tile backend emits CTL1 format for the cuda_tile HAL driver.
#cuda_tile_target = #hal.executable.target<"cuda_tile", "cuda-tile-fb", {
  target_arch = "sm_86"
}>
#device = #hal.device.target<"cuda_tile", [#cuda_tile_target]> : !hal.device

module @cuda_tile_matmul_test attributes {hal.device.targets = [#device]} {

  hal.executable.source private @matmul_exe attributes {
    objects = #hal.executable.objects<{
      #cuda_tile_target = [
        #hal.executable.object<{path = "matmul_128x128.cubin"}>
      ]
    }>
  } {
    hal.executable.export public @matmul_f32 ordinal(0)
        layout(#hal.pipeline.layout<constants = 0, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer>
        ]>) count(%device: !hal.device) -> (index, index, index) {
      // Grid = (2,2,1): 4 CTAs covering 128x128 with 64x64 tiles.
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      hal.return %c2, %c2, %c1 : index, index, index
    } attributes {
      // cuda_tile manages threads internally — block = {1,1,1}.
      workgroup_size = [1 : index, 1 : index, 1 : index]
    }
  }

  func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %result = flow.dispatch @matmul_exe::@matmul_f32(%A, %B)
        : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}
