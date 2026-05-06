// All-reduce sum test for cuda_new NCCL support.
// Compile: iree-compile --iree-hal-target-backends=cuda_tile \
//            --iree-cuda-tile-enable-codegen=false \
//            -o allreduce_f32.vmfb allreduce_f32.mlir
// Run (2 ranks):
//   mpirun -n 2 bash -c \
//     'CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_RANK \
//      iree-run-module --device=cuda_new \
//        --module=allreduce_f32.vmfb \
//        --function=all_reduce_sum \
//        --input=4xf32=...'

func.func @all_reduce_sum(%input: tensor<4xf32>) -> tensor<4xf32> {
  %channel = flow.channel.default : !flow.channel
  %empty = tensor.empty() : tensor<4xf32>
  %result = flow.collective.all_reduce sum, f32, %empty, %input, %channel
    : (tensor<4xf32>, tensor<4xf32>, !flow.channel) -> %empty as tensor<4xf32>
  return %result : tensor<4xf32>
}
