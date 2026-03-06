# # Sync stuff
# iree-compile    --iree-hal-target-device=local   --iree-hal-local-target-device-backends=vmvx   /scratch2/merlin/samples/robotic-NN/mlir/phases_fastnet/fastdepth.1.input.mlir -o=fastdepth_sync.vmfb --compile-from=input  

# iree-compile   --iree-hal-target-device=local   --iree-hal-local-target-device-backends=vmvx   /scratch2/merlin/samples/robotic-NN/mlir/phases_dronet/dronet.1.input.mlir -o=dronet_sync.vmfb --compile-from=input  

# iree-compile     --iree-hal-target-device=local   --iree-hal-local-target-device-backends=vmvx   our_pipeline_async.mlir -o=our_pipeline_sync.vmfb 
# # Async stuff
# iree-compile    --iree-hal-target-device=local --iree-execution-model=async-external  --iree-execution-model=async-external --iree-hal-local-target-device-backends=vmvx   /scratch2/merlin/samples/robotic-NN/mlir/phases_fastnet/fastdepth.1.input.mlir -o=fastdepth_async.vmfb --compile-from=input  --dump-compilation-phases-to=/scratch2/merlin/samples/robotic-NN/mlir/phases_fastnet
iree-compile    --iree-hal-target-device=local --iree-execution-model=async-external  --iree-hal-local-target-device-backends=vmvx   workload_mlir/fastdepth.mlir -o=fastdepth_async.vmfb 

# iree-compile   --iree-hal-target-device=local --iree-execution-model=async-external --iree-execution-model=async-external  --iree-hal-local-target-device-backends=vmvx   /scratch2/merlin/samples/robotic-NN/mlir/phases_dronet/dronet.1.input.mlir -o=dronet_async.vmfb --compile-from=input  --dump-compilation-phases-to=/scratch2/merlin/samples/robotic-NN/mlir/phases_dronet
iree-compile   --iree-hal-target-device=local --iree-execution-model=async-external  --iree-hal-local-target-device-backends=vmvx   workload_mlir/dronet.mlir -o=dronet_async.vmfb 

iree-compile     --iree-hal-target-device=local --iree-execution-model=async-external  --iree-hal-local-target-device-backends=vmvx   async_mlir/our_pipeline_async.mlir -o=our_pipeline_async.vmfb --dump-compilation-phases-to=/scratch2/merlin/samples/robotic-NN/mlir/phases_pipelinels

# MLP Stuff
iree-compile    --iree-hal-target-device=local --iree-execution-model=async-external --iree-hal-local-target-device-backends=vmvx     workload_mlir/mlp.mlir -o=simple_mlp_async.vmfb --dump-compilation-phases-to=/scratch2/merlin/samples/robotic-NN/mlir/phases_simple_mlp

iree-compile     --iree-hal-target-device=local --iree-execution-model=async-external  --iree-hal-local-target-device-backends=vmvx   async_mlir/our_pipeline_full_async.mlir -o=our_pipeline_full_async.vmfb 

iree-compile     --iree-hal-target-device=local --iree-execution-model=async-external --iree-hal-local-target-device-backends=vmvx   async_mlir/our_pipeline_full_async_looped.mlir -o=our_pipeline_full_async_looped.vmfb 