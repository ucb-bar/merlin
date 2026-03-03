

# iree-run-module   --device=local-sync   --module=dronet_sync.vmfb --module=fastdepth_sync.vmfb --module=our_pipeline_sync.vmfb   --function=run   --input=1x3x224x224xf32=0

# iree-run-module   --device=local-task   --module=dronet_async.vmfb --module=fastdepth_async.vmfb --module=our_pipeline_async.vmfb   --function=run   --input=1x3x224x224xf32=0

# iree-run-module   --device=local-task   --module=dronet_async.vmfb --module=fastdepth_async.vmfb --module=simple_mlp_async.vmfb --module=our_pipeline_full_async.vmfb   --function=run   --input=1x3x224x224xf32=0 --input=1x10xf32=0

iree-run-module   --device=local-task   --module=dronet_async.vmfb --module=fastdepth_async.vmfb --module=simple_mlp_async.vmfb --module=our_pipeline_full_async_looped.vmfb   --function=run_looped   --input=1x3x224x224xf32=0 --input=1x10xf32=0