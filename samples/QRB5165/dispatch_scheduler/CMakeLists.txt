# Qualcomm QRB5165 (Snapdragon SA8155 / 865) dispatch-scheduler entry.
#
# Aliases the target-neutral merlin-dispatch-scheduler under
# samples/common/dispatch_scheduler/. The binary runs heterogeneous schedules
# across CPU_P/CPU_E (local-task or local-sync) plus QNN_GPU (Adreno via
# libQnnGpu.so) and QNN_HTA (Hexagon via libQnnHta.so) when those drivers are
# present at runtime.
#
# This directory exists as the canonical Qualcomm-target home for the entry; the
# same binary is also reachable via the legacy
# samples/SpacemiTX60/dispatch_scheduler/ alias.

# Re-export the target-neutral target as already built by
# samples/common/dispatch_scheduler/. No new sources here — keep this directory
# as a hint to anyone navigating the tree.
