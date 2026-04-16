e_mlp_opu

Script done on 2026-04-14 18:55:13-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 18:58:09-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_bmm_4x64x64_opu0-bench_model_mt_bmm_4x64x64_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_bmm_4x64x64_opu0-bench_model_mt_bmm_4x64x64_opu   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_bmm_4x64x64_opu0-bench_model_mt_bmm_4x64x64_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_bmm_4x64x64_opu0-bench_model_mt_bmm_4x64x64_opu
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_bmm_4x64x64, Variant: OPU
Input shape: 4x64x32 (8192 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243365)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_reduction_256x64x32_f32 wg_count=16,1,1
bash: line 1: 2515284 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_bmm_4x64x64_opu0-bench_model_mt_bmm_4x64x64_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_bmm_4x64x64_opu0-bench_model_mt_bmm_4x64x64_opu

Script done on 2026-04-14 19:00:04-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:02:43-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_bmm_4x64x64_rvv0-bench_model_mt_bmm_4x64x64_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_bmm_4x64x64_rvv0-bench_model_mt_bmm_4x64x64_rvv   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_bmm_4x64x64_rvv0-bench_model_mt_bmm_4x64x64_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_bmm_4x64x64_rvv0-bench_model_mt_bmm_4x64x64_rvv
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_bmm_4x64x64, Variant: RVV
Input shape: 4x64x32 (8192 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243375)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_reduction_256x64x32_f32 wg_count=16,1,1
bash: line 1: 2539792 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_bmm_4x64x64_rvv0-bench_model_mt_bmm_4x64x64_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_bmm_4x64x64_rvv0-bench_model_mt_bmm_4x64x64_rvv

Script done on 2026-04-14 19:04:38-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:07:13-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_cvt_64_opu0-bench_model_mt_cvt_64_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_cvt_64_opu0-bench_model_mt_cvt_64_opu   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_cvt_64_opu0-bench_model_mt_cvt_64_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_cvt_64_opu0-bench_model_mt_cvt_64_opu
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_cvt_64, Variant: OPU
Input shape: 64 (64 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243373)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=380 ret=0
Warmup iter 1/2 done (cycle=1238963180, delta=1136719807)
Warmup iter 2/2 enter (cycle=1273026144)
[vm_invoke] #2 entering
[apply] #2 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=337 ret=0
Warmup iter 2/2 done (cycle=2411746329, delta=1138720185)
Benchmark START (10 iterations)
Bench iter 1/10
[vm_invoke] #3 entering
[apply] #3 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=323 ret=0
Bench iter 2/10
[vm_invoke] #4 entering
[apply] #4 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=326 ret=0
Bench iter 3/10
[vm_invoke] #5 entering
[apply] #5 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=328 ret=0
Bench iter 4/10
[vm_invoke] #6 entering
[apply] #6 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0bash: line 1: 2563727 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_cvt_64_opu0-bench_model_mt_cvt_64_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_cvt_64_opu0-bench_model_mt_cvt_64_opu

Script done on 2026-04-14 19:09:09-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:11:58-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_cvt_64_rvv0-bench_model_mt_cvt_64_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_cvt_64_rvv0-bench_model_mt_cvt_64_rvv   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_cvt_64_rvv0-bench_model_mt_cvt_64_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_cvt_64_rvv0-bench_model_mt_cvt_64_rvv
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_cvt_64, Variant: RVV
Input shape: 64 (64 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243385)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=413 ret=0
Warmup iter 1/2 done (cycle=1238963026, delta=1136719641)
Warmup iter 2/2 enter (cycle=1273026166)
[vm_invoke] #2 entering
[apply] #2 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=306 ret=0
Warmup iter 2/2 done (cycle=2411745691, delta=1138719525)
Benchmark START (10 iterations)
Bench iter 1/10
[vm_invoke] #3 entering
[apply] #3 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=335 ret=0
Bench iter 2/10
[vm_invoke] #4 entering
[apply] #4 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=370 ret=0
Bench iter 3/10
[vm_invoke] #5 entering
[apply] #5 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=318 ret=0
Bench iter 4/10
[vm_invoke] #6 entering
[apply] #6 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0bash: line 1: 2593727 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_cvt_64_rvv0-bench_model_mt_cvt_64_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_cvt_64_rvv0-bench_model_mt_cvt_64_rvv

Script done on 2026-04-14 19:13:53-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:16:27-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_divf_vv_64_opu0-bench_model_mt_divf_vv_64_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_divf_vv_64_opu0-bench_model_mt_divf_vv_64_opu   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_divf_vv_64_opu0-bench_model_mt_divf_vv_64_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_divf_vv_64_opu0-bench_model_mt_divf_vv_64_opu
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_divf_vv_64, Variant: OPU
Input shape: 64 (64 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243390)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=1641 ret=0
Warmup iter 1/2 done (cycle=1238962952, delta=1136719562)
Warmup iter 2/2 enter (cycle=1273026165)
[vm_invoke] #2 entering
[apply] #2 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=1601 ret=0
Warmup iter 2/2 done (cycle=2411746563, delta=1138720398)
Benchmark START (10 iterations)
Bench iter 1/10
[vm_invoke] #3 entering
[apply] #3 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=1549 ret=0
Bench iter 2/10
[vm_invoke] #4 entering
[apply] #4 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=1587 ret=0
Bench iter 3/10
[vm_invoke] #5 entering
[apply] #5 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=1579 ret=0
Bench iter 4/10
[vm_invoke] #6 entering
[apply] #6 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0bash: line 1: 2617656 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_divf_vv_64_opu0-bench_model_mt_divf_vv_64_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_divf_vv_64_opu0-bench_model_mt_divf_vv_64_opu

Script done on 2026-04-14 19:18:22-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:20:58-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_divf_vv_64_rvv0-bench_model_mt_divf_vv_64_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_divf_vv_64_rvv0-bench_model_mt_divf_vv_64_rvv   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_divf_vv_64_rvv0-bench_model_mt_divf_vv_64_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_divf_vv_64_rvv0-bench_model_mt_divf_vv_64_rvv
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_divf_vv_64, Variant: RVV
Input shape: 64 (64 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243444)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=1605 ret=0
Warmup iter 1/2 done (cycle=1238962965, delta=1136719521)
Warmup iter 2/2 enter (cycle=1273026147)
[vm_invoke] #2 entering
[apply] #2 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=1549 ret=0
Warmup iter 2/2 done (cycle=2411746365, delta=1138720218)
Benchmark START (10 iterations)
Bench iter 1/10
[vm_invoke] #3 entering
[apply] #3 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=1579 ret=0
Bench iter 2/10
[vm_invoke] #4 entering
[apply] #4 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=1610 ret=0
Bench iter 3/10
[vm_invoke] #5 entering
[apply] #5 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=1605 ret=0
Bench iter 4/10
[vm_invoke] #6 entering
[apply] #6 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0bash: line 1: 2641929 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_divf_vv_64_rvv0-bench_model_mt_divf_vv_64_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_divf_vv_64_rvv0-bench_model_mt_divf_vv_64_rvv

Script done on 2026-04-14 19:22:53-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:25:27-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_erf_64xf32_opu0-bench_model_mt_erf_64xf32_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_erf_64xf32_opu0-bench_model_mt_erf_64xf32_opu   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_erf_64xf32_opu0-bench_model_mt_erf_64xf32_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_erf_64xf32_opu0-bench_model_mt_erf_64xf32_opu
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_erf_64xf32, Variant: OPU
Input shape: 64 (64 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243379)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=3300 ret=0
Warmup iter 1/2 done (cycle=1238962865, delta=1136719486)
Warmup iter 2/2 enter (cycle=1273026150)
[vm_invoke] #2 entering
[apply] #2 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=3266 ret=0
Warmup iter 2/2 done (cycle=2411746121, delta=1138719971)
Benchmark START (10 iterations)
Bench iter 1/10
[vm_invoke] #3 entering
[apply] #3 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=3288 ret=0
Bench iter 2/10
[vm_invoke] #4 entering
[apply] #4 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=3278 ret=0
Bench iter 3/10
[vm_invoke] #5 entering
[apply] #5 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=3282 ret=0
Bench iter 4/10
[vm_invoke] #6 entering
[apply] #6 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0bash: line 1: 2665649 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_erf_64xf32_opu0-bench_model_mt_erf_64xf32_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_erf_64xf32_opu0-bench_model_mt_erf_64xf32_opu

Script done on 2026-04-14 19:27:23-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:29:57-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_erf_64xf32_rvv0-bench_model_mt_erf_64xf32_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_erf_64xf32_rvv0-bench_model_mt_erf_64xf32_rvv   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_erf_64xf32_rvv0-bench_model_mt_erf_64xf32_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_erf_64xf32_rvv0-bench_model_mt_erf_64xf32_rvv
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_erf_64xf32, Variant: RVV
Input shape: 64 (64 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243396)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=3372 ret=0
Warmup iter 1/2 done (cycle=1238962786, delta=1136719390)
Warmup iter 2/2 enter (cycle=1273026144)
[vm_invoke] #2 entering
[apply] #2 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=3338 ret=0
Warmup iter 2/2 done (cycle=2411746039, delta=1138719895)
Benchmark START (10 iterations)
Bench iter 1/10
[vm_invoke] #3 entering
[apply] #3 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=3314 ret=0
Bench iter 2/10
[vm_invoke] #4 entering
[apply] #4 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=3317 ret=0
Bench iter 3/10
[vm_invoke] #5 entering
[apply] #5 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0 cyc=3318 ret=0
Bench iter 4/10
[vm_invoke] #6 entering
[apply] #6 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_64_f32 wg_count=1,1,1
[dc] o=0 wg=0,0,0bash: line 1: 2689497 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_erf_64xf32_rvv0-bench_model_mt_erf_64xf32_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_erf_64xf32_rvv0-bench_model_mt_erf_64xf32_rvv

Script done on 2026-04-14 19:31:51-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:34:26-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_matmul_64x128_opu0-bench_model_mt_matmul_64x128_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_matmul_64x128_opu0-bench_model_mt_matmul_64x128_opu   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_matmul_64x128_opu0-bench_model_mt_matmul_64x128_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_matmul_64x128_opu0-bench_model_mt_matmul_64x128_opu
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_matmul_64x128, Variant: OPU
Input shape: 64x128 (8192 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243397)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_reduction_64x128x128_f32 wg_count=16,1,1
bash: line 1: 2714777 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_matmul_64x128_opu0-bench_model_mt_matmul_64x128_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_matmul_64x128_opu0-bench_model_mt_matmul_64x128_opu

Script done on 2026-04-14 19:36:21-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:38:55-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_matmul_64x128_opu_llm0-bench_model_mt_matmul_64x128_opu_llm-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_matmul_64x128_opu_llm0-bench_model_mt_matmul_64x128_opu_llm   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_matmul_64x128_opu_llm0-bench_model_mt_matmul_64x128_opu_llm-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_matmul_64x128_opu_llm0-bench_model_mt_matmul_64x128_opu_llm
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_matmul_64x128, Variant: OPU_LLM
Input shape: 64x128 (8192 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243380)
[vm_invoke] #1 entering
[apply] #1 ord=2 bindings=2
[dn] o=2 sym=_encoding_1_encode_64x128xf32_to_64x128xf32 wg_count=16,1,1
[dc] o=2 wg=0,0,0 cyc=1398 ret=0
[dc] o=2 wg=1,0,0 cyc=1322 ret=0
[dc] o=2 wg=2,0,0 cyc=1262 ret=0
[dc] o=2 wg=3,0,0 cyc=1208 ret=0
[dc] o=2 wg=4,0,0 cyc=1250 ret=0
[dc] o=2 wg=5,0,0 cyc=1310 ret=0
[dc] o=2 wg=6,0,0 cyc=1256 ret=0
[dc] o=2 wg=7,0,0 cyc=1214 ret=0
[dc] o=2 wg=8,0,0 cyc=1250 ret=0
[dc] o=2 wg=9,0,0 cyc=1334 ret=0
[dc] o=2 wg=10,0,0 cyc=1238 ret=0
[dc] o=2 wg=11,0,0 cyc=1250 ret=0
[dc] o=2 wg=12,0,0 cyc=1316 ret=0
[dc] o=2 wg=13,0,0 cyc=1250 ret=0
[dc] o=2 wg=14,bash: line 1: 2738272 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_matmul_64x128_opu_llm0-bench_model_mt_matmul_64x128_opu_llm-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_matmul_64x128_opu_llm0-bench_model_mt_matmul_64x128_opu_llm

Script done on 2026-04-14 19:40:50-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:43:24-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_matmul_64x128_rvv0-bench_model_mt_matmul_64x128_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_matmul_64x128_rvv0-bench_model_mt_matmul_64x128_rvv   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_matmul_64x128_rvv0-bench_model_mt_matmul_64x128_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_matmul_64x128_rvv0-bench_model_mt_matmul_64x128_rvv
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_matmul_64x128, Variant: RVV
Input shape: 64x128 (8192 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243382)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_reduction_64x128x128_f32 wg_count=16,1,1
bash: line 1: 2762120 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_matmul_64x128_rvv0-bench_model_mt_matmul_64x128_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_matmul_64x128_rvv0-bench_model_mt_matmul_64x128_rvv

Script done on 2026-04-14 19:45:19-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:47:53-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_mm_i8_64x128_opu0-bench_model_mt_mm_i8_64x128_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_mm_i8_64x128_opu0-bench_model_mt_mm_i8_64x128_opu   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_mm_i8_64x128_opu0-bench_model_mt_mm_i8_64x128_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_mm_i8_64x128_opu0-bench_model_mt_mm_i8_64x128_opu
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_mm_i8_64x128, Variant: OPU
Input shape: 64x128 (8192 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243405)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_8192_f32xi8 wg_count=2,1,1
[dc] o=0 wg=0,0,0 cyc=11578 ret=0
[dc] o=0 wg=1,0,0 cyc=11334 ret=0
[apply] #2 ord=1 bindings=2
[dn] o=1 sym=main_dispatch_1_reduction_64x128x128_i8xi32 wg_count=16,1,1
[dc] o=1 wg=0,0,0 cyc=41050 ret=0
[dc] o=1 wg=1,0,0 cyc=41007 ret=0
[dc] o=1 wg=2,0,0 cyc=40926 ret=0
[dc] o=1 wg=3,0,0 cyc=41122 ret=0
[dc] o=1 wg=4,0,0 cyc=41003 ret=0
[dc] o=1 wg=5,0,0 cyc=41089 ret=0
[dc] o=1 wg=6,0,0 cyc=40803 ret=0
[dc] o=1 wg=7,0,0 cyc=41013 ret=0
[dc] o=1 wg=8,0,0 cyc=40898 ret=0
[dc] o=1 wg=9,0,0 cyc=41030 ret=0
[dc] o=1 wg=10,0,0 cyc=41009bash: line 1: 2785931 Killed                  ./FireSim-xilinx_alveo_u250 +permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_mm_i8_64x128_opu0-bench_model_mt_mm_i8_64x128_opu-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_mm_i8_64x128_opu0-bench_model_mt_mm_i8_64x128_opu

Script done on 2026-04-14 19:49:48-07:00 [COMMAND_EXIT_CODE="137"]
tail: /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog: file truncated
Script started on 2026-04-14 19:52:23-07:00 [COMMAND="stty intr ^] &&  ./FireSim-xilinx_alveo_u250 +permissive   +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0  +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_mm_i8_64x128_rvv0-bench_model_mt_mm_i8_64x128_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE  +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default  +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off +prog0=merlin-bench-bench_model_mt_mm_i8_64x128_rvv0-bench_model_mt_mm_i8_64x128_rvv   && stty intr ^c" TERM="screen" TTY="/dev/pts/44" COLUMNS="80" LINES="24"]
+domain found: 0x0000
+bus found: 0x42
+device found: 0x00
+function found: 0x0
+bar found: 0x0
Using: 0000:42:00.0, BAR ID: 0, PCI Vendor ID: 0x10ee, PCI Device ID: 0x903f
Opening /sys/bus/pci/devices/0000:42:00.0/vendor
Opening /sys/bus/pci/devices/0000:42:00.0/device
examining xdma/.
examining xdma/..
examining xdma/xdma0_h2c_0
Using xdma write queue: /dev/xdma0_h2c_0
Using xdma read queue: /dev/xdma0_c2h_0
widget_registry_t::add_widget(StreamEngine)
cpu2fpga: 0, fpga2cpu: 1
UART0 is here (stdin/stdout).
TraceRV 0: Tracing disabled, since +tracefile was not provided.
command line for program 0. argc=24:
+permissive +macaddr0=00:12:6D:00:00:02 +niclog0=niclog0 +trace-select=1 +trace-start=0 +trace-end=-1 +trace-output-format=0 +dwarf-file-name=merlin-bench-bench_model_mt_mm_i8_64x128_rvv0-bench_model_mt_mm_i8_64x128_rvv-dwarf +autocounter-readrate=0 +autocounter-filename-base=AUTOCOUNTERFILE +print-start=0 +print-end=-1 +linklatency0=6405 +netbw0=200 +shmemportname0=default +domain=0x0000 +bus=0x42 +device=0x00 +function=0x0 +bar=0x0 +pci-vendor=0x10ee +pci-device=0x903f +permissive-off merlin-bench-bench_model_mt_mm_i8_64x128_rvv0-bench_model_mt_mm_i8_64x128_rvv
FireSim fingerprint: 0x46697265
TracerV: Trigger enabled from 0 to 18446744073709551615 cycles
Commencing simulation.
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
tsibridge_t::tick skipping tick
Model: mt_mm_i8_64x128, Variant: RVV
Input shape: 64x128 (8192 elements, f32, 1 inputs)
Warmup START (2 iterations)
Warmup iter 1/2 enter (cycle=102243420)
[vm_invoke] #1 entering
[apply] #1 ord=0 bindings=2
[dn] o=0 sym=main_dispatch_0_elementwise_8192_f32xi8 wg_count=2,1,1
[dc] o=0 wg=0,0,0 cyc=11637 ret=0
[dc] o=0 wg=1,0,0 cyc=11409 ret=0
[apply] #2 ord=1 bindings=2
[dn] o=1 sym=main_dispatch_1_reduction_64x128x128_i8xi32 wg_count=16,1,1
[dc] o=1 wg=0,0,0 cyc=41360 ret=0
[dc] o=1 wg=
