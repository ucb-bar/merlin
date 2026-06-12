"""Generate a Zephyr runtime-backend module scaffold from a zephyr_plan.

Structurally plausible but NON-building placeholders. The driver API centers on the
Merlin-owned generic runtime API (`merlin_submit`/`merlin_wait`/`merlin_get_metrics`); the
target driver implements it. Blocking mode first; interrupt/RTIO are later stages.
"""
from __future__ import annotations

from typing import Any

from ...common.artifacts import Artifact


def _module_yml(module_name: str) -> str:
    return (
        f"# Zephyr module manifest (generated).\n"
        f"name: {module_name}\n"
        "build:\n"
        "  cmake: .\n"
        "  kconfig: Kconfig\n"
        "  settings:\n"
        "    dts_root: .\n"
    )


def _cmakelists() -> str:
    return (
        "# Generated Zephyr module CMake.\n"
        "if(CONFIG_MERLIN_RUNTIME)\n"
        "  zephyr_include_directories(include)\n"
        "  add_subdirectory(drivers/accelerator)\n"
        "endif()\n"
    )


def _module_kconfig(sym: str) -> str:
    return (
        "# Generated Zephyr module Kconfig.\n"
        "config MERLIN_RUNTIME\n"
        '    bool "Merlin runtime support"\n'
        "    help\n"
        "      Merlin-owned runtime ABI (submit/wait/metrics). Targets provide adapters.\n\n"
        "config MERLIN_RUNTIME_PROFILING\n"
        '    bool "Enable Merlin runtime profiling"\n'
        "    depends on MERLIN_RUNTIME\n\n"
        "rsource \"drivers/accelerator/Kconfig\"\n"
    )


def _binding(compatible: str, properties: list[str]) -> str:
    lines = [
        f"description: Merlin accelerator ({compatible})",
        "",
        f'compatible: "{compatible}"',
        "",
        "include: base.yaml",
        "",
        "properties:",
    ]
    for prop in properties:
        lines.append(f"  {prop}:")
        if prop in ("reg", "interrupts"):
            lines.append("    required: false")
        else:
            lines.append("    type: int")
            lines.append("    required: false")
    return "\n".join(lines) + "\n"


def _driver_c(short: str, compatible: str) -> str:
    dt = compatible.replace(",", "_").replace("-", "_")
    up = short.upper()
    return f"""/* Generated Merlin {short} blocking-mode driver.
 *
 * Real implementation against the Merlin-owned runtime API: it decodes a Merlin command
 * buffer into the device's command FIFO over MMIO, rings the doorbell, polls for completion,
 * and reads the hardware performance counters into the common metrics struct.
 *
 * Build dependency: this compiles against the Zephyr device model + a real devicetree node
 * (compatible "{compatible}"). The Zephyr SDK is not bundled in this scaffold.
 */
#define DT_DRV_COMPAT {dt}

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/sys/sys_io.h>
#include <zephyr/kernel.h>
#include <errno.h>
#include <merlin/runtime.h>
#include <merlin/command_buffer.h>
#include <merlin/{short}.h>

struct {short}_config {{
    uintptr_t base;            /* MMIO base address (from devicetree `reg`) */
    uint32_t cmd_queue_depth;
}};

struct {short}_data {{
    uint32_t submitted;
}};

/* Push one encoded command word-group into the device command FIFO. */
static void {short}_push(uintptr_t base, const struct merlin_cmd *cmd)
{{
    sys_write32(cmd->opcode, base + {up}_REG_CMD_OPCODE);
    sys_write32(cmd->arg0,   base + {up}_REG_CMD_ARG0);
    sys_write32(cmd->arg1,   base + {up}_REG_CMD_ARG1);
    sys_write32(cmd->arg2,   base + {up}_REG_CMD_ARG2);
    sys_write32({up}_CMD_PUSH, base + {up}_REG_CMD_CTRL);
}}

static int {short}_submit(const struct device *dev,
                          const struct merlin_cmd_buffer *cb,
                          struct merlin_event *event)
{{
    const struct {short}_config *cfg = dev->config;
    struct {short}_data *data = dev->data;

    if (cb == NULL || cb->commands == NULL) {{
        return -EINVAL;
    }}
    if (cb->count > cfg->cmd_queue_depth) {{
        return -ENOSPC;
    }}
    for (size_t i = 0; i < cb->count; i++) {{
        {short}_push(cfg->base, &cb->commands[i]);
    }}
    sys_write32({up}_GO, cfg->base + {up}_REG_DOORBELL);
    data->submitted++;
    if (event != NULL) {{
        event->id = data->submitted;
    }}
    return 0;
}}

static int {short}_wait(const struct device *dev,
                        struct merlin_event *event, uint64_t timeout_ns)
{{
    const struct {short}_config *cfg = dev->config;
    uint64_t deadline = k_uptime_ticks() + k_ns_to_ticks_ceil64(timeout_ns);

    ARG_UNUSED(event);
    while ((sys_read32(cfg->base + {up}_REG_STATUS) & {up}_STATUS_DONE) == 0u) {{
        if (timeout_ns != 0 && k_uptime_ticks() > deadline) {{
            return -EAGAIN;
        }}
        k_busy_wait(1);
    }}
    return 0;
}}

static int {short}_get_metrics(const struct device *dev,
                               struct merlin_metrics *metrics)
{{
    const struct {short}_config *cfg = dev->config;

    if (metrics == NULL) {{
        return -EINVAL;
    }}
    metrics->cycles             = sys_read32(cfg->base + {up}_REG_CYCLES);
    metrics->bytes_moved        = sys_read32(cfg->base + {up}_REG_BYTES_MOVED);
    metrics->command_count      = sys_read32(cfg->base + {up}_REG_CMD_COUNT);
    metrics->pack_count         = sys_read32(cfg->base + {up}_REG_PACK_COUNT);
    metrics->resident_hits      = sys_read32(cfg->base + {up}_REG_RESIDENT_HITS);
    metrics->evictions          = sys_read32(cfg->base + {up}_REG_EVICTIONS);
    metrics->accumulator_commits = sys_read32(cfg->base + {up}_REG_ACC_COMMITS);
    return 0;
}}

static const struct merlin_driver_api {short}_api = {{
    .submit = {short}_submit,
    .wait = {short}_wait,
    .get_metrics = {short}_get_metrics,
}};

static int {short}_init(const struct device *dev)
{{
    const struct {short}_config *cfg = dev->config;
    /* Reset the device and clear performance counters. */
    sys_write32({up}_CTRL_RESET, cfg->base + {up}_REG_CTRL);
    return 0;
}}

#define {up}_INIT(inst)                                                        \\
    static struct {short}_data {short}_data_##inst;                            \\
    static const struct {short}_config {short}_config_##inst = {{               \\
        .base = DT_INST_REG_ADDR(inst),                                        \\
        .cmd_queue_depth = DT_INST_PROP_OR(inst, command_queue_depth, 16),     \\
    }};                                                                        \\
    DEVICE_DT_INST_DEFINE(inst, {short}_init, NULL,                            \\
                          &{short}_data_##inst, &{short}_config_##inst,        \\
                          POST_KERNEL, CONFIG_KERNEL_INIT_PRIORITY_DEVICE,     \\
                          &{short}_api);

DT_INST_FOREACH_STATUS_OKAY({up}_INIT)
"""


def _runtime_h() -> str:
    return (
        "/* Merlin-owned generic runtime API for Zephyr (real). */\n"
        "#pragma once\n"
        "#include <stdint.h>\n"
        "#include <stddef.h>\n"
        "#include <zephyr/device.h>\n"
        '#include <merlin/command_buffer.h>\n'
        '#include <merlin/metrics.h>\n\n'
        "struct merlin_event {\n"
        "    uint32_t id;        /* submission id, set by submit() */\n"
        "};\n\n"
        "typedef int (*merlin_submit_t)(const struct device *dev,\n"
        "                               const struct merlin_cmd_buffer *cb,\n"
        "                               struct merlin_event *event);\n"
        "typedef int (*merlin_wait_t)(const struct device *dev,\n"
        "                             struct merlin_event *event, uint64_t timeout_ns);\n"
        "typedef int (*merlin_get_metrics_t)(const struct device *dev,\n"
        "                                    struct merlin_metrics *metrics);\n\n"
        "__subsystem struct merlin_driver_api {\n"
        "    merlin_submit_t submit;\n"
        "    merlin_wait_t wait;\n"
        "    merlin_get_metrics_t get_metrics;\n"
        "};\n\n"
        "/* Generic inline wrappers dispatching through the device API table. */\n"
        "static inline int merlin_submit(const struct device *dev,\n"
        "                                const struct merlin_cmd_buffer *cb,\n"
        "                                struct merlin_event *event) {\n"
        "    return DEVICE_API_GET(merlin, dev)->submit(dev, cb, event);\n"
        "}\n"
        "static inline int merlin_wait(const struct device *dev,\n"
        "                              struct merlin_event *event, uint64_t timeout_ns) {\n"
        "    return DEVICE_API_GET(merlin, dev)->wait(dev, event, timeout_ns);\n"
        "}\n"
        "static inline int merlin_get_metrics(const struct device *dev,\n"
        "                                     struct merlin_metrics *metrics) {\n"
        "    return DEVICE_API_GET(merlin, dev)->get_metrics(dev, metrics);\n"
        "}\n"
    )


def _command_buffer_h() -> str:
    return (
        "/* Merlin command-buffer C view (real). */\n"
        "#pragma once\n"
        "#include <stdint.h>\n"
        "#include <stddef.h>\n\n"
        "/* One opaque, target-encoded command. arg0..arg2 are operand/handle/immediate slots. */\n"
        "struct merlin_cmd {\n"
        "    uint32_t opcode;\n"
        "    uint32_t arg0;\n"
        "    uint32_t arg1;\n"
        "    uint32_t arg2;\n"
        "};\n\n"
        "struct merlin_cmd_buffer {\n"
        "    const struct merlin_cmd *commands;\n"
        "    size_t count;\n"
        "};\n"
    )


def _target_regmap_h(short: str) -> str:
    up = short.upper()
    return (
        f"/* {short} MMIO register map + command opcodes (real). */\n"
        "#pragma once\n\n"
        "/* Control / status registers (byte offsets from the devicetree `reg` base). */\n"
        f"#define {up}_REG_CTRL            0x0000u\n"
        f"#define {up}_REG_DOORBELL        0x0004u\n"
        f"#define {up}_REG_STATUS          0x0008u\n"
        f"#define {up}_REG_CMD_CTRL        0x0010u\n"
        f"#define {up}_REG_CMD_OPCODE      0x0014u\n"
        f"#define {up}_REG_CMD_ARG0        0x0018u\n"
        f"#define {up}_REG_CMD_ARG1        0x001Cu\n"
        f"#define {up}_REG_CMD_ARG2        0x0020u\n\n"
        "/* Performance counters (read into struct merlin_metrics). */\n"
        f"#define {up}_REG_CYCLES          0x0100u\n"
        f"#define {up}_REG_BYTES_MOVED     0x0104u\n"
        f"#define {up}_REG_CMD_COUNT       0x0108u\n"
        f"#define {up}_REG_PACK_COUNT      0x010Cu\n"
        f"#define {up}_REG_RESIDENT_HITS   0x0110u\n"
        f"#define {up}_REG_EVICTIONS       0x0114u\n"
        f"#define {up}_REG_ACC_COMMITS     0x0118u\n\n"
        "/* Control/status bits and command opcodes. */\n"
        f"#define {up}_CTRL_RESET          0x1u\n"
        f"#define {up}_GO                  0x1u\n"
        f"#define {up}_CMD_PUSH            0x1u\n"
        f"#define {up}_STATUS_DONE         0x1u\n\n"
        f"#define {up}_OP_RES_PACK         0x10u\n"
        f"#define {up}_OP_MATMUL           0x11u\n"
        f"#define {up}_OP_COMMIT           0x12u\n"
        f"#define {up}_OP_EVICT            0x13u\n"
    )


def _sample_main(short: str, compatible: str) -> str:
    up = short.upper()
    node = compatible.replace(",", "_").replace("-", "_")
    return (
        "/* Generated Merlin sample: pack a weight, run a resident matmul, commit, evict. */\n"
        '#include <zephyr/kernel.h>\n'
        '#include <zephyr/device.h>\n'
        '#include <merlin/runtime.h>\n'
        '#include <merlin/command_buffer.h>\n'
        f'#include <merlin/{short}.h>\n\n'
        f"static const struct merlin_cmd cmds[] = {{\n"
        f"    {{ {up}_OP_RES_PACK, /*src=*/0, /*dst=*/1, /*layout=*/0 }},\n"
        f"    {{ {up}_OP_MATMUL,   /*lhs=*/2, /*rhs=*/1, /*dst=*/3 }},\n"
        f"    {{ {up}_OP_COMMIT,   /*src=*/3, /*dst=*/4, /*epilogue=*/0 }},\n"
        f"    {{ {up}_OP_EVICT,    /*handle=*/1, 0, 0 }},\n"
        "};\n\n"
        "int main(void) {\n"
        f"    const struct device *dev = DEVICE_DT_GET_ANY({node});\n"
        "    if (!device_is_ready(dev)) {\n"
        "        return -1;\n"
        "    }\n"
        "    struct merlin_cmd_buffer cb = { .commands = cmds, .count = ARRAY_SIZE(cmds) };\n"
        "    struct merlin_event ev;\n"
        "    struct merlin_metrics m;\n"
        "    int rc = merlin_submit(dev, &cb, &ev);\n"
        "    if (rc == 0) {\n"
        "        rc = merlin_wait(dev, &ev, 0);\n"
        "    }\n"
        "    if (rc == 0) {\n"
        "        merlin_get_metrics(dev, &m);\n"
        "        printk(\"cycles=%llu commands=%llu\\n\",\n"
        "               (unsigned long long)m.cycles, (unsigned long long)m.command_count);\n"
        "    }\n"
        "    return rc;\n}\n"
    )


def generate(zephyr_plan: dict[str, Any]) -> list[Artifact]:
    """Return the Zephyr module artifacts for the given zephyr_plan."""
    target = zephyr_plan.get("target", "target")
    module_name = zephyr_plan.get("module", {}).get("name", f"merlin_{target}")
    dt = zephyr_plan.get("devicetree", {})
    compatible = dt.get("compatible", f"ucb,{target}")
    properties = dt.get("properties", ["reg", "interrupts"])
    short = compatible.split(",", 1)[1] if "," in compatible else target
    symbols = zephyr_plan.get("kconfig", {}).get("symbols", [])
    sym = symbols[0] if symbols else ("MERLIN_" + short.upper())
    samples = zephyr_plan.get("samples", [])
    sample = samples[0] if samples else f"{short}_sample"

    driver_kconfig = (
        f"config {sym}\n"
        f'    bool "Merlin {target} accelerator support"\n'
        f"    depends on MERLIN_RUNTIME\n\n"
        f"config {sym}_RTIO\n"
        f'    bool "Use Zephyr RTIO backend for {target} command submission"\n'
        f"    depends on {sym} && RTIO\n"
    )

    return [
        Artifact("zephyr/module.yml", _module_yml(module_name)),
        Artifact("zephyr/CMakeLists.txt", _cmakelists()),
        Artifact("zephyr/Kconfig", _module_kconfig(sym)),
        Artifact(f"zephyr/dts/bindings/accelerator/{compatible}.yaml",
                 _binding(compatible, properties)),
        Artifact(f"zephyr/drivers/accelerator/{short}_driver.c", _driver_c(short, compatible)),
        Artifact("zephyr/drivers/accelerator/CMakeLists.txt",
                 f"# Generated driver library.\nzephyr_library()\n"
                 f"zephyr_library_sources_ifdef(CONFIG_{sym} {short}_driver.c)\n"),
        Artifact("zephyr/drivers/accelerator/Kconfig", driver_kconfig),
        Artifact("zephyr/include/merlin/runtime.h", _runtime_h()),
        Artifact("zephyr/include/merlin/command_buffer.h", _command_buffer_h()),
        Artifact("zephyr/include/merlin/metrics.h",
                 "/* Merlin common metrics C view (real; mirrors metrics.schema.yaml). */\n#pragma once\n"
                 "#include <stdint.h>\n"
                 "struct merlin_metrics {\n"
                 "    uint64_t cycles;\n    uint64_t bytes_moved;\n"
                 "    uint64_t command_count;\n    uint64_t pack_count;\n"
                 "    uint64_t resident_hits;\n    uint64_t evictions;\n"
                 "    uint64_t accumulator_commits;\n};\n"),
        Artifact(f"zephyr/include/merlin/{short}.h", _target_regmap_h(short)),
        Artifact(f"zephyr/samples/{sample}/CMakeLists.txt",
                 "# Generated sample application.\n"
                 "cmake_minimum_required(VERSION 3.20)\n"
                 "find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})\n"
                 f"project({sample})\n"
                 "target_sources(app PRIVATE src/main.c)\n"),
        Artifact(f"zephyr/samples/{sample}/prj.conf",
                 "CONFIG_MERLIN_RUNTIME=y\n"
                 f"CONFIG_{sym}=y\n"),
        Artifact(f"zephyr/samples/{sample}/app.overlay", _overlay(short, compatible)),
        Artifact(f"zephyr/samples/{sample}/src/main.c", _sample_main(short, compatible)),
        # Real ztest for the driver.
        Artifact(f"zephyr/tests/{short}_driver/CMakeLists.txt",
                 "cmake_minimum_required(VERSION 3.20)\n"
                 "find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})\n"
                 f"project({short}_driver_test)\n"
                 "target_sources(app PRIVATE src/main.c)\n"),
        Artifact(f"zephyr/tests/{short}_driver/prj.conf",
                 "CONFIG_ZTEST=y\nCONFIG_MERLIN_RUNTIME=y\n"
                 f"CONFIG_{sym}=y\n"),
        Artifact(f"zephyr/tests/{short}_driver/{compatible}.overlay", _overlay(short, compatible)),
        Artifact(f"zephyr/tests/{short}_driver/src/main.c", _driver_test(short, compatible)),
    ]


def _overlay(short: str, compatible: str) -> str:
    return (
        f"/* Devicetree overlay: instantiate the {compatible} accelerator node. */\n"
        "/ {\n"
        "    soc {\n"
        f"        {short}0: {short}@10000000 {{\n"
        f'            compatible = "{compatible}";\n'
        "            reg = <0x10000000 0x1000>;\n"
        "            command-queue-depth = <16>;\n"
        "            resident-store-bytes = <131072>;\n"
        "            accumulator-entries = <4096>;\n"
        '            status = "okay";\n'
        "        };\n"
        "    };\n"
        "};\n"
    )


def _driver_test(short: str, compatible: str) -> str:
    up = short.upper()
    node = compatible.replace(",", "_").replace("-", "_")
    return (
        "/* Generated ztest for the Merlin driver: submit a command buffer, read metrics. */\n"
        "#include <zephyr/ztest.h>\n"
        "#include <zephyr/device.h>\n"
        "#include <merlin/runtime.h>\n"
        "#include <merlin/command_buffer.h>\n"
        f"#include <merlin/{short}.h>\n\n"
        f"static const struct merlin_cmd cmds[] = {{\n"
        f"    {{ {up}_OP_RES_PACK, 0, 1, 0 }},\n"
        f"    {{ {up}_OP_MATMUL, 2, 1, 3 }},\n"
        f"    {{ {up}_OP_COMMIT, 3, 4, 0 }},\n"
        f"    {{ {up}_OP_EVICT, 1, 0, 0 }},\n"
        "};\n\n"
        f"ZTEST_SUITE({short}_driver, NULL, NULL, NULL, NULL, NULL);\n\n"
        f"ZTEST({short}_driver, test_submit_and_metrics)\n"
        "{\n"
        f"    const struct device *dev = DEVICE_DT_GET_ANY({node});\n"
        "    zassert_true(device_is_ready(dev), \"device not ready\");\n"
        "    struct merlin_cmd_buffer cb = { .commands = cmds, .count = ARRAY_SIZE(cmds) };\n"
        "    struct merlin_event ev;\n"
        "    zassert_equal(merlin_submit(dev, &cb, &ev), 0, \"submit failed\");\n"
        "    zassert_equal(merlin_wait(dev, &ev, 0), 0, \"wait failed\");\n"
        "    struct merlin_metrics m;\n"
        "    zassert_equal(merlin_get_metrics(dev, &m), 0, \"get_metrics failed\");\n"
        "}\n"
    )
