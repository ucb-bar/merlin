import json
import re
import os


PRIMARY_DEVICE = "@device_a"
SECONDARY_DEVICE = "@device_b"
PRIMARY_TARGET = "#device_target_primary"
SECONDARY_TARGET = "#device_target_secondary"


def _parse_schedule(dispatches):
    schedule = {}
    for key, val in dispatches.items():
        hw = val.get("hardware_target", "CPU_P")
        d_id = val.get("id")
        if d_id is None:
            match = re.search(r"dispatch_(\d+)", key)
            if match:
                d_id = int(match.group(1))
        if d_id is not None:
            schedule[d_id] = hw
    return schedule


def _strip_affinity(prefix):
    cleaned = re.sub(r"stream\.affinity\s*=\s*#hal\.device\.affinity<@[^>]+>\s*,?\s*", "", prefix)
    cleaned = cleaned.replace("{  }", "{}")
    return cleaned.strip()


def _attach_affinity(prefix, affinity_attr):
    """Ensure prefix has an attribute block with the given affinity."""
    match = re.search(r"\{([^}]*)\}", prefix)
    if match:
        existing = match.group(1).strip()
        attrs = [a.strip() for a in existing.split(",") if a.strip()]
        attrs = [a for a in attrs if not a.startswith("stream.affinity")]
        attrs.append(affinity_attr)
        new_block = "{ " + ", ".join(attrs) + " }"
        return prefix[: match.start()] + new_block + prefix[match.end():]
    return prefix.rstrip() + f" {{ {affinity_attr} }}"


def patch_split_personality(json_path, mlir_path, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    schedule = _parse_schedule(data.get("dispatches", {}))

    with open(mlir_path, "r") as f:
        lines = f.readlines()

    target_def_line = None
    for line in lines:
        if "#hal.device.target" in line and "=" in line:
            target_def_line = line
            break

    if not target_def_line:
        print("❌ Error: Could not find #hal.device.target definition.")
        return

    primary_device_old = "@__device_0"
    for line in lines:
        if "stream.affinity.default" in line:
            m = re.search(r"affinity<(@[\w\d_$]+)>", line)
            if m:
                primary_device_old = m.group(1)
            break

    target_rhs = target_def_line.split("=", 1)[1].strip()
    primary_target_line = f"{PRIMARY_TARGET} = {target_rhs}\n"
    secondary_target_line = f"{SECONDARY_TARGET} = {target_rhs}\n"

    print(f"ℹ️  Primary Target RHS:  {target_rhs.strip()}")
    print(f"ℹ️  Old Primary Device: {primary_device_old}")
    print(f"ℹ️  Using devices: {PRIMARY_DEVICE}, {SECONDARY_DEVICE}")

    with open(output_path, "w") as out:
        targets_written = False
        globals_written = False

        for line in lines:
            if not targets_written and "#hal.device.target" in line and "=" in line:
                out.write(primary_target_line)
                out.write(secondary_target_line)
                targets_written = True
                continue

            if targets_written and "#hal.device.target" in line:
                continue

            if "module attributes" in line:
                line = re.sub(r"affinity<@[^>]+>", f"affinity<{PRIMARY_DEVICE}>", line)
                out.write(line)
                out.write(f"  util.global private {PRIMARY_DEVICE} = {PRIMARY_TARGET} : !hal.device\n")
                out.write(f"  util.global private {SECONDARY_DEVICE} = {SECONDARY_TARGET} : !hal.device\n")
                globals_written = True
                continue

            if "util.global" in line and ("!hal.device" in line or "#device_target" in line):
                continue

            line = line.replace(primary_device_old, PRIMARY_DEVICE)
            line = line.replace("@device_e", SECONDARY_DEVICE)
            line = line.replace("@device_ab", SECONDARY_DEVICE)

            if "hal.devices.get" in line:
                line = re.sub(r"%\w+\s*=\s*hal\.devices\.get[^\n]*", f"%device_a = util.global.load {PRIMARY_DEVICE} : !hal.device", line)

            if "hal.fence.create" in line and "%device_" in line:
                line = re.sub(r"%device_\w+", "%device_a", line)

            if "flow.dispatch" in line:
                match = re.search(r"dispatch_(\d+)", line)
                d_id = int(match.group(1)) if match else None
                hw_target = schedule.get(d_id, "CPU_P") if d_id is not None else "CPU_P"
                target = SECONDARY_DEVICE if hw_target == "CPU_E" else PRIMARY_DEVICE
                affinity_attr = f"stream.affinity = #hal.device.affinity<{target}>"

                if " : " in line:
                    prefix, suffix = line.split(" : ", 1)
                    prefix = _strip_affinity(prefix)
                    prefix = _attach_affinity(prefix, affinity_attr)
                    out.write(f"{prefix} : {suffix}")
                    continue

            out.write(line)

    print(f"✅ Saved split-device schedule to: {output_path}")


if __name__ == "__main__":
    patch_split_personality("combined_schedule.json", "mlp.mlir", "mlp_pinned.mlir")