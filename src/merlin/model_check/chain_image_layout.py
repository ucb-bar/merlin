"""Explicit complete-chain linker layout and retained runtime compilation."""

from __future__ import annotations

from pathlib import Path

from .chain_image_objects import pin, reopen


def command(program, output_name):
    rows = [row for row in program["recipe"]["commands"] if Path(row["requested_output"]).name == output_name]
    if len(rows) != 1:
        raise ValueError("retained runtime has no unique selected compilation")
    return rows[0]


def compile_argv(original, source, output, *, includes=(), defines=()):
    """Retain selected ABI flags; replace only this command's source/output."""
    argv, skip = [], False
    for value in original["argv"]:
        if skip:
            skip = False
            continue
        if value == "-o":
            skip = True
        elif value not in {row["path"] for row in original["inputs"] if row["path"].endswith(".c")}:
            argv.append(value)
    if "-c" not in argv:
        raise ValueError("runtime command is not a C object compilation")
    for key, value in defines:
        argv = [arg for arg in argv if not arg.startswith("-D" + key + "=")]
        argv.append(f"-D{key}={value}")
    return [
        *argv,
        "-fno-fast-math",
        "-fno-finite-math-only",
        "-ffp-contract=off",
        *("-I" + str(p) for p in includes),
        str(source),
        "-o",
        str(output),
    ]


def layout(selection, plan, references):
    selected = selection["layout"]
    if len(selected["weights"]) != len(plan["programs"]):
        raise ValueError("layout omits a program's complete weights")
    weights, pins = [], []
    for program, begin in zip(plan["programs"], selected["weights"], strict=True):
        path = program["production"]["root"] / "cgen" / "weights.bin"
        record = pin(path)
        if record["bytes"] <= 0:
            raise ValueError("retained weights are empty")
        pins.append(record)
        weights.append({"begin": begin, "bytes": record["bytes"]})
    ranges = [
        *weights,
        {"begin": selected["inputs"], "bytes": plan["input_blob"]["bytes"]},
        {"begin": selected["references"], "bytes": references["blob"]["bytes"]},
        selected["arena"],
    ]
    for item in ranges:
        if (
            type(item["begin"]) is not int
            or type(item["bytes"]) is not int
            or item["begin"] < 0
            or item["bytes"] <= 0
            or item["begin"] + item["bytes"] >= 1 << 63
            or item["begin"] % 64
        ):
            raise ValueError("chain storage needs finite aligned absolute extents")
    ordered = sorted(ranges, key=lambda row: row["begin"])
    if any(a["begin"] + a["bytes"] > b["begin"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("simultaneously live chain storage overlaps")
    if (
        selected["arena"]["bytes"]
        < max(p["production"]["receipt"]["inputs"]["arena_mb"] for p in plan["programs"]) * 1024**2
    ):
        raise ValueError("chain arena is smaller than a retained program's selected demand")
    return {**selected, "weights": weights, "source_pins": pins}


def linker_script(original, destination, storage):
    """Extend the original stock split-weights layout; unknown scripts refuse."""
    text = original.read_text()
    marker = "  . = MERLIN_WEIGHTS_BASE;"
    if text.count(marker) != 1 or text.count(".weights :") != 1:
        raise ValueError("retained linker script lacks the supported split-weights section ABI")
    prefix, suffix = text.split(marker)
    if suffix.splitlines()[-1].strip() != "}":
        raise ValueError("unsupported original linker script suffix")
    # Preserve the original boot address, entry, stack and all low sections.
    lines = [prefix]
    for index, row in enumerate(storage["weights"]):
        lines.append(f" . = {row['begin']}; .chain_weights_{index} : {{ *(.merlin_chain_weights_{index}) }}")
    for name in ("inputs", "references"):
        lines.append(f" . = {storage[name]}; .chain_{name} : {{ *(.merlin_chain_{name}) }}")
    lines.append("}")
    destination.write_text("\n".join(lines) + "\n")
    return destination


def final_link(original, *, objects, script, output, wraps):
    direct = {row["path"] for row in original["inputs"]}
    argv, skip = [], False
    for value in original["argv"]:
        if skip:
            skip = False
        elif value in {"-o", "-T"}:
            skip = True
        elif value in direct or value.startswith(("-Wl,--wrap=", "-Wl,--trace-symbol=")):
            continue
        else:
            argv.append(value)
    libraries = [reopen(row) for row in original["inputs"] if Path(row["path"]).suffix == ".a"]
    return [
        *argv,
        "-T",
        str(script),
        *(str(path) for path in objects),
        *(str(p) for p in libraries),
        *("-Wl,--wrap=" + name for name in sorted(set(wraps))),
        "-o",
        str(output),
    ], libraries
