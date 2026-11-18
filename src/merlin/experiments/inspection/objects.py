"""Compare actual allocated object sections across a symbol-only rewrite."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.process import invocation_records as invocation
from merlin.common.process import invocation_records as invocation_record
from merlin.common.record_formats import is_identifier, source_tokens
from merlin.compiler.lower.model.compilation_recipe import verify_completed_recipe

# ---- chain_image_sections ----------------------------------------------------------------------


def allocated(path: Path):
    """Read ELF64 section tables structurally, including NOBITS extents."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if header[:6] != b"\x7fELF\x02\x01":
            raise ValueError("chain runtime requires little-endian ELF64 objects")
        table = struct.unpack_from("<Q", header, 40)[0]
        size, count, strings = struct.unpack_from("<HHH", header, 58)
        if size != 64 or not 0 < count < 65535 or strings >= count:
            raise ValueError("unsupported extended ELF section roster")
        stream.seek(table)
        sections = [struct.unpack("<IIQQQQIIQQ", stream.read(64)) for _ in range(count)]
        names = sections[strings]
        stream.seek(names[4])
        names = stream.read(names[5])
        result = {}
        for row in sections:
            name, kind, flags, _, offset, length, *_ = row
            if not flags & 2:
                continue
            label = names[name:].split(b"\0", 1)[0].decode("ascii")
            if label in result:
                raise ValueError("duplicate allocated object section")
            digest = None
            if kind != 8:  # ELF SHT_NOBITS has no stored bytes.
                stream.seek(offset)
                remaining, hasher = length, hashlib.sha256()
                while remaining:
                    block = stream.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise ValueError("truncated allocated object section")
                    hasher.update(block)
                    remaining -= len(block)
                digest = hasher.hexdigest()
            result[label] = {"kind": kind, "flags": flags, "bytes": length, "sha256": digest}
        return result


def unchanged(source, product, *, weight_section=None):
    before, after = allocated(source), allocated(product)
    if weight_section is not None:
        if ".data" not in before:
            raise ValueError("weights object has no selected data section")
        # objcopy's explicit section flags preserve ALLOC/WRITE and initialized data.
        before[weight_section] = before.pop(".data")
    if before != after:
        raise ValueError("symbol namespace changed actual allocated instructions/data/storage")
    return {"status": "allocated_sections_unchanged", "sections": after}


def symbol_bytes(path, symbol):
    """Read one defined ELF64 symbol through its actual section index/extent."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if header[:6] != b"\x7fELF\x02\x01":
            raise ValueError("unsupported object ABI for table layout proof")
        table = struct.unpack_from("<Q", header, 40)[0]
        stride, count, _ = struct.unpack_from("<HHH", header, 58)
        if stride != 64 or not 0 < count < 65535:
            raise ValueError("unsupported object section roster")
        stream.seek(table)
        sections = [struct.unpack("<IIQQQQIIQQ", stream.read(64)) for _ in range(count)]
        found = []
        for section in sections:
            if section[1] != 2:  # SHT_SYMTAB
                continue
            if section[9] != 24 or section[6] >= len(sections) or section[5] % 24:
                raise ValueError("invalid object symbol table")
            strings = sections[section[6]]
            stream.seek(strings[4])
            names = stream.read(strings[5])
            stream.seek(section[4])
            symbols = stream.read(section[5])
            for at in range(0, len(symbols), 24):
                name, _, _, index, offset, length = struct.unpack_from("<IBBHQQ", symbols, at)
                if names[name:].split(b"\0", 1)[0].decode("utf-8") != symbol:
                    continue
                if not 0 < index < len(sections) or not length:
                    raise ValueError("selected table symbol has no stored definition")
                stored = sections[index]
                if stored[1] == 8 or offset + length > stored[5]:
                    raise ValueError("selected table symbol exceeds its section")
                stream.seek(stored[4] + offset)
                found.append(stream.read(length))
        if len(found) != 1:
            raise ValueError("selected table symbol is absent or ambiguous")
        return found[0]


# ---- chain_image_objects -----------------------------------------------------------------------
# Reopen and namespace completed model objects without compiling model code.
#
# Only symbol tables and a weight section name change. Existing instruction and
# data sections remain byte-for-byte inputs to the new ordinary final link.

EXPORTED_TABLES = ("MERLIN_ARGS", "MERLIN_INPUT_PTR", "MERLIN_OUTPUT_PTR")
COMMON_OBJECTS = frozenset(
    {
        "merlin_model.o",
        "mlir_rt.o",
        "crt.o",
        "console.o",
        "libc_min.o",
        "malloc.o",
        "host_math.o",
    }
)


def pin(path: Path) -> dict:
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("chain image inputs must be canonical ordinary files")
    if not path.is_file():
        raise ValueError("chain image input is absent")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def reopen(record: dict) -> Path:
    path = Path(record["path"])
    actual = pin(path)
    if any(actual[key] != record[key] for key in ("path", "sha256")) or (
        "bytes" in record and actual["bytes"] != record["bytes"]
    ):
        raise ValueError("chain image selected input changed")
    return path


def payload(record):
    path = reopen(record)
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != record["sha256"]:
        raise ValueError("selected bytes changed while being read")
    return value


def completed_program(receipt_pin: dict) -> dict:
    """Read an original completed link, preserving any exact-restoration record."""
    receipt = json.loads(payload(receipt_pin))
    recovery_pins = []
    if receipt.get("schema") == "merlin.completed_model_link_revalidation.v1":
        from merlin.verification.references.saved_model import verify

        receipt, recipe, recovery_pins = verify(receipt_pin)
    elif receipt.get("schema") != "merlin.baremetal-saved-model.v1" or receipt.get("status") != "compiled":
        raise ValueError("chain packaging requires a completed ordinary model build")
    output = receipt["output"]
    recipe_pin = output["compilation_recipe"]
    if not recovery_pins:
        recipe = verify_completed_recipe(
            reopen(recipe_pin),
            executable=Path(output["elf"]),
            expected_recipe_sha256=recipe_pin["sha256"],
        )
    link = recipe["commands"][-1]
    root = Path(output["elf"]).parent
    objects = [row for row in link["inputs"] if Path(row["path"]).suffix == ".o"]
    if len({row["path"] for row in objects}) != len(objects):
        raise ValueError("chain program repeats a final-link object")
    by_name = {Path(row["path"]).name: row for row in objects}
    if (
        len(by_name) != len(objects)
        or not {"model.o", "model_call.o", "model_main.o", "weights_blob.o"} <= by_name.keys()
    ):
        raise ValueError("chain program has no unambiguous ordinary model object roster")
    return {
        "receipt": receipt,
        "receipt_pin": receipt_pin,
        "recipe": recipe,
        "recipe_pin": recipe_pin,
        "link": link,
        "root": root,
        "objects": objects,
        "by_name": by_name,
        "completed_link_revalidation_pins": recovery_pins,
    }


def _nm(nm, obj, *, timeout, directory=None):
    argv = [str(nm), "--print-size", "--format=posix", str(obj)]
    if directory is None:
        return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=timeout)
    return invocation.run(
        argv,
        directory=directory,
        stage="chain_image_symbol_inventory",
        inputs=(obj,),
        dependencies=(Path(__file__),),
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )


def symbols(nm: Path, obj: Path, *, timeout: float, directory=None) -> dict[str, str]:
    """Parse the selected binutils POSIX symbol format, never disassembly text."""
    result = _nm(nm, obj, timeout=timeout, directory=directory)
    found = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) not in (2, 3, 4) or len(fields[1]) != 1:
            raise ValueError("unsupported selected symbol-table format")
        name, kind = fields[:2]
        if name in found and found[name] != kind:
            raise ValueError("ambiguous duplicate object symbol")
        found[name] = kind
    return found


def table_sizes(nm: Path, obj: Path, *, timeout: float, directory=None) -> dict[str, int]:
    result = _nm(nm, obj, timeout=timeout, directory=directory)
    rows = [line.split() for line in result.stdout.splitlines()]
    sizes = {row[0]: int(row[3], 16) for row in rows if len(row) == 4 and row[0] in EXPORTED_TABLES}
    if set(sizes) != set(EXPORTED_TABLES) or any(size <= 0 for size in sizes.values()):
        raise ValueError("compiled tables have no exact symbol extents")
    return sizes


def wrap_names(argv: list[str]) -> tuple[str, ...]:
    names = []
    for value in argv:
        if value.startswith("-Wl,--wrap="):
            name = value.partition("=")[2]
            if not name or "," in name:
                raise ValueError("unsupported original linker wrap declaration")
            names.append(name)
    return tuple(names)


def namespace(program: dict, index: int, directory: Path, *, nm: Path, objcopy: Path, timeout: float) -> dict:
    """Expose compiled runtime tables and retain program-local symbol custody."""
    prefix = f"merlin_chain_p{index}_"
    directory.mkdir()
    selected = [row for row in program["objects"] if Path(row["path"]).name not in COMMON_OBJECTS]
    tables, defined = {}, set()
    for row in selected:
        obj = reopen(row)
        table = symbols(nm, obj, timeout=timeout, directory=directory)
        tables[str(obj)] = table
        defined.update(name for name, kind in table.items() if kind.isupper() and kind not in {"U", "N"})
    main = Path(program["by_name"]["model_main.o"]["path"])
    sizes = table_sizes(nm, main, timeout=timeout, directory=directory)
    if not set(EXPORTED_TABLES) <= tables[str(main)].keys():
        raise ValueError("retained driver does not expose the original generated argument/pointer tables")
    defined.update(EXPORTED_TABLES)
    mapping = {name: prefix + name for name in defined}
    wraps = []
    for name in (*wrap_names(program["link"]["argv"]), *program.get("profile_wraps", ())):
        if name in defined or "__wrap_" + name in defined:
            mapping[name] = prefix + name
            mapping["__wrap_" + name] = "__wrap_" + prefix + name
            mapping["__real_" + name] = "__real_" + prefix + name
            wraps.append(prefix + name)
        else:
            wraps.append(name)
    map_file = directory / "symbols.map"
    map_file.write_text("".join(f"{old} {new}\n" for old, new in sorted(mapping.items())))
    products, observations = [], []
    for ordinal, row in enumerate(selected):
        source = reopen(row)
        destination = directory / f"o{ordinal}_{source.name}"
        argv = [str(objcopy), "--redefine-syms=" + str(map_file)]
        if source == main:
            argv.extend("--globalize-symbol=" + mapping[name] for name in EXPORTED_TABLES)
        if source.name == "weights_blob.o":
            argv += ["--rename-section", f".data=.merlin_chain_weights_{index},alloc,load,data,contents"]
        argv += [str(source), str(destination)]
        invocation.run(
            argv,
            directory=directory,
            stage="chain_image_object_namespace",
            inputs=(source, map_file),
            outputs=(destination,),
            dependencies=(Path(__file__),),
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        observations.append(
            unchanged(
                source,
                destination,
                weight_section=(f".merlin_chain_weights_{index}" if source.name == "weights_blob.o" else None),
            )
        )
        products.append(pin(destination))
    return {
        "prefix": prefix,
        "objects": products,
        "mapping": pin(map_file),
        "wraps": wraps,
        "original_objects": selected,
        "section_observations": observations,
        "table_sizes": sizes,
        "invocations": [pin(p) for p in sorted(directory.glob("invocations/*/invocation.json"))],
        "exported_tables": list(EXPORTED_TABLES),
        "weights_section": f".merlin_chain_weights_{index}",
    }


# ---- chain_image_layout ------------------------------------------------------------------------
# Explicit complete-chain linker layout and retained runtime compilation.


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
        pins.append(record)
        weights.append({"begin": begin, "bytes": record["bytes"]})
    ranges = [
        *weights,
        {"begin": selected["inputs"], "bytes": plan["input_blob"]["bytes"]},
        {"begin": selected["references"], "bytes": references["blob"]["bytes"]},
        selected["arena"],
    ]
    for index, item in enumerate(ranges):
        if (
            type(item["begin"]) is not int
            or type(item["bytes"]) is not int
            or item["begin"] < 0
            or item["bytes"] < 0
            or (item["bytes"] == 0 and index >= len(weights))
            or item["begin"] + item["bytes"] >= 1 << 63
            or item["begin"] % 64
        ):
            raise ValueError("chain storage needs finite aligned absolute extents")
    ordered = sorted((row for row in ranges if row["bytes"]), key=lambda row: row["begin"])
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
        if row["bytes"]:
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


# ---- chain_image_counter -----------------------------------------------------------------------
# Compile counter-only support from an explicitly selected original host CSR.
#
# This reuses a completed program's compiler/ABI flags and source declaration.
# It adds no profiling wrappers and establishes no counter frequency or physical
# execution origin. The original source and compilation remain independently bound.

KIND = "host_source_csr_v1"
SCHEMA = "merlin.chain_counter.host_csr.v1"
SYMBOL = "merlin_chain_host_counter"


def _declares_counter(source):
    """Find the supported complete inline-assembly production outside comments and literals."""
    tokens = [token.text for token in source_tokens(source)]
    return any(
        tokens[index : index + 7] == ["__asm__", "volatile", "(", '"csrr %0, mcycle"', ":", '"=r"', "("]
        and is_identifier(tokens[index + 7])
        and tokens[index + 8 : index + 10] == [")", ")"]
        for index in range(max(0, len(tokens) - 9))
    )


def prepare(counter, program, directory, *, nm, timeout):
    if set(counter) != {"kind", "program", "contract"} or counter["kind"] != KIND:
        raise ValueError("source counter selection needs kind, original program and contract")
    contract = json.loads(payload(counter["contract"]))
    if set(contract) != {"schema", "csr", "source"} or contract["schema"] != SCHEMA or contract["csr"] != "mcycle":
        raise ValueError("unsupported original host counter contract")
    command = command(program, "model_main.o")
    sources = [row for row in command["inputs"] if row["path"].endswith(".c")]
    if len(sources) != 1 or any(contract["source"][k] != sources[0][k] for k in ("path", "sha256")):
        raise ValueError("counter source is not the selected program's original compiled host driver")
    reopen(contract["source"])
    original = payload(sources[0]).decode("utf-8")
    if not _declares_counter(original):
        raise ValueError("original selected host driver does not declare the supported mcycle read")
    if not any(arg.startswith("-march=rv64") for arg in command["argv"]):
        raise ValueError("source mcycle counter requires the original selected RV64 ABI")
    reopen(command["executable"])
    directory.mkdir()
    source, product = directory / "counter.c", directory / "counter.o"
    source.write_text(
        "/* Counter-only support bound to the original selected host CSR. */\n"
        "#include <stdint.h>\n"
        f"uint64_t {SYMBOL}(void) {{\n"
        '  uint64_t ticks; __asm__ volatile("csrr %0, mcycle" : "=r"(ticks) : : "memory");\n'
        "  return ticks;\n}\n"
    )
    invocation_record.run(
        compile_argv(command, source, product),
        directory=directory,
        stage="chain_counter_compile",
        inputs=(*(reopen(row) for row in command["inputs"]), reopen(counter["contract"]), source),
        outputs=(product,),
        dependencies=(Path(__file__), Path(__file__)),
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    if symbols(nm, product, timeout=timeout, directory=directory).get(SYMBOL) != "T":
        raise ValueError("compiled counter support lacks its actual defining symbol")
    return {
        "kind": KIND,
        "symbol": SYMBOL,
        "contract": counter["contract"],
        "original_source": sources[0],
        "compiler": command["executable"],
        "source": pin(source),
        "object": pin(product),
        "invocations": [pin(p) for p in sorted(directory.glob("invocations/*/invocation.json"))],
        "scope": "Original selected mcycle semantics only; no physical counter frequency/execution qualification.",
    }


def verify(record):
    if record is not None:
        for key in ("contract", "original_source", "compiler", "source", "object"):
            reopen(record[key])
        for row in record["invocations"]:
            invocation_record.verify(reopen(row))
