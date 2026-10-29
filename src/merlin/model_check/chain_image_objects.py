"""Reopen and namespace completed model objects without compiling model code.

Only symbol tables and a weight section name change. Existing instruction and
data sections remain byte-for-byte inputs to the new ordinary final link.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from merlin.common import invocation_record as invocation
from merlin.common.digest import sha256_file
from merlin.llvmlower.compilation_recipe import verify_completed_recipe

from .chain_image_sections import unchanged

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
        from merlin.compile.completed_model_link import verify

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
