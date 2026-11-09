"""Independently selected portable recipe joined to its observed compiled object.

The object is selected by the ordinary builder's fixed output location and exact
source/compiler/flags, never by the link trace's claimed defining filename.
"""

from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.runtime.host_math import host_math_recipe, runtime_symbol_bindings
from merlin.targetgen.host_linkage_contract import runtime_recipe


def select(policy, contracts, compiler, recipe_path):
    identities = [runtime_recipe(contract, policy) for contract in contracts]
    if not identities or any(identity != identities[0] for identity in identities):
        raise ValueError("runtime supplier has no singular reviewed selected recipe")
    identity = identities[0]
    work = recipe_path.parent
    source, obj = work / "host_math/host_math.c", work / "host_math/host_math.o"
    if any(path.is_symlink() or path.resolve(strict=True) != path for path in (source, obj)):
        raise ValueError("runtime supplier needs canonical builder-owned files")
    if source.read_bytes() != host_math_recipe(policy).source.encode():
        raise ValueError("actual runtime source differs from selected portable recipe")
    if sha256_file(source) != identity["source_sha256"]:
        raise ValueError("reviewed runtime source differs from actual compiled source")
    symbols = sorted({symbol for contract in contracts for symbol in contract["symbols"]})
    return {
        **identity,
        "symbols": symbols,
        "wrapper_symbols": runtime_symbol_bindings(policy, symbols),
        "source": str(source),
        "object": str(obj),
        "object_sha256": sha256_file(obj),
        "compiler": compiler["gcc"],
        "compiler_sha256": compiler["gcc_sha256"],
    }


def verify_compile(record, selected, compiler):
    """Require the actual returned compilation, not a fabricated post-link trace."""
    source, obj = Path(selected["source"]), Path(selected["object"])
    commands = [command for command in record["commands"][:-1] if command["output"]["path"] == str(obj)]
    if len(commands) != 1:
        raise ValueError("runtime supplier has no unique returned object compilation")
    command = commands[0]
    expected = [
        compiler["gcc_requested"],
        *compiler["gcc_cflags"],
        *selected["compile_flags"],
        "-c",
        str(source),
        "-o",
        str(obj),
    ]
    if (
        command["argv"] != expected
        or command["executable"]["path"] != compiler["gcc"]
        or command["executable"]["sha256"] != compiler["gcc_sha256"]
        or len(command["inputs"]) != 1
        or command["inputs"][0]["path"] != str(source)
        or command["inputs"][0]["sha256"] != selected["source_sha256"]
        or command["output"]["sha256"] != selected["object_sha256"]
        or sha256_file(obj) != selected["object_sha256"]
        or any(record["commands"][-1]["argv"].count(flag) != 1 for flag in selected["link_flags"])
    ):
        raise ValueError("runtime supplier compilation or wrapper link flags differ from selected recipe")
