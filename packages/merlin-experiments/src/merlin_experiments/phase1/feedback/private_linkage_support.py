"""Private exact-source requirement joined to an independently selected link supplier.

This proves an archive member or selected runtime object defined a requested symbol in the completed link.
It does not prove a source call reaches that symbol or numerical equivalence.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from merlin.common.digest import is_sha256, sha256_file
from merlin.compile.host_lane import require_host_isa_dts
from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.llvmlower.compilation_recipe import verify_completed_recipe
from merlin.llvmlower.link_supplier_trace import trace_symbol_flags
from merlin.mining.registry import load_rvv_package
from merlin.runtime.backends import spike, spike_model
from merlin.targetgen.host_linkage_contract import RUNTIME_SUPPLIER, runtime_recipe, validate_source_linkage_contract

FIELD = "host_linkage_support"
PENDING = "reviewed_source_linkage_requirement_pending_build"
LINKED = "reviewed_source_linkage_supplier_verified"
SCOPE = "exact reviewed source ordinals and actual defining supplier; source-call and numerical proof excluded"
_BUILD = {"capture_tree_sha256", "candidate_tree_sha256", "elf_sha256"}
_OCCURRENCE = {"ordinal", "profile", "capability_spec_sha256", "declaration", "schema", "operation", "contract"}
_TOP = {
    "status",
    "scope",
    "raw_source_sha256",
    "normalized_source_sha256",
    "n_source_operations",
    "count",
    "occurrences",
    "linked_build",
    "selection",
    "recipe_sha256",
    "host_math_policy",
}


def begin(raw_sha256: str, normalized_sha256: str, n_operations: int) -> dict[str, Any]:
    """A mandatory empty-or-complete source record, including for no-policy builds."""
    if (
        not is_sha256(raw_sha256)
        or not is_sha256(normalized_sha256)
        or type(n_operations) is not int
        or n_operations < 1
    ):
        raise ValueError("linkage source identity is incomplete")
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "n_source_operations": n_operations,
        "count": 0,
        "occurrences": [],
        "linked_build": None,
        "selection": None,
        "recipe_sha256": None,
        "host_math_policy": None,
    }


def record(
    witness: dict[str, Any],
    row: Mapping[str, Any],
    admission: Mapping[str, Any],
    source_rows: Mapping[int, Mapping[str, Any]],
    linalg_witness: Mapping[str, Any],
) -> None:
    """Retain only exact reviewed host ordinals with an already-proved source body."""
    requirement = admission.get("linkage_requirement")
    if requirement is None:
        return
    body = admission.get("source_body_proof")
    ordinals = row.get("ordinals")
    proved = linalg_witness.get("occurrences")
    if (
        witness.get("status") != PENDING
        or admission.get("status") != "admitted"
        or admission.get("reviewed") is not True
        or not isinstance(requirement, Mapping)
        or set(requirement) != {"profile", "capability_spec_sha256", "declaration", "contract"}
        or not isinstance(body, Mapping)
        or not isinstance(ordinals, list)
        or type(row.get("count")) is not int
        or len(ordinals) != row["count"]
        or not isinstance(proved, list)
        or (requirement["profile"], requirement["capability_spec_sha256"], requirement["declaration"])
        != (body.get("profile"), body.get("capability_spec_sha256"), body.get("declaration"))
    ):
        raise ValueError("linkage requirement has no exact reviewed source occurrence roster")
    contract = validate_source_linkage_contract(body.get("schema"), body.get("operation"), requirement["contract"])
    seen = {item["ordinal"] for item in witness["occurrences"]}
    for ordinal in ordinals:
        matches = [
            item
            for item in proved
            if item.get("ordinal") == ordinal
            and item.get("schema") == body.get("schema")
            and item.get("profile") == requirement["profile"]
            and item.get("capability_spec_sha256") == requirement["capability_spec_sha256"]
            and item.get("declaration") == requirement["declaration"]
            and item.get("operation") == body.get("operation")
        ]
        if (
            type(ordinal) is not int
            or ordinal < 0
            or ordinal >= witness["n_source_operations"]
            or ordinal in seen
            or source_rows.get(ordinal) is not row
            or len(matches) != 1
            or matches[0].get("linkage_requirement") != contract
        ):
            raise ValueError("linkage requirement differs from exact proved source ordinal")
        witness["occurrences"].append(
            {
                "ordinal": ordinal,
                "profile": requirement["profile"],
                "capability_spec_sha256": requirement["capability_spec_sha256"],
                "declaration": requirement["declaration"],
                "schema": body["schema"],
                "operation": body["operation"],
                "contract": deepcopy(contract),
            }
        )
        seen.add(ordinal)
    witness["occurrences"].sort(key=lambda item: item["ordinal"])
    witness["count"] = len(witness["occurrences"])


def symbols(source: Mapping[str, Any]) -> list[str] | None:
    """Return the unique requested roster, or None for the unchanged default build."""
    witness = source.get(FIELD)
    if not isinstance(witness, Mapping) or witness.get("status") != PENDING:
        raise ValueError("source has no pending linkage roster")
    occurrences = witness.get("occurrences")
    if not isinstance(occurrences, list) or witness.get("count") != len(occurrences):
        raise ValueError("source linkage roster is incomplete")
    selected = sorted(
        {
            symbol
            for item in occurrences
            for symbol in validate_source_linkage_contract(item["schema"], item["operation"], item["contract"])[
                "symbols"
            ]
        }
    )
    if selected:
        trace_symbol_flags(selected)
    return selected or None


def build_options(source: dict[str, Any], policy: str) -> dict[str, list[str]]:
    """Bind original selected policy and split explicit archive/runtime requests."""
    from merlin.runtime.host_math import host_math_recipe

    symbols(source)  # Validate the complete pending source roster first.
    witness = source[FIELD]
    archive, runtime = set(), set()
    recipe = host_math_recipe(policy)
    wrapped = {flag.removeprefix("-Wl,--wrap=") for flag in recipe.link_flags if flag.startswith("-Wl,--wrap=")}
    for item in witness["occurrences"]:
        contract = validate_source_linkage_contract(item["schema"], item["operation"], item["contract"])
        if contract["supplier"] == RUNTIME_SUPPLIER:
            runtime_recipe(contract, policy)
            runtime.update(contract["symbols"])
        else:
            if wrapped.intersection(contract["symbols"]):
                raise ValueError(
                    "selected portable wrapper requires a matching portable-provider contract; "
                    "archive relabeling is forbidden"
                )
            archive.update(contract["symbols"])
    if runtime:
        if witness["host_math_policy"] not in (None, policy):
            raise ValueError("runtime supplier policy changed after original source selection")
        witness["host_math_policy"] = policy
    return {
        **({"math_archive_symbols": sorted(archive)} if archive else {}),
        **({"host_math_symbols": sorted(runtime)} if runtime else {}),
    }


def _selected_archive(
    package: Path,
    expected_tree_sha256: str,
    vlen: int | None,
    dts: Path,
    expected_dts_sha256: str,
) -> dict[str, Any]:
    """Re-select the actual driver/archive from the pinned host package, never trace text."""
    if (
        not package.is_absolute()
        or package.is_symlink()
        or strict_tree_sha256(package)["sha256"] != expected_tree_sha256
    ):
        raise ValueError("linkage host package differs from selected bytes")
    pkg = load_rvv_package(package)
    if not dts.is_absolute() or dts.is_symlink() or sha256_file(dts) != expected_dts_sha256:
        raise ValueError("linkage host DTS differs from selected bytes")
    isas = require_host_isa_dts(pkg.cflags, dts, expected_sha256=expected_dts_sha256)
    marches = [flag.removeprefix("-march=") for flag in pkg.cflags if flag.startswith("-march=")]
    if len(isas) != 1 or len(marches) != 1:
        raise ValueError("linkage host package has no singular selected ISA")
    vlen = vlen if pkg.backend == "rvv" else None
    flags = spike_model._harness_cflags(list(pkg.cflags))  # noqa: PLC2701 -- actual builder flag owner
    if pkg.backend == "rvv" and vlen is not None:
        from merlin.runtime.backends.zephyr_model import march_with_vlen

        flags = march_with_vlen(flags, vlen)
    gcc = spike.gcc_path()
    archive, archive_sha, driver_sha = spike_model._selected_libm_archive(  # noqa: PLC2701 -- actual selected archive owner
        gcc, flags, ()
    )
    return {
        "host_package": str(package),
        "host_package_tree_sha256": expected_tree_sha256,
        "dts": str(dts),
        "dts_sha256": expected_dts_sha256,
        "host_isa": isas[0],
        "simulator_isa": marches[0],
        "backend": pkg.backend,
        "vlen": vlen,
        "gcc_requested": str(gcc),
        "gcc": str(gcc.resolve(strict=True)),
        "gcc_sha256": driver_sha,
        "gcc_cflags": flags,
        "archive": str(archive),
        "archive_sha256": archive_sha,
    }


def _verify_recipe(entry: Mapping[str, Any], selection: Mapping[str, Any], selected_symbols: list[str]) -> str:
    binding = entry.get("compilation_recipe")
    if not isinstance(binding, Mapping) or binding.get("status") != "completed_compilation_inputs_verified":
        raise ValueError("linked build has no verified compilation recipe")
    recipe = Path(str(binding.get("recipe_path", "")))
    elf = Path(str(binding.get("elf_path", "")))
    if not recipe.is_absolute() or not elf.is_absolute() or binding.get("elf_sha256") != entry.get("elf_sha256"):
        raise ValueError("linkage proof has no exact linked image")
    runtime = selection.get("runtime")
    portable = runtime["symbols"] if runtime else ()
    expected = {symbol: Path(selection["archive"]) for symbol in selected_symbols if symbol not in portable}
    if runtime:
        expected.update({symbol: Path(runtime["object"]) for symbol in runtime["wrapper_symbols"].values()})
    record = verify_completed_recipe(
        recipe,
        executable=elf,
        expected_recipe_sha256=binding["recipe_sha256"],
        expected_link_suppliers=expected,
    )
    link = record["commands"][-1]
    driver = link["executable"]
    tail = link["argv"][1 + len(selection["gcc_cflags"]) :]
    if (
        driver["path"] != selection["gcc"]
        or driver["sha256"] != selection["gcc_sha256"]
        or link["argv"][: 1 + len(selection["gcc_cflags"])] != [selection["gcc_requested"], *selection["gcc_cflags"]]
        or any(flag.startswith(("-march=", "-mabi=", "-mcmodel=", "-L", "-B", "--sysroot")) for flag in tail)
        or not any(
            item["path"] == selection["archive"] and item["sha256"] == selection["archive_sha256"]
            for item in link["inputs"]
        )
        or sha256_file(Path(selection["archive"])) != selection["archive_sha256"]
    ):
        raise ValueError("actual link differs from independently selected GCC, ISA flags or archive")
    if runtime:
        from .private_runtime_math_supplier import verify_compile

        verify_compile(record, runtime, selection)
    return binding["recipe_sha256"]


def link(
    source: dict[str, Any],
    entry: Mapping[str, Any],
    *,
    host_package: Path,
    host_package_tree_sha256: str,
    vlen: int | None,
    dts: Path,
    dts_sha256: str,
    host_isa: str,
    simulator_isa: str,
    linked_build: Mapping[str, str],
    host_math_policy: str = "native",
) -> None:
    witness = source.get(FIELD)
    if not isinstance(witness, dict) or witness.get("status") != PENDING or set(linked_build) != _BUILD:
        raise ValueError("linkage source or linked build is incomplete")
    selected_symbols = symbols(source)
    if selected_symbols is None:
        witness.update(status=LINKED, linked_build=dict(linked_build))
        return
    contracts = [
        validate_source_linkage_contract(item["schema"], item["operation"], item["contract"])
        for item in witness["occurrences"]
    ]
    selected = _selected_archive(host_package, host_package_tree_sha256, vlen, dts, dts_sha256)
    if (selected["host_isa"], selected["simulator_isa"]) != (host_isa, simulator_isa):
        raise ValueError("linkage build differs from independently selected host ISA")
    runtime_contracts = [contract for contract in contracts if contract["supplier"] == RUNTIME_SUPPLIER]
    if runtime_contracts:
        from .private_runtime_math_supplier import select as select_runtime

        if witness["host_math_policy"] != host_math_policy:
            raise ValueError("linked runtime policy differs from original selected source/reference policy")
        selected["runtime"] = select_runtime(
            host_math_policy, runtime_contracts, selected, Path(entry["compilation_recipe"]["recipe_path"])
        )
    if any(
        contract["archive_sha256"] != selected["archive_sha256"]
        for contract in contracts
        if contract["supplier"] != RUNTIME_SUPPLIER
    ):
        raise ValueError("reviewed linkage archive differs from selected compiler archive")
    recipe_sha = _verify_recipe(entry, selected, selected_symbols)
    witness.update(status=LINKED, linked_build=dict(linked_build), selection=selected, recipe_sha256=recipe_sha)


def link_compiled(
    source: dict[str, Any],
    entry: Mapping[str, Any],
    receipt: Mapping[str, Any],
    host_package: Path | None,
    host_package_tree_sha256: str | None,
    catalog: Path,
    board: str,
    dts: Path,
    linked_build: Mapping[str, str],
) -> None:
    """Select the same actual host/board inputs as the canonical saved-model build."""
    if host_package is None and host_package_tree_sha256 is None:
        return  # Historical direct helper callers cannot complete the v10 roster.
    if host_package is None or host_package_tree_sha256 is None:
        raise ValueError("linkage host package selection is incomplete")
    inputs = receipt.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("package") != str(host_package)
        or inputs.get("package_tree") != strict_tree_sha256(host_package)
        or inputs["package_tree"]["sha256"] != host_package_tree_sha256
    ):
        raise ValueError("linked build used a different selected host package")
    from merlin.runtime.boards import load_boards

    selected_board = load_boards(catalog)[board]
    link(
        source,
        entry,
        host_package=host_package,
        host_package_tree_sha256=host_package_tree_sha256,
        vlen=selected_board.vlen,
        dts=dts,
        dts_sha256=sha256_file(dts),
        host_isa=inputs["host_isa"],
        simulator_isa=inputs["simulator_isa"],
        linked_build=linked_build,
        host_math_policy=inputs.get("host_math_policy", "native"),
    )


def linked_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """Recheck every roster, source/candidate/capture/ELF join and current supplier bytes."""
    witness = source.get(FIELD)
    if (
        not isinstance(witness, Mapping)
        or set(witness) != _TOP
        or witness.get("status") != LINKED
        or witness.get("scope") != SCOPE
        or witness.get("raw_source_sha256") != source.get("source_sha256")
        or witness.get("normalized_source_sha256") != source.get("normalized_source_sha256")
        or not is_sha256(source.get("source_sha256"))
        or not is_sha256(source.get("normalized_source_sha256"))
        or type(witness.get("n_source_operations")) is not int
        or witness["n_source_operations"] < 1
        or type(source.get("n_source_operations")) is not int
        or witness.get("n_source_operations") != source.get("n_source_operations")
        or entry.get("source_sha256") != source.get("source_sha256")
        or not isinstance(witness.get("occurrences"), list)
        or type(witness.get("count")) is not int
        or witness.get("count") != len(witness["occurrences"])
        or not isinstance(witness.get("linked_build"), Mapping)
        or set(witness["linked_build"]) != _BUILD
        or witness["linked_build"].get("candidate_tree_sha256") != candidate_sha256
        or witness["linked_build"].get("capture_tree_sha256") != entry.get("capture_tree_sha256")
        or witness["linked_build"].get("elf_sha256") != entry.get("elf_sha256")
    ):
        return False
    occurrences = witness["occurrences"]
    linalg = source.get("linalg_host_support")
    proved = linalg.get("occurrences") if isinstance(linalg, Mapping) else None
    if not isinstance(proved, list):
        return False
    expected = [
        {
            "ordinal": item.get("ordinal"),
            "profile": item.get("profile"),
            "capability_spec_sha256": item.get("capability_spec_sha256"),
            "declaration": item.get("declaration"),
            "schema": item.get("schema"),
            "operation": item.get("operation"),
            "contract": item.get("linkage_requirement"),
        }
        for item in proved
        if isinstance(item, Mapping) and item.get("linkage_requirement") is not None
    ]
    if expected != occurrences:
        return False
    seen = set()
    try:
        for item in occurrences:
            if not isinstance(item, Mapping) or set(item) != _OCCURRENCE:
                return False
            ordinal = item["ordinal"]
            if (
                type(ordinal) is not int
                or not 0 <= ordinal < witness["n_source_operations"]
                or ordinal in seen
                or not is_sha256(item["capability_spec_sha256"])
                or len(
                    [
                        row
                        for row in proved
                        if row.get("ordinal") == ordinal
                        and row.get("profile") == item["profile"]
                        and row.get("capability_spec_sha256") == item["capability_spec_sha256"]
                        and row.get("declaration") == item["declaration"]
                        and row.get("schema") == item["schema"]
                        and row.get("operation") == item["operation"]
                    ]
                )
                != 1
            ):
                return False
            validate_source_linkage_contract(item["schema"], item["operation"], item["contract"])
            seen.add(ordinal)
        selected_symbols = sorted({symbol for item in occurrences for symbol in item["contract"]["symbols"]})
        if not selected_symbols:
            return (
                witness.get("selection") is None
                and witness.get("recipe_sha256") is None
                and witness.get("host_math_policy") is None
            )
        selection = witness.get("selection")
        if not isinstance(selection, Mapping) or not is_sha256(witness.get("recipe_sha256")):
            return False
        package = Path(selection["host_package"])
        actual = _selected_archive(
            package,
            selection["host_package_tree_sha256"],
            selection["vlen"],
            Path(selection["dts"]),
            selection["dts_sha256"],
        )
        runtime_contracts = [
            item["contract"] for item in occurrences if item["contract"]["supplier"] == RUNTIME_SUPPLIER
        ]
        if not runtime_contracts and witness["host_math_policy"] is not None:
            return False
        if runtime_contracts:
            from .private_runtime_math_supplier import select as select_runtime

            actual["runtime"] = select_runtime(
                witness["host_math_policy"], runtime_contracts, actual, Path(entry["compilation_recipe"]["recipe_path"])
            )
        if actual != selection or any(
            item["contract"]["archive_sha256"] != actual["archive_sha256"]
            for item in occurrences
            if item["contract"]["supplier"] != RUNTIME_SUPPLIER
        ):
            return False
        return _verify_recipe(entry, actual, selected_symbols) == witness["recipe_sha256"]
    except (KeyError, OSError, TypeError, ValueError):
        return False
