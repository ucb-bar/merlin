"""Explicitly bypass only byte-recognized generated group trace wrappers.

Staged outside the source tree during the model-study build freeze. No model,
device, weight or retained profiling object is rewritten. This supplies link
binding observations, never a numerical, effect or physical timing verdict.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

_OWNED = object()


class _ObservationOwner:
    def __init__(self, record, *, supports=(), _owner=None):
        if _owner is not _OWNED:
            raise ValueError("live policy owner required")
        self._record = deepcopy(record)
        self._digest = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).digest()
        self._supports = tuple(supports)
        self._owner = _owner

    def record(self):
        if (
            self._owner is not _OWNED
            or hashlib.sha256(
                json.dumps(self._record, sort_keys=True).encode()
            ).digest()
            != self._digest
        ):
            raise ValueError("live policy observation changed")
        for support in self._supports:
            _live(support)
        _verify_inputs(self._record)
        return deepcopy(self._record)


def _live(value):
    if type(value) is not _ObservationOwner:
        raise ValueError(
            "serialized records do not reconstruct live policy observations"
        )
    return value.record()


from merlin.common import invocation_record
from merlin.llvmlower.link_supplier_trace import observe_link_suppliers
from merlin.model_check import chain_image_layout as layout
from merlin.model_check import chain_image_objects as objects
from merlin.runtime import whole_model_readback as readback

RETAIN = "retain"
BYPASS = "bypass_verified_neutral"
RESET = (
    "void merlin_group_profile_reset(void) {\n"
    "  merlin_group_count = 0;\n  merlin_group_dropped = 0;\n}\n"
)
HOOKS = frozenset(
    (
        "merlin_group_profile_now",
        "merlin_group_profile_reset",
        "merlin_group_profile_dump",
    )
)


def selected(value=RETAIN, *, group_profile=False):
    if (
        type(value) is not str
        or value not in (RETAIN, BYPASS)
        or type(group_profile) is not bool
    ):
        raise ValueError("select an explicit retained group trace policy")
    if value == BYPASS and group_profile:
        raise ValueError(
            "bypassing retained trace conflicts with explicit group profiling"
        )
    return value == BYPASS


def _symbol(value):
    if type(value) is not str or not value.isascii() or not value.isidentifier():
        raise ValueError("generated profiling needs ordinary C symbol names")
    return value


def wraps(argv):
    found = []
    for value in argv:
        if type(value) is not str or value.startswith("@"):
            raise ValueError("link policy requires explicit argument tokens")
        parts = value[4:].split(",") if value.startswith("-Wl,") else [value]
        if any(p.startswith("@") for p in parts):
            raise ValueError("link response files are outside this policy")
        mentions = [p for p in parts if p == "--wrap" or p.startswith("--wrap=")]
        if mentions:
            if len(parts) != 1 or not value.startswith("-Wl,--wrap="):
                raise ValueError("unsupported linker wrap spelling")
            found.append(_symbol(value.removeprefix("-Wl,--wrap=")))
    if len(set(found)) != len(found):
        raise ValueError("repeated linker wrap declaration")
    return tuple(found)


def recognize(source, routing, undefined, bindings=None):
    """Derive prototypes from observed callees and compare complete generated C."""
    rows = list(routing.get("routed") or [])
    for row in rows:
        _symbol(row.get("symbol"))
        if row.get("group") is not None and (
            type(row["group"]) is not int or row["group"] < 0
        ):
            raise ValueError("unknown original group label")
    callees = readback.callees(
        rows,
        routing.get("signatures") or {},
        set(undefined),
        routing.get("call_buffers"),
        bindings or {},
    )
    for callee in callees:
        _symbol(callee.symbol)
        if callee.routed_symbol is not None:
            _symbol(callee.routed_symbol)
    variants = []
    for label, names in [
        ("source_route_order", readback.group_names(rows)),
        ("observed_callee_only", []),
    ]:
        expected = readback.render_group_profile(callees, names)
        variants.append((label, "current_exact", expected))
        if expected.count(RESET) == 1:
            variants.append(
                (label, "known_without_reset", expected.replace(RESET, "", 1))
            )
    matches = [
        (label, form) for label, form, expected in variants if source == expected
    ]
    if not matches:
        raise ValueError(
            "retained profiling source is not an exact known neutral generator form"
        )
    return tuple(c.symbol for c in callees), matches[0]


def _compile_flags(command, source, product):
    allowed = {
        "-O0",
        "-O1",
        "-O2",
        "-O3",
        "-Os",
        "-Og",
        "-g",
        "-c",
        "-o",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-stack-protector",
        "-ffunction-sections",
        "-fdata-sections",
        "-fno-lto",
        "-fno-fast-math",
        "-fno-finite-math-only",
        "-ffp-contract=off",
    }
    argv = command["argv"]
    if (
        argv.count("-c") != 1
        or argv.count("-o") != 1
        or argv.count(str(product)) != 1
        or argv[argv.index("-o") + 1] != str(product)
    ):
        raise ValueError("profiling object has no supported original C compilation")
    for index, value in enumerate(argv[1:], 1):
        if value in {str(source), str(product)} or argv[index - 1] == "-o":
            continue
        if value not in allowed and not value.startswith(
            ("-march=", "-mabi=", "-mcmodel=", "-std=")
        ):
            raise ValueError(
                "unknown profiling compiler flag; do not infer neutral object behavior"
            )
    if argv.count(str(source)) != 1:
        raise ValueError("profiling source is not the unique original C input")


def prepare(program, directory, *, nm, timeout):
    """Read a completed production; return pins and a finite removal roster."""
    directory.mkdir()
    original_wraps = wraps(program["link"]["argv"])
    profile = program["by_name"].get("group_profile.o")
    observations = [program["receipt_pin"], program["recipe_pin"]]
    if profile is None:
        model = objects.reopen(program["by_name"]["model.o"])
        table = objects.symbols(nm, model, timeout=timeout, directory=directory)
        undefined = {name for name, kind in table.items() if kind == "U"}
        sidecar = program["receipt"]["output"]["device_sidecar"]
        routing = json.loads(objects.payload(sidecar))
        selected_callees = readback.callees(
            routing.get("routed") or [],
            routing.get("signatures") or {},
            undefined,
            routing.get("call_buffers"),
        )
        if (
            set(original_wraps) & {c.symbol for c in selected_callees}
            or set(table) & HOOKS
        ):
            raise ValueError(
                "group interception has no known retained generated object"
            )
        return _ObservationOwner(
            {
                "schema": "merlin.retained_group_trace_policy.v1",
                "mode": BYPASS,
                "status": "already_unprofiled",
                "profiling_wraps": [],
                "original_wraps": list(original_wraps),
                "inputs": [*observations, objects.pin(model), sidecar],
            },
            _owner=_OWNED,
        )
    profile_path = objects.reopen(profile)
    model = program["by_name"]["model.o"]
    model_path = objects.reopen(model)
    model_table = objects.symbols(nm, model_path, timeout=timeout, directory=directory)
    undefined = {name for name, kind in model_table.items() if kind == "U"}
    sidecar = program["receipt"]["output"]["device_sidecar"]
    routing = json.loads(objects.payload(sidecar))
    bindings, more = {}, []
    missing = {
        row["symbol"]
        for row in routing.get("routed", ())
        if row["symbol"] not in undefined
        and "_mlir_ciface_" + row["symbol"] not in undefined
    }
    if missing:
        from merlin.llvmlower.group_profile_sibling import completed_sibling_bindings

        bindings, more = completed_sibling_bindings(program, routing, undefined)
        if missing - bindings.keys():
            from merlin.model_check.chain_image_profile_descriptor import (
                completed_descriptor_bindings,
            )

            other, extra = completed_descriptor_bindings(program, routing, undefined)
            if bindings.keys() & other.keys():
                raise ValueError("ambiguous profiling source-to-call binding")
            bindings.update(other)
            more.extend(extra)
    command = layout.command(program, "group_profile.o")
    sources = [p for p in command["inputs"] if p["path"].endswith(".c")]
    if len(sources) != 1 or len(command["inputs"]) != 1:
        raise ValueError(
            "profiling support lacks one complete original generated source"
        )
    source = objects.reopen(sources[0])
    objects.reopen(command["executable"])
    _compile_flags(command, source, profile_path)
    symbols, form = recognize(
        objects.payload(sources[0]).decode("utf-8"), routing, undefined, bindings
    )
    if any(original_wraps.count(name) != 1 for name in symbols):
        raise ValueError(
            "generated profiling callee has no exact original wrap declaration"
        )
    table = objects.symbols(nm, profile_path, timeout=timeout, directory=directory)
    generated = {"__wrap_" + name for name in symbols}
    definitions = {
        name
        for name, kind in table.items()
        if kind.isupper() and kind not in {"U", "N"}
    }
    if definitions - generated - HOOKS or not generated <= definitions:
        raise ValueError(
            "profiling object defines unrelated public functions or omits a wrapper"
        )
    permitted_undefined = {"__real_" + name for name in symbols} | {
        "htif_puts",
        "htif_putc",
    }
    if {name for name, kind in table.items() if kind == "U"} - permitted_undefined:
        raise ValueError("profiling object has an unknown external side effect")
    main = program["by_name"].get("model_main.o")
    main_code = set()
    if main is not None:
        main_table = objects.symbols(
            nm, objects.reopen(main), timeout=timeout, directory=directory
        )
        main_code = {name for name, kind in main_table.items() if kind.upper() == "T"}
    blocked = generated | {"__real_" + name for name in symbols} | HOOKS
    # Whole-object reference closure is conservative. The old driver may use
    # hooks; it is not the chain's merlin_invoke forward root. Other original
    # program code may not reach it or refer directly to generated profiling.
    checked = []
    for row in program["objects"]:
        name = Path(row["path"]).name
        if name in {"group_profile.o", "model_main.o", "weights_blob.o"}:
            continue
        path = objects.reopen(row)
        own = (
            model_table
            if row == model
            else objects.symbols(nm, path, timeout=timeout, directory=directory)
        )
        if set(own) & blocked:
            raise ValueError(
                "original forward/runtime object directly references profiling support"
            )
        if (
            name not in objects.COMMON_OBJECTS
            and {s for s, k in own.items() if k == "U"} & main_code
        ):
            raise ValueError(
                "original forward code may enter the retained profiled driver"
            )
        checked.append(row)
    observations += [
        profile,
        model,
        sidecar,
        sources[0],
        command["executable"],
        *more,
        *checked,
    ]
    for pin in observations:
        objects.reopen(pin)
    return _ObservationOwner(
        {
            "schema": "merlin.retained_group_trace_policy.v1",
            "mode": BYPASS,
            "status": "verified_neutral",
            "profiling_wraps": list(symbols),
            "original_wraps": list(original_wraps),
            "retained_semantic_wraps": [s for s in original_wraps if s not in symbols],
            "generator_form": {"labels": form[0], "form": form[1]},
            "inputs": observations,
            "generator": objects.pin(Path(readback.__file__)),
            "policy_source": objects.pin(Path(__file__).resolve()),
            "profile_object": profile,
            "forward_reference_objects": checked,
            "original_objects": program["objects"],
            "scope": "Known generated group wrappers only; original objects and arithmetic unchanged. "
            "Other instrumentation and explicit overall counter boundaries are retained.",
        },
        _owner=_OWNED,
    )


def rewrite(argv, support, namespaces, *, nm, timeout):
    """Remove only qualified flags; retain objects, resolve unused __real refs."""
    if len(support) != len(namespaces):
        raise ValueError("trace policy omitted an original program")
    original = list(argv)
    original_wraps = wraps(original)
    removals, aliases, targets, inputs = [], {}, {}, []
    for owner, namespace in zip(support, namespaces, strict=True):
        record = _live(owner)
        if "original_objects" in record:
            selected_original = [
                p
                for p in record["original_objects"]
                if Path(p["path"]).name not in objects.COMMON_OBJECTS
            ]
            if namespace.get("original_objects") != selected_original:
                raise ValueError("namespace substituted an original program object")
        mapping = dict(
            line.split()
            for line in objects.payload(namespace["mapping"]).decode().splitlines()
        )
        inputs.append(namespace["mapping"])
        for old in record["profiling_wraps"]:
            target = namespace["prefix"] + old
            expected = {
                old: target,
                "__wrap_" + old: "__wrap_" + target,
                "__real_" + old: "__real_" + target,
            }
            if any(mapping.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "namespace does not preserve original profiling linkage"
                )
            if original_wraps.count(target) != 1:
                raise ValueError("actual final link lacks the qualified unique wrapper")
            suppliers = []
            for row in namespace["objects"]:
                own = objects.symbols(nm, objects.reopen(row), timeout=timeout)
                if own.get(target) == "T":
                    suppliers.append(row)
            if len(suppliers) != 1:
                raise ValueError("unwrapped target lacks one actual defining object")
            if target in targets:
                raise ValueError("program namespaces overlap")
            targets[target] = suppliers[0]
            aliases["__real_" + target] = target
            removals.append(target)
    if len(targets) > 64:
        raise ValueError(
            "bounded supplier observation needs at most 64 unwrapped callees"
        )
    final = [
        value
        for value in original
        if value not in {"-Wl,--wrap=" + s for s in removals}
    ]
    definitions = set()
    for value in original:
        parts = value[4:].split(",") if value.startswith("-Wl,") else [value]
        for index, part in enumerate(parts):
            assignment = (
                parts[index + 1]
                if part == "--defsym" and index + 1 < len(parts)
                else part.removeprefix("--defsym=")
                if part.startswith("--defsym=")
                else None
            )
            if assignment is not None:
                key, separator, _value = assignment.partition("=")
                if not separator:
                    raise ValueError("unsupported original linker symbol definition")
                definitions.add(_symbol(key))
    if definitions & (
        set(aliases) | set(targets) | {"__wrap_" + name for name in targets}
    ):
        raise ValueError(
            "original final link already overrides a generated call binding"
        )
    for alias, target in aliases.items():
        final.insert(final.index("-o"), "-Wl,--defsym," + alias + "=" + target)
    if wraps(final) != tuple(s for s in original_wraps if s not in removals):
        raise ValueError("semantic linker wraps changed")
    record = {
        "schema": "merlin.retained_group_trace_link_policy.v1",
        "mode": BYPASS,
        "original_argv": original,
        "final_policy_argv": final,
        "removed_wraps": removals,
        "retained_wraps": list(wraps(final)),
        "real_aliases": aliases,
        "expected_suppliers": targets,
        "inputs": inputs,
        "support": [_live(owner) for owner in support],
        "scope": "Link-time bypass of verified generated group wrappers; no object rewrite. "
        "Global ROI counter calls and other instrumentation are not removed.",
    }
    return (
        final,
        {symbol: objects.reopen(pin) for symbol, pin in targets.items()},
        _ObservationOwner(record, supports=support, _owner=_OWNED),
    )


def _verify_inputs(record):
    for pin in record.get("inputs", ()):
        objects.reopen(pin)
    if "generator" in record:
        objects.reopen(record["generator"])
    if "policy_source" in record:
        objects.reopen(record["policy_source"])


def verify_final(owner, elf, *, argv, result, inputs, nm, timeout, directory):
    """Observe actual target suppliers and alias addresses after the real link."""
    record = _live(owner)
    if result.returncode != 0 or list(map(str, result.args)) != list(argv):
        raise ValueError("policy has no matching successful final link")

    def without_traces(values):
        clean = []
        for value in values:
            if value.startswith("-Wl,--trace-symbol="):
                _symbol(value.removeprefix("-Wl,--trace-symbol="))
            else:
                clean.append(value)
        return clean

    if without_traces(argv) != without_traces(record["final_policy_argv"]):
        raise ValueError(
            "actual link changed more than the selected group trace policy"
        )
    expected = {
        symbol: objects.reopen(pin)
        for symbol, pin in record["expected_suppliers"].items()
    }
    if any("-Wl,--wrap=" + name in argv for name in record["removed_wraps"]):
        raise ValueError("actual final link still intercepts a selected group")
    if wraps(argv) != tuple(record["retained_wraps"]):
        raise ValueError("actual final link changed semantic wrapping")
    for alias, target in record["real_aliases"].items():
        if argv.count("-Wl,--defsym," + alias + "=" + target) != 1:
            raise ValueError("actual final link omitted a direct real alias")
    suppliers = (
        observe_link_suppliers(
            argv=argv,
            input_identities=inputs,
            expected_suppliers=expected,
            stderr=result.stderr,
        )
        if expected
        else None
    )
    observed = invocation_record.run(
        [str(nm), "--print-size", "--format=posix", str(elf)],
        directory=directory,
        stage="chain_image_profile_bypass_symbols",
        inputs=(elf,),
        dependencies=(Path(__file__),),
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    wanted = (
        set(record["real_aliases"]) | set(expected) | {"__wrap_" + s for s in expected}
    )
    values = {}
    for line in observed.stdout.splitlines():
        row = line.split()
        if row and row[0] in wanted:
            if len(row) not in (3, 4) or row[0] in values:
                raise ValueError("final ELF has ambiguous target or profiling symbols")
            values[row[0]] = (row[1], int(row[2], 16))
    for alias, target in record["real_aliases"].items():
        wrapper = "__wrap_" + target
        if (
            target not in values
            or alias not in values
            or wrapper not in values
            or values[target][0] != "T"
            or values[wrapper][0] != "T"
            or values[alias][1] != values[target][1]
            or values[target][1] == values[wrapper][1]
        ):
            raise ValueError(
                "actual linked target is not distinct from retained tracing wrapper"
            )
    return {
        "policy": record,
        "actual_argv": list(argv),
        "elf": objects.pin(elf),
        "observed_symbols": values,
        "link_suppliers": suppliers,
        "scope": "Selected forward external calls bind directly to their original target suppliers. "
        "Generated wrappers remain in the image but are not interposed. No numerical/timing verdict.",
    }
