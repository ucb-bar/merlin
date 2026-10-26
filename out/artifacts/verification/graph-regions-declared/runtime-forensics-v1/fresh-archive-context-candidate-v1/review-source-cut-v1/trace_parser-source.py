"""Strict structural reader for the reviewed original-loader development trace.

No process is executed. This NEW observational policy uses explicit loader/OS/
module-reporting trust; it does not change the older denial inspector or prove
Linux path-search denial. Every raw line remains retained by the controller.
"""

from __future__ import annotations


class TraceRefusal(ValueError):
    pass


def parse_axioms(text, roots):
    """Consume the complete #print axioms buffer, including wrapped lists."""
    if len(roots) != len(set(roots)):
        raise TraceRefusal("duplicate requested exact audit roots")
    remaining, found = text.strip(), {}
    allowed = {"propext", "Classical.choice", "Quot.sound"}
    while remaining:
        if not remaining.startswith("'"):
            raise TraceRefusal("unexpected exact-root audit text")
        name, delimiter, rest = remaining[1:].partition("'")
        if not delimiter or name not in roots or name in found:
            raise TraceRefusal("unknown/duplicate/missing quoted audit root")
        rest = rest.lstrip()
        if rest.startswith("does not depend on any axioms"):
            found[name] = []
            remaining = rest.removeprefix("does not depend on any axioms").lstrip()
        elif rest.startswith("depends on axioms:"):
            rest = rest.removeprefix("depends on axioms:").lstrip()
            if not rest.startswith("["):
                raise TraceRefusal("missing axiom-list opening bracket")
            values, delimiter, rest = rest[1:].partition("]")
            if not delimiter:
                raise TraceRefusal("unterminated axiom list")
            axioms = [part.strip() for part in values.split(",")] if values.strip() else []
            if set(axioms) - allowed or len(set(axioms)) != len(axioms):
                raise TraceRefusal("non-standard/duplicate axiom in exact root")
            found[name] = axioms
            remaining = rest.lstrip()
        else:
            raise TraceRefusal("unknown exact-root audit predicate")
    if set(found) != set(roots):
        raise TraceRefusal("missing exact-root audit")
    return found


def zero_namespace(value):
    if value == "[0]":
        return ""
    name, separator, suffix = value.rpartition(" [")
    if not separator or suffix != "0]":
        raise TraceRefusal("unknown loader namespace syntax: " + value)
    return name


def optional_miss(body, known):
    """Known pinned Lean interpreter dlsym fallback diagnostics, not dlopen."""
    marker = ": error: symbol lookup error: undefined symbol: "
    owner, separator, symbol = body.partition(marker)
    if not separator:
        return None
    if not symbol.endswith(" (fatal)"):
        raise TraceRefusal("unknown symbol diagnostic syntax")
    symbol = symbol.removesuffix(" (fatal)")
    if owner not in known or not symbol or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in symbol):
        raise TraceRefusal("unbound symbol-diagnostic owner or nonidentifier")
    return owner, symbol


def baseline_misses(raw, known):
    result = set()
    for line in raw.splitlines():
        prefix, colon, body = line.partition(":")
        if not colon or not prefix.strip().isdigit():
            raise TraceRefusal("baseline has a non-loader diagnostic line")
        miss = optional_miss(body.strip(), known)
        if miss is not None:
            result.add(miss)
    return result


def parse(raw, runtime_refs, executable, interpreter, allowed_misses, expected_pid=None, project_initializers=()):
    known = {ref["path"] for ref in runtime_refs}
    aliases = {}
    for path in known:
        basename = path.rsplit("/", 1)[-1]
        if basename in aliases:
            raise TraceRefusal("ambiguous original runtime basename")
        aliases[basename] = path

    def bound(name):
        if name in known:
            return name
        if name in aliases:
            return aliases[name]
        raise TraceRefusal("unknown library/main in observed loader trace: " + name)

    initialized, finalized, linked, program, transferred = set(), set(), set(), [], []
    pids, probes, searches, symbol_misses = set(), [], [], []
    empty_fini, metadata_rows = 0, 0
    for number, line in enumerate(raw.splitlines(), 1):
        prefix, colon, tail = line.partition(":")
        if not colon or not prefix.strip().isascii() or not prefix.strip().isdigit():
            raise TraceRefusal(f"non-loader stderr line {number}: {line}")
        pid = int(prefix.strip())
        pids.add(pid)
        if expected_pid is not None and pid != expected_pid:
            raise TraceRefusal("unexpected process in loader trace")
        body = tail.strip()
        if not body:
            continue
        miss = optional_miss(body, known)
        if miss is not None:
            initializer = miss[0].rsplit("/", 1)[-1] == "libleanshared.so" and miss[1] in project_initializers
            if miss not in allowed_misses and not initializer:
                raise TraceRefusal("unreviewed optional symbol diagnostic: " + str(miss))
            symbol_misses.append(list(miss))
        elif body.startswith("calling init: "):
            initialized.add(bound(body.removeprefix("calling init: ").strip()))
        elif body.startswith("calling fini:"):
            name = zero_namespace(body.removeprefix("calling fini:").strip())
            if name:
                finalized.add(bound(name))
            else:
                empty_fini += 1
        elif body.startswith("initialize program: "):
            program.append(bound(body.removeprefix("initialize program: ").strip()))
        elif body.startswith("transferring control: "):
            transferred.append(bound(body.removeprefix("transferring control: ").strip()))
        elif body.startswith("file="):
            value, separator, action = body.removeprefix("file=").partition(";")
            if not separator:
                raise TraceRefusal("missing loader file-action delimiter")
            name = bound(zero_namespace(value.strip()))
            action = action.strip()
            if action == "generating link map":
                linked.add(name)
            elif action.startswith("needed by "):
                bound(zero_namespace(action.removeprefix("needed by ").strip()))
            else:
                raise TraceRefusal("unreviewed file/dynamic-load action: " + action)
        elif body.startswith("find library="):
            value, separator, action = body.removeprefix("find library=").partition(";")
            if not separator or action.strip() != "searching":
                raise TraceRefusal("unknown library-search syntax")
            bound(zero_namespace(value.strip()))
        elif body.startswith("search path="):
            searches.append(body)
        elif body.startswith("trying file="):
            probes.append(body.removeprefix("trying file="))
        elif body == "(no such file)":
            if not probes:
                raise TraceRefusal("missing-file diagnostic without preceding probe")
        elif body.startswith(("dynamic:", "entry:")):
            fields = body.split()
            labels = ["dynamic:", "base:", "size:"] if body.startswith("dynamic:") else ["entry:", "phdr:", "phnum:"]
            if len(fields) != 6 or fields[::2] != labels:
                raise TraceRefusal("unreviewed link-map metadata layout")
            for label, value in zip(labels, fields[1::2], strict=True):
                try:
                    int(value, 10 if label == "phnum:" else 16)
                except ValueError as error:
                    raise TraceRefusal("invalid link-map metadata value") from error
            metadata_rows += 1
        else:
            raise TraceRefusal(f"unreviewed loader line {number}: {body}")
    if len(pids) != 1:
        raise TraceRefusal("missing or multiple loader processes")
    expected_libraries = known - {executable}
    if initialized != expected_libraries or finalized != expected_libraries:
        raise TraceRefusal("original library init/fini closure differs")
    # The canonical loader is initialized but supplied by the kernel/direct
    # invocation, not by an ordinary needed-library link-map diagnostic.
    if linked != known - {interpreter}:
        raise TraceRefusal("original link-map closure differs")
    if program != [executable] or transferred != [executable] or empty_fini != 1:
        raise TraceRefusal("missing/duplicate/wrong main init/transfer/fini")
    return dict(
        pid=next(iter(pids)),
        init_files=sorted(initialized),
        fini_files=sorted(finalized),
        link_maps=sorted(linked),
        main=executable,
        empty_main_fini_namespace_zero=empty_fini,
        searches=searches,
        attempted_paths=probes,
        optional_symbol_misses=symbol_misses,
        link_map_metadata_rows=metadata_rows,
        dynamic_loads_observed=False,
        formal_OS_default_directory_denial=False,
        future_late_loads_ruled_out=False,
        trust=["OS/kernel/vDSO", "original pinned loader diagnostics", "host transport"],
    )
