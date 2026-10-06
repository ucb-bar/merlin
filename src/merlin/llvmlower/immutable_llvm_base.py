"""Explicit immutable address binding through a retained public LLVM wrapper.

This changes a helper's private parameter/liveness choice and retains its arithmetic.
The source address is passed once to an out-of-line implementation. Codegen may
still spill/rematerialize it; emitted complete-body measurements govern use.
No target, workload, numerical permission or default policy is supplied.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .late_quant_rne import _functions, _tokens


@dataclass(frozen=True)
class ImmutableBaseBinding:
    """Symbols identify an explicit source binding, never a strategy selector."""

    function: str
    global_symbol: str
    implementation: str


def _symbol(name):
    if (
        not isinstance(name, str)
        or not name
        or name[0].isdigit()
        or not all(c.isascii() and (c.isalnum() or c in "_.$-") for c in name)
    ):
        raise ValueError("simple explicit LLVM symbol required")
    return "@" + name


def _readonly_uses(body, global_symbol):
    """Close each derived address through GEPs to non-pointer loads only."""
    statements = [[t.text for t in _tokens(line)] for line in body.splitlines()]
    pending, seen, direct = [global_symbol], set(), 0
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        for statement in statements:
            for index, token in enumerate(statement):
                if token != value or (index == 0 and len(statement) > 1 and statement[1] == "="):
                    continue
                if len(statement) < 4 or statement[1] != "=" or index < 3 or statement[index - 1] != "ptr":
                    raise ValueError("immutable address escape or unsupported use")
                if statement[2] == "getelementptr":
                    pending.append(statement[0])
                    direct += value == global_symbol
                elif statement[2] == "load":
                    comma = statement.index(",")
                    if "ptr" in statement[3:comma]:
                        raise ValueError("pointer-valued immutable loads are unsupported")
                else:
                    raise ValueError("immutable address escape or unsupported use")
    if not direct:
        raise ValueError("selected function has no direct immutable GEP")
    return direct, sorted(seen - {global_symbol})


def _context_refusal(body, attributes, tokens, function):
    """Refuse LLVM-defined observations tied to the original function frame.

    Intrinsic semantics, rather than an opaque callee's spelling, establish
    these refusals. Unknown calls/assembly receive no purity or frame fact:
    their existing explicit provider contract must cover helper placement.
    """
    if any(
        word in attributes
        for word in (
            "returns_twice",
            "naked",
            "presplitcoroutine",
            '"presplitcoroutine"',
            "personality",
            "prologue",
            "prefix",
            "gc",
        )
    ):
        raise ValueError("function frame or calling-context contract unsupported")
    words = [token.text for token in _tokens(body)]
    if any(word in words for word in ("musttail", "callbr", "invoke")):
        raise ValueError("function frame or calling-context contract unsupported")
    sensitive = (
        "@llvm.returnaddress",
        "@llvm.frameaddress",
        "@llvm.addressofreturnaddress",
        "@llvm.localaddress",
        "@llvm.localescape",
        "@llvm.localrecover",
        "@llvm.stacksave",
        "@llvm.stackrestore",
        "@llvm.read_register",
        "@llvm.write_register",
        "@llvm.eh.",
        "@llvm.coro.",
        "@llvm.experimental.deoptimize",
        "@llvm.experimental.guard",
        "@llvm.experimental.patchpoint",
        "@llvm.experimental.stackmap",
    )
    if any(word.startswith(sensitive) for word in words):
        raise ValueError("function frame or calling-context intrinsic unsupported")
    whole = [token.text for token in tokens]
    if any(whole[index : index + 3] == ["blockaddress", "(", function] for index in range(len(whole) - 2)):
        raise ValueError("escaping original block identity unsupported")


def bind_immutable_llvm_bases(source, *, bindings=(), expected_source_sha256=None):
    """Preserve all original operations inside an explicit private helper.

    Supported public definitions have a plain void return and named opaque
    pointer parameters; alternate calling conventions/varargs/address spaces
    refuse. Constant globals, complete nonescaping readonly address uses and
    collision-free hidden symbols are proved before transactional mutation.
    Original attributes remain on both definitions; alwaysinline conflicts
    refuse. The ordinary build must verify the returned whole LLVM module.

    The wrapper preserves the original symbol/signature and supplies the same
    constant address. The extra implementation has external linkage with hidden
    visibility: an internal sole-caller function permits argument specialization
    that can erase the explicit base. A build must seal its additional borrowed
    pointer ABI and callers. There is no FP/memory motion or added alias/effect fact.
    Frame-observing intrinsic/attribute/block identities refuse. Opaque calls
    and assembly retain their original provider obligations; their spelling
    never grants purity or frame independence for helper placement.
    Empty bindings preserve source bytes without parsing or requiring a pin.
    """
    digest = hashlib.sha256(source.encode()).hexdigest()
    report = {"schema": "immutable_llvm_base_binding_v1", "source_sha256": digest, "routes": []}
    if not bindings:
        return source, report
    if expected_source_sha256 != digest:
        raise ValueError("source witness changed")
    tokens = _tokens(source)
    symbols = {t.text for t in tokens if t.text.startswith("@")}
    extents = {}
    for body in _functions(tokens):
        start = max(t.start for t in tokens if t.text == "define" and t.start < body[0].start)
        opening = max((t for t in tokens if t.text == "{" and t.start < body[0].start), key=lambda t: t.start)
        closing = next(t for t in tokens if t.text == "}" and t.start > body[-1].end)
        header = [t for t in tokens if start <= t.start < opening.start]
        names = [t.text for t in header if t.text.startswith("@")]
        if len(names) != 1 or names[0] in extents:
            raise ValueError("unsupported or ambiguous function definition")
        extents[names[0]] = (start, opening, closing, header)
    planned, selected = [], set()
    for binding in bindings:
        if not isinstance(binding, ImmutableBaseBinding):
            raise ValueError("typed immutable base binding required")
        function, global_symbol, implementation = map(
            _symbol, (binding.function, binding.global_symbol, binding.implementation)
        )
        if function in selected or implementation in symbols:
            raise ValueError("duplicate function or implementation symbol collision")
        selected.add(function)
        symbols.add(implementation)
        declarations = [
            line for line in source.splitlines() if _tokens(line) and _tokens(line)[0].text == global_symbol
        ]
        if len(declarations) != 1:
            raise ValueError("unique constant global declaration required")
        declaration = [t.text for t in _tokens(declarations[0])]
        if (
            declaration[1:2] != ["="]
            or "constant" not in declaration
            or any(x in declaration for x in ("thread_local", "addrspace", "externally_initialized", "alias", "ifunc"))
        ):
            raise ValueError("ordinary immutable constant global required")
        if function not in extents:
            raise ValueError("selected function definition unavailable")
        start, opening, closing, header = extents[function]
        h = [t.text for t in header]
        symbol_index = h.index(function)
        if h[:symbol_index] not in (["define", "void"], ["define", "internal", "void"]):
            raise ValueError("plain void helper definition required")
        if h[symbol_index + 1] != "(":
            raise ValueError("unsupported helper parameters")
        end = h.index(")", symbol_index + 2)
        parameters = h[symbol_index + 2 : end]
        args = []
        while parameters:
            if len(parameters) < 2 or parameters[0] != "ptr" or not parameters[1].startswith("%"):
                raise ValueError("plain named opaque pointer parameters required")
            args.append(parameters[1])
            parameters = parameters[2:]
            if parameters:
                if parameters[0] != ",":
                    raise ValueError("unsupported helper parameter attributes")
                parameters = parameters[1:]
        attributes = source[header[end].end : opening.start]
        resolved = [t.text for t in _tokens(attributes)]
        for attribute in tuple(resolved):
            if attribute.startswith("#"):
                groups = [
                    line
                    for line in source.splitlines()
                    if [t.text for t in _tokens(line)][:2] == ["attributes", attribute]
                ]
                if len(groups) != 1:
                    raise ValueError("attribute group unavailable")
                resolved += [t.text for t in _tokens(groups[0])]
        if "alwaysinline" in resolved:
            raise ValueError("alwaysinline conflicts with explicit private binding")
        body = source[opening.end : closing.start]
        _context_refusal(body, resolved, tokens, function)
        if any(t.text == function for t in _tokens(body)):
            raise ValueError("recursive selected helper unsupported")
        if any(t.text == "%immutable_base" for t in _tokens(body)) or "%immutable_base" in args:
            raise ValueError("private parameter SSA collision")
        count, derived = _readonly_uses(body, global_symbol)
        edits = [(t.start, t.end) for t in _tokens(body) if t.text == global_symbol]
        changed_body = body
        for left, right in reversed(edits):
            changed_body = changed_body[:left] + "%immutable_base" + changed_body[right:]
        parameters = ", ".join("ptr " + arg for arg in args)
        extra = (parameters + ", " if parameters else "") + "ptr %immutable_base"
        private = (
            "define hidden void " + implementation + "(" + extra + ") noinline" + attributes + "{" + changed_body + "}"
        )
        call = ", ".join([*("ptr " + arg for arg in args), "ptr " + global_symbol])
        wrapper = source[start : opening.start] + "{\n  call void " + implementation + "(" + call + ")\n  ret void\n}"
        planned.append((start, closing.end, wrapper + "\n" + private))
        report["routes"].append(
            {
                "function": binding.function,
                "global_symbol": binding.global_symbol,
                "implementation": binding.implementation,
                "implementation_linkage": "external_hidden",
                "extra_borrowed_pointer_ABI": "Original arguments followed by the current constant-global base",
                "constant_declaration_sha256": hashlib.sha256(declarations[0].encode()).hexdigest(),
                "original_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "private_body_sha256": hashlib.sha256(changed_body.encode()).hexdigest(),
                "direct_GEPs": count,
                "closed_derived_pointer_SSA": derived,
                "original_operations_and_order_retained": True,
                "floating_memory_alias_effect_permissions_added": False,
            }
        )
    changed = source
    for left, right, replacement in sorted(planned, reverse=True):
        changed = changed[:left] + replacement + changed[right:]
    report["rewritten_sha256"] = hashlib.sha256(changed.encode()).hexdigest()
    return changed, report
