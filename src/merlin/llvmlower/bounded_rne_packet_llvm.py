"""CPU packet emission for pure scalar helpers returning proved RNE lanes.

No symbol naming convention selects a function. The typed return aggregate must
contain fully proved scalar bounded-RNE results within the explicit lane budget,
and the function
must contain only straight-line pure scalar arithmetic. The original arithmetic
is retained as dead code until ordinary LLVM DCE; only return operands change.
The explicit CPU policy assumes FP exception flags are not observable, matching
the scalar bounded-RNE legalization contract; strict/constrained FP refuses.
"""

from __future__ import annotations

import hashlib

from .late_quant_rne import _functions, _identity, _instruction, _match, _tokens


def rewrite_packet_helpers(source: str, *, host_isa: str | None = None, max_lanes: int = 4):
    """Legalize pure return packets under an explicit CPU/lane policy.

    The default recognizes the original two-through-four lane domain. Larger
    packets require an explicit budget, bounded at eight floating temporaries
    under the supported CPU policy. This is a code-generation option, not a
    profitability or alias proof; compiled register pressure must be measured.
    """
    if type(max_lanes) is not int or not 2 <= max_lanes <= 8:
        raise ValueError("explicit packet lane budget must be an integer in [2,8]")
    report = {
        "schema": "bounded_rne_packet_llvm_v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "routes": [],
    }
    if host_isa is None:
        return source, report
    if host_isa != "rv64gc":
        raise ValueError("explicit supported host CPU ISA required")
    tokens = _tokens(source)
    if any(t.text == "strictfp" or t.text.startswith("@llvm.experimental.constrained.") for t in tokens):
        report["refusal"] = "strict or constrained FP"
        return source, report
    functions = _functions(tokens)
    used = {_identity(t.text) for t in tokens if t.text.startswith("%")}
    edits = []
    serial = 0
    for body in functions:
        if not body:
            continue
        # Parse statement boundaries structurally; SSA identifiers never contain
        # the bare opcode token, so operands cannot masquerade as statements.
        starts = [
            i
            for i in range(len(body))
            if body[i].text == "ret" or (body[i].text.startswith("%") and i + 1 < len(body) and body[i + 1].text == "=")
        ]
        if not starts or any(
            t.text
            in (
                "br",
                "switch",
                "phi",
                "load",
                "store",
                "alloca",
                "invoke",
                "callbr",
                "fence",
                "atomicrmw",
                "cmpxchg",
                "unreachable",
                "resume",
                "indirectbr",
                "cleanupret",
                "catchret",
                "catchswitch",
            )
            for t in body
        ):
            continue
        if any(
            t.text == "call" and (i < 2 or body[i - 1].text != "=" or not body[i - 2].text.startswith("%"))
            for i, t in enumerate(body)
        ):
            continue
        definitions = {}
        aggregate = {}
        returned = None
        legal = True
        for pos, i in enumerate(starts):
            end = starts[pos + 1] if pos + 1 < len(starts) else len(body)
            stmt = body[i:end]
            words = [t.text for t in stmt]
            if words[0] == "ret":
                if pos != len(starts) - 1 or len(words) < 5 or words[1] != "{" or "}" not in words:
                    legal = False
                    break
                close = words.index("}")
                types = words[2:close]
                if len(words) != close + 2 or not words[-1].startswith("%"):
                    legal = False
                    break
                returned = (types, words[-1])
                continue
            if len(words) > 3 and words[2] == "insertvalue":
                if words[3] != "{" or "}" not in words:
                    legal = False
                    break
                close = words.index("}")
                types = words[4:close]
                rest = words[close + 1 :]
                if len(rest) != 6 or rest[1] != "," or rest[4] != "," or not rest[5].isdigit():
                    legal = False
                    break
                aggregate[_identity(words[0])] = {
                    "types": types,
                    "previous": rest[0],
                    "dtype": rest[2],
                    "value": rest[3],
                    "lane": int(rest[5]),
                    "start": stmt[0].start,
                    "value_token": stmt[close + 4],
                }
                continue
            op = _instruction(body, i)
            if op is None:
                # Multiplication is allowed only in its exact, flag-free form.
                if len(words) == 7 and words[2:4] == ["fmul", "float"] and words[5] == ",":
                    continue
                legal = False
                break
            if op.end != stmt[-1].end or (
                op.opcode == "call" and op.callee not in ("llvm.minimum.f32", "llvm.maximum.f32")
            ):
                legal = False
                break
            definitions[_identity(op.result)] = op
        if not legal or returned is None:
            continue
        typelist, head = returned
        if len(typelist) % 2 != 1 or any(x != "," for x in typelist[1::2]):
            continue
        dtypes = typelist[::2]
        width = len(dtypes)
        if not 2 <= width <= max_lanes or len(set(dtypes)) != 1:
            continue
        dtype = dtypes[0]
        chain = []
        seen = set()
        for _ in range(width):
            agg = aggregate.get(_identity(head))
            if agg is None or agg["types"] != typelist or agg["dtype"] != dtype or agg["lane"] in seen:
                break
            chain.append(agg)
            seen.add(agg["lane"])
            head = agg["previous"]
        if len(chain) != width or head not in ("poison", "undef") or seen != set(range(width)):
            continue
        chain.sort(key=lambda a: a["lane"])
        proofs = []
        for agg in chain:
            op = definitions.get(_identity(agg["value"]))
            proof = _match(op, definitions) if op else None
            if proof is None or proof["integer_dtype"] != dtype:
                break
            proofs.append(proof)
        if len(proofs) != width or any(p["bounds"] != proofs[0]["bounds"] for p in proofs):
            continue
        insertion = min(a["start"] for a in chain)
        # Every argument to the new packet must already exist before the first
        # returned-aggregate construction. No loads are moved by this rewrite.
        locations = {_identity(body[i].text): body[i].start for i in starts if body[i].text.startswith("%")}
        if any(locations.get(_identity(p["raw_input"]), -1) >= insertion for p in proofs):
            continue

        def local_names(number):
            prefix = f"merlin.packet.{number}"
            return [
                prefix + ".asm",
                *[f"{prefix}.wide{i}" for i in range(width)],
                *[f"{prefix}.lane{i}" for i in range(width)],
            ]

        while any(name in used for name in local_names(serial)):
            serial += 1
        prefix = f"merlin.packet.{serial}"
        used.update(local_names(serial))
        serial += 1
        integer_struct = "{ " + ", ".join(["i32"] * width) + " }"
        lo, hi = proofs[0]["bounds"]
        asm = [f"fmax.s ft{i}, ${width + i}, ${2 * width}" for i in range(width)]
        asm += [f"fmin.s ft{i}, ft{i}, ${2 * width + 1}" for i in range(width)]
        asm += [f"fcvt.w.s ${i}, ft{i}, rne" for i in range(width)]
        constraints = ",".join(["=r"] * width + ["f"] * (width + 2) + [f"~{{ft{i}}}" for i in range(width)])
        operands = ", ".join([*(f"float {p['raw_input']}" for p in proofs), f"float {lo:.6e}", f"float {hi:.6e}"])
        lines = [f'%{prefix}.asm = call {integer_struct} asm "' + r"\0A".join(asm) + f'", "{constraints}"({operands})']
        for lane, agg in enumerate(chain):
            lines += [
                f"%{prefix}.wide{lane} = extractvalue {integer_struct} %{prefix}.asm, {lane}",
                f"%{prefix}.lane{lane} = trunc i32 %{prefix}.wide{lane} to {dtype}",
            ]
            tok = agg["value_token"]
            assert tok.text == agg["value"]
            edits.append((tok.start, tok.end, f"%{prefix}.lane{lane}"))
        indent = source[source.rfind("\n", 0, insertion) + 1 : insertion]
        if indent.strip():
            raise ValueError("packet helper instruction must start a line")
        edits.append((insertion, insertion, ("\n" + indent).join(lines) + "\n" + indent))
        report["routes"].append(
            {
                "lanes": width,
                "integer_dtype": dtype,
                "bounds": proofs[0]["bounds"],
                "raw_inputs": [p["raw_input"] for p in proofs],
                "selection": "pure straight-line return aggregate and complete typed RNE proofs",
            }
        )
    for a, b, replacement in sorted(edits, reverse=True):
        source = source[:a] + replacement + source[b:]
    report["rewritten_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return source, report
