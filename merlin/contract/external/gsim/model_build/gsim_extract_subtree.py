#!/usr/bin/env python3
"""Extract a self-contained FIRRTL circuit rooted at a chosen module.

Given a large chipyard-emitted ``TestHarness`` ``.fir`` (single circuit, thousands
of module defs, leading annotation blob), produce a new ``.fir`` whose ``circuit``
top is a *lower* module (e.g. ``ChipTop`` / ``DigitalTop``) and that contains only
the modules transitively reachable from that root via ``inst <name> of <Module>``.

Motivation: GSIM's ``splitArray`` chokes on the chipyard test-harness plumbing
(TSI<->TileLink serial bridge ``SerialRAM``/``tsi2tl``, ``SimTSI`` blackbox,
harness success/assertion nodes). Emitting the DUT one level below ``TestHarness``
sidesteps that plumbing (GSIM_ENGINE.md sec 5/6). This is a boundary-only,
simulation-semantics-preserving reslice: no module *body* is edited, we only drop
unreachable modules and rewrite the ``circuit`` header + strip the annotation blob
(annotations only drive dedup/transforms, not sim semantics).

No regex. Structured line scanning + string ops only.

Usage:
    gsim_extract_subtree.py <input.fir> <RootModule> <output.fir>
"""
import sys


def is_module_header(line):
    """A top-level module/extmodule declaration is indented exactly 2 spaces."""
    if not line.startswith("  "):
        return None
    if line.startswith("   "):  # 3+ spaces -> body line, not a header
        return None
    body = line[2:]
    for kw in ("module ", "extmodule ", "intmodule "):
        if body.startswith(kw):
            # name is the token after the keyword, before " :" / whitespace
            rest = body[len(kw):]
            # name ends at first space or ':'
            name = rest.strip().split(" ")[0].split(":")[0].strip()
            kind = kw.strip()
            return kind, name
    return None


def instance_target(line):
    """Return the module name referenced by an ``inst <x> of <Module>`` line, else None."""
    s = line.strip()
    if not s.startswith("inst "):
        return None
    marker = " of "
    idx = s.find(marker)
    if idx < 0:
        return None
    after = s[idx + len(marker):].strip()
    # target ends at whitespace or '@' (source-locator)
    tok = after.split(" ")[0]
    tok = tok.split("@")[0]
    return tok.strip()


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    in_path, root, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(in_path, "r") as f:
        lines = f.readlines()

    # 1) Index module boundaries: name -> (start_index, end_index_exclusive)
    headers = []  # (name, kind, start_index)
    for i, line in enumerate(lines):
        hdr = is_module_header(line)
        if hdr is not None:
            kind, name = hdr
            headers.append((name, kind, i))

    if not headers:
        print("ERROR: no module headers found", file=sys.stderr)
        sys.exit(1)

    bounds = {}
    kinds = {}
    for j, (name, kind, start) in enumerate(headers):
        end = headers[j + 1][2] if j + 1 < len(headers) else len(lines)
        bounds[name] = (start, end)
        kinds[name] = kind

    if root not in bounds:
        print(f"ERROR: root module {root!r} not found. Sample modules: "
              f"{[h[0] for h in headers[:5]]} ...", file=sys.stderr)
        sys.exit(1)

    # 2) Transitive closure of instance references from root
    reachable = set()
    stack = [root]
    while stack:
        m = stack.pop()
        if m in reachable:
            continue
        reachable.add(m)
        if m not in bounds:
            # referenced module not defined in this file (shouldn't happen for a
            # complete emission); record and continue
            print(f"WARN: referenced module {m!r} has no definition", file=sys.stderr)
            continue
        start, end = bounds[m]
        for k in range(start + 1, end):
            tgt = instance_target(lines[k])
            if tgt is not None and tgt not in reachable:
                stack.append(tgt)

    # 3) Emit new circuit. Preserve original module definition order for determinism.
    emitted = [name for (name, _, _) in headers if name in reachable]
    n_ext = sum(1 for name in emitted if kinds.get(name) == "extmodule")

    with open(out_path, "w") as out:
        # keep the original FIRRTL version line (lines[0])
        version_line = lines[0] if lines[0].startswith("FIRRTL version") else "FIRRTL version 3.3.0\n"
        out.write(version_line)
        out.write(f"circuit {root} :\n")
        for name in emitted:
            start, end = bounds[name]
            out.writelines(lines[start:end])

    print(f"root                = {root}")
    print(f"modules in file     = {len(headers)}")
    print(f"modules reachable   = {len(reachable)}  (emitted {len(emitted)})")
    print(f"  of which extmodule = {n_ext}")
    print(f"extmodules kept     = "
          f"{sorted(name for name in emitted if kinds.get(name)=='extmodule')[:40]}")
    print(f"wrote               = {out_path}")


if __name__ == "__main__":
    main()
