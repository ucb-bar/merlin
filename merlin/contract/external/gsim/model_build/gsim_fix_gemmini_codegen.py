#!/usr/bin/env python3
"""Work around two GSIM *codegen* defects in the Gemmini-accelerator emission.

These are distinct from the vector-IO accessor issue (see gsim_fix_vector_io.py).
They are genuine bugs in GSIM's C++ backend that leave the emitted .cpp
referencing identifiers that were never declared, so it will not compile:

  (1) ORPHAN OUTPUT-ALIAS SETTERS.  GSIM emits set_io$$csrs$$ren / $$wen /
      $$wdata / $$value whose backing members are absent -- the real ports are
      the arrays io$$ptw$$customCSRs$$csrs$$*[4].  These are duplicate accessor
      stubs for an output alias.  Fix: neutralise the four bodies to a no-op
      (you never *drive* an accelerator output anyway).  Purely at the IO
      boundary; the datapath is untouched.

  (2) CROSS-PARTITION TEMP LEAK.  The reset-init block references
      im2col$_modulo_block_done_T / _modulo_block_save_T, which GSIM only
      declared as *locals inside a different partition function*.  Fix: declare
      them locally (=0) at the point of use.  Value judgement, documented:
      these back the im2col (image-to-column) unit and are reset to 0 -- which
      is the natural reset value and is independent of the load/store/ex
      controller FSM state that the M2 waterfall observes.

No regex (repo rule): structured string ops only.

Usage:  gsim_fix_gemmini_codegen.py <ModuleName> <gsim_out_dir>
"""
import os
import sys


def declared_members(htext):
    names = set()
    for line in htext.splitlines():
        s = line.strip()
        if "//" in s:
            s = s[:s.index("//")].strip()
        if not s.endswith(";") or "(" in s:
            continue
        toks = s[:-1].split(None, 1)
        if len(toks) != 2:
            continue
        etype, rest = toks
        if not (etype.startswith("uint") or etype.startswith("int")):
            continue
        name = rest.split("[")[0].strip()
        names.add(name)
    return names


def setter_decls(htext):
    """name -> param_type for every 'void set_<name>(<type> val);'."""
    out = {}
    for line in htext.splitlines():
        s = line.strip()
        if s.startswith("void set_") and s.endswith(" val);"):
            name = s[len("void set_"):s.index("(")]
            ptype = s[s.index("(") + 1:s.index(" val);")]
            out[name] = ptype
    return out


def fix(module, out_dir):
    cls = "S" + module
    hpath = os.path.join(out_dir, module + ".h")
    cpath = os.path.join(out_dir, module + "0.cpp")
    htext = open(hpath).read()
    ctext = open(cpath).read()

    members = declared_members(htext)
    setters = setter_decls(htext)

    # (1) orphan setters whose backing member is not declared anywhere.
    orphans = sorted(n for n in setters if n not in members)
    n_orphan = 0
    for name in orphans:
        ptype = setters[name]
        old = ("void %s::set_%s(%s val) {\n"
               "  if (%s != val) { \n"
               "    %s = val;\n"
               "  }\n"
               "}") % (cls, name, ptype, name, name)
        new = ("void %s::set_%s(%s val) {\n"
               "  (void)val;  // [gsim_fix_gemmini] orphan output-alias accessor: backing member absent; no-op\n"
               "}") % (cls, name, ptype)
        if old in ctext:
            ctext = ctext.replace(old, new)
            n_orphan += 1

    # (2) cross-partition temp leak in the reset-init block.  Prefix-agnostic:
    #     works for both the accel-only emission (im2col$_modulo_block_*_T) and
    #     the full-SoC emission (system$...$gemmini$im2col$_modulo_block_*_T).
    #     Find reset-init assignment lines `LHS = RHS;` whose RHS is a single
    #     identifier ending in a leaked temp suffix and is NOT a declared member,
    #     then declare those temps locally (=0) right before their first use.
    suffixes = ("_modulo_block_done_T", "_modulo_block_save_T")
    clines = ctext.splitlines(keepends=True)
    leaked = []          # preserve first-seen order
    first_use_idx = None
    for idx, line in enumerate(clines):
        s = line.strip()
        if not s.endswith(";") or " = " not in s:
            continue
        rhs = s[:-1].split(" = ", 1)[1].strip()
        # RHS must be a bare identifier (no operators/spaces/parens)
        if any(c in rhs for c in " ()*+-/&|^<>!%[]") :
            continue
        if not rhs.endswith(suffixes):
            continue
        if rhs in members:
            continue
        if rhs not in leaked:
            leaked.append(rhs)
        if first_use_idx is None:
            first_use_idx = idx
    n_temp = 0
    if leaked and first_use_idx is not None:
        indent = clines[first_use_idx][:len(clines[first_use_idx]) - len(clines[first_use_idx].lstrip())]
        decl = (indent + "uint16_t "
                + ", ".join("%s = 0" % t for t in leaked)
                + ";  // [gsim_fix_gemmini] cross-partition temps: reset-init to 0\n")
        clines.insert(first_use_idx, decl)
        ctext = "".join(clines)
        n_temp = len(leaked)

    open(hpath, "w").write(htext)
    open(cpath, "w").write(ctext)
    print("gemmini codegen fix: neutralised %d orphan setter(s) %s; temp-leak shims: %d"  # target-ok: a per-target model-build helper; this target IS its subject
          % (n_orphan, orphans, n_temp))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    fix(sys.argv[1], sys.argv[2])
