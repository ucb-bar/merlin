#!/usr/bin/env python3
"""Repair GSIM's malformed top-level accessor stubs for AGGREGATE (vector) IO ports.

Why this exists
---------------
GSIM (OpenXiangShan/gsim) compiles chirrtl to C++. For every top-level input it
emits ``void set_<port>(<T> val)`` and for every top-level output
``<T> get_<port>()``. That is correct for the RISC-V cores GSIM ships (Rocket,
BOOM, XiangShan) whose top IO is scalar (clock/reset/a few difftest bits).

Accelerator-only modules (the Atlas MXU ``SystolicArray``, a gemmini ``Mesh``,
etc.) expose WIDE VECTOR ports at the top, e.g. ``io.computeReq.bits.act`` is a
32-wide vector. GSIM declares the member correctly as an array
(``uint8_t io$$..$$act[32];``) but still emits a *scalar* accessor whose body does
``if (member != val) { member = val; ... }`` -- which does not compile
(``comparison between pointer and integer`` / ``array type ... is not assignable``).

This post-processor rewrites ONLY those broken accessor stubs. It is
semantics-preserving at the model boundary:
  * setter: element-wise ``memcpy(member, val, sizeof(member))`` in place of the
    scalar assignment; the SAME ``activeFlags[...] |= ...`` node-activation lines
    are preserved verbatim (now run unconditionally, which is conservatively
    correct -- it marks the dependent nodes for re-evaluation).
  * getter: gains an ``int i`` index and returns ``member[i]``.
The internal datapath / register model in the .cpp is NOT touched -- only the
hand-off accessor functions that GSIM got wrong.

No regex (repo standing rule): pure structured string ops.

Usage:  gsim_fix_vector_io.py <ModuleName> <gsim_out_dir>
        (edits <dir>/<Module>.h and <dir>/<Module>0.cpp in place)
"""
import sys
import os


def find_class_name(header_text):
    for line in header_text.splitlines():
        s = line.strip()
        if s.startswith("class S") and s.endswith("{"):
            # "class SSystolicArray {"
            return s[len("class "):-1].strip()
    raise RuntimeError("could not find 'class S...' in header")


def array_members(header_text):
    """Return dict name -> (elem_type, ndims) for public array data members."""
    out = {}
    for line in header_text.splitlines():
        s = line.strip()
        if "//" in s:            # drop trailing "// width = ..." comment
            s = s[:s.index("//")].strip()
        if not s.endswith(";"):
            continue
        if "[" not in s or "(" in s:  # skip function decls
            continue
        if "$NEXT" in s:
            continue
        toks = s[:-1].split(None, 1)  # split off type
        if len(toks) != 2:
            continue
        etype, rest = toks
        if not (etype.startswith("uint") or etype.startswith("int")):
            continue
        # rest looks like  name[32]  or  name[32][32]
        name = rest[:rest.index("[")]
        ndims = rest.count("[")
        out[name] = (etype, ndims)
    return out


def accessor_members(header_text):
    """Names that have set_/get_ accessors declared in the header."""
    sets, gets = set(), set()
    for line in header_text.splitlines():
        s = line.strip()
        if s.startswith("void set_") and s.endswith(");"):
            name = s[len("void set_"):s.index("(")]
            sets.add(name)
        elif " get_" in s and s.endswith(");"):
            # e.g. "uint16_t get_io$$..();"
            after = s[s.index(" get_") + len(" get_"):]
            name = after[:after.index("(")]
            gets.add(name)
    return sets, gets


def repair(module, out_dir):
    hpath = os.path.join(out_dir, module + ".h")
    cpath = os.path.join(out_dir, module + "0.cpp")
    with open(hpath) as f:
        htext = f.read()
    with open(cpath) as f:
        ctext = f.read()

    cls = find_class_name(htext)
    members = array_members(htext)
    sets, gets = accessor_members(htext)

    broken_setters = sorted(n for n in sets if n in members)
    broken_getters = sorted(n for n in gets if n in members)

    changed = []

    # --- setters ---
    for name in broken_setters:
        etype, _ = members[name]
        # header decl: "void set_<name>(<etype> val);"
        old_decl = "void set_%s(%s val);" % (name, etype)
        new_decl = "void set_%s(const %s* val);" % (name, etype)
        if old_decl in htext:
            htext = htext.replace(old_decl, new_decl)
        # cpp def head: signature + scalar guard + scalar assign.
        old_head = ("void %s::set_%s(%s val) {\n"
                    "  if (%s != val) { \n"
                    "    %s = val;") % (cls, name, etype, name, name)
        # brace-preserving replacement: keep one '{' where the 'if (...) {' was.
        new_head = ("void %s::set_%s(const %s* val) {\n"
                    "  {  // [gsim_fix_vector_io] vector port: copy elements, mark active\n"
                    "    memcpy(%s, val, sizeof(%s));") % (cls, name, etype, name, name)
        if old_head in ctext:
            ctext = ctext.replace(old_head, new_head)
            changed.append("set_" + name)
        else:
            print("  WARN: setter body pattern not found for", name, file=sys.stderr)

    # --- getters (1-D only; return member[i]) ---
    for name in broken_getters:
        etype, ndims = members[name]
        if ndims != 1:
            print("  SKIP getter (ndims=%d) %s" % (ndims, name), file=sys.stderr)
            continue
        old_decl = "%s get_%s();" % (etype, name)
        new_decl = "%s get_%s(int i);" % (etype, name)
        if old_decl in htext:
            htext = htext.replace(old_decl, new_decl)
        old_def = ("%s %s::get_%s() {\n"
                   "  return %s;\n}") % (etype, cls, name, name)
        new_def = ("%s %s::get_%s(int i) {\n"
                   "  return %s[i];\n}") % (etype, cls, name, name)
        if old_def in ctext:
            ctext = ctext.replace(old_def, new_def)
            changed.append("get_" + name)
        else:
            print("  WARN: getter body pattern not found for", name, file=sys.stderr)

    with open(hpath, "w") as f:
        f.write(htext)
    with open(cpath, "w") as f:
        f.write(ctext)

    print("repaired %d vector-IO accessor(s) in %s:" % (len(changed), module))
    for c in changed:
        print("   ", c)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    repair(sys.argv[1], sys.argv[2])
