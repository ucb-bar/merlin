"""Let CSE see through provenance: drop `prov.*` metadata so identical work is computed once.

WHAT IS BROKEN. Every op a capture emits carries `prov.*` attributes -- `prov.region_id`,
`prov.fqn`, `prov.op`, `prov.family`, ... -- which exist so a measurement can be joined back to the
model graph (see :mod:`~merlin.llvmlower.op_profile`). MLIR's `cse` compares operations by their
whole attribute DICTIONARY, so two ops that compute exactly the same value from exactly the same
operands are NOT common subexpressions as far as the pass is concerned the moment their provenance
differs -- and `prov.region_id` is unique per captured region BY CONSTRUCTION. The metadata that
exists to explain the cost is therefore also creating it.

MEASURED on ``small_llama_int8_consistent`` (the prepared int8 module, `canonicalize,cse` run on it
in isolation):

    with prov      163 -> 145 linalg.generic
    without prov   163 -> 112 linalg.generic       (33 more, 23% of what survives)

and the duplicates are not marginal ops. The rotary embedding is captured as
``cat(cos(f), cos(f))`` / ``cat(sin(f), sin(f))``: 8 ``math.cos`` generics over ``8x16xf32`` where
one would do, and 8 ``math.sin``. `math.cos`/`math.sin` have no inline expansion on this target, so
each element is a libm call -- 1024 ``cosf`` and 1024 ``sinf`` calls per inference where 128 each
suffice. The K1 per-op profile prices those 16 ops at 1.53% of whole-model runtime, and the inverse
frequency table (`math.powf`, 4 identical generics) at another 0.35%. The full list of classes that
collapse is cos, sin, powf, the ``linalg.index``-derived position/arange vectors, and the causal-mask
compare/select pair.

WHY THIS IS THE GENERAL FORM. Nothing here knows what a rotary embedding is. The rule is
"provenance is metadata; it must not decide which arithmetic gets executed", and it holds for any
capture of any model on any target: a frontend that emits the same subgraph twice (a shared position
table, a mask reused by every layer, a scale recomputed per head) pays for it once the tags differ.
The dedup itself is done by upstream `cse`, which the pipeline already runs first -- this rewrite only
removes the metadata that was hiding the duplicates from it.

MEASURED ON THE EMITTED CODE (K1 binary, ``hand_v0_int8`` + ``perop_register_block``,
``promote_buffers_to_stack``; baseline vs the same build with this feature):

    forward                  37,525 -> 35,109 instructions
    cosf32 / sinf32 calls       128 -> 16 each
    @memrefCopy calls            24 -> 12   (the rotary concat's two halves become one value)
    model.o                 215,344 -> 193,096 bytes
    forward vector fraction   0.3712 -> 0.3869   (scalar compute 17,088 -> 15,491)
    vwmacc                      152 -> 152       (the contractions stay vectorized)

and on spike the model's output is BIT-IDENTICAL to the baseline (max abs diff 0.0), gating ok on
``tiers=['fp32', 'w8a8']``.

WHEN IT RUNS, AND WHY THAT IS A CORRECTNESS PROPERTY. The strip is the LAST prepared-module rewrite
(:func:`zephyr_model._strip_provenance`). It round-trips the module through the MLIR printer, and
that print is not parseable by the xDSL reader the preparation layer's own derivations use --
``kernels.shapes.observe_contractions`` returns an EMPTY list on a module it cannot parse rather than
raising. Run earlier, it took the observed contraction count 19 -> 0, emptied the per-op block table,
silently dropped ``perop_register_block``, and left every contraction to ``convert-linalg-to-loops``:
measured as ``vwmacc`` 152 -> 0 in the emitted ``forward``, with the build, the numerics and the gate
all still passing. ``merlin/tests/rvv/test_prov_cse.py`` pins the order.

WHAT IT COSTS. The stripped module can no longer be joined to the model graph, so a build with this
feature is not a build you can PROFILE per-op (``op_profile``'s table falls back to the MLIR op name).
That is why it is a feature and not the default. The count of attributes removed is printed, so a
build that stripped nothing is visible rather than silent.

Default OFF: without it the prepared module is byte-identical, so the frozen baseline is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

#: Feature name, as it appears in a package's ``compiler_features``.
FEATURE = "cse_through_provenance"

#: Attribute-name prefix this rewrite removes. Everything the capture stamps for join purposes lives
#: under it; nothing the lowering READS does.
PROV_PREFIX = "prov."

#: The rewrite, as it runs in the m2m venv (the only interpreter with the MLIR bindings). It walks the
#: whole module -- not just ``@forward`` -- because a duplicate can sit in any function, and deletes
#: every attribute whose name starts with :data:`PROV_PREFIX`. It does NOT run `cse` itself: the
#: pipeline's own leading ``canonicalize,cse`` does the dedup, so the only thing that changes here is
#: metadata and every downstream derivation (contraction shapes, the per-op block table, the register
#: group width) sees structurally the same module it saw before.
_REWRITE_SRC = '''
def strip_prov(module):
    """Delete every `prov.*` attribute in the module. Returns (ops_touched, attrs_removed)."""
    ops = attrs = 0

    def walk(op):
        nonlocal ops, attrs
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    at = inner.operation.attributes
                    names = [at[i].name for i in range(len(at))]
                    doomed = [n for n in names if n.startswith("PROV_PREFIX")]
                    if not doomed:
                        continue
                    for n in doomed:
                        del at[n]
                    ops += 1
                    attrs += len(doomed)

    walk(module.operation)
    return ops, attrs
'''


def rewrite_prepared_file(prepared: "Any", work: "Any" = None) -> Path:
    """Strip `prov.*` from a prepared module, returning the path to the stripped one.

    Runs in the m2m venv, the same way :func:`perop_blocks.tag_prepared_mlir` does — the prepared
    module is text on disk and the MLIR bindings live in that interpreter. Fails closed: a rewrite
    that produces no file, or removes nothing, is an error rather than a silent pass-through, because
    "the feature was enabled and changed nothing" is exactly the inert-lever failure this repo keeps
    re-learning.
    """
    import subprocess

    from .toolchain import m2m_python

    prepared = Path(prepared)
    work = Path(work) if work is not None else prepared.parent
    work.mkdir(parents=True, exist_ok=True)
    out = work / "model.prov_stripped.mlir"
    script = work / "_strip_prov.py"
    script.write_text(
        "import sys\n"
        "from torch_mlir import ir\n"
        + _REWRITE_SRC.replace("PROV_PREFIX", PROV_PREFIX) +
        "\nsrc, dst = sys.argv[1], sys.argv[2]\n"
        "ctx = ir.Context()\n"
        "ctx.allow_unregistered_dialects = True\n"
        "mod = ir.Module.parse(open(src).read(), ctx)\n"
        "with ctx:\n"
        "    ops, attrs = strip_prov(mod)\n"
        "open(dst, 'w').write(str(mod.operation))\n"
        "print('OK cse_through_provenance stripped', attrs, 'attributes from', ops, 'ops')\n",
        encoding="utf-8")
    proc = subprocess.run([str(m2m_python()), str(script), str(prepared), str(out)],
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(
            f"{FEATURE}: provenance strip failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    print(proc.stdout.strip())
    if " stripped 0 attributes" in proc.stdout:
        raise RuntimeError(
            f"{FEATURE} was requested but the prepared module carries no {PROV_PREFIX}* attributes, "
            f"so nothing was hiding duplicates from cse and the feature cannot do what it claims. "
            f"Refusing to report it as applied.")
    return out


def _feature():
    from .impr_features import ImprFeature
    return ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "drop the capture's `prov.*` provenance attributes from the prepared module so upstream "
            "`cse` -- which compares whole attribute dictionaries and therefore treats two identical "
            "ops with different `prov.region_id` as distinct -- can collapse work the frontend emitted "
            "more than once. MEASURED on small_llama_int8_consistent: running canonicalize+cse on the "
            "prepared module collapses 163 -> 145 linalg.generic with the tags and 163 -> 112 without, "
            "i.e. 33 more (23%). The duplicates are the rotary `cat(cos(f), cos(f))` / "
            "`cat(sin(f), sin(f))` pairs -- 8 identical math.cos and 8 identical math.sin generics, "
            "1024 libm cosf + 1024 sinf calls per inference where 128 each suffice, priced at 1.53% of "
            "K1 whole-model runtime -- plus the inverse-frequency math.powf table (0.35%), the "
            "linalg.index position vectors and the causal-mask compare/select. Structure-keyed: names "
            "no op, model or target; the dedup is done by upstream cse, this only removes the metadata "
            "hiding it. ON THE EMITTED K1 CODE: forward 37,525 -> 35,109 instructions, cosf32/sinf32 "
            "call sites 128 -> 16 each, @memrefCopy 24 -> 12, model.o 215,344 -> 193,096 bytes, "
            "forward vector fraction 0.3712 -> 0.3869, vwmacc unchanged at 152, and the spike output "
            "BIT-IDENTICAL (gate ok on tiers=['fp32','w8a8']). Runs LAST in the preparation layer: its "
            "MLIR-printer round-trip is unparseable by the xDSL reader the block-table derivation uses, "
            "which degrades to 'no contractions observed' and silently drops the register block. "
            "COST: the stripped module can no longer be joined to the model graph, so a "
            "build with this feature cannot be profiled per-op. Default-off; baseline byte-identical."
        ),
    )


def ensure_registered() -> str:
    """Register the feature if it is not already. Idempotent, so importing from several entry points
    is safe. Returns the feature name."""
    from .impr_features import known, register
    if FEATURE not in known():
        register(_feature())
    return FEATURE
