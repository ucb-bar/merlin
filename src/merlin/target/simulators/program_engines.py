"""Declared program-output values: byte widths, decoding and output specifications.

Independent of model processes and simulator execution. program_oracle reexports the
same functions/classes; already-loaded legacy overrides remain visible to helper calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

from merlin.common.paths import ext_path  # noqa: F401 — legacy/context resolver export

# ---- program_values ----------------------------------------------------------------------------


class OracleUnavailable(RuntimeError):
    """Raised when the program oracle cannot run (model venv / cosim / arc artifacts absent)."""


def _context():
    # Preserve legacy monkeypatch dispatch without importing the execution layer.
    return sys.modules.get("merlin.development.targetgen.oracles.program_oracle", sys.modules[__name__])


#: IEEE binary formats numpy decodes natively, keyed by the ``(exp_bits, mant_bits)`` split the shared
#: format registry declares for them. A DERIVED lookup, not a dtype-name table: a registry entry whose
#: split matches one of these IS that machine float however it is spelled (``f32``/``fp32``/
#: ``torch.float32``), and one that matches none is widened into the machine float sharing its exponent
#: field (below) or refused — never silently reinterpreted as raw bytes.
_IEEE_NATIVE: dict[tuple[int, int], str] = {(5, 10): "<f2", (8, 23): "<f4", (11, 52): "<f8"}


def _dtype_key(dtype: object) -> str:
    """A declared dtype reduced to its registry spelling (``torch.`` prefix dropped)."""
    key = str(dtype)
    return key[len("torch.") :] if key.startswith("torch.") else key


def _element_bytes(dtype: object) -> int:
    """Bytes ONE element of a declared tensor dtype occupies in DRAM.

    DERIVED from the shared format registry (:func:`merlin.common.target_data.storage_bits`), which is
    also what sizes the operand preloads — never a substring test on the spelling. This function and
    :func:`_decode_output` are the readback's two halves and they used to carry SEPARATE ad-hoc rules:
    the window was sized ``2 if "16" in dtype else (4 if "32" in dtype else 1)`` while the decode named
    only ``bf16`` and ``i32`` and read everything else as ``int8``. For an ``f32`` result that read 4
    bytes per element and decoded 1, so the reshape onto the declared shape raised a bare numpy
    ``ValueError`` ("cannot reshape array of size 1024 into shape (16,16)") that the ladder classified as
    a tool crash. One derived width means the two cannot disagree again.

    A spelling the registry does not know RAISES rather than defaulting to one byte: an unknown width is
    UNKNOWN, and substituting a default is what turned a decodable result into an uninterpretable crash.
    """
    from merlin.common import target_data as qf

    key = _context()._dtype_key(dtype)
    try:
        bits = qf.storage_bits(key)
    except KeyError as exc:
        raise _context().OracleUnavailable(
            f"cannot size output dtype {dtype!r}: it is neither a format registered in "
            f"src/merlin/resources/schemas/quant_formats.registry.yaml nor a plain machine width, so how many bytes "
            f"one element occupies is UNKNOWN — register the format rather than assuming a width"
        ) from exc
    if bits % 8:
        raise _context().OracleUnavailable(
            f"output dtype {dtype!r} stores {bits} bit(s) per element, which is not a whole number of "
            f"bytes; this readback addresses whole bytes in DRAM and cannot say where one element ends"
        )
    return bits // 8


def _decode_elements(raw: bytes, dtype: object, width: int):
    """The flat element array ``raw`` encodes, decoded as the DECLARED format.

    Every branch is chosen from what the format registry says about the format (its kind, its
    exponent/mantissa split, its signedness, whether it is sub-byte packed), so a format plugs in as a
    registry entry. A format whose bytes alone do NOT determine a value — a block-scaled element, whose
    scale plane this output window does not carry, or a packed sub-byte element — is refused, because
    reading its bytes as small integers produces numbers that compare against a golden and mean nothing.
    """
    import numpy as np

    from merlin.common import target_data as qf

    key = _context()._dtype_key(dtype)
    if qf.has(key):
        fmt = qf.get(key)
        if fmt.is_float:
            if fmt.kind == "float_ieee":
                native = _context()._IEEE_NATIVE.get((fmt.exp_bits, fmt.mant_bits))
                if native:
                    return np.frombuffer(raw, dtype=native)
                # A truncated IEEE float (bf16 is f32's exponent field with a short mantissa): widen it
                # into the machine float that shares its exponent field by shifting the stored bits up
                # by the MANTISSA DIFFERENCE. The shift is derived from the two splits; it used to be a
                # literal 16 that only bf16 could ever be right for.
                wider = next(
                    (
                        (mb, nat)
                        for (eb, mb), nat in sorted(_context()._IEEE_NATIVE.items())
                        if eb == fmt.exp_bits and mb > fmt.mant_bits
                    ),
                    None,
                )
                if wider is None:
                    raise _context().OracleUnavailable(
                        f"output dtype {dtype!r} (E{fmt.exp_bits}M{fmt.mant_bits}) is not a machine "
                        f"float and shares an exponent field with none, so its stored bits cannot be "
                        f"widened to a value without a decoder for that format"
                    )
                mant_bits, native = wider
                stored = np.frombuffer(raw, dtype=f"<u{width}")
                held = stored.astype(f"<u{np.dtype(native).itemsize}")
                return (held << (mant_bits - fmt.mant_bits)).view(native)
            if fmt.kind == "fp_ocp" and not fmt.pack_bits:
                from merlin.common.fp8_codec import ocp_decode

                codes = np.frombuffer(raw, dtype=f"<u{width}")
                return np.asarray(
                    [ocp_decode(int(c), fmt.exp_bits, fmt.mant_bits, signed=bool(fmt.signed)) for c in codes],
                    dtype=np.float32,
                )
            raise _context().OracleUnavailable(
                f"output dtype {dtype!r} is a {fmt.kind} format whose element value needs a scale "
                f"plane this output window does not carry, so its DRAM bytes alone do not determine a "
                f"value"
            )
        if fmt.pack_bits and fmt.pack_bits != fmt.element_bits:
            raise _context().OracleUnavailable(
                f"output dtype {dtype!r} packs {fmt.element_bits}-bit elements into {fmt.pack_bits} "
                f"bit(s); this readback cannot address a packed element"
            )
        # An INTEGER element under a PER-BLOCK scale is the float branch's case with the kinds swapped,
        # and it needs the same refusal: the block scales are a SEPARATE plane that this single output
        # window provably does not carry, so the codes in it do not determine values. Returning them as
        # small integers is the quiet half of the bug this function exists to fix -- the reshape stays
        # legal, the numbers reach the compare, and every one of them is wrong by its block's scale.
        # Only PER-BLOCK is refused: a per-channel/per-tensor scale is a property of the tensor rather
        # than a plane inside it, and its codes ARE what the corresponding golden compares (19 shipped
        # capsules declare an i8 output on exactly those terms).
        if fmt.granularity == "per_block":
            raise _context().OracleUnavailable(
                f"output dtype {dtype!r} is a {fmt.kind} format whose elements carry a per-block "
                f"{getattr(fmt.scale, 'kind', None)!r} scale (block={getattr(fmt.scale, 'block', None)}), "
                f"and that scale plane is not inside this output window — so these bytes do not "
                f"determine a value. Read the scales alongside the codes, or declare an output dtype "
                f"whose bytes are self-describing"
            )
        return np.frombuffer(raw, dtype=f"<{'i' if fmt.signed else 'u'}{width}")
    # Not a registered format: a plain machine scalar (``i32``/``u8``/``i64``). Its family comes from the
    # spelling's own prefix, parsed structurally, and its width from _element_bytes above.
    family = next(
        (
            f
            for p, f in (("float", "f"), ("uint", "u"), ("int", "i"), ("f", "f"), ("u", "u"), ("i", "i"))
            if key.startswith(p) and key[len(p) :].isdigit()
        ),
        None,
    )
    if family is None:
        raise _context().OracleUnavailable(
            f"output dtype {dtype!r} names no element type this readback can decode; its value is UNKNOWN"
        )
    if family == "f":
        raise _context().OracleUnavailable(
            f"output dtype {dtype!r} names a {width * 8}-bit machine float with no registered "
            f"exponent/mantissa split, so its stored bits cannot be decoded to a value"
        )
    return np.frombuffer(raw, dtype=f"<{family}{width}")


def _decode_output(raw: bytes, shape: list[int], dtype: str, physical: dict | None, *, name: str = "output"):
    """The declared output tensor the readback bytes hold, decoded at its DECLARED element width.

    ``name`` is the tensor's own name and is carried only so a refusal can say WHICH output it is about:
    a readback that cannot be reconciled with its declaration is reported as a named, sized refusal
    rather than as ``arr.reshape(shape)``'s bare numpy ``ValueError``. That message ("cannot reshape
    array of size 1024 into shape (16,16)") reached the verdict with its integers redacted — "cannot
    reshape array of size # into shape (#,#)" — and named neither the tensor nor the dtype, so nothing
    in the artifact said what had gone wrong.
    """
    import numpy as np

    dims = [int(d) for d in shape]
    n = 1
    for d in dims:
        n *= d
    width = _context()._element_bytes(dtype)
    if len(raw) != n * width:
        raise _context().OracleUnavailable(
            f"output {name!r}: the oracle read back {len(raw)} byte(s), but the tensor is declared "
            f"{dims} of {dtype} — {n} element(s) x {width} B = {n * width} B. That is "
            f"{len(raw) / width:g} element(s) at the declared width, so the readback window and the "
            f"declared tensor disagree and no reshape of these bytes onto {dims} is the declared "
            f"result. This is a harness fault, not a numeric one"
        )
    arr = _context()._decode_elements(raw, dtype, width).reshape(dims)
    # physical->logical layout, DECLARED by the emitting backend (not a constant here). An MXU that
    # writes an [2R, C] tensor as two stacked R-row banks declares ``{"unstack_row_halves": 2}`` to un-stack it.
    halves = (physical or {}).get("unstack_row_halves")
    if halves and dims[0] % halves == 0:
        h = dims[0] // halves
        arr = np.concatenate([arr[i * h : (i + 1) * h] for i in range(halves)], axis=1)
    return arr


def _resolve_out_specs(target: str, cb: dict | None, bundle: dict) -> dict[str, dict]:
    """EVERY output tensor spec, ``{name: {base, shape, dtype, physical}}`` — from the cb
    (generation-declared) or the program's own golden, in declaration order.

    A command buffer may declare more than one output: an interface module with two commit ops (one
    resident weight, two activations) produces two result tensors, and capturing only the first reports
    the second as never written no matter what the kernel did. Each output DRAM base is the harness-owned
    address (stamped by ``capsule_dram.inject_bases``, the same map the agent's kernel was told to store
    to). A missing base is an actionable grading error (the layout could not be applied), NOT a bare
    ``KeyError``. Target-agnostic: the names and layouts are whatever the emitting backend declared.

    THE ELEMENT WIDTH IS RESOLVED THE WAY THE REST OF THE HARNESS RESOLVES IT. It sizes both the DRAM
    read window (:func:`_out_nbytes`) and the decode (:func:`_decode_output`), and a command that names a
    destination may re-declare the container its result lands in (``attributes.output_dtype`` — a
    movement IS a container widening, and a commit's readout dtype is not the accumulator's). This read
    ``tensors[name]["dtype"]`` alone and defaulted it to ``"bf16"``, while every other reader of the same
    buffer goes through :func:`merlin.runtime.engine.command_buffer.declared_output_dtypes`. One buffer with two
    resolution rules is a disagreement waiting to happen, and it is not a wrong number when it happens:
    a mis-sized window fails inside numpy and reaches the verdict as ``tool_crash``. The default is gone
    for the same reason the width is derived — an undeclared element type is UNKNOWN, and ``bf16`` is a
    2-byte guess that is wrong by 2x for the f32 results this corpus is full of."""
    from merlin.runtime.engine.command_buffer import declared_output_dtypes

    specs: dict[str, dict] = {}
    declared = [(n, t) for n, t in ((cb or {}).get("tensors") or {}).items() if t.get("role") == "output"]
    resolved = declared_output_dtypes(cb or {})
    for name, t in declared:
        # An output the submission declared but gave no address to cannot be captured. That is not a
        # tool error: it is exactly the "you never wrote this output" verdict the numeric compare
        # reports, and reporting it there names the tensor. So it is skipped, not raised on — unless NO
        # declared output has an address, which really is a layout failure the caller must hear about.
        if t.get("base") is None:
            continue
        dtype = resolved.get(name) or t.get("dtype")
        if not dtype:
            raise _context().OracleUnavailable(
                f"{target}: the command buffer declares output tensor {name!r} (shape "
                f"{list(t.get('shape') or [])}) but states no element type for it — neither the "
                f"tensor's own 'dtype' nor an 'output_dtype' attribute on the command that writes it. "
                f"How many bytes one element occupies is therefore UNKNOWN, so neither the DRAM read "
                f"window nor the decode can be sized; declare the dtype rather than have one assumed"
            )
        specs[name] = {
            "base": int(t["base"]),
            "shape": list(t["shape"]),
            "dtype": str(dtype),
            "physical": t.get("physical"),
        }
    if declared and not specs:
        raise _context().OracleUnavailable(
            f"{target}: no declared output tensor has a DRAM base — the harness DRAM layout was not "
            f"applied (capsule_dram.inject_bases); cannot read the result back"
        )
    if not specs and bundle.get("output"):
        o = bundle["output"]
        specs[_context()._output_name(cb)] = {
            "base": int(o["base"]),
            "shape": list(o["shape"]),
            "dtype": o["dtype"],
            "physical": o.get("physical"),
        }
    if not specs:
        raise _context().OracleUnavailable(f"{target}: no output tensor declared in cb or program golden")
    return specs


def _resolve_out_spec(target: str, cb: dict | None, bundle: dict) -> dict:
    """The FIRST output tensor spec ``{base, shape, dtype, physical}``. A runner that captures one memory
    region reads this; the cosim path captures every declared output via :func:`_resolve_out_specs`."""
    return next(iter(_context()._resolve_out_specs(target, cb, bundle).values()))


def _out_nbytes(out_spec: dict) -> int:
    """Bytes to read back for a declared output tensor — its element count times the width
    :func:`_element_bytes` DERIVES, so the window this opens is exactly what :func:`_decode_output`
    consumes. The two used to carry independent ad-hoc width rules and disagreed 4:1 on an f32 result."""
    n = 1
    for d in out_spec["shape"]:
        n *= int(d)
    return n * _context()._element_bytes(out_spec["dtype"])


def _output_name(cb: dict | None) -> str:
    return next((n for n, t in (cb.get("tensors") or {}).items() if t.get("role") == "output"), "Y0") if cb else "Y0"


# ---- program_engine_policy ---------------------------------------------------------------------
# Read-only program-engine discovery and lineage checks, independent of evaluation.
#
# The execution oracle reexports these exact objects. Already-loaded legacy overrides
# remain effective without importing the oracle to inspect an engine's availability.


def _context():
    return sys.modules.get("merlin.development.targetgen.oracles.program_oracle", sys.modules[__name__])


def _model_venv_python(model_ext: str) -> Path:
    """Resolve the model project's own interpreter through the external-project registry."""
    root = _context().ext_path(model_ext)
    py = root / ".venv" / "bin" / "python"
    if not py.is_file():
        raise _context().OracleUnavailable(f"model venv python absent: {py} (run `uv sync` in {root})")
    return py


_DEFAULT_MODEL_PYTHON = _model_venv_python


def selected_model_venv_python(model_ext: str) -> Path:
    """Honor an already-loaded execution module's resolver override without importing it."""
    legacy = sys.modules.get("merlin.development.targetgen.oracles.program_oracle")
    resolver = getattr(legacy, "_model_venv_python", _DEFAULT_MODEL_PYTHON)
    if resolver is not _DEFAULT_MODEL_PYTHON:
        return resolver(model_ext)
    return _model_venv_python(model_ext)


#: Elaborated-RTL engines, each as (ext_path key suffix, conventional wrapper filename). Every engine
#: exposes the SAME ``run_program``; merlin never names a binary or a target. Adding an engine is one row
#: here plus a wrapper beside its build — the priority among them lives in ``rtl_engine_policy``.
#: The engine whose home layout -- and therefore whose build receipt / adoption record --
#: :mod:`gsim_emulator` owns. Named here because the lineage question is only answerable for the homes
#: that module lays out; every other engine is registered and judged by whoever built it.
_LINEAGE_ENGINE = "gsim"

#: The flavour this program-driven path can actually import and call. The other flavour
#: (a standalone binary taking a linked ELF) is a real build of the same engine that this
#: particular route cannot drive -- a distinction the probe must state rather than flatten.
_WRAPPER_FLAVOUR = "wrapper"

_RTL_ENGINES: dict[str, tuple[str, str]] = {
    "vcs": ("vcs", "vcs_run.py"),
    "gsim": ("gsim", "gsim_run.py"),
    "verilator": ("vsim", "verilator_run.py"),
}


def _rtl_engine_dir(target: str, engine: str) -> Path | None:
    """Where ``engine``'s build for ``target`` lives, or None when nothing registers one.

    Three sources, in precedence order:

    1. ``MERLIN_EXT_<TARGET>_<SUFFIX>`` (process env or ``.env``) — the machine-specific registration.
    2. The run's derived home ``out/build/rtl_engines/<target>/<engine>/``.
    3. The installed home under the repository-anchored build root.

    (2) is new and it is the point. Registration by env var alone means an engine that IS BUILT AND
    WORKING on this machine is invisible to the policy until somebody adds a line to a gitignored file
    — which is exactly what happened: a cycle-exact, 32x-faster GSIM engine sat beside its conventional
    wrapper, fully built, and every cert ran on Verilator because no ``MERLIN_EXT_<TARGET>_GSIM`` line
    existed. Nothing reported that; the engine was simply never considered. A derived home makes
    "install it where it belongs" the way to register an engine, and the env var the exception.
    """
    suffix, _ = _context()._RTL_ENGINES[engine]
    try:
        return Path(_context().ext_path(f"{target}_{suffix}"))
    except KeyError:
        pass
    from merlin.target.simulators.gsim_emulator import engine_home, installed_engine_home

    # Mirror binary discovery: a preinstalled simulator is a machine asset,
    # not something that must be rebuilt under each invocation's output root.
    homes = (engine_home(target, engine), installed_engine_home(target, engine))
    for home in homes:
        if (home / _context()._RTL_ENGINES[engine][1]).is_file():
            return home
    return next((home for home in homes if home.is_dir()), None)


def _rtl_engine_probe(target: str, engine: str):
    """``() -> (available, reason)`` for one engine: its dir must be registered AND hold its wrapper."""

    def probe():
        from merlin.target.simulators.gsim_emulator import engine_home

        _, fname = _context()._RTL_ENGINES[engine]
        d = _context()._rtl_engine_dir(target, engine)
        if d is None:
            # BOTH places, named. "not registered" alone sent every reader to the env var and none of
            # them to the directory an engine can simply be installed into.
            return False, (
                f"no MERLIN_EXT_{target.upper()}_{_context()._RTL_ENGINES[engine][0].upper()} "
                f"registered and no build at {engine_home(target, engine)}"
            )
        w = d / fname
        if not w.is_file():
            # PRESENCE IS THE HOME-LAYOUT MODULE'S FACT TOO, not just lineage. Statting one filename
            # made a home holding a BUILT engine of the other flavour report the engine as ABSENT --
            # the same two-modules-disagree defect as the blind probe and the bypassed lineage gate,
            # in a third place. The answer stays False, because it must: this path calls
            # ``run_program(words, preload, reads)`` on an imported wrapper, and the binary flavour
            # takes a LINKED ELF instead (``<emu> <elf> +loadmem=<elf> +max-cycles=N``). Assembling an
            # ELF out of program words is the compiler's job upstream, not a probe's, so a home
            # without the wrapper genuinely cannot be driven from here and passing it would trade a
            # clean unavailable for a crash inside the runner import.
            #
            # What changes is the REASON. "absent" sent every reader looking for a missing build;
            # naming the flavour that IS there sends them to the right question -- which is whether
            # this target's own backend drives that engine, as it does for the binary flavour today.
            if engine == _LINEAGE_ENGINE:
                from merlin.target.simulators.gsim_emulator import resolve as _built

                r = _built(target)
                if r.ok and r.flavour and r.flavour != _WRAPPER_FLAVOUR:
                    return False, (
                        f"{engine} IS built for {target}, as the {r.flavour} flavour at {r.path} -- "
                        f"but NOT as {fname}. This program-driven path imports {fname} and calls "
                        f"run_program(words, preload, reads); the {r.flavour} flavour is driven with a "
                        f"linked ELF instead, so it cannot be run from assembled program words here. "
                        f"The engine is not missing: the target's own backend runs it."
                    )
            return False, f"{fname} absent under {d}"
        # FINDING THE WRAPPER PROVES AN ENGINE IS BUILT, NOT THAT ITS BYTES MAY CERTIFY. The home-layout
        # module owns the lineage question for the homes it lays out, and asking it here is what stops
        # two modules disagreeing about one fact with the OPTIMISTIC one having the last word.
        #
        # Measured 2026-09-04: with MERLIN_GSIM_REQUIRE_RECEIPT=1 set, a target whose engine carries only
        # an adoption record -- lineage `adopted`, never built-and-bound -- had gsim_emulator.probe()
        # answer False and STILL certified on it, because this probe asked only whether a file existed.
        # A provenance gate that the selection path routes around is not a gate.
        #
        # The same lineage gate applies to this engine's explicitly selected
        # wrapper directory, not only its conventional derived home.
        if engine == _LINEAGE_ENGINE:
            from merlin.target.simulators.gsim_emulator import resolve_wrapper

            # Validate the wrapper we actually selected, including explicit
            # external homes. A separately registered ELF-driven binary cannot
            # confer provenance on this program-driven engine.
            r = resolve_wrapper(target, home=d, source=f"selected:{d}")
            if r is not None and r.refused:
                return False, r.reason
            if r is not None:
                return r.ok, r.reason
        return True, f"{fname} at {d}"

    return probe


def select_rtl_engine(target: str) -> dict:
    """Pick this target's elaborated-RTL engine (see :mod:`rtl_engine_policy`). Raises
    :class:`OracleUnavailable` when none can run, so the tier reports unavailable rather than silently
    resolving to a lower-fidelity oracle."""
    from merlin.target.simulators import engine_policy as _pol

    try:
        return _pol.select(target, {e: _context()._rtl_engine_probe(target, e) for e in _context()._RTL_ENGINES})
    except _pol.NoEngineAvailable as exc:
        raise OracleUnavailable(str(exc)) from exc
