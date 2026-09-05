"""A semantics for the COMMAND BUFFER, given by the same lowering-to-``smt`` idea as the interface.

Why this surface and not the ``runtime`` dialect. There are two backend paths and they are not the
same shape. The in-tree ``TargetPackage`` path lowers ``interface -> target -> runtime ->
command_buffer``, with ``runtime_lowering`` constructing the ``runtime.*`` ops itself. The
capsule-bench path — *what an experiment agent actually writes* — is a subprocess with CLI
entrypoints that emits ``command_buffer.json`` DIRECTLY, and the target-dialect contract states in
so many words that the intermediate MLIR is "a recommendation, not a gate". So a checker built on the
``runtime`` dialect would be structurally unable to see the compiler under evaluation. The command
buffer is the ABI both paths converge on, it is schema-validated, and the in-tree path yields it for
free via ``emit_command_buffer``.

**This module mirrors** :func:`merlin.runtime.simulator.simulate`. That is the whole contract: where
the two disagree, a CORRECT backend gets refuted, which is worse than not checking at all. Every
method here names the reference line it mirrors, and the differential test in
``merlin/tests/ir/test_cb_semantics.py`` runs both engines on concrete inputs and requires them to
agree before any refutation is trusted.

**Fail closed, in the idiom already established by** :mod:`merlin.runtime.reference`: an opcode with
no encoding raises rather than being skipped. That module records why, and the reason is worth
repeating — silently dropping an unmodelled opcode once produced an empty output map, which read
downstream as "the kernel never wrote its output", indistinguishable from a real dropped store and
unfixable by any submission.
"""
from __future__ import annotations

from typing import Any

from .smt_semantics import Encoder, Tensor, UnsupportedSemantics

#: Opcodes this encoder has a bit-exact integer definition for.
ENCODABLE_OPCODES = frozenset({
    "RES_PACK", "MATMUL", "MATMUL_RESIDENT", "COMMIT", "EVICT",
})

#: Opcodes that provably cannot change a committed value, so skipping them is sound rather than
#: convenient. ``EVICT`` frees a residency handle; residency is a performance property by
#: construction. Listing these explicitly is what lets an unknown opcode raise instead of being
#: quietly ignored.
NO_NUMERIC_EFFECT = frozenset({"EVICT"})

#: Epilogue stages with an exact integer encoding. ``acc_scale`` is deliberately absent: it is an
#: IEEE-754 f32 round-trip (``Tensor.requant_acc_scale``), and this package refuses float rather than
#: approximating it, because a bit-exact float check rejects correct backends.
ENCODABLE_EPILOGUE = frozenset({"bias_add", "bias", "requant", "relu"})

#: Signed range by dtype spelling, for the readout narrow.
_SIGNED_RANGE = {"i8": (-128, 127), "i16": (-32768, 32767)}

#: The reference's own default when a COMMIT carries no ``output_dtype``. It is ``i8`` — i.e. absent
#: means SATURATE — and the reference narrows only on an exact ``"i8"`` match, so ``i16``/``i32`` are
#: passthroughs there. Mirroring either of these wrongly would refute a correct backend.
_COMMIT_DEFAULT_DTYPE = "i8"


def safe_k_bound(operand_width: int, acc_width: int) -> int:
    """Largest contraction extent K for which the accumulation provably cannot overflow.

    This exists because the reference engine and this encoder genuinely disagree about overflow.
    ``Tensor.matmul`` documents "accumulated in i32" but accumulates in UNBOUNDED Python ints — it
    tags the dtype without enforcing it — while :meth:`Encoder.matmul` wraps mod 2**acc_width. The two
    agree exactly as long as no sum ever leaves the accumulator's range, and that is a checkable side
    condition rather than a hope:

        |a*b| <= 2**(2*w-2)  for signed w-bit operands, so  K * 2**(2*w-2) <= 2**(acc-1) - 1.

    For i8 operands into an i32 accumulator that is K <= 131071 — far beyond any shape we verify, but
    the point is that it is CHECKED. Beyond the bound the honest verdict is an abstention, because the
    two engines are then answering different questions.
    """
    product_max = 2 ** (2 * operand_width - 2)
    return (2 ** (acc_width - 1) - 1) // product_max


class CommandBufferEncoder:
    """Symbolically executes a command buffer, mirroring the reference simulator."""

    def __init__(self, enc: Encoder, cb: dict[str, Any], *, acc_width: int = 32) -> None:
        self.enc = enc
        self.cb = cb
        self.acc_width = acc_width
        self.env: dict[str, Tensor] = {}
        self.resident: dict[str, str] = {}
        #: ``cb["params"]["requant_shift"]`` else 4 — the reference's default (simulator.py:111).
        self.default_shift = int((cb.get("params") or {}).get("requant_shift", 4))

    # -- leaves ----------------------------------------------------------------------------------
    def declare_leaves(self, shared: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """Declare a symbolic tensor per DECLARED tensor, reusing any caller-supplied terms.

        ``shared`` lets the two sides of a refinement query be built over the SAME symbols; encoding
        each side over independent symbols would make the query trivially satisfiable and the check
        meaningless. Intermediates (``W_res``, ``acc0``) are deliberately NOT declared here: they are
        produced-then-referenced names that never appear in ``tensors``, exactly as the reference
        treats them.
        """
        shared = dict(shared or {})
        for name, spec in (self.cb.get("tensors") or {}).items():
            if name in shared:
                self.env[name] = shared[name]
                continue
            shape = list(spec.get("shape") or ())
            if len(shape) != 2:
                raise UnsupportedSemantics(
                    f"tensor {name!r} has rank {len(shape)}; only rank-2 tensors are encoded")
            dtype = str(spec.get("dtype") or "")
            width = _width_of(dtype, name)
            self.env[name] = self.enc.symbolic_tensor(name, shape[0], shape[1], width)
        return dict(self.env)

    # -- execution -------------------------------------------------------------------------------
    def run(self) -> dict[str, Tensor]:
        """Walk the commands in order; return the DECLARED outputs, as the reference does."""
        committed: dict[str, Tensor] = {}
        for i, cmd in enumerate(self.cb.get("commands") or []):
            op = str(cmd.get("opcode") or "")
            operands = cmd.get("operands") or {}
            attrs = cmd.get("attributes") or {}
            if op in NO_NUMERIC_EFFECT:
                if op == "EVICT":
                    self.resident.pop(operands.get("handle", ""), None)
                continue
            if op not in ENCODABLE_OPCODES:
                raise UnsupportedSemantics(
                    f"command {i} uses opcode {op!r}, which this encoder has no definition for. "
                    f"Encodable: {sorted(ENCODABLE_OPCODES)}. Refusing rather than skipping it — a "
                    f"skipped command silently changes what the query is about.")
            if op == "RES_PACK":
                self._res_pack(operands, attrs)
            elif op in ("MATMUL", "MATMUL_RESIDENT"):
                self._matmul(operands)
            elif op == "COMMIT":
                dst, value = self._commit(operands, attrs)
                committed[dst] = value

        declared = list(self.cb.get("outputs") or ())
        if not declared:
            return committed
        missing = [d for d in declared if d not in committed]
        if missing:
            raise UnsupportedSemantics(
                f"declared outputs never committed: {missing}. The buffer cannot be compared against "
                f"a specification that expects them.")
        return {d: committed[d] for d in declared}

    # -- per-opcode, each mirroring its reference branch ------------------------------------------
    def _res_pack(self, operands: dict, attrs: dict) -> None:
        """simulator.py:119 — a value-identical packed copy, plus residency bookkeeping."""
        if "scale" in operands:
            raise UnsupportedSemantics(
                "RES_PACK with a 'scale' operand dequantizes per channel to f32 "
                "(Tensor.dequant_per_channel); float is refused here, not approximated")
        src, dst = operands["src"], operands["dst"]
        self.env[dst] = self._get(src)
        self.resident[dst] = src

    def _matmul(self, operands: dict) -> None:
        """simulator.py:132 — ``env[dst] = env[lhs].matmul(env[rhs])``."""
        lhs, rhs, dst = self._get(operands["lhs"]), self._get(operands["rhs"]), operands["dst"]
        bound = safe_k_bound(max(lhs.width, rhs.width), self.acc_width)
        if lhs.cols > bound:
            raise UnsupportedSemantics(
                f"contraction K={lhs.cols} exceeds the overflow-free bound {bound} for "
                f"{max(lhs.width, rhs.width)}-bit operands in an i{self.acc_width} accumulator. "
                f"Beyond it this encoder wraps while the reference engine accumulates in unbounded "
                f"integers, so the two answer different questions — abstaining rather than guessing.")
        self.env[dst] = self.enc.matmul(lhs, rhs, acc_width=self.acc_width)

    def _commit(self, operands: dict, attrs: dict) -> tuple[str, Tensor]:
        """simulator.py:154 — epilogue stages IN ORDER, then the dtype narrow."""
        src, dst = operands["src"], operands["dst"]
        t = self._get(src)
        shift = int(attrs.get("requant_shift", self.default_shift))
        for stage in (attrs.get("epilogue") or []):
            stage = str(stage)
            if stage not in ENCODABLE_EPILOGUE:
                raise UnsupportedSemantics(
                    f"epilogue stage {stage!r} has no exact integer encoding here "
                    f"(encodable: {sorted(ENCODABLE_EPILOGUE)}); 'acc_scale' in particular is an "
                    f"IEEE-754 f32 round-trip and is refused rather than approximated")
            if stage in ("bias_add", "bias"):
                t = self.enc.add_bias(t, self._get(self._bias_name(operands, attrs, dst)))
            elif stage == "requant":
                t = self.enc.requant(t, shift)
            elif stage == "relu":
                t = self.enc.relu(t)
        # The narrow happens AFTER the whole epilogue, unconditionally, and the reference tests an
        # exact "i8" — so anything else is a passthrough THERE and must be here too.
        dtype = str(attrs.get("output_dtype", _COMMIT_DEFAULT_DTYPE))
        if dtype == "i8":
            lo, hi = _SIGNED_RANGE["i8"]
            t = self.enc.saturate(t, lo, hi, 8)
        self.env[dst] = t
        return dst, t

    # -- helpers ---------------------------------------------------------------------------------
    def _bias_name(self, operands: dict, attrs: dict, dst: str) -> str:
        """Mirror ``commandbuffer.bias_tensor_name``: attributes first, then operands, else raise."""
        for source in (attrs, operands):
            name = source.get("bias")
            if name:
                return str(name)
        raise UnsupportedSemantics(
            f"COMMIT {dst!r} lists a bias epilogue stage but names no bias tensor in either its "
            f"attributes or its operands")

    def _get(self, name: str) -> Tensor:
        try:
            return self.env[name]
        except KeyError:
            raise UnsupportedSemantics(
                f"command references tensor {name!r}, which is neither declared nor produced by an "
                f"earlier command") from None


def _width_of(dtype: str, name: str) -> int:
    from .smt_semantics import _INT_WIDTH

    try:
        return _INT_WIDTH[dtype]
    except KeyError:
        raise UnsupportedSemantics(
            f"tensor {name!r} has dtype {dtype!r}, which has no bitvector encoding here. Float "
            f"datapaths are refused: reassociation is a legal backend choice, so a bit-exact float "
            f"check would reject correct backends.") from None


def encode_command_buffer(enc: Encoder, cb: dict[str, Any], *,
                          shared: dict[str, Tensor] | None = None,
                          acc_width: int = 32) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Encode ``cb``; return ``(declared outputs, the leaf tensors used)``."""
    cbe = CommandBufferEncoder(enc, cb, acc_width=acc_width)
    leaves = cbe.declare_leaves(shared)
    return cbe.run(), leaves
