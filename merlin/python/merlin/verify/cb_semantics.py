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
    "VECTOR_MAP", "BIAS_ADD", "VREDUCE", "ATTENTION_QK", "ATTENTION_PV", "MOVEMENT",
})

#: Integer opcodes that ARE encodable in principle but are not built here yet. Named with the reason
#: rather than left to fall through to "unknown": a reader deciding whether to extend the encoder
#: needs to know the difference between "we cannot" and "we have not".
DEFERRED_OPCODES = {
    "CONV2D": "encodable in principle — the im2col geometry is concrete, so it unrolls — but it needs "
              "the reference's index map factored out so the two engines cannot disagree about a "
              "padding edge; not built",
    "BATCHED_MATMUL": "encodable in principle as a per-batch loop of 2-D matmuls, but this encoder's "
                      "Tensor is rank-2 and would need a batch layer; not built",
}

#: Opcodes the reference simulator itself implements in float, so a bit-exact check would be the
#: wrong specification rather than merely an expensive one. Named individually so an abstention can
#: say WHICH class it fell into instead of "unknown".
FLOAT_ONLY_OPCODES = frozenset({"RMSNORM", "SOFTMAX", "GELU", "SOFTCAP", "ROPE"})

#: In the command-buffer schema's opcode enum but with no branch in ``simulate`` at all — the
#: reference raises ``unknown opcode`` on them, so there is nothing here to mirror.
UNIMPLEMENTED_OPCODES = frozenset({
    "LAYERNORM", "GEGLU", "ATTENTION_FULL", "CONV", "MATMUL_BATCHED",
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

#: The reference's default when a COMMIT carries no ``output_dtype``: ``i32`` — absent means DO NOT
#: NARROW.
#:
#: This was ``i8`` and had to change WITH the engine. The runtime defaulted to ``i8`` and narrowed
#: on an exact match, while ``capsule_golden`` defaulted to ``i32`` and narrowed any width; they
#: diverged for 77 days and a correct backend omitting the attribute failed L0 on 85 of 130
#: capsules. Both runtime COMMIT sites now route through the shared ``_narrow_int_readout``.
#:
#: The rule this constant serves is unchanged: MIRROR the engine the corpus grades against,
#: whatever it says. An encoder that keeps its own opinion refutes correct backends — which is
#: what this line would have started doing the moment the engine moved.
_COMMIT_DEFAULT_DTYPE = "i32"


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
                raise UnsupportedSemantics(_why_not_encodable(i, op))
            if op == "RES_PACK":
                self._res_pack(operands, attrs)
            elif op in ("MATMUL", "MATMUL_RESIDENT"):
                self._matmul(operands)
            elif op == "COMMIT":
                dst, value = self._commit(operands, attrs)
                committed[dst] = value
            elif op == "VECTOR_MAP":
                self._vector_map(operands, attrs)
            elif op == "VREDUCE":
                self._vreduce(operands, attrs)
            elif op == "BIAS_ADD":
                dst, value = self._bias_add(operands, attrs)
                committed[dst] = value
            elif op in ("ATTENTION_QK", "ATTENTION_PV"):
                dst, value = self._attention(op, operands, attrs)
                committed[dst] = value
            elif op == "MOVEMENT":
                dst, value = self._movement(operands, attrs)
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
        # The narrow happens AFTER the whole epilogue, unconditionally, and now follows the engine's
        # SHARED rule: saturate to any integer width below the accumulator, derived from the dtype
        # rather than tested against "i8". The old exact-i8 test could not express i16/i4/u8 at all.
        t = self._narrow(t, attrs, default=_COMMIT_DEFAULT_DTYPE)
        self.env[dst] = t
        return dst, t

    def _vector_map(self, operands: dict, attrs: dict) -> None:
        """simulator.py:194 — elementwise combine, then an optional activation."""
        combine = str(attrs.get("combine", "add"))
        dst = operands["dst"]
        if combine == "identity":
            t = self._get(operands["lhs"])
        else:
            a, b = self._get(operands["lhs"]), self._get(operands["rhs"])
            if (a.rows, a.cols) != (b.rows, b.cols):
                raise UnsupportedSemantics(
                    f"VECTOR_MAP operands differ in shape: {(a.rows, a.cols)} vs {(b.rows, b.cols)}")
            if combine == "add":
                t = self._elementwise(a, b, self.enc.smt.BVAddOp)
            elif combine == "mul":
                t = self._elementwise(a, b, self.enc.smt.BVMulOp)
            else:
                raise UnsupportedSemantics(
                    f"VECTOR_MAP combine {combine!r} has no encoding here "
                    f"(the reference defines identity/add/mul)")
        for stage in (attrs.get("activation") or []):
            if str(stage) != "relu":
                raise UnsupportedSemantics(
                    f"VECTOR_MAP activation {stage!r} has no encoding here "
                    f"(the reference defines relu only)")
            t = self.enc.relu(t)
        self.env[dst] = t

    def _vreduce(self, operands: dict, attrs: dict) -> None:
        """simulator.py:505 — total sum to a length-1 tensor.

        The reference keeps the SOURCE dtype label while summing in unbounded ints, so a mod-2^w sum
        at the source width would disagree with it for any non-trivial element count. The sum is
        therefore taken at the accumulator width, with the same overflow bound the contraction uses.
        """
        rop = str(attrs.get("op", "sum"))
        if rop != "sum":
            raise UnsupportedSemantics(
                f"VREDUCE op {rop!r} has no encoding here (the reference defines sum only)")
        src = self._get(operands["src"])
        n = src.rows * src.cols
        bound = safe_k_bound(src.width, self.acc_width)
        if n > bound:
            raise UnsupportedSemantics(
                f"VREDUCE over {n} elements exceeds the overflow-free bound {bound} for "
                f"{src.width}-bit elements in an i{self.acc_width} accumulator")
        acc = None
        for (r, c) in sorted(src.elems):
            term = self.enc.sign_extend(src.at(r, c), src.width, self.acc_width)
            acc = term if acc is None else self.enc.smt.BVAddOp(acc, term).results[0]
        self.env[operands["dst"]] = Tensor(1, 1, self.acc_width,
                                           {(0, 0): acc if acc is not None
                                            else self.enc.const(0, self.acc_width)})

    def _bias_add(self, operands: dict, attrs: dict) -> tuple[str, Tensor]:
        """simulator.py:218 — the UNFUSED per-column add. Note the default dtype is i32, not i8."""
        src, dst = operands["src"], operands["dst"]
        t = self.enc.add_bias(self._get(src), self._get(self._bias_name(operands, attrs, dst)))
        t = self._narrow(t, attrs, default="i32")
        self.env[dst] = t
        return dst, t

    def _attention(self, op: str, operands: dict, attrs: dict) -> tuple[str, Tensor]:
        """simulator.py:321 / :356 — the two attention matmuls, default dtype i32.

        QK contracts over the trailing head dim of BOTH operands, so K is transposed first; PV is a
        plain matmul. Getting that transpose wrong would silently compare the wrong contraction, so
        it mirrors the reference's own index expression.
        """
        if op == "ATTENTION_QK":
            q, k, dst = self._get(operands["q"]), self._get(operands["k"]), operands["dst"]
            if q.cols != k.cols:
                raise UnsupportedSemantics(
                    f"ATTENTION_QK head-dim mismatch: {(q.rows, q.cols)} vs {(k.rows, k.cols)}")
            k_t = Tensor(k.cols, k.rows, k.width,
                         {(j, i): k.at(i, j) for i in range(k.rows) for j in range(k.cols)})
            t = self.enc.matmul(q, k_t, acc_width=self.acc_width)
        else:
            pt, vt, dst = self._get(operands["p"]), self._get(operands["v"]), operands["dst"]
            if pt.cols != vt.rows:
                raise UnsupportedSemantics(
                    f"ATTENTION_PV key-count mismatch: {(pt.rows, pt.cols)} vs {(vt.rows, vt.cols)}")
            t = self.enc.matmul(pt, vt, acc_width=self.acc_width)
        for stage in (attrs.get("epilogue") or []):
            stage = str(stage)
            if stage == "requant":
                t = self.enc.requant(t, int(attrs.get("requant_shift", self.default_shift)))
            elif stage == "relu":
                t = self.enc.relu(t)
            else:
                raise UnsupportedSemantics(
                    f"{op} epilogue stage {stage!r} has no exact integer encoding here "
                    f"(the reference accepts acc_scale/requant/relu; acc_scale is float)")
        t = self._narrow(t, attrs, default="i32")
        self.env[dst] = t
        return dst, t

    def _movement(self, operands: dict, attrs: dict) -> tuple[str, Tensor]:
        """simulator.py:485 — a load/store round-trip that carries values UNCHANGED.

        Deliberately no clamp and no requantize: only the container dtype widens. A movement
        capsule's whole point is that the data survives the trip bit-for-bit, so narrowing here would
        refute exactly the programs it exists to check.
        """
        if "src" not in operands:
            raise UnsupportedSemantics("MOVEMENT needs operands src/dst")
        src, dst = operands["src"], operands["dst"]
        t = self._get(src)
        self.env[dst] = t
        return dst, t

    def _elementwise(self, a: Tensor, b: Tensor, op_cls) -> Tensor:
        out = {k: op_cls(v, b.elems[k]).results[0] for k, v in a.elems.items()}
        return Tensor(a.rows, a.cols, a.width, out)

    def _narrow(self, t: Tensor, attrs: dict, *, default: str) -> Tensor:
        """The readout narrow, mirroring the reference's rule: an EXACT "i8" match, nothing else.

        The default differs per opcode in the reference (COMMIT defaults to i8, the vector and
        attention family to i32), so it is passed in rather than assumed. Mirroring either wrongly
        would refute a correct backend.
        """
        bits = _narrow_bits(str(attrs.get("output_dtype", default)))
        if bits is None:
            return t
        return self.enc.saturate(t, -(1 << (bits - 1)), (1 << (bits - 1)) - 1, bits)

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


def _why_not_encodable(index: int, op: str) -> str:
    """Say WHICH class an unencodable opcode fell into — 'unknown' is not an actionable diagnostic."""
    if op in FLOAT_ONLY_OPCODES:
        return (f"command {index} uses {op!r}, which the reference simulator itself computes in "
                f"float. A bit-exact check on a float datapath is the wrong specification, not "
                f"merely an expensive one: reassociation is a legal backend choice, so it would "
                f"reject correct backends. Abstaining.")
    if op in DEFERRED_OPCODES:
        return (f"command {index} uses {op!r}: {DEFERRED_OPCODES[op]}. This is a gap in THIS encoder, "
                f"not a defect in the buffer.")
    if op in UNIMPLEMENTED_OPCODES:
        return (f"command {index} uses {op!r}, which is in the command-buffer schema's enum but has "
                f"no branch in the reference simulator at all — there is nothing here to mirror.")
    return (f"command {index} uses opcode {op!r}, which this encoder has no definition for. "
            f"Encodable: {sorted(ENCODABLE_OPCODES)}. Refusing rather than skipping it — a skipped "
            f"command silently changes what the query is about.")


def _narrow_bits(dtype: str) -> int | None:
    """Bits to saturate to, or None when nothing narrows — mirroring ``_narrow_int_readout``.

    Derived from the spelling rather than tested against "i8", so a target narrowing to i16/i4/u8 is
    handled instead of silently passed through. A width at or above the accumulator has nothing to
    narrow; a non-integer spelling is not this function's business (floats are refused earlier).
    """
    if not dtype or dtype[0] not in ("i", "u") or not dtype[1:].isdigit():
        return None
    bits = int(dtype[1:])
    return None if bits >= 32 else bits


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
