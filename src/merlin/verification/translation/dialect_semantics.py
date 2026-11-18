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

**This module mirrors** :func:`merlin.runtime.engine.simulator.simulate`. That is the whole contract: where
the two disagree, a CORRECT backend gets refuted, which is worse than not checking at all. Every
method here names the reference line it mirrors, and the differential test in
``tests/ir/test_cb_semantics.py`` runs both engines on concrete inputs and requires them to
agree before any refutation is trusted.

**Fail closed, in the idiom already established by** :mod:`merlin.runtime.engine.reference`: an opcode with
no encoding raises rather than being skipped. That module records why, and the reason is worth
repeating — silently dropping an unmodelled opcode once produced an empty output map, which read
downstream as "the kernel never wrote its output", indistinguishable from a real dropped store and
unfixable by any submission.
"""

from __future__ import annotations

from typing import Any

from merlin.verification.translation.smt_encoding import (
    Encoded,
    Encoder,
    Tensor,
    UnsupportedSemantics,
    _elem_width,
    _shape,
)

# ---- cb_semantics ------------------------------------------------------------------------------

#: Opcodes this encoder has a bit-exact integer definition for.
ENCODABLE_OPCODES = frozenset(
    {
        "RES_PACK",
        "MATMUL",
        "MATMUL_RESIDENT",
        "COMMIT",
        "EVICT",
        "VECTOR_MAP",
        "BIAS_ADD",
        "VREDUCE",
        "ATTENTION_QK",
        "ATTENTION_PV",
        "MOVEMENT",
    }
)

#: Integer opcodes that ARE encodable in principle but are not built here yet. Named with the reason
#: rather than left to fall through to "unknown": a reader deciding whether to extend the encoder
#: needs to know the difference between "we cannot" and "we have not".
DEFERRED_OPCODES = {
    "CONV2D": "encodable in principle — the im2col geometry is concrete, so it unrolls — but it needs "
    "the reference's index map factored out so the two engines cannot disagree about a "
    "padding edge; not built",
    "BATCHED_MATMUL": "encodable in principle as a per-batch loop of 2-D matmuls, but this encoder's "
    "Tensor is rank-2 and would need a batch layer; not built",
    "ROW_MAJOR_VIEW": "the reference changes a produced buffer's logical shape without copying, but this "
    "encoder does not model the corresponding shape/index and output-ownership change; not built",
}

#: Opcodes the reference simulator itself implements in float, so a bit-exact check would be the
#: wrong specification rather than merely an expensive one. Named individually so an abstention can
#: say WHICH class it fell into instead of "unknown".
FLOAT_ONLY_OPCODES = frozenset({"RMSNORM", "SOFTMAX", "GELU", "SOFTCAP", "ROPE", "MAXPOOL2D", "RESIDUAL_ADD"})

#: In the command-buffer schema's opcode enum but with no branch in ``simulate`` at all — the
#: reference raises ``unknown opcode`` on them, so there is nothing here to mirror.
UNIMPLEMENTED_OPCODES = frozenset(
    {
        "LAYERNORM",
        "GEGLU",
        "ATTENTION_FULL",
        "CONV",
        "MATMUL_BATCHED",
        "K_CHAIN",
        "DEPTHWISE_CONV2D",
    }
)

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
                raise UnsupportedSemantics(f"tensor {name!r} has rank {len(shape)}; only rank-2 tensors are encoded")
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
            self._refuse_unmodelled_step(i, op, attrs)
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

                # Mirror simulator.py:539-543. The reference adds any tensor DECLARED with role "output"
                # that exists in the environment, whichever opcode produced it -- vector-family results
                # (VECTOR_MAP, VREDUCE) never pass through COMMIT and would otherwise be invisible.
                #
                # Omitting this step made the checker refute 39 archived submissions that the numeric grade
                # PASSED: a movement capsule whose spec says `merlin_iface.movement` is legitimately
                # implemented by the backend as an identity VECTOR_MAP, whose result this encoder then could
                # not see at all. An encoder that disagrees with the engine it mirrors refutes correct work,
                # which is the one outcome this module exists to prevent.
        for name, spec in (self.cb.get("tensors") or {}).items():
            if (spec or {}).get("role") == "output" and name in self.env and name not in committed:
                committed[name] = self.env[name]

        declared = list(self.cb.get("outputs") or ())
        if not declared:
            return committed
        missing = [d for d in declared if d not in committed]
        if missing:
            raise UnsupportedSemantics(
                f"declared outputs never committed: {missing}. The buffer cannot be compared against "
                f"a specification that expects them."
            )
        return {d: committed[d] for d in declared}

        # -- per-opcode, each mirroring its reference branch ------------------------------------------

    def _res_pack(self, operands: dict, attrs: dict) -> None:
        """simulator.py:119 — a value-identical packed copy, plus residency bookkeeping."""
        if "scale" in operands:
            raise UnsupportedSemantics(
                "RES_PACK with a 'scale' operand dequantizes per channel to f32 "
                "(Tensor.dequant_per_channel); float is refused here, not approximated"
            )
        src, dst = operands["src"], operands["dst"]
        self.env[dst] = self._get(src)
        self.resident[dst] = src

    def _matmul(self, operands: dict) -> None:
        """simulator.py:132 — ``env[dst] = env[lhs].matmul(env[rhs])``."""
        lhs = self._get(self._operand(operands, "lhs", opcode="MATMUL"))
        rhs = self._get(self._operand(operands, "rhs", opcode="MATMUL"))
        dst = self._operand(operands, "dst", opcode="MATMUL")
        bound = safe_k_bound(max(lhs.width, rhs.width), self.acc_width)
        if lhs.cols > bound:
            raise UnsupportedSemantics(
                f"contraction K={lhs.cols} exceeds the overflow-free bound {bound} for "
                f"{max(lhs.width, rhs.width)}-bit operands in an i{self.acc_width} accumulator. "
                f"Beyond it this encoder wraps while the reference engine accumulates in unbounded "
                f"integers, so the two answer different questions — abstaining rather than guessing."
            )
        self.env[dst] = self.enc.matmul(lhs, rhs, acc_width=self.acc_width)

    def _commit(self, operands: dict, attrs: dict) -> tuple[str, Tensor]:
        """simulator.py:154 — epilogue stages IN ORDER, then the dtype narrow."""
        src, dst = operands["src"], operands["dst"]
        t = self._get(src)
        shift = int(attrs.get("requant_shift", self.default_shift))
        for stage in attrs.get("epilogue") or []:
            stage = str(stage)
            if stage not in ENCODABLE_EPILOGUE:
                raise UnsupportedSemantics(
                    f"epilogue stage {stage!r} has no exact integer encoding here "
                    f"(encodable: {sorted(ENCODABLE_EPILOGUE)}); 'acc_scale' in particular is an "
                    f"IEEE-754 f32 round-trip and is refused rather than approximated"
                )
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
        combine = self._combine_of(attrs, "VECTOR_MAP")
        dst = self._operand(operands, "dst", opcode="VECTOR_MAP")
        if combine == "identity":
            t = self._get(self._operand(operands, "src", "lhs", opcode="VECTOR_MAP(identity)"))
        else:
            a = self._get(self._operand(operands, "lhs", opcode="VECTOR_MAP"))
            b = self._get(self._operand(operands, "rhs", opcode="VECTOR_MAP"))
            if (a.rows, a.cols) != (b.rows, b.cols):
                raise UnsupportedSemantics(
                    f"VECTOR_MAP operands differ in shape: {(a.rows, a.cols)} vs {(b.rows, b.cols)}"
                )
            if combine == "add":
                t = self._elementwise(a, b, self.enc.smt.BVAddOp)
            elif combine == "mul":
                t = self._elementwise(a, b, self.enc.smt.BVMulOp)
            else:
                raise UnsupportedSemantics(
                    f"VECTOR_MAP combine {combine!r} has no encoding here (the reference defines identity/add/mul)"
                )
        for stage in attrs.get("activation") or []:
            if str(stage) != "relu":
                raise UnsupportedSemantics(
                    f"VECTOR_MAP activation {stage!r} has no encoding here (the reference defines relu only)"
                )
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
            raise UnsupportedSemantics(f"VREDUCE op {rop!r} has no encoding here (the reference defines sum only)")
        src = self._get(operands["src"])
        n = src.rows * src.cols
        bound = safe_k_bound(src.width, self.acc_width)
        if n > bound:
            raise UnsupportedSemantics(
                f"VREDUCE over {n} elements exceeds the overflow-free bound {bound} for "
                f"{src.width}-bit elements in an i{self.acc_width} accumulator"
            )
        acc = None
        for r, c in sorted(src.elems):
            term = self.enc.sign_extend(src.at(r, c), src.width, self.acc_width)
            acc = term if acc is None else self.enc.smt.BVAddOp(acc, term).results[0]
        self.env[operands["dst"]] = Tensor(
            1, 1, self.acc_width, {(0, 0): acc if acc is not None else self.enc.const(0, self.acc_width)}
        )

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
                raise UnsupportedSemantics(f"ATTENTION_QK head-dim mismatch: {(q.rows, q.cols)} vs {(k.rows, k.cols)}")
            k_t = Tensor(k.cols, k.rows, k.width, {(j, i): k.at(i, j) for i in range(k.rows) for j in range(k.cols)})
            t = self.enc.matmul(q, k_t, acc_width=self.acc_width)
        else:
            pt, vt, dst = self._get(operands["p"]), self._get(operands["v"]), operands["dst"]
            if pt.cols != vt.rows:
                raise UnsupportedSemantics(
                    f"ATTENTION_PV key-count mismatch: {(pt.rows, pt.cols)} vs {(vt.rows, vt.cols)}"
                )
            t = self.enc.matmul(pt, vt, acc_width=self.acc_width)
        for stage in attrs.get("epilogue") or []:
            stage = str(stage)
            if stage == "requant":
                t = self.enc.requant(t, int(attrs.get("requant_shift", self.default_shift)))
            elif stage == "relu":
                t = self.enc.relu(t)
            else:
                raise UnsupportedSemantics(
                    f"{op} epilogue stage {stage!r} has no exact integer encoding here "
                    f"(the reference accepts acc_scale/requant/relu; acc_scale is float)"
                )
        t = self._narrow(t, attrs, default="i32")
        self.env[dst] = t
        return dst, t

    def _movement(self, operands: dict, attrs: dict) -> tuple[str, Tensor]:
        """simulator.py:485 — a load/store round-trip that carries values UNCHANGED.

        Deliberately no clamp and no requantize: only the container dtype widens. A movement
        capsule's whole point is that the data survives the trip bit-for-bit, so narrowing here would
        refute exactly the programs it exists to check.
        """
        src = self._operand(operands, "src", "lhs", opcode="MOVEMENT")
        dst = self._operand(operands, "dst", opcode="MOVEMENT")
        t = self._get(src)
        self.env[dst] = t
        return dst, t

        #: Opcodes for which an ``op`` attribute is part of the modelled semantics, so seeing one is not a
        #: sign the buffer is saying something this encoder cannot read. VREDUCE selects its reduction with
        #: it; VECTOR_MAP is handled by :meth:`_combine_of`, which refuses rather than guesses.

    _OP_ATTR_IS_MODELLED = frozenset({"VREDUCE", "VECTOR_MAP"})

    def _refuse_unmodelled_step(self, i: int, opcode: str, attrs: dict) -> None:
        """Abstain when a command carries an ``op`` step this encoder does not model.

        FOUND BY ADJUDICATING A CONTRADICTION, and it is the worst outcome in the vocabulary: a false
        VERIFIED. Some submissions write the buffer at RoCC instruction level -- the semantic opcode is
        reused across several commands and the real step lives in an ``op`` attribute
        (``config_ex`` / ``config_ld`` / ``mvin`` / ``preload`` / ``compute_preloaded`` / ``config_st``
        / ``mvout`` / ``flush``). This encoder read none of those, so three RES_PACKs collapsed to one,
        two COMMITs to one, and the result happened to be the function the specification asked for.
        Verdict: verified. On hardware the same program saturated every output at 127 -- INT8_MAX --
        against a golden expecting i32, because the store width came from a config step nothing here
        looked at.

        The COMMIT in that buffer also carried no ``output_dtype``, so the encoder supplied its own
        default and that default is what matched the spec. A default must never be what decides a
        verdict: it is the checker agreeing with itself.
        """
        if opcode in self._OP_ATTR_IS_MODELLED or "op" not in attrs:
            return
        raise UnsupportedSemantics(
            f"command {i} ({opcode}) carries op={attrs['op']!r}, a step this encoder does not model. "
            f"Buffers written at instruction level reuse one semantic opcode across several commands "
            f"and put the real step here; ignoring it collapses them into one operation and can report "
            f"a program VERIFIED that saturates on hardware"
        )

    def _operand(self, operands: dict, *keys: str, opcode: str):
        """Read the first spelling of a REQUIRED operand that this buffer supplies, or abstain.

        Two defects met here. A bare ``operands["lhs"]`` raised ``KeyError``, which the ablation
        counted as a checker ERROR — our defect, and indistinguishable from a crash in the solver.
        A missing or differently-spelled operand is a property of the SUBMITTED buffer, so it is an
        abstention that names both what was wanted and what the buffer actually carries.

        The alias list is not invention: the version-1 kernel ABI (``legacy/kernel_abi_v1.yaml``,
        ``movement_src``) defines the movement source as "the MOVEMENT/identity-VECTOR_MAP command's
        ``src`` (or ``lhs``) operand", and
        neither engine implemented the documented alternative, so a buffer written to the contract
        crashed the checker rather than being checked.
        """
        for k in keys:
            if k in operands:
                return operands[k]
        wanted = " or ".join(repr(k) for k in keys)
        raise UnsupportedSemantics(f"{opcode} needs operand {wanted}; this buffer supplies {sorted(operands)}")

    def _combine_of(self, attrs: dict, opcode: str) -> str:
        """The VECTOR_MAP combine, refusing to GUESS when the buffer spells it somewhere else.

        Some submissions carry ``op: "identity"`` / ``op: "copy"`` where the engines read ``combine``.
        ``attrs.get("combine", "add")`` then silently answered "add" — and that is the dangerous
        reading, not the crash it happened to cause here: a buffer carrying ``op: "identity"`` plus an
        ``rhs`` would have been encoded as an ADDITION, agreed with by a reference that defaults the
        same way, and reported VERIFIED while the hardware moved data. The schema types ``op`` as a
        free string (``command_buffer.schema.json:203``) and nothing defines it for VECTOR_MAP, so the
        only safe reading of an unexplained ``op`` is no reading at all.
        """
        if "combine" in attrs:
            return str(attrs["combine"])
        if "op" in attrs:
            raise UnsupportedSemantics(
                f"{opcode} carries attribute op={attrs['op']!r} and no 'combine'; the engines read "
                f"'combine' and nothing defines 'op' for this opcode, so defaulting to 'add' could "
                f"silently encode an addition where the buffer meant {attrs['op']!r}"
            )
        return "add"

    def _elementwise(self, a: Tensor, b: Tensor, op_cls) -> Tensor:
        out = {k: op_cls(v, b.elems[k]).results[0] for k, v in a.elems.items()}
        return Tensor(a.rows, a.cols, a.width, out)

    def _narrow(self, t: Tensor, attrs: dict, *, default: str) -> Tensor:
        """The readout narrow, mirroring ``_narrow_int_readout`` in both engines.

        SIGNEDNESS COMES FROM THE SPELLING, not from an assumption. This saturated to the signed range
        for every width, so a ``u8`` readout was clamped to [-128, 127] where all three engines clamp
        it to [0, 255] -- the encoder disagreeing with its own oracle, which refutes a correct backend.
        The engines derive the range from the ``i``/``u`` prefix; so does this now.

        The default is passed in rather than assumed because it is a property of the opcode, and the
        docstring here used to say COMMIT defaulted to ``i8``. It does not any more: every narrowing
        opcode defaults to ``i32`` (absent means do not narrow), and `validate_command_buffer` refuses
        an absent declaration outright, so the default is a backstop rather than a policy.
        """
        rng = _narrow_range(str(attrs.get("output_dtype", default)))
        if rng is None:
            return t
        lo, hi, bits = rng
        return self.enc.saturate(t, lo, hi, bits)

        # -- helpers ---------------------------------------------------------------------------------

    def _bias_name(self, operands: dict, attrs: dict, dst: str) -> str:
        """Mirror ``commandbuffer.bias_tensor_name``: attributes first, then operands, else raise."""
        for source in (attrs, operands):
            name = source.get("bias")
            if name:
                return str(name)
        raise UnsupportedSemantics(
            f"COMMIT {dst!r} lists a bias epilogue stage but names no bias tensor in either its "
            f"attributes or its operands"
        )

    def _get(self, name: str) -> Tensor:
        try:
            return self.env[name]
        except KeyError:
            raise UnsupportedSemantics(
                f"command references tensor {name!r}, which is neither declared nor produced by an earlier command"
            ) from None


def _why_not_encodable(index: int, op: str) -> str:
    """Say WHICH class an unencodable opcode fell into — 'unknown' is not an actionable diagnostic."""
    if op in FLOAT_ONLY_OPCODES:
        return (
            f"command {index} uses {op!r}, which the reference simulator itself computes in "
            f"float. A bit-exact check on a float datapath is the wrong specification, not "
            f"merely an expensive one: reassociation is a legal backend choice, so it would "
            f"reject correct backends. Abstaining."
        )
    if op in DEFERRED_OPCODES:
        return (
            f"command {index} uses {op!r}: {DEFERRED_OPCODES[op]}. This is a gap in THIS encoder, "
            f"not a defect in the buffer."
        )
    if op in UNIMPLEMENTED_OPCODES:
        return (
            f"command {index} uses {op!r}, which is in the command-buffer schema's enum but has "
            f"no branch in the reference simulator at all — there is nothing here to mirror."
        )
    return (
        f"command {index} uses opcode {op!r}, which this encoder has no definition for. "
        f"Encodable: {sorted(ENCODABLE_OPCODES)}. Refusing rather than skipping it — a skipped "
        f"command silently changes what the query is about."
    )


def _narrow_range(dtype: str) -> tuple[int, int, int] | None:
    """``(lo, hi, bits)`` to saturate to, or None when nothing narrows — mirroring the engines.

    Derived entirely from the spelling rather than tested against "i8", so a target narrowing to
    i16/i4/u8 is handled instead of silently passed through, and an UNSIGNED container gets the
    unsigned range instead of the signed one. A width at or above the accumulator has nothing to
    narrow; a non-integer spelling is not this function's business (floats are refused earlier, and
    both engines pass one through unchanged).
    """
    if not dtype or dtype[0] not in ("i", "u") or not dtype[1:].isdigit():
        return None
    bits, signed = int(dtype[1:]), dtype[0] == "i"
    if bits >= 32:
        return None
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1, bits
    return 0, (1 << bits) - 1, bits


def _width_of(dtype: str, name: str) -> int:
    from merlin.verification.translation.smt_encoding import _INT_WIDTH

    try:
        return _INT_WIDTH[dtype]
    except KeyError:
        raise UnsupportedSemantics(
            f"tensor {name!r} has dtype {dtype!r}, which has no bitvector encoding here. Float "
            f"datapaths are refused: reassociation is a legal backend choice, so a bit-exact float "
            f"check would reject correct backends."
        ) from None


def encode_command_buffer(
    enc: Encoder, cb: dict[str, Any], *, shared: dict[str, Tensor] | None = None, acc_width: int = 32
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Encode ``cb``; return ``(declared outputs, the leaf tensors used)``."""
    cbe = CommandBufferEncoder(enc, cb, acc_width=acc_width)
    leaves = cbe.declare_leaves(shared)
    return cbe.run(), leaves

    # Preserve historical object lookup; implementation and completion state live here.


# ---- linalg_semantics --------------------------------------------------------------------------
# A semantics for the ``linalg`` SOURCE module, given by the same lowering-to-``smt`` idea.
#
# This is the source half of translation validation. :mod:`merlin.verification.translation.smt_encoding` encodes the
# ``interface`` program a pass *produced*; this module encodes the ``linalg`` program that pass
# *consumed*. With both, :func:`merlin.verification.translation.refinement.validate_pass` states a per-compilation theorem
# about the pass itself — *this source and this output compute the same function on every integer
# input at this shape* — rather than the weaker target-vs-respecification claim that comes from
# re-deriving what the pass "should have done" from the function signature.
#
# **Why the distinction is not pedantry.** A specification re-derived from the signature shares the
# compiler author's mental model of the workload. If that model is wrong — the workload is not actually
# ``A @ W``, or the zero points are not actually zero — a re-derived spec and a buggy pass agree, and
# the query returns ``unsat`` while the program is wrong. Encoding the source IR removes the author
# from the loop: the only inputs are the two artifacts.
#
# **Scope.** Same as the interface encoder, and for the same reasons: rank-2 integer tensors, concrete
# extents taken from the IR's own types (so every query stays quantifier-free), no poison, no memory
# model, no float. Anything outside it raises :class:`UnsupportedSemantics` naming the construct —
# never skipped, because a skipped op is a silently weakened theorem.
#
# An accumulator initialized directly from ``tensor.empty`` is *not* in this domain. MLIR specifies
# its contents as unspecified, and a contraction reads its ``outs`` accumulator. Treating that value as
# zero would prove equivalence to an invented source program. An explicit ``linalg.fill`` of an integer
# constant defines every element and is encoded; the usual zero fill makes the source-to-interface
# claim unconditional at the supported shape.
#
# **Fail closed on zero points in particular.** ``linalg.quantized_matmul`` computes
# ``out[m][n] += (a[m][k] - zp_a) * (w[k][n] - zp_w)``. Reading the zero points off the actual operands
# is the whole point: assuming they are zero would reintroduce the author's model through the back
# door, and would silently validate a pass that dropped a *non-zero* zero point — the exact defect the
# op exists to express. A zero point that is not a resolvable integer constant is an ABSTENTION.
#
# One ``linalg.generic`` subset is encoded: rank-2 signed i8 operands, i32 accumulation, ordinary
# matmul affine maps and iterators, and a body that sign-extends, multiplies, adds the accumulator,
# and yields the sum. Every structural part is checked; provenance tags are not evidence of semantics.

#: Top-level ops with a value semantics here. ``linalg.generic`` is conditional on its exact body,
#: maps, iterators, and types; this is a candidate inventory, not a promise about all generics.
ENCODABLE_OPS = frozenset(
    {
        "linalg.quantized_matmul",
        "linalg.matmul",
        "linalg.generic",
        "linalg.fill",
        "tensor.empty",
        "arith.constant",
        "func.return",
    }
)


def _const_int(op) -> int:
    """The integer an ``arith.constant`` carries, or raise.

    Read structurally off the value attribute rather than off the printed form: a float or an index
    constant has no bitvector reading here and must abstain instead of being coerced.
    """
    attr = op.properties.get("value")
    if attr is None:
        attr = op.attributes.get("value")
    value = getattr(attr, "value", None)
    data = getattr(value, "data", None)
    if not isinstance(data, int) or isinstance(data, bool):
        raise UnsupportedSemantics(f"arith.constant carries {attr!r}, which has no integer bitvector reading here")
    return data


class _Uninitialized:
    """Marker for ``tensor.empty``: a shape with unspecified contents, never implicit zero."""

    __slots__ = ("rows", "cols", "width")

    def __init__(self, rows: int, cols: int, width: int) -> None:
        self.rows, self.cols, self.width = rows, cols, width


def _sub_const(enc: Encoder, term, value: int, width: int):
    """``term - value`` at ``width`` bits, via addition of the negated constant.

    The ``smt`` dialect xDSL ships has no subtract op, and the zero point is a known integer, so the
    subtraction is done as ``term + (-value)`` — exact in two's complement, mod 2**width, which is
    exactly what ``arith.subi`` on an ``i<width>`` does.
    """
    if not -(2 ** (width - 1)) <= value < 2 ** (width - 1):
        raise UnsupportedSemantics(f"zero point {value} is not a signed {width}-bit value")
    return enc.smt.BVAddOp(term, enc.const((-value) & ((1 << width) - 1), width)).results[0]


def _contract(enc: Encoder, lhs: Tensor, rhs: Tensor, zp_lhs: int, zp_rhs: int, acc_width: int) -> Tensor:
    """``out[m][n] = sum_k (lhs[m][k] - zp_lhs) * (rhs[k][n] - zp_rhs)``.

    Two paths, because the cheap one is only *exact* when the zero points vanish.

    * Both zero points zero: this is a plain contraction, so :meth:`Encoder.matmul` applies, and it
      multiplies at twice the OPERAND width rather than at the accumulator width. That is exact (a
      product of two w-bit signed values fits in 2w bits) and it is the difference between a query
      that refutes and one that never returns — bit-blasted multiplier area scales as width**2.
    * Either zero point non-zero: the subtraction can leave the operand's own width, so the whole
      computation is done at ``acc_width``, mirroring what ``linalg`` itself does (extend, subtract,
      multiply and accumulate in the result's integer type, wrapping). Exact, and much more
      expensive; the shape has to be correspondingly smaller for a refutation to land.
    """
    if lhs.cols != rhs.rows:
        raise UnsupportedSemantics(f"contraction extent mismatch: {lhs.rows}x{lhs.cols} @ {rhs.rows}x{rhs.cols}")
    if zp_lhs == 0 and zp_rhs == 0:
        return enc.matmul(lhs, rhs, acc_width=acc_width)

    lhs_w = {
        (m, k): _sub_const(enc, enc.sign_extend(lhs.at(m, k), lhs.width, acc_width), zp_lhs, acc_width)
        for m in range(lhs.rows)
        for k in range(lhs.cols)
    }
    rhs_w = {
        (k, n): _sub_const(enc, enc.sign_extend(rhs.at(k, n), rhs.width, acc_width), zp_rhs, acc_width)
        for k in range(rhs.rows)
        for n in range(rhs.cols)
    }
    out: dict[tuple[int, int], Any] = {}
    for m in range(lhs.rows):
        for n in range(rhs.cols):
            acc = None
            for k in range(lhs.cols):
                prod = enc.smt.BVMulOp(lhs_w[(m, k)], rhs_w[(k, n)]).results[0]
                acc = prod if acc is None else enc.smt.BVAddOp(acc, prod).results[0]
            out[(m, n)] = acc if acc is not None else enc.const(0, acc_width)
    return Tensor(lhs.rows, rhs.cols, acc_width, out)


def _add_init(enc: Encoder, acc: Tensor, init) -> Tensor:
    """Fold the destination-passing ``outs`` operand into the contraction result."""
    if isinstance(init, _Uninitialized):
        raise UnsupportedSemantics(
            "contraction accumulates into tensor.empty with unspecified contents; "
            "an explicit linalg.fill of zero is required for an unconditional equivalence proof"
        )
    if (init.rows, init.cols) != (acc.rows, acc.cols):
        raise UnsupportedSemantics(f"init is {init.rows}x{init.cols} but the contraction is {acc.rows}x{acc.cols}")
    if init.width != acc.width:
        raise UnsupportedSemantics(
            f"init is {init.width} bits but the accumulator is {acc.width}; refusing to widen an "
            f"initialised accumulator implicitly"
        )
    out = {k: enc.smt.BVAddOp(v, init.at(*k)).results[0] for k, v in acc.elems.items()}
    return Tensor(acc.rows, acc.cols, acc.width, out)


def _generic_matmul(op, acc_width: int) -> tuple:
    """Use the shared parsed-body recognizer before constructing the SMT reading."""
    from merlin.compiler.frontend.patterns.linalg_patterns import (
        InvalidLinalgPattern,
        recognize_signed_i8_i32_matmul,
    )

    if acc_width != 32:
        raise UnsupportedSemantics("linalg.generic signed i8 matmul is encoded only with i32 accumulation")
    try:
        return recognize_signed_i8_i32_matmul(op)
    except InvalidLinalgPattern as exc:
        raise UnsupportedSemantics(str(exc)) from exc


def encode_linalg(enc: Encoder, module, *, acc_width: int = 32) -> Encoded:
    """Interpret a ``linalg`` source module, returning its returned values and its symbolic inputs.

    This is the SPECIFICATION side of source-to-target validation, so it declares the leaves: the
    source module's block arguments are the program's inputs, and the target encoding is bound onto
    them (``encode_interface(..., shared=...)``). Two sides encoded over independent symbols would
    make the refinement query trivially satisfiable, so which side declares the leaves is not an
    implementation detail — it is what makes the result mean anything.

    ``inputs`` lists the RANK-2 TENSOR block arguments in order. A scalar argument (a runtime zero
    point is exactly this) is legal ``linalg`` but is not a tensor input; it gets no symbolic tensor
    and is not a constant, so anything that reads it abstains rather than guessing a value.

    Outputs are keyed ``ret0, ret1, ...`` in ``func.return`` operand order, which is the order the
    function signature declares and therefore the only correspondence a consumer can rely on.
    """
    env: dict[Any, Any] = {}
    consts: dict[Any, int] = {}
    outputs: dict[str, Tensor] = {}
    inputs: list[Tensor] = []

    funcs = [op for op in module.walk() if op.name == "func.func"]
    if len(funcs) != 1:
        raise UnsupportedSemantics(f"expected exactly one func.func; found {len(funcs)}")
    func = funcs[0]
    if len(func.body.blocks) != 1:
        raise UnsupportedSemantics(f"source function has {len(func.body.blocks)} blocks; only one is encoded")

    block = func.body.block
    operations = list(block.ops)
    if not operations or operations[-1].name != "func.return":
        raise UnsupportedSemantics("source function must end in func.return")
    for i, arg in enumerate(block.args):
        if getattr(arg.type, "get_shape", None) is None:
            # A scalar leaf is legal linalg (a runtime zero point is exactly this) but it is not a
            # tensor input, so it gets no symbolic tensor and is NOT a constant. Anything that reads
            # it must abstain, which is what the zero-point resolution below does.
            continue
        rows, cols = _shape(arg.type)
        env[arg] = enc.symbolic_tensor(f"arg{i}", rows, cols, _elem_width(arg.type), acc_width)
        inputs.append(env[arg])

    def _zero_point(value, which: str) -> int:
        if value not in consts:
            raise UnsupportedSemantics(
                f"the {which} zero point of a quantized contraction is not a resolvable integer "
                f"constant (it is produced by "
                f"{getattr(getattr(value, 'owner', None), 'name', 'a block argument')!r}); "
                f"abstaining rather than assuming it is zero"
            )
        return consts[value]

    def _tensor(value, which: str) -> Tensor:
        t = env.get(value)
        if isinstance(t, _Uninitialized):
            raise UnsupportedSemantics(
                f"the {which} operand of a contraction is an uninitialised tensor.empty; it has no value to contract"
            )
        if t is None:
            raise UnsupportedSemantics(f"the {which} operand of a contraction is undefined here")
        return t

    for op in block.ops:
        name = op.name
        if name == "arith.constant":
            consts[op.results[0]] = _const_int(op)
        elif name == "tensor.empty":
            rows, cols = _shape(op.results[0].type)
            env[op.results[0]] = _Uninitialized(rows, cols, _elem_width(op.results[0].type))
        elif name == "linalg.fill":
            if len(op.operands) != 2 or len(op.results) != 1:
                raise UnsupportedSemantics("linalg.fill must have one scalar, one destination, and one result")
            scalar, destination = op.operands
            if scalar not in consts:
                raise UnsupportedSemantics("linalg.fill scalar is not a resolvable integer constant")
            if destination not in env:
                raise UnsupportedSemantics("linalg.fill destination is undefined here")
            rows, cols = _shape(op.results[0].type)
            width = _elem_width(op.results[0].type)
            dest = env[destination]
            if (dest.rows, dest.cols, dest.width) != (rows, cols, width):
                raise UnsupportedSemantics("linalg.fill result and destination have different shape or element type")
            if str(scalar.type) != str(op.results[0].type.element_type):
                raise UnsupportedSemantics(
                    f"linalg.fill scalar type {scalar.type} differs from output element type "
                    f"{op.results[0].type.element_type}"
                )
            literal = enc.const(consts[scalar], width)
            env[op.results[0]] = Tensor(rows, cols, width, {(r, c): literal for r in range(rows) for c in range(cols)})
        elif name in ("linalg.quantized_matmul", "linalg.matmul", "linalg.generic"):
            res_width = _elem_width(op.results[0].type)
            if res_width != acc_width:
                raise UnsupportedSemantics(
                    f"{name} returns an i{res_width} tensor but the accumulator is i{acc_width}; "
                    f"refusing rather than silently re-widening the contraction"
                )
            if name == "linalg.generic":
                lhs, rhs, init_value = _generic_matmul(op, acc_width)
                zp_lhs = zp_rhs = 0
                init = env.get(init_value)
            elif name == "linalg.quantized_matmul":
                if len(op.operands) != 5:
                    raise UnsupportedSemantics(
                        f"linalg.quantized_matmul with {len(op.operands)} operands; expected "
                        f"lhs, rhs, zp_lhs, zp_rhs, init"
                    )
                zp_lhs = _zero_point(op.operands[2], "lhs")
                zp_rhs = _zero_point(op.operands[3], "rhs")
                init = env.get(op.operands[4])
                lhs, rhs = op.operands[:2]
            else:
                if len(op.operands) != 3:
                    raise UnsupportedSemantics(
                        f"linalg.matmul with {len(op.operands)} operands; expected lhs, rhs, init"
                    )
                zp_lhs = zp_rhs = 0
                init = env.get(op.operands[2])
                lhs, rhs = op.operands[:2]
            if init is None:
                raise UnsupportedSemantics(f"{name} init operand is undefined here")
            acc = _contract(enc, _tensor(lhs, "lhs"), _tensor(rhs, "rhs"), zp_lhs, zp_rhs, acc_width)
            env[op.results[0]] = _add_init(enc, acc, init)
        elif name == "func.return":
            for i, value in enumerate(op.operands):
                outputs[f"ret{i}"] = _tensor(value, f"return operand {i}")
        else:
            raise UnsupportedSemantics(
                f"no semantics for {name!r} in a linalg source module (encodable: {sorted(ENCODABLE_OPS)})"
            )

    if not outputs:
        raise UnsupportedSemantics("the source module returns no tensors; nothing to validate")
    return Encoded(outputs=outputs, inputs=inputs)

    # Preserve historical object lookup; implementation and completion state live here.


# ---- merlin_iface_semantics --------------------------------------------------------------------
# A deliberately narrow value semantics for the frozen ``merlin_iface`` text grammar.
#
# The OOT compiler consumes/produces this contract text, while the in-tree proof interpreter reads
# ``interface.*`` xDSL ops. Convert the custom assembly to MLIR's generic form, parse the resulting
# SSA graph, then interpret only tensor leaves, resident pack, matmul, commit, and evict. The generic
# form conversion is spelling only; every parsed operation and attribute is checked here. A new
# mnemonic, an unknown attribute, or an arbitrary leaf alias abstains.
#
# This proves integer VALUES at a concrete shape. ``resident_pack`` is value preserving in this
# algebra; physical layout, DMA, and device execution require separate checks.


def parse_merlin_iface(text: str):
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.parser import Parser

    from merlin.target.contract.interfaces.grammar import to_generic_form

    try:
        generic = to_generic_form(text)
        ctx = Context(allow_unregistered=True)
        ctx.load_dialect(Builtin)
        return Parser(ctx, generic).parse_module()
    except Exception as exc:
        # xDSL parser errors are checker abstentions. The original text is still hashed in the
        # receipt, so a failed conversion cannot be mistaken for the successfully parsed artifact.
        raise UnsupportedSemantics(f"merlin_iface text cannot be parsed completely: {exc}") from exc


def _string_attr(attrs: dict, key: str) -> str:
    from xdsl.dialects.builtin import StringAttr

    value = attrs.get(key)
    if not isinstance(value, StringAttr):
        raise UnsupportedSemantics(f"merlin_iface {key!r} must be a string attribute")
    return value.data


def _kind(op) -> str:
    return _string_attr(op.attributes, "op_name__")


def _only_attrs(op, allowed: set[str]) -> None:
    unknown = set(op.attributes) - allowed - {"op_name__"}
    if unknown or op.properties or op.regions:
        raise UnsupportedSemantics(f"{_kind(op)} has unencoded attributes/properties/regions: {sorted(unknown)}")


def merlin_iface_signature(text: str) -> dict[str, Any]:
    """Typed declaration summary for receipts; the semantic walk remains the authority."""
    module = parse_merlin_iface(text)
    leaves = []
    outputs = []
    for op in module.body.block.ops:
        name = _kind(op)
        if name == "merlin_iface.tensor" and len(op.results) == 1:
            leaves.append({"name": _string_attr(op.attributes, "name"), "type": str(op.results[0].type)})
        elif name == "merlin_iface.commit" and len(op.results) == 1:
            outputs.append({"name": _string_attr(op.attributes, "name"), "type": str(op.results[0].type)})
    return {"grammar": "merlin_iface", "inputs": leaves, "outputs": outputs}


def encode_merlin_iface(enc: Encoder, module, *, shared: list[Tensor], acc_width: int = 32) -> Encoded:
    """Interpret actual ``merlin_iface`` SSA, binding ``name=argN`` to source argument N."""
    from xdsl.dialects.builtin import ArrayAttr

    if set(module.attributes) != {"merlin_iface.version", "merlin_iface.target", "merlin_iface.abi_version"}:
        raise UnsupportedSemantics("merlin_iface module metadata differs from the frozen grammar")
    if _string_attr(module.attributes, "merlin_iface.version") != "0.1":
        raise UnsupportedSemantics("unsupported merlin_iface grammar version")
    _string_attr(module.attributes, "merlin_iface.target")
    _string_attr(module.attributes, "merlin_iface.abi_version")
    if len(module.body.blocks) != 1:
        raise UnsupportedSemantics("merlin_iface module must have one block")
    if module.properties or module.body.block.args:
        raise UnsupportedSemantics("merlin_iface module has unencoded properties or block arguments")

    env: dict[Any, Tensor] = {}
    resident: set[Any] = set()
    declared: set[int] = set()
    outputs: dict[str, Tensor] = {}
    for op in module.body.block.ops:
        name = _kind(op)
        if name == "merlin_iface.tensor":
            _only_attrs(op, {"name", "role"})
            if op.operands or len(op.results) != 1:
                raise UnsupportedSemantics("merlin_iface.tensor must declare one leaf value")
            label = _string_attr(op.attributes, "name")
            if _string_attr(op.attributes, "role") not in ("input", "weight"):
                raise UnsupportedSemantics("merlin_iface.tensor role is not an input or weight")
            if not label.startswith("arg") or not label[3:].isdigit() or label != f"arg{int(label[3:])}":
                raise UnsupportedSemantics(f"merlin_iface leaf {label!r} has no source argument binding")
            index = int(label[3:])
            if index >= len(shared) or index in declared:
                raise UnsupportedSemantics(f"merlin_iface leaf {label!r} is duplicate or outside source arguments")
            shape = _shape(op.results[0].type)
            width = _elem_width(op.results[0].type)
            leaf = shared[index]
            if (leaf.rows, leaf.cols, leaf.width) != (*shape, width):
                raise UnsupportedSemantics(f"merlin_iface leaf {label!r} type disagrees with source argument")
            env[op.results[0]] = leaf
            declared.add(index)
        elif name == "merlin_iface.resident_pack":
            _only_attrs(op, {"layout"})
            if len(op.operands) != 1 or len(op.results) != 1:
                raise UnsupportedSemantics("merlin_iface.resident_pack expects one tensor and one handle")
            _string_attr(op.attributes, "layout")
            if (
                str(op.results[0].type) != "!merlin_iface.resident"
                or op.operands[0] not in env
                or not str(op.operands[0].type).startswith("tensor<")
            ):
                raise UnsupportedSemantics("merlin_iface.resident_pack has an unbound source or wrong result type")
            env[op.results[0]] = env[op.operands[0]]
            resident.add(op.results[0])
        elif name == "merlin_iface.matmul":
            _only_attrs(op, set())
            if len(op.operands) != 2 or len(op.results) != 1:
                raise UnsupportedSemantics("merlin_iface.matmul expects two operands and one accumulator")
            lhs, rhs = op.operands
            if (
                lhs not in env
                or rhs not in resident
                or not str(lhs.type).startswith("tensor<")
                or str(rhs.type) != "!merlin_iface.resident"
                or str(op.results[0].type) != f"!merlin_iface.acc<i{acc_width}>"
            ):
                raise UnsupportedSemantics("merlin_iface.matmul has an unbound operand or wrong accumulator type")
            env[op.results[0]] = enc.matmul(env[lhs], env[rhs], acc_width=acc_width)
        elif name == "merlin_iface.commit":
            _only_attrs(op, {"name", "epilogue", "output_dtype"})
            if (
                len(op.operands) != 1
                or len(op.results) != 1
                or op.operands[0] not in env
                or str(op.operands[0].type) != f"!merlin_iface.acc<i{acc_width}>"
            ):
                raise UnsupportedSemantics("merlin_iface.commit has an unbound accumulator or wrong arity")
            epilogue = op.attributes.get("epilogue")
            if not isinstance(epilogue, ArrayAttr) or epilogue.data:
                raise UnsupportedSemantics("merlin_iface.commit epilogue is not an empty list")
            if _string_attr(op.attributes, "output_dtype") != f"i{acc_width}":
                raise UnsupportedSemantics("merlin_iface.commit output dtype is not the accumulator dtype")
            value = env[op.operands[0]]
            if (*_shape(op.results[0].type), _elem_width(op.results[0].type)) != (value.rows, value.cols, value.width):
                raise UnsupportedSemantics("merlin_iface.commit result type disagrees with the accumulator")
            label = _string_attr(op.attributes, "name")
            if label in outputs:
                raise UnsupportedSemantics(f"merlin_iface.commit output {label!r} appears twice")
            outputs[label] = value
            env[op.results[0]] = value
        elif name == "merlin_iface.evict":
            _only_attrs(op, set())
            if len(op.operands) != 1 or op.results or op.operands[0] not in resident:
                raise UnsupportedSemantics("merlin_iface.evict references no live resident handle")
            resident.remove(op.operands[0])
        else:
            raise UnsupportedSemantics(f"no value semantics for {name!r} in merlin_iface")
    if declared != set(range(len(shared))):
        raise UnsupportedSemantics("merlin_iface does not declare every source argument exactly once")
    if len(outputs) != 1:
        raise UnsupportedSemantics(
            f"merlin_iface value proof requires exactly one committed output; found {len(outputs)}"
        )
    return Encoded(outputs=outputs, inputs=shared)

    # Preserve historical object lookup; implementation and completion state live here.
