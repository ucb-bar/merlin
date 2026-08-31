"""Execute a Merlin command buffer with real integer arithmetic.

This is the Merlin-owned simulator harness. It interprets the target-independent command
buffer, performs the actual tensor math (via :mod:`merlin.runtime.tensor`), records real
metrics and a trace, and returns the committed output tensors. The ToyNPU opcodes
(RES_PACK / MATMUL_RESIDENT / COMMIT / EVICT) are handled directly; the design generalizes to
target-supplied semantics.

Every whole-op opcode here (RMSNORM / SOFTMAX / GELU / SOFTCAP / ROPE / ATTENTION_QK / ATTENTION_PV /
CONV2D / MOVEMENT / …) implements the SAME definition as the golden engine's op of that name
(:mod:`merlin.targetgen.capsule_golden`), so the functional tier and the numeric oracle cannot
disagree about what a conformant command buffer means. An opcode with no definition here raises
``SimulationError`` rather than being skipped.
"""
from __future__ import annotations

import math
from typing import Any

from .commandbuffer import (apply_pool_stage, conv_im2col, conv_out_dims,
                            materialize_inputs, pool_params, validate_command_buffer)
from .metrics import Metrics
from .tensor import Tensor


class SimulationError(RuntimeError):
    pass


#: The ONLY attributes ``CONV2D`` implements. Anything else raises, naming it — a conv whose stride
#: or dilation was quietly ignored produces a full, plausible-looking tensor of wrong numbers, which
#: is the failure mode the whole fail-closed interface grammar exists to prevent. Widening the
#: supported subset means adding the key here AND the code that honours it, in that order.
_CONV2D_ATTRS = frozenset({"kernel", "stride", "padding", "dilation", "layout",
                           "epilogue", "output_dtype", "acc_scale", "requant_shift", "semantic",
                           # Pooling fused onto the conv's store path (the fused conv loop's
                           # pool_size/pool_stride/pool_padding, configured by config_st). Listed here
                           # AND honoured in the CONV2D branch below -- the rejection above is what
                           # forces those two to land together, so a pool parameter can never arrive
                           # and be quietly ignored.
                           "pool_in_dims", "pool_size", "pool_stride", "pool_padding",
                           "pool_pad_value"})


def _pool(t: Tensor, stage: str, attrs: dict[str, Any], op: str) -> Tensor:
    """Apply a pooling epilogue stage, translating the runtime's ``ValueError`` into this engine's
    ``SimulationError``. The arithmetic and the attribute parsing stay in
    :func:`merlin.runtime.commandbuffer.apply_pool_stage` -- the single definition the golden and the
    reference call too -- so this wrapper only reconciles the exception type callers of ``simulate``
    are documented to catch."""
    try:
        return apply_pool_stage(t, stage, attrs, op=op)
    except ValueError as e:
        raise SimulationError(str(e)) from e


def _conv_geom(attrs: dict[str, Any], key: str, arity: int, default: list[int]) -> tuple[int, ...]:
    """A conv geometry vector (``stride`` / ``padding`` / ``dilation``) as ``arity`` ints.

    A vector of the wrong length is REJECTED rather than padded or truncated: ``padding`` is
    ``[top, left, bottom, right]``, and silently reading a 2-element ``[h, w]`` as ``[top, left]``
    with zero bottom/right would change the output geometry without saying so.
    """
    v = attrs.get(key, default)
    if not isinstance(v, list) or len(v) != arity:
        raise SimulationError(
            f"CONV2D {key} must be a list of {arity} int(s) (got {v!r})")
    return tuple(int(x) for x in v)


def _narrow_int_readout(t: Tensor, dtype: str, op: str) -> Tensor:
    """Saturating readout of the i32 accumulator into the command's DECLARED integer output dtype.

    Same definition as the golden engine's ``_narrow_to_dtype`` (bitwidth parsed structurally from the
    token, ``i8`` routed through :meth:`Tensor.to_i8` so it stays byte-identical to the historical
    path, width >= 32 read out whole), so the functional tier and the numeric oracle cannot disagree
    about a narrowed conv. A NON-integer dtype raises instead of passing through: this engine's matmul
    is integer, so a float readout it "supported" by ignoring would be a wrong answer.
    """
    kind, digits = (dtype[:1], dtype[1:]) if dtype else ("", "")
    if kind not in ("i", "u") or not digits.isdigit():
        raise SimulationError(
            f"{op} output_dtype {dtype!r} is not an integer dtype; this integer engine has no "
            f"definition for it")
    bits, signed = int(digits), kind == "i"
    if bits >= 32:
        return t
    if dtype == "i8":
        return t.to_i8()
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    return Tensor(t.shape, [lo if x < lo else (hi if x > hi else x) for x in t.data], dtype)


def simulate(cb: dict[str, Any], inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a command buffer. Returns {outputs, metrics, trace}.

    ``outputs`` maps committed tensor names to nested lists; ``metrics`` is the common metric
    dict; ``trace`` is an ordered list of event dicts.
    """
    problems = validate_command_buffer(cb)
    if problems:
        raise SimulationError("; ".join(problems))

    env: dict[str, Tensor] = materialize_inputs(cb, inputs)
    resident: dict[str, str] = {}        # handle -> source tensor name
    metrics = Metrics()
    trace: list[dict[str, Any]] = []
    outputs: dict[str, list] = {}

    default_shift = int(cb.get("params", {}).get("requant_shift", 4))

    for cmd in cb.get("commands", []):
        op = cmd["opcode"]
        ops = cmd.get("operands", {})
        attrs = cmd.get("attributes", {})
        metrics.command_count += 1

        if op == "RES_PACK":
            src, dst = ops["src"], ops["dst"]
            t = env[src]
            if "scale" in ops:                # int8 weight-only: dequantize per channel at pack time
                t = t.dequant_per_channel(env[ops["scale"]], int(attrs.get("dequant_axis", 1)))
            env[dst] = t                      # packed copy (identical values, or dequantized weight)
            resident[dst] = src
            metrics.pack_count += 1
            metrics.bytes_moved += t.nbytes
            metrics.bytes_read += t.nbytes
            metrics.cycles += len(t.data)
            trace.append({"name": "resident_pack", "object": dst, "count": 1})

        elif op in ("MATMUL_RESIDENT", "MATMUL"):
            lhs, rhs, dst = ops["lhs"], ops["rhs"], ops["dst"]
            if rhs in resident:
                metrics.resident_hits += 1
            else:
                # A non-resident RHS is re-fetched on every dispatch; charge the
                # fetch exactly like RES_PACK does (bytes + one cycle per element).
                # This is what makes residency exploitable at sufficient reuse.
                metrics.resident_misses += 1
                metrics.bytes_read += env[rhs].nbytes
                metrics.bytes_moved += env[rhs].nbytes
                metrics.cycles += len(env[rhs].data)
            acc = env[lhs].matmul(env[rhs])
            env[dst] = acc
            m, k = env[lhs].shape
            _, n = env[rhs].shape
            metrics.cycles += base_matmul_cycles(m, k, n)
            metrics.bytes_read += env[lhs].nbytes
            metrics.dispatch_count += 1
            trace.append({"name": "resident_hit" if rhs in resident else "resident_miss",
                          "object": rhs, "count": 1})

        elif op == "COMMIT":
            src, dst = ops["src"], ops["dst"]
            shift = int(attrs.get("requant_shift", default_shift))
            t = env[src]
            for stage in attrs.get("epilogue", []):
                if stage == "bias_add" or stage == "bias":
                    bias_name = ops.get("bias")
                    if bias_name is not None:
                        t = t.add_bias(env[bias_name])
                elif stage == "requant":
                    t = t.requant(shift)
                elif stage == "acc_scale":
                    t = t.requant_acc_scale(float(attrs.get("acc_scale", 1.0)))
                elif stage == "relu":
                    t = t.relu()
                elif stage == "maxpool":
                    # Pooling fused into the accumulator readout -- where the store-path ABI configures
                    # it (config_st's pool_stride/pool_size/pool_out_dim/porows/pocols/orows/ocols/
                    # upad/lpad, issued just before the mvout). The accumulator here is [M, N] with no
                    # spatial extent of its own, so the window and the extent its rows unflatten to come
                    # off THIS command's attributes, through the same parser the golden and the
                    # reference use -- one definition, so the three engines cannot disagree.
                    t = _pool(t, stage, attrs, f"COMMIT {dst!r}")
                else:
                    raise SimulationError(f"unknown epilogue stage '{stage}'")
            if attrs.get("output_dtype", "i8") == "i8":
                t = t.to_i8()
            env[dst] = t
            outputs[dst] = t.to_list()
            metrics.accumulator_commits += 1
            metrics.bytes_written += t.nbytes
            metrics.bytes_moved += t.nbytes
            metrics.cycles += len(t.data)
            trace.append({"name": "accumulator_commit", "object": dst, "count": 1})

        elif op == "EVICT":
            handle = ops["handle"]
            resident.pop(handle, None)
            metrics.evictions += 1
            metrics.cycles += 4
            trace.append({"name": "eviction", "object": handle, "count": 1})

        elif op == "VECTOR_MAP":
            # Elementwise combine of two equal-shape vectors + optional activation (SIMD family),
            # or an identity copy (combine="identity") for pure data movement.
            combine = attrs.get("combine", "add")
            dst = ops["dst"]
            if combine == "identity":
                a = env[ops["lhs"]]
                b = a
                t = Tensor(a.shape, list(a.data), a.dtype)
            else:
                a, b = env[ops["lhs"]], env[ops["rhs"]]
                t = a.ew_add(b) if combine == "add" else a.ew_mul(b)
            for stage in attrs.get("activation", []):
                if stage == "relu":
                    t = t.relu()
                else:
                    raise SimulationError(f"unknown vector activation '{stage}'")
            env[dst] = t
            metrics.bytes_read += a.nbytes + b.nbytes
            metrics.bytes_written += t.nbytes
            metrics.bytes_moved += a.nbytes + b.nbytes + t.nbytes
            metrics.cycles += len(t.data)
            trace.append({"name": "vector_map", "object": dst, "count": len(t.data)})

        elif op == "BATCHED_MATMUL":
            # O[b] = A[b] @ W[b] for a batch of independent 2-D matmuls (weight differs per batch, so no
            # residency reuse). Slice the flat (batch,m,k)/(batch,k,n) tensors per batch and matmul each.
            a, w, dst = env[ops["a"]], env[ops["w"]], ops["dst"]
            batch, m, kdim = a.shape
            _, _, n = w.shape
            out: list = []
            for bb in range(batch):
                asl = Tensor((m, kdim), a.data[bb * m * kdim:(bb + 1) * m * kdim], a.dtype)
                wsl = Tensor((kdim, n), w.data[bb * kdim * n:(bb + 1) * kdim * n], w.dtype)
                out.extend(asl.matmul(wsl).data)
            env[dst] = Tensor((batch, m, n), out, "i32")
            outputs[dst] = [[[out[bb * m * n + i * n + j] for j in range(n)]
                             for i in range(m)] for bb in range(batch)]
            metrics.dispatch_count += batch
            metrics.cycles += batch * base_matmul_cycles(m, kdim, n)
            trace.append({"name": "batched_matmul", "object": dst, "count": batch})

        elif op in ("RMSNORM", "SOFTMAX", "GELU", "SOFTCAP", "ROPE"):
            # Row-wise vector ops the backends emit as whole-op mnemonics. Without them a CONFORMANT
            # command buffer dies 'unknown opcode' in the functional tier and the capsule is scored as a
            # runner crash — a tooling gap masquerading as a submission failure. Semantics match the
            # golden engine's definitions of the same ops (same eps/theta/half-split conventions), so the
            # functional tier and the numeric oracle cannot disagree.
            src, dst = ops["src"], ops["dst"]
            t = env[src]
            rows, cols = t.shape
            src_rows = [[float(v) for v in t.data[r * cols:(r + 1) * cols]] for r in range(rows)]
            out_rows: list[list[float]] = []
            if op == "RMSNORM":
                gname = ops.get("gamma")
                if gname is None:
                    raise SimulationError("RMSNORM needs operands src/gamma/dst")
                g = [float(v) for v in env[gname].data]
                if len(g) != cols:
                    raise SimulationError(
                        f"RMSNORM gamma has {len(g)} element(s), expected {cols}")
                eps = float(attrs.get("eps", 1e-5))
                for row in src_rows:
                    inv = 1.0 / math.sqrt(sum(v * v for v in row) / cols + eps)
                    out_rows.append([v * inv * g[c] for c, v in enumerate(row)])
            elif op == "SOFTMAX":
                for row in src_rows:
                    mx = max(row)
                    ex = [math.exp(v - mx) for v in row]
                    s = sum(ex)
                    out_rows.append([e / s for e in ex])
            elif op == "GELU":
                # tanh-approximation GELU (the form the vector datapath implements).
                k = math.sqrt(2.0 / math.pi)
                for row in src_rows:
                    out_rows.append([0.5 * v * (1.0 + math.tanh(k * (v + 0.044715 * v * v * v)))
                                     for v in row])
            elif op == "SOFTCAP":
                cap = attrs.get("softcap", attrs.get("cap"))
                if cap is None:
                    raise SimulationError("SOFTCAP needs a 'softcap' attribute")
                cap = float(cap)
                for row in src_rows:
                    out_rows.append([cap * math.tanh(v / cap) for v in row])
            else:                                                     # ROPE
                if cols % 2:
                    raise SimulationError(
                        f"ROPE width {cols} is odd; the half-split convention needs an even width")
                half = cols // 2
                theta = float(attrs.get("rope_theta", attrs.get("theta", 10000.0)))
                freq = [theta ** (-(i / half)) for i in range(half)]
                for p, row in enumerate(src_rows):
                    cos = [math.cos(p * f) for f in freq]
                    sin = [math.sin(p * f) for f in freq]
                    out_rows.append([
                        row[c] * cos[c % half]
                        + (-row[c + half] if c < half else row[c - half]) * sin[c % half]
                        for c in range(cols)])
            res = Tensor((rows, cols), [v for row in out_rows for v in row], "f32")
            env[dst] = res
            outputs[dst] = res.to_list()
            metrics.bytes_read += t.nbytes
            metrics.bytes_written += res.nbytes
            metrics.bytes_moved += t.nbytes + res.nbytes
            metrics.cycles += len(res.data)
            trace.append({"name": op.lower(), "object": dst, "count": len(res.data)})

        elif op == "ATTENTION_QK":
            # S = Q @ K^T for an attention score block: q is [m, d], k is [n, d] (K stored ROW-per-key,
            # so the contraction is over the trailing head dim of BOTH operands), s is [m, n]. Same
            # semantics the golden engine computes (``attention_qk``: ``q.matmul(transpose(k))``) and the
            # same q/k/dst operand contract the backends emit, so the functional tier agrees with the
            # oracle instead of raising 'unknown opcode' on a conformant command buffer.
            q, k, dst = ops["q"], ops["k"], ops["dst"]
            qt, kt = env[q], env[k]
            (m, d), (n, d2) = qt.shape, kt.shape
            if d != d2:
                raise SimulationError(
                    f"ATTENTION_QK head-dim mismatch: {q}{qt.shape} vs {k}{kt.shape}")
            # Transpose K [n, d] -> [d, n] so the shared matmul primitive does the contraction.
            k_t = Tensor((d, n), [kt.data[i * d + j] for j in range(d) for i in range(n)], kt.dtype)
            t = qt.matmul(k_t)
            for stage in attrs.get("epilogue", []):
                if stage == "acc_scale":
                    t = t.requant_acc_scale(float(attrs.get("acc_scale", 1.0)))
                elif stage == "requant":
                    t = t.requant(int(attrs.get("requant_shift", default_shift)))
                elif stage == "relu":
                    t = t.relu()
                else:
                    raise SimulationError(f"unknown epilogue stage '{stage}'")
            if attrs.get("output_dtype", "i32") == "i8":
                t = t.to_i8()
            env[dst] = t
            outputs[dst] = t.to_list()
            metrics.dispatch_count += 1
            metrics.bytes_read += qt.nbytes + kt.nbytes
            metrics.bytes_written += t.nbytes
            metrics.bytes_moved += qt.nbytes + kt.nbytes + t.nbytes
            metrics.cycles += base_matmul_cycles(m, d, n)
            trace.append({"name": "attention_qk", "object": dst, "count": 1})

        elif op == "ATTENTION_PV":
            # O = P @ V, the SECOND matmul of flash attention and the sibling of ATTENTION_QK: p is
            # [m, s] probabilities over s keys, v is [s, d] values, o is [m, d]. Unlike QK there is no
            # transpose — the contraction is p's trailing axis against v's leading axis, exactly what
            # the golden engine computes (``attention_pv``: ``p.matmul(v)``). The interface grammar did
            # not define this op until now, so all 7 shipped flash-attention capsules parsed without it
            # and the runner scored a program that never multiplied by V.
            p, v, dst = ops["p"], ops["v"], ops["dst"]
            pt, vt = env[p], env[v]
            (m, s), (s2, d) = pt.shape, vt.shape
            if s != s2:
                raise SimulationError(
                    f"ATTENTION_PV key-count mismatch: {p}{pt.shape} vs {v}{vt.shape}")
            t = pt.matmul(vt)
            for stage in attrs.get("epilogue", []):
                if stage == "acc_scale":
                    t = t.requant_acc_scale(float(attrs.get("acc_scale", 1.0)))
                elif stage == "requant":
                    t = t.requant(int(attrs.get("requant_shift", default_shift)))
                elif stage == "relu":
                    t = t.relu()
                else:
                    raise SimulationError(f"unknown epilogue stage '{stage}'")
            if attrs.get("output_dtype", "i32") == "i8":
                t = t.to_i8()
            env[dst] = t
            outputs[dst] = t.to_list()
            metrics.dispatch_count += 1
            metrics.bytes_read += pt.nbytes + vt.nbytes
            metrics.bytes_written += t.nbytes
            metrics.bytes_moved += pt.nbytes + vt.nbytes + t.nbytes
            metrics.cycles += base_matmul_cycles(m, s, d)
            trace.append({"name": "attention_pv", "object": dst, "count": 1})

        elif op == "CONV2D":
            # dst = im2col(ifm) @ resolve(weight). The activation is NHWC, the weight is ALREADY
            # im2col-packed [Kh*Kw*Ci, Co], and the result is the [N*Ho*Wo, Co] im2col product — the
            # exact definition the golden engine computes (``capsule_golden`` conv2d branch:
            # ``im2col(ifm, ...).matmul(w)``), reached through the SAME :func:`conv_im2col` so the two
            # cannot drift on a padding edge or a column order. Undefined until now, which left all 3
            # shipped conv capsules with resident_pack + evict and no compute at all.
            #
            # The supported subset is deliberately narrow and every parameter outside it is REJECTED BY
            # NAME below rather than ignored: silently ignoring a conv parameter is the failure this
            # opcode exists to close, and a conv that quietly drops its stride is a wrong answer that
            # still looks like a plausible tensor.
            ifm_n, w_n, dst = ops["ifm"], ops["weight"], ops["dst"]
            ifm, w = env[ifm_n], env[w_n]
            unknown = sorted(set(attrs) - _CONV2D_ATTRS)
            if unknown:
                raise SimulationError(
                    f"CONV2D does not implement attribute(s) {unknown}; it implements "
                    f"{sorted(_CONV2D_ATTRS)}. Applying a conv with a parameter this engine ignored "
                    f"would be a wrong answer that still looks like a tensor")
            layout = str(attrs.get("layout", "nhwc"))
            if layout != "nhwc":
                raise SimulationError(f"CONV2D layout {layout!r} unsupported (nhwc only)")
            kernel = attrs.get("kernel")
            if not isinstance(kernel, list) or len(kernel) != 4:
                raise SimulationError(
                    f"CONV2D needs kernel = [kh, kw, ci, co]; got {kernel!r}")
            kh, kw, ci, co = (int(v) for v in kernel)
            stride = _conv_geom(attrs, "stride", 2, [1, 1])
            padding = _conv_geom(attrs, "padding", 4, [0, 0, 0, 0])   # [top, left, bottom, right]
            dilation = _conv_geom(attrs, "dilation", 2, [1, 1])
            if len(ifm.shape) != 4:
                raise SimulationError(
                    f"CONV2D activation {ifm_n}{ifm.shape} is not rank-4 NHWC")
            if ifm.shape[3] != ci:
                raise SimulationError(
                    f"CONV2D channel mismatch: {ifm_n}{ifm.shape} has C={ifm.shape[3]} but "
                    f"kernel declares ci={ci}")
            cols = conv_im2col(ifm, kh=kh, kw=kw, ci=ci, stride=stride, padding=padding,
                               dilation=dilation, layout=layout)
            if w.shape != (kh * kw * ci, co):
                raise SimulationError(
                    f"CONV2D weight {w_n}{w.shape} is not the im2col-packed "
                    f"[Kh*Kw*Ci, Co] = [{kh * kw * ci}, {co}] the kernel attribute declares")
            if w_n in resident:
                metrics.resident_hits += 1
            else:
                metrics.resident_misses += 1
                metrics.bytes_read += w.nbytes
                metrics.bytes_moved += w.nbytes
                metrics.cycles += len(w.data)
            t = cols.matmul(w)
            for stage in attrs.get("epilogue", []):
                if stage == "acc_scale":
                    t = t.requant_acc_scale(float(attrs.get("acc_scale", 1.0)))
                elif stage == "requant":
                    t = t.requant(int(attrs.get("requant_shift", default_shift)))
                elif stage == "relu":
                    t = t.relu()
                elif stage == "maxpool":
                    # Pooling fused onto the conv, the form the hardware actually offers (the fused conv
                    # loop takes pool_size/pool_stride/pool_padding and stores the POOLED window). The
                    # conv result is already contracted to [N*Ho*Wo, Co], so the declared
                    # ``pool_in_dims`` is cross-checked against the (Ho, Wo) THIS branch just derived and
                    # a disagreement raises: the capsule golden has only the flat 2-D tensor to work
                    # from and must trust the declaration, so if the two ever diverged the golden and
                    # the simulator would pool different images and blame the numbers.
                    Ho, Wo = conv_out_dims(ifm.shape[1], ifm.shape[2], kh, kw,
                                           stride, padding, dilation)
                    try:
                        p_in = pool_params(attrs, op=f"CONV2D {dst!r}")["pool_in_dims"]
                    except ValueError as e:
                        raise SimulationError(str(e)) from e
                    if p_in != (Ho, Wo):
                        raise SimulationError(
                            f"CONV2D {dst!r}: pool_in_dims {list(p_in)} disagrees with the conv's own "
                            f"output extent [{Ho}, {Wo}]; the pooled image is the conv's output, so the "
                            f"declaration is wrong (or the conv geometry is)")
                    t = _pool(t, stage, attrs, f"CONV2D {dst!r}")
                else:
                    # bias_add lands here on purpose: the conv2d op carries no bias operand, so there
                    # is nothing to add and pretending otherwise would fabricate a result.
                    raise SimulationError(
                        f"CONV2D does not implement epilogue stage '{stage}'")
            t = _narrow_int_readout(t, str(attrs.get("output_dtype", "i32")), "CONV2D")
            env[dst] = t
            outputs[dst] = t.to_list()
            p, kdim = cols.shape
            metrics.dispatch_count += 1
            metrics.bytes_read += ifm.nbytes
            metrics.bytes_written += t.nbytes
            metrics.bytes_moved += ifm.nbytes + t.nbytes
            metrics.cycles += len(cols.data) + base_matmul_cycles(p, kdim, co)
            trace.append({"name": "conv2d", "object": dst, "count": 1})

        elif op == "MOVEMENT":
            # A load->store round-trip through the accelerator: values are carried unchanged, only the
            # container dtype widens (operand dtype in, accumulate dtype out — see
            # corpus_spec.build_movement). The golden engine agrees exactly (``movement``:
            # ``src.to_list()``), so this must NOT clamp or requantize: a movement capsule's whole point
            # is that the data survives the trip bit-for-bit. Undefined until now, which made all 5
            # shipped movement capsules parse to ZERO commands — their only op vanished.
            if "src" not in ops:
                raise SimulationError("MOVEMENT needs operands src/dst")
            src, dst = ops["src"], ops["dst"]
            t = env[src]
            moved = Tensor(t.shape, list(t.data), str(attrs.get("output_dtype", t.dtype)))
            env[dst] = moved
            outputs[dst] = moved.to_list()
            metrics.bytes_read += t.nbytes
            metrics.bytes_written += moved.nbytes
            metrics.bytes_moved += t.nbytes + moved.nbytes
            metrics.cycles += len(moved.data)
            trace.append({"name": "movement", "object": dst, "count": len(moved.data)})

        elif op == "VREDUCE":
            src, dst = env[ops["src"]], ops["dst"]
            rop = attrs.get("op", "sum")
            if rop != "sum":
                raise SimulationError(f"unknown vector reduce '{rop}'")
            t = src.reduce_sum()
            env[dst] = t
            metrics.bytes_read += src.nbytes
            metrics.cycles += len(src.data)
            trace.append({"name": "vector_reduce", "object": dst, "count": len(src.data)})

        else:
            raise SimulationError(f"unknown opcode '{op}'")

    # Vector-family outputs are tensors declared with role "output" (matmul outputs come from
    # COMMIT and are already collected above; this sweep is a no-op for those).
    for name, spec in cb.get("tensors", {}).items():
        if spec.get("role") == "output" and name in env and name not in outputs:
            outputs[name] = env[name].to_list()

    # Surface EXACTLY the declared model outputs when the buffer names them (a chained layer's
    # committed output is an intermediate, not a result).
    declared = cb.get("outputs")
    if declared:
        outputs = {k: v for k, v in outputs.items() if k in set(declared)}
    return {"outputs": outputs, "metrics": metrics.as_dict(), "trace": trace}


def base_matmul_cycles(m: int, k: int, n: int) -> int:
    """A simple but real cycle model: one MAC-tile per 64 MACs, plus fixed overhead."""
    return 16 + (m * k * n) // 64
