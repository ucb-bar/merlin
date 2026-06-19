"""Execute a Merlin command buffer with real integer arithmetic.

This is the Merlin-owned simulator harness. It interprets the target-independent command
buffer, performs the actual tensor math (via :mod:`merlin.runtime.tensor`), records real
metrics and a trace, and returns the committed output tensors. The ToyNPU opcodes
(RES_PACK / MATMUL_RESIDENT / COMMIT / EVICT) are handled directly; the design generalizes to
target-supplied semantics.
"""
from __future__ import annotations

from typing import Any

from .commandbuffer import materialize_inputs, validate_command_buffer
from .metrics import Metrics
from .tensor import Tensor


class SimulationError(RuntimeError):
    pass


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
            env[dst] = t                      # packed copy has identical values
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

    return {"outputs": outputs, "metrics": metrics.as_dict(), "trace": trace}


def base_matmul_cycles(m: int, k: int, n: int) -> int:
    """A simple but real cycle model: one MAC-tile per 64 MACs, plus fixed overhead."""
    return 16 + (m * k * n) // 64
