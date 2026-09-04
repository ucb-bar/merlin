"""Measure a systolic array's fill/drain depth from its circuit, instead of assuming a law.

A fill/drain depth is an intercept, not a rate: it is paid once per weight reload and it dominates
small tiles. It is also the term most easily assumed. The obvious closed form for a square
weight-stationary array is ``2*DIM`` -- rows plus columns -- and a slightly better one is ``2*DIM-2``.
Both are guesses about a pipeline whose actual length is a property of the emitted circuit, and the
gap between them is two cycles on one array and much larger on another microarchitecture.

So this reads the depth out of the circuit: the length of the output-valid delay-line register chain,
counted directly in the IR. Nothing is fitted and no law is applied.

WHY A LAW IS STILL WORTH KEEPING, and why this module reports BOTH. A measured depth is the truth for
the design that was elaborated; a law is what lets the model answer for a design that has not been
elaborated yet -- a different mesh dimension in a design-space sweep, where there is no circuit to
read. Keeping the two apart, and reporting when they disagree, is what tells you whether the law may
be extrapolated at all. A law that matches the circuit on the one point it was checked against is
evidence, not proof; a law that does not match is refuted for this family and must not be swept with.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class HandshakeUnavailable(RuntimeError):
    """The circuit could not be read, so the fill/drain depth is UNKNOWN -- never a default."""


@dataclass(frozen=True)
class FillDepth:
    """A fill/drain depth, its source, and how it compares with the law it would otherwise be given."""

    dim: int
    measured_cycles: int
    #: What the named law predicts for this dimension, or None when no law was offered.
    law_cycles: int | None
    law: str | None
    #: Double-buffering the circuit carries. A second slot lets the next reload overlap the current
    #: compute, which is why it belongs beside the depth rather than in a separate report.
    weight_buffer_slots: int
    accumulator_banks: int
    source: str

    @property
    def law_agrees(self) -> bool | None:
        """True/False when a law was offered, None when none was."""
        return None if self.law_cycles is None else self.law_cycles == self.measured_cycles

    def claim(self) -> str:
        if self.law_cycles is None:
            return f"fill/drain {self.measured_cycles} cycles, measured from {self.source}"
        if self.law_agrees:
            return (f"fill/drain {self.measured_cycles} cycles, measured; the law {self.law!r} agrees "
                    f"at DIM={self.dim} -- evidence it may extrapolate, not proof")
        return (f"fill/drain {self.measured_cycles} cycles, measured; the law {self.law!r} predicts "
                f"{self.law_cycles} and is REFUTED for this design -- do not sweep with it")


def _instance_ports(line: str) -> tuple[str, dict[str, str]]:
    """``(instance name, {port: driving value})`` for one ``hw.instance`` line, else ``("", {})``.

    Parsed structurally -- membership and ``partition`` only, no pattern matching -- because the
    result list on the left of the ``=`` is a comma-separated series of the instance's own results
    and must not be mistaken for its name. The name is the quoted string after ``hw.instance``.
    """
    if "hw.instance" not in line:
        return "", {}
    _, _, after = line.strip().partition("hw.instance")
    after = after.strip()
    if after.count('"') < 2:
        return "", {}
    _, _, rest = after.partition('"')
    name, _, rest = rest.partition('"')
    _, _, args = rest.partition("(")
    ports: dict[str, str] = {}
    for part in args.split(","):
        port, sep, value = part.partition(":")
        if not sep:
            continue
        value = value.strip()
        if value.startswith("%"):
            ports[port.strip()] = value.split(":")[0].strip().lstrip("%")
    return name, ports


def _valid_path_depth(bodies: "dict[str, list[str]]", array_module: str) -> tuple[str, int]:
    """The longest register-stage path a valid signal takes through ``array_module``.

    The upstream pass reads the depth off a register chain whose OWN NAME contains "valid". That is
    a naming convention, not a structural fact, and a conformant design loses it: firtool named this
    array's chain ``%r_256_0 ... %r_1115_0`` and put the word "valid" only on the SIGNAL each stage
    samples. The pass then reports "no output-valid delay-line found", which reads as an unreadable
    circuit when the delay line is plainly there -- 257 registers of it.

    So the path is walked instead of name-matched: a stage is a register, and the valid signal
    crosses a submodule through the ports whose names carry the handshake's own "valid" (the port
    names are the DESIGN's vocabulary, unlike the register names, which the emitter invents). The
    answer is the deepest such path reaching a module output, which is the same physical quantity --
    cycles from an input valid to the output valid -- that the named chain measures where one exists.
    """
    body = bodies[array_module]
    regs: dict[str, str] = {}
    instances: dict[str, dict[str, str]] = {}
    outputs: list[str] = []
    for line in body:
        stripped = line.strip()
        if stripped.startswith("%") and "seq.firreg" in stripped:
            lhs, _, rhs = stripped.partition("=")
            _, _, driver = rhs.partition("seq.firreg")
            head = driver.split()
            if head:
                regs[lhs.strip().lstrip("%")] = head[0].lstrip("%")
        elif "hw.instance" in stripped:
            name, ports = _instance_ports(stripped)
            if name:
                instances[name] = ports
        elif stripped.startswith("hw.output"):
            _, _, values = stripped.partition("hw.output")
            outputs = [v.strip().lstrip("%") for v in values.split(",") if v.strip()]

    depth_of: dict[str, int] = {}

    def depth(token: str) -> int:
        seen = depth_of.get(token)
        if seen is not None:
            return seen
        depth_of[token] = 0          # break combinational cycles rather than recursing forever
        if token in regs:
            result = 1 + depth(regs[token])
        else:
            owner, sep, _ = token.partition(".")
            ports = instances.get(owner) if sep else None
            result = max((depth(v) for port, v in (ports or {}).items()
                          if "valid" in port.lower()), default=0)
        depth_of[token] = result
        return result

    best, where = 0, ""
    for token in outputs:
        found = depth(token)
        if found > best:
            best, where = found, token
    return where, best


def measure_fill_depth(target: str, *, law: str | None = "systolic_2d",
                       hw_mlir: Any = None) -> FillDepth:
    """Read the fill/drain depth from the target's elaborated circuit, and check any offered law.

    ``law`` names a fill law from :mod:`merlin.perf.record` to cross-check against; pass None to skip
    the comparison. Raises rather than returning a default when the circuit is unreachable: an
    unmeasurable intercept is UNKNOWN, and a pipeline silently given a depth of zero reads as an array
    that fills instantly."""
    from merlin.targetgen.rtl import mlc_bridge
    path = hw_mlir if hw_mlir is not None else mlc_bridge.core_hw_mlir(target)
    if path is None:
        raise HandshakeUnavailable(
            f"no elaborated circuit is resolvable for {target!r}; the fill/drain depth is UNKNOWN")
    # MODULE NAMES ARE DERIVED, NOT DEFAULTED. `infer_handshake_depth` takes the array/mesh/PE module
    # names and defaults them to one family's spelling; calling it without them asks every target's
    # circuit for another target's module. That produced "array module @SystolicArray not found" on a
    # design whose array container the facts name `Mesh` -- a refusal about the wrong thing, which
    # reads as "this circuit is unreadable" rather than "we looked for the wrong module". The names
    # come from the target's own discovered array fact; absent one, the defaults still apply.
    # The pass distinguishes the outer array WRAPPER from the inner MESH, and the discovered array
    # fact names the mesh (its `container`) -- so the derived name goes to `mesh_module`, never to
    # `array_module`. Mapping it to the wrapper broke a target whose circuit reads correctly under
    # the defaults (atlas: dim=32, fill_drain_depth=62), which is why the wrapper name is only
    # retried, not replaced: a design that carries no separate wrapper is then asked for its mesh.
    names: dict[str, str] = {}
    try:
        from merlin.targetgen.rtl import facts as _facts
        arrays = ((_facts.load_facts(target) or {}).get("facts") or {}).get("arrays") or ()
        discovered = arrays[0] if arrays else {}
        if discovered.get("container"):
            names["mesh_module"] = str(discovered["container"])
        if discovered.get("element"):
            names["pe_module"] = str(discovered["element"])
    except Exception:  # noqa: BLE001 - no facts bundle just means the defaults stand
        names = {}
    walked: tuple[str, int] | None = None
    try:
        with mlc_bridge._mlc_cwd():
            import importlib
            module = importlib.import_module("mlc.passes.infer_handshake_depth")
            infer_handshake_depth = module.infer_handshake_depth

            def _attempt(**kw: Any) -> Any:
                return infer_handshake_depth(str(path), **kw)

            try:
                facts = _attempt(**names)
            except ValueError as missing:
                # Only a MISSING WRAPPER is retried with the derived mesh name; any other refusal is
                # about this design and must surface unchanged rather than be masked by a retry.
                if "array module" not in str(missing) or "mesh_module" not in names:
                    raise
                names = {**names, "array_module": names["mesh_module"]}
                facts = _attempt(**names)
    except ValueError as unnamed:
        # A design whose delay line the emitter did not NAME "valid" is not an unreadable circuit.
        # Walk the valid path instead, and let every other refusal fall through unchanged.
        if "output-valid delay-line" not in str(unnamed):
            raise HandshakeUnavailable(
                f"the circuit for {target!r} could not be read for a fill/drain depth: "
                f"{type(unnamed).__name__}: {unnamed}") from unnamed
        try:
            with mlc_bridge._mlc_cwd():
                import importlib
                module = importlib.import_module("mlc.passes.infer_handshake_depth")
                lines = Path(path).read_text().split("\n")
                bodies = {name: lines[start:end]
                          for name, start, end in module._iter_modules(lines)}
                array_module = names.get("array_module") or names.get("mesh_module") or ""
                if array_module not in bodies:
                    raise HandshakeUnavailable(
                        f"the array module for {target!r} is not in its own circuit: "
                        f"{array_module!r}")
                where, depth = _valid_path_depth(bodies, array_module)
                if depth <= 0:
                    raise HandshakeUnavailable(
                        f"{target!r} has no valid path through @{array_module}, named or walked, "
                        f"so its fill/drain depth is UNKNOWN")
                mesh_module = names.get("mesh_module", array_module)
                pe_module = names.get("pe_module", "PE")
                pe_instances = module._count_pe_instances(
                    bodies.get(mesh_module, bodies[array_module]), pe_module)
                if pe_instances <= 0:
                    raise HandshakeUnavailable(
                        f"no @{pe_module} instances in @{mesh_module}, so {target!r} has no "
                        f"measurable array dimension")
                from math import isqrt
                dim = isqrt(pe_instances)
                if dim * dim != pe_instances:
                    raise HandshakeUnavailable(
                        f"@{pe_module} count {pe_instances} in @{mesh_module} is not a square mesh")
                walked = (f"@{array_module}.%{where} valid path "
                          f"({pe_instances} @{pe_module} instances)", depth)
                facts = SimpleNamespace(
                    dim=dim, fill_drain_depth=depth,
                    weight_buffer_slots=module._weight_slot_count(bodies[array_module]),
                    accumulator_banks=module._accsel_banks(bodies[array_module]),
                    array_module=array_module, valid_register_family=where,
                    pe_instances=pe_instances)
        except HandshakeUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - an unwalkable circuit is UNKNOWN, not a default
            raise HandshakeUnavailable(
                f"the circuit for {target!r} could not be walked for a fill/drain depth: "
                f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - a circuit we cannot read is UNKNOWN, not a default
        raise HandshakeUnavailable(
            f"the circuit for {target!r} could not be read for a fill/drain depth: "
            f"{type(exc).__name__}: {exc}") from exc

    law_cycles = None
    if law is not None:
        from merlin.perf.record import fill_cycles
        law_cycles = int(fill_cycles(law, facts.dim))
    source = (walked[0] if walked is not None else
              f"{facts.array_module}.{facts.valid_register_family} "
              f"({facts.pe_instances} MAC instances)")
    return FillDepth(dim=int(facts.dim), measured_cycles=int(facts.fill_drain_depth),
                     law_cycles=law_cycles, law=law,
                     weight_buffer_slots=int(facts.weight_buffer_slots),
                     accumulator_banks=int(facts.accumulator_banks),
                     source=source)
