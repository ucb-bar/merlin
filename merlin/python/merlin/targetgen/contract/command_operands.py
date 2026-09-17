"""Audit a command's operand slots against the operands its opcode DECLARES in the ABI.

WHY THIS EXISTS. ``command_buffer.schema.json`` types an operand map as
``{"type": "object", "additionalProperties": {"type": "string"}}`` -- any keys, any arity. The ABI,
meanwhile, declares exact operands per opcode (``DEPTHWISE_CONV2D: {src, weight, dst}``). Nothing
compared the two, so a command could name operands its opcode has never heard of and validate.

MEASURED. Compiling the whole-model capsule ``SY_model_smolvla`` (8,234 ``linalg.generic``, 812
``linalg.reduce``, 525 ``linalg.transpose``, 1 ``linalg.matmul``) through a backend produced ONE
command: ``DEPTHWISE_CONV2D`` with 817 operands named ``arg0..arg816`` -- every entry-point argument
of the model attached to a single opcode that the ABI defines as an NCHW depthwise convolution over
``{src, weight, dst}``. It passed schema validation. A whole model collapsed to one mislabelled
command is the most degenerate output a backend can produce, and the contract called it valid.

This is the operand-NAME half of a hole whose value half is already closed: ``schemas``'
``validate_command_buffer`` records that "JSON Schema types an operand slot as a bare string, so a
value naming nothing ... validates", and states the rule this module extends -- *a validator the
submitter is told to run must fail on what the runner will refuse, or it is not the contract*.

SURFACES, DOES NOT GATE. ``audit`` returns findings; it raises nothing and no caller is obliged to
act. That is deliberate. ``epilogue_applicability`` is the precedent: it went from advisory to gating
and immediately failed ten capsules on a plane the other arms had never been assessed on, which
invalidated a cross-arm comparison. Make this decisive first -- observe that it agrees with graded
outcomes -- and only then let a caller refuse on it.

The operand vocabulary is DERIVED from ``command_buffer_abi.yaml`` at call time, never listed here:
adding an opcode to the ABI extends this audit with no edit, and no opcode name is baked into code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["declared_operands", "audit", "applies", "undeclared_opcodes", "OPTIONAL_MARKER"]

# How the ABI spells an operand a command MAY omit, e.g. ``bias: "tensor (optional)"``.
OPTIONAL_MARKER = "(optional)"


def _abi_path(contract: str | Path | None = None) -> Path:
    from merlin.common.paths import repo_root

    if contract is not None:
        return Path(contract)
    return repo_root() / "merlin" / "contract" / "command_buffer_abi.yaml"


def declared_operands(*, contract: str | Path | None = None) -> dict[str, dict[str, bool]]:
    """``{OPCODE: {operand_name: is_required}}`` exactly as the ABI declares it.

    Optionality is read structurally from the operand's type phrase (``"tensor (optional)"``), not
    pattern-matched: the phrase is split on the marker token and the remainder is the type.
    """
    import yaml

    path = _abi_path(contract)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    opcodes = doc.get("opcodes")
    if not isinstance(opcodes, dict):
        raise ValueError(f"{path}: no 'opcodes' mapping to derive operand names from")

    out: dict[str, dict[str, bool]] = {}
    for opcode, spec in opcodes.items():
        operands = (spec or {}).get("operands")
        if not isinstance(operands, dict):
            # Fail closed: an opcode whose operands the ABI does not state is NOT silently
            # treated as unconstrained -- it is absent, and audit() reports it as underivable.
            continue
        out[str(opcode)] = {str(name): OPTIONAL_MARKER not in str(kind) for name, kind in operands.items()}
    return out


def applies(cb: Any) -> bool:
    """Is this an ABI-vocabulary command buffer at all?

    Two other artifact families share the ``capsule.command_buffer.json`` file name and must not be
    reported as defective by an audit that does not describe them: an ISA-level command stream
    (numeric ``opcode``, e.g. an atlas ``PushWeight`` at opcode 0) and a SIMT warp descriptor (no
    ``commands`` list at all, carrying ``kernel`` instead). Both are discriminated STRUCTURALLY --
    by the shape of what is there, not by any target or kind literal.
    """
    if not isinstance(cb, dict):
        return False
    commands = cb.get("commands")
    if not isinstance(commands, list):
        return False
    return all(isinstance(c, dict) and isinstance(c.get("opcode"), str) for c in commands) if commands else True


def undeclared_opcodes(*, contract: str | Path | None = None, schema: str | Path | None = None) -> list[str]:
    """Opcodes the schema admits but the ABI declares no operands for -- OUR specification gap.

    MEASURED at the time of writing: 12 of the 25 enumerated opcodes, including ``ATTENTION_FULL``.
    For those, a command's operand names cannot be checked by anyone, and a submitter cannot learn
    the expected spelling from the contract. This is a gap to close in the ABI, never a finding
    against a submission.
    """
    import json

    from merlin.common.paths import repo_root

    path = (
        Path(schema)
        if schema is not None
        else (repo_root() / "merlin" / "contract" / "schemas" / "command_buffer.schema.json")
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    enum = doc["properties"]["commands"]["items"]["properties"]["opcode"].get("enum") or []
    return sorted(set(enum) - set(declared_operands(contract=contract)))


def audit(cb: Any, *, contract: str | Path | None = None) -> list[str]:
    """Findings about operand slots that disagree with the opcode's declared operands.

    One string per problem, naming the command index and the opcode. Returns [] when every command's
    operand names are exactly the declared required ones plus any subset of the declared optional
    ones, AND [] for a buffer that is not an ABI command buffer (see :func:`applies`) -- describing
    another artifact family as broken would be a false finding, not rigour.

    An opcode the ABI declares no operands for yields a finding that says so IN THOSE TERMS: it is a
    hole in our specification, not a defect in the buffer. Never raises.
    """
    findings: list[str] = []
    if not applies(cb):
        return findings

    declared = declared_operands(contract=contract)

    for i, command in enumerate(cb["commands"]):
        opcode = command.get("opcode")
        where = f"command {i} ({opcode})"
        spec = declared.get(str(opcode))
        if spec is None:
            findings.append(
                f"{where}: NOT CHECKABLE -- command_buffer_abi.yaml declares no operands for this "
                f"opcode, so no operand spelling can be required of a submitter. This is a gap in "
                f"our contract, not a defect in this buffer."
            )
            continue
        operands = command.get("operands")
        if not isinstance(operands, dict):
            findings.append(f"{where}: 'operands' is not an object")
            continue

        given = set(operands)
        required = {name for name, req in spec.items() if req}
        allowed = set(spec)

        missing = sorted(required - given)
        unknown = sorted(given - allowed)
        if missing:
            findings.append(f"{where}: missing required operand(s) {missing}; the ABI declares {sorted(allowed)}")
        if unknown:
            shown = unknown if len(unknown) <= 6 else unknown[:6] + [f"... and {len(unknown) - 6} more"]
            findings.append(
                f"{where}: {len(unknown)} operand(s) the opcode does not declare: {shown}; "
                f"the ABI declares {sorted(allowed)}"
            )
    return findings
