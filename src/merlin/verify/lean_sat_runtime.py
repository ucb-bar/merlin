"""Original pinned SAT executable for generated bit-vector obligations.

The solver supplies an LRAT certificate checked by the same selected Lean
toolchain. It is not proof authority. No ambient PATH or default sibling binary
may supply this runtime dependency, and candidates cannot select tactic options.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

NATIVE_AXIOMS = frozenset({"Lean.ofReduceBool", "Lean.trustCompiler"})
NATIVE_ASSUMPTION = "originally admitted Lean native reflection, Lean compiler and LRAT certificate checker are trusted"


def _tokens(source):
    """Lex identifiers in generated Lean, excluding strings and nested comments."""
    tokens, word = [], []
    index, depth = 0, 0
    while index < len(source):
        pair = source[index:index + 2]
        if depth:
            if pair == "/-":
                depth += 1
                index += 2
            elif pair == "-/":
                depth -= 1
                index += 2
            else:
                index += 1
            continue
        if pair in {"/-", "--"} or source[index] == '"':
            if word:
                tokens.append("".join(word))
                word = []
            if pair == "/-":
                depth, index = 1, index + 2
            elif pair == "--":
                newline = source.find("\n", index)
                index = len(source) if newline < 0 else newline + 1
            else:
                index += 1
                while index < len(source):
                    if source[index] == "\\":
                        index += 2
                    elif source[index] == '"':
                        index += 1
                        break
                    else:
                        index += 1
            continue
        character = source[index]
        if character.isalnum() or character in "_.'":
            word.append(character)
        elif word:
            tokens.append("".join(word))
            word = []
        index += 1
    if word:
        tokens.append("".join(word))
    return tuple(tokens)


def validate_sat_solver(solver, files):
    """Require one original canonical executable already in the file inventory."""
    if solver is None:
        return
    if type(solver) is not tuple or len(solver) != 2:
        raise ValueError("Lean SAT selection requires an exact path/SHA256 pair")
    path, identity = solver
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.resolve() != path
        or any(owner.is_symlink() for owner in (path, *path.parents))
        or type(identity) is not str
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
        or solver not in files
    ):
        raise ValueError("Lean SAT solver is outside its exact original file inventory")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError("selected Lean SAT solver is unavailable or nonexecutable")


def prepared_sat_solver(executable, runtime_files, files, *, explicit=None):
    """Select only an explicitly supplied runtime file in Lean's reviewed layout."""
    selected = explicit if explicit is not None else executable.parent / "cadical"
    matches = [path for path in runtime_files if path == selected]
    if not matches:
        if explicit is not None:
            raise ValueError("explicit Lean SAT executable is outside its runtime inventory")
        return None
    if len(matches) != 1:
        raise ValueError("Lean SAT runtime selection is duplicated")
    solver = next(row for row in files if row[0] == selected)
    validate_sat_solver(solver, files)
    return solver


def proof_runtime_arguments(query, toolchain):
    """Fixed host options, checked before any execution or cached completion."""
    tokens = _tokens(query)
    if any(pair == ("set_option", "sat.solver") for pair in zip(tokens, tokens[1:])):
        raise ValueError("generated proof cannot override the original pinned SAT solver")
    if "bv_decide" not in tokens:
        return ()
    if toolchain.sat_solver is None:
        raise FileNotFoundError("generated bit-vector proof requires an explicitly selected pinned SAT solver")
    validate_sat_solver(toolchain.sat_solver, toolchain.files)
    if not toolchain.allow_native_reflection:
        raise FileNotFoundError("bit-vector proof requires original Lean native-reflection admission")
    # Lean's sat.solver is an exact executable string. A nonempty absolute value
    # prevents both sibling discovery and PATH fallback, also after relocation.
    return ("-Dsat.solver=" + str(toolchain.sat_solver[0]),)


def completion_assurance(query, toolchain, axioms, assumptions, allowed_axioms):
    """Qualify actual reported axioms; no extra axiom is globally admitted."""
    native = bool(set(axioms) & NATIVE_AXIOMS)
    if native:
        if not toolchain.allow_native_reflection or "bv_decide" not in _tokens(query):
            raise ValueError("Lean obligation depends on unadmitted native-reflection axioms")
        proof_runtime_arguments(query, toolchain)
    if not set(axioms) <= allowed_axioms | (NATIVE_AXIOMS if native else frozenset()):
        raise ValueError("Lean obligation depends on an unapproved axiom")
    if native:
        return "lean_native_reflection", tuple(dict.fromkeys((*assumptions, NATIVE_ASSUMPTION)))
    return "lean_kernel", assumptions


def runtime_stage(argv):
    return "lean_native_reflection" if any(value.startswith("-Dsat.solver=") for value in argv) else "lean_kernel"


def runtime_receipt(receipt, options):
    """Failed requested native checks also describe their actual runtime scope."""
    if not options:
        return receipt
    return replace(
        receipt,
        method="lean_native_reflection",
        assumptions=tuple(dict.fromkeys((*receipt.assumptions, NATIVE_ASSUMPTION))),
    )
