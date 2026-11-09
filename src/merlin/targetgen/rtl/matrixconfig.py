"""Read matrix-unit geometry from the selected Scala configuration declarations.

Ported from OPU-VME 80e00803d; hardware values remain caller-supplied.
"""

from __future__ import annotations

from .matrixparse import _uint_literal
from .scala_parse import (
    _VECTOR_UNIT_MIXIN_SUFFIX,
    _balanced_after,
    _bit_literal,
    _split_top_level,
    _strip_comments,
    scala_param_names,
)


def _find_def(text: str, name: str) -> int:
    """Index of ``def <name>`` not followed by more identifier characters (``def a`` is not ``def ab``)."""
    i = text.find(f"def {name}")
    while i >= 0:
        end = i + len("def ") + len(name)
        if end >= len(text) or not (text[end].isalnum() or text[end] == "_"):
            return i
        i = text.find(f"def {name}", end)
    return -1


def _params_flag(params_text: str, params_object: str, member: str, flag: str, *, depth: int = 0) -> bool | None:
    """The value of ``flag`` in ``params_object.member``, following ``.copy(...)`` bases to the class default.

    None when it cannot be grounded (a member that is not declared, a base that cannot be followed).
    """
    if depth > 16:
        return None
    body = _balanced_after(_strip_comments(params_text), f"object {params_object}")
    if body is None:
        return None
    i = _find_def(body, member)
    if i < 0:
        return None
    rhs = body[body.find("=", i) + 1 :].lstrip()
    call_head = rhs.split("(", 1)[0].strip()
    args = _balanced_after(rhs, call_head, "(", ")")
    if args is None:
        return None
    for arg in _split_top_level(args):
        key, sep, val = arg.partition("=")
        if sep and key.strip() == flag:
            v = val.strip()
            return True if v == "true" else False if v == "false" else None
    if call_head.endswith(".copy"):
        return _params_flag(params_text, params_object, call_head[: -len(".copy")].strip(), flag, depth=depth + 1)
    # A constructor call: the class's declared default.
    decl = _balanced_after(_strip_comments(params_text), f"case class {call_head}", "(", ")")
    if decl is None:
        return None
    for arg in _split_top_level(decl):
        name, _, rest = arg.partition(":")
        if name.strip() == flag:
            default = rest.partition("=")[2].strip()
            return True if default == "true" else False if default == "false" else None
    return None


def matrix_unit_params(
    config_text: str,
    config_class: str,
    *,
    mixin_text: str,
    params_text: str,
    params_object: str,
    enable_flag: str,
) -> tuple[dict[str, int], list[str]]:
    """``({vLen, dLen, ...}, notes)`` for the vector-unit mixin of ``config_class`` that ENABLES the unit.

    A heterogeneous SoC instantiates several vector units, and only some carry the matrix unit; reading the
    first one (what a plain VectorUnit lookup does) can return the wrong unit's geometry. Here each mixin's
    params argument (``VectorParams.opuMxParams``) is resolved against ``params_object`` and followed through
    its ``.copy`` chain to ``enable_flag``. Exactly the mixins whose flag is true are candidates; if they
    disagree on geometry, or none is found, the result is empty and the notes say why.
    """
    notes: list[str] = []
    body = _balanced_after(_strip_comments(config_text), f"class {config_class} extends Config(", "(", ")")
    if body is None:
        return {}, [f"config {config_class!r} not found"]
    picks: list[dict[str, int]] = []
    for mixin in _split_top_level(body, "+"):
        head = mixin.strip().removeprefix("new ").strip().split("(", 1)[0]
        simple = head.strip().rsplit(".", 1)[-1]
        if not simple.endswith(_VECTOR_UNIT_MIXIN_SUFFIX):
            continue
        call = _balanced_after(mixin, head.strip(), "(", ")")
        args = _split_top_level(call) if call else []
        names = scala_param_names(mixin_text, f"class {simple}")
        bound: dict[str, str] = {}
        position = 0
        for arg in args:
            key, sep, rhs = arg.partition("=")
            if sep and key.strip().isidentifier() and not rhs.startswith("="):
                bound[key.strip()] = rhs.strip()
            else:
                if position < len(names):
                    bound[names[position]] = arg.strip()
                position += 1
        enabled = None
        for token in bound.values():
            parts = token.split(".")
            # A host SoC may use a fully qualified Scala reference such as
            # saturn.common.VectorParams.opuMxParams. The object and member
            # are still resolved against the pinned params source above.
            if len(parts) >= 2 and parts[-2] == params_object and all(part.isidentifier() for part in parts):
                enabled = _params_flag(params_text, params_object, parts[-1], enable_flag)
        if enabled is None:
            notes.append(f"{simple}: could not resolve {enable_flag} from its params")
            continue
        if not enabled:
            notes.append(f"{simple}: {enable_flag} is false")
            continue
        ints: dict[str, int] = {}
        for k, v in bound.items():
            got = _bit_literal(v) if v.startswith('"') else _uint_literal(v)
            if got is not None:
                ints[k] = got
        picks.append(ints)
    if not picks:
        return {}, notes + [f"no vector unit in {config_class} enables {enable_flag}"]
    if any(p != picks[0] for p in picks[1:]):
        return {}, notes + [f"vector units enabling {enable_flag} disagree on geometry: {picks}"]
    return picks[0], notes
