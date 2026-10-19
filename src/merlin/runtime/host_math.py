"""Explicit portable host libm evaluation policies.

These policies specify returned values, not errno or floating-point exception
flag equivalence to the platform's float libm. Native is emission-neutral.
Callers must qualify their original numerical contract when changing precision.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HostMathRecipe:
    source: str | None = None
    compile_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = ()


def runtime_symbol_bindings(policy: str, symbols: Sequence[str]) -> dict[str, str]:
    """Select GNU wrapper definitions from an explicit runtime recipe."""
    from merlin.llvmlower.link_supplier_trace import trace_symbol_flags

    trace_symbol_flags(symbols)  # Validate the original, bounded C-symbol roster.
    recipe = host_math_recipe(policy)
    wrapped = {flag.removeprefix("-Wl,--wrap=") for flag in recipe.link_flags if flag.startswith("-Wl,--wrap=")}
    if recipe.source is None or not set(symbols) <= wrapped:
        raise ValueError("requested runtime symbols are not wrapped by the selected host math policy")
    return {symbol: "__wrap_" + symbol for symbol in symbols}


def host_math_recipe(policy: str = "native") -> HostMathRecipe:
    """Return a GNU-compatible link recipe; no implicit policy selection."""
    if policy == "native":
        return HostMathRecipe()
    if policy == "portable_trig":
        return HostMathRecipe(
            source=Path(__file__).with_name("portable_trig.c").read_text(),
            compile_flags=("-fno-builtin", "-fno-fast-math", "-ffp-contract=off"),
            link_flags=("-Wl,--wrap=sinf", "-Wl,--wrap=cosf"),
        )
    if policy == "portable_exp_trig":
        return HostMathRecipe(
            source=(
                Path(__file__).with_name("portable_trig.c").read_text()
                + "\n"
                + Path(__file__).with_name("portable_exp.c").read_text()
            ),
            compile_flags=("-std=c11", "-fno-builtin", "-fno-fast-math", "-ffp-contract=off"),
            link_flags=("-Wl,--wrap=expf", "-Wl,--wrap=sinf", "-Wl,--wrap=cosf"),
        )
    if policy == "portable_neural_f32":
        base = host_math_recipe("portable_exp_trig")
        return HostMathRecipe(
            source=(
                base.source
                + "\n"
                + Path(__file__).with_name("portable_pow.c").read_text()
                + "\n"
                + Path(__file__).with_name("portable_tanh.c").read_text()
            ),
            compile_flags=base.compile_flags,
            link_flags=(*base.link_flags, "-Wl,--wrap=powf", "-Wl,--wrap=tanhf"),
        )
    if policy != "expf_via_double":
        raise ValueError(
            "host_math_policy must be 'native', 'portable_trig', 'portable_exp_trig', "
            "'portable_neural_f32', or 'expf_via_double'"
        )
    return HostMathRecipe(
        source=("#include <math.h>\nfloat __wrap_expf(float x) { return (float)exp((double)x); }\n"),
        compile_flags=("-fno-builtin", "-fno-fast-math", "-ffp-contract=off"),
        link_flags=("-Wl,--wrap=expf",),
    )


def build_host_math(
    policy: str,
    work: Path,
    compiler: str | Path,
    cflags: Sequence[str],
    run: Callable,
) -> tuple[list[Path], tuple[str, ...]]:
    """Compile only an explicitly requested runtime object.

    The caller includes returned objects in its normal linked-byte identity.
    Native creates no files and invokes no compiler, preserving default bytes.
    """
    recipe = host_math_recipe(policy)
    if recipe.source is None:
        return [], ()
    work.mkdir(parents=True, exist_ok=True)
    source, obj = work / "host_math.c", work / "host_math.o"
    source.write_text(recipe.source)
    run([compiler, *cflags, *recipe.compile_flags, "-c", source, "-o", obj])
    if not obj.is_file():
        raise RuntimeError("host math compiler did not produce an object")
    return [obj], recipe.link_flags
