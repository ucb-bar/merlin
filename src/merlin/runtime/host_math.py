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
    if policy != "expf_via_double":
        raise ValueError("host_math_policy must be 'native', 'portable_trig', or 'expf_via_double'")
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
