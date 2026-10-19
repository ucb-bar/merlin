"""Explicit portable host libm evaluation policies.

Native is emission-neutral. Precision-changing policies require qualification
against the original numerical contract. Guarded call selection instead requires
provider-reviewed equality of returned bits, errno and floating effects in its
selected domain, plus actual endpoint binding to a pinned library archive.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import sha256_file


@dataclass(frozen=True)
class HostMathRecipe:
    source: str | None = None
    compile_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = ()
    library_sha256: str | None = None


@dataclass(frozen=True)
class GuardedF32Call:
    """Provider-reviewed equivalence for two fixed library entry points.

    The provider proves equal return bits, errno and floating effects throughout
    the finite magnitude interval, and at every explicit constant point. This
    data is a proof obligation, not a theorem inferred from symbol names or test
    samples. The normal link must bind both endpoints to the pinned archive.
    All other values execute the original entry point.
    """

    source_symbol: str
    body_symbol: str
    max_magnitude_bits: int
    library_sha256: str
    exact_points: tuple[tuple[int, int], ...] = ()

    def validate(self) -> None:
        names = (self.source_symbol, self.body_symbol)
        if (
            any(
                type(name) is not str
                or not name.isascii()
                or not name.isidentifier()
                or name.startswith(("__wrap_", "__real_"))
                for name in names
            )
            or len(set(names)) != 2
        ):
            raise ValueError("guarded math requires distinct C symbol names")
        if type(self.max_magnitude_bits) is not int or not 0 <= self.max_magnitude_bits <= 0x7F7FFFFF:
            raise ValueError("guarded math requires a finite binary32 magnitude bound")
        if (
            type(self.library_sha256) is not str
            or len(self.library_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.library_sha256)
        ):
            raise ValueError("guarded math requires the reviewed library SHA256")
        if (
            type(self.exact_points) is not tuple
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or any(type(word) is not int or not 0 <= word < 2**32 for word in pair)
                for pair in self.exact_points
            )
            or len({pair[0] for pair in self.exact_points}) != len(self.exact_points)
        ):
            raise ValueError("guarded math requires distinct exact binary32 points")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "schema": "merlin.guarded_f32_call.v1",
            "source_symbol": self.source_symbol,
            "body_symbol": self.body_symbol,
            "max_magnitude_bits": self.max_magnitude_bits,
            "library_sha256": self.library_sha256,
            "exact_points": [list(pair) for pair in self.exact_points],
            "provider_obligation": "return_bits_errno_floating_effects_for_all_supported_rounding_modes",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GuardedF32Call":
        fields = {
            "schema",
            "source_symbol",
            "body_symbol",
            "max_magnitude_bits",
            "library_sha256",
            "exact_points",
            "provider_obligation",
        }
        if (
            type(data) is not dict
            or set(data) != fields
            or data["schema"] != "merlin.guarded_f32_call.v1"
            or data["provider_obligation"] != "return_bits_errno_floating_effects_for_all_supported_rounding_modes"
            or type(data["exact_points"]) is not list
            or any(type(pair) is not list for pair in data["exact_points"])
        ):
            raise ValueError("guarded math needs an explicit complete provider contract")
        result = cls(
            data["source_symbol"],
            data["body_symbol"],
            data["max_magnitude_bits"],
            data["library_sha256"],
            tuple(tuple(pair) for pair in data["exact_points"]),
        )
        result.validate()
        return result


def _guarded_recipe(guard: GuardedF32Call) -> HostMathRecipe:
    if type(guard) is not GuardedF32Call:
        raise ValueError("guarded_f32 requires a typed provider contract")
    guard.validate()
    points = "".join(
        f"  if (word.u == {before}u) {{ word.u = {after}u; return word.f; }}\n" for before, after in guard.exact_points
    )
    return HostMathRecipe(
        source=(
            "#include <stdint.h>\n"
            f"extern float __real_{guard.source_symbol}(float);\n"
            f"extern float {guard.body_symbol}(float);\n"
            f"float __wrap_{guard.source_symbol}(float x) {{\n"
            "  union { float f; uint32_t u; } word = { .f = x };\n"
            + points
            + f"  if ((word.u & 0x7fffffffu) <= {guard.max_magnitude_bits}u) return {guard.body_symbol}(x);\n"
            f"  return __real_{guard.source_symbol}(x);\n}}\n"
        ),
        compile_flags=("-fno-builtin", "-fno-fast-math", "-ffp-contract=off"),
        link_flags=(f"-Wl,--wrap={guard.source_symbol}",),
        library_sha256=guard.library_sha256,
    )


def host_math_recipe(policy: str = "native", *, guarded_math: GuardedF32Call | None = None) -> HostMathRecipe:
    """Return a GNU-compatible link recipe; no implicit policy selection."""
    if policy == "guarded_f32":
        return _guarded_recipe(guarded_math)
    if guarded_math is not None:
        raise ValueError("guarded math requires explicit guarded_f32 host_math_policy")
    if policy == "native":
        return HostMathRecipe()
    if policy == "portable_trig":
        return HostMathRecipe(
            source=Path(__file__).with_name("portable_trig.c").read_text(),
            compile_flags=("-fno-builtin", "-fno-fast-math", "-ffp-contract=off"),
            link_flags=("-Wl,--wrap=sinf", "-Wl,--wrap=cosf"),
        )
    if policy != "expf_via_double":
        raise ValueError("host_math_policy must be 'native', 'portable_trig', 'expf_via_double', or 'guarded_f32'")
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
    *,
    guarded_math: GuardedF32Call | None = None,
) -> tuple[list[Path], tuple[str, ...]]:
    """Compile only an explicitly requested runtime object.

    The caller includes returned objects in its normal linked-byte identity.
    Native creates no files and invokes no compiler, preserving default bytes.
    """
    recipe = host_math_recipe(policy, guarded_math=guarded_math)
    if recipe.source is None:
        return [], ()
    work.mkdir(parents=True, exist_ok=True)
    source, obj = work / "host_math.c", work / "host_math.o"
    source.write_text(recipe.source)
    run([compiler, *cflags, *recipe.compile_flags, "-c", source, "-o", obj])
    if not obj.is_file():
        raise RuntimeError("host math compiler did not produce an object")
    return [obj], recipe.link_flags


def selected_libm_archive(
    gcc: Path,
    flags: list[str],
    link_flags: tuple[str, ...],
    *,
    runner: Callable,
    error_type: type[Exception] = ValueError,
) -> tuple[Path, str, str]:
    """Resolve the selected driver's archive before naming it as a link input.

    A path found by a later diagnostic query is not evidence for an earlier
    ``-lm`` link. The returned absolute archive is used directly by this build.
    """
    # The current host-math policy adds only symbol wrapping. A future search
    # override must get its own exact resolution contract, not silently select
    # the driver's default archive in place of the originally requested one.
    if any(
        flag in {"-L", "-B", "--sysroot", "-isysroot", "-Xlinker"}
        or flag.startswith(("-L", "-B", "--sysroot=", "-isysroot", "-specs="))
        for flag in flags
    ) or any(not flag.startswith("-Wl,--wrap=") for flag in link_flags):
        raise error_type("unsupported math-library search or link override")
    driver = gcc.resolve(strict=True)
    if not driver.is_file():
        raise error_type("selected math-library driver is not a regular file")
    driver_sha = sha256_file(driver)
    selected = runner(
        [gcc, *flags, *link_flags, "-nostdlib", "-nostartfiles", "-print-file-name=libm.a"]
    ).stdout.strip()
    candidate = Path(selected)
    if "\n" in selected or not candidate.is_absolute() or candidate.name != "libm.a" or not candidate.is_file():
        raise error_type("selected driver did not resolve a regular libm.a archive")
    archive = candidate.resolve(strict=True)
    if not archive.is_file() or sha256_file(driver) != driver_sha:
        raise error_type("selected math-library driver or archive changed during resolution")
    # A thin archive points at external member files: its own digest would not
    # close the library bytes selected by the linker.
    with archive.open("rb") as stream:
        if stream.read(8) != b"!<arch>\n":
            raise error_type("selected libm.a is not a self-contained regular archive")
    return archive, sha256_file(archive), driver_sha
