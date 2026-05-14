"""Global configuration for the benchmark framework."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchConfig:
    build_dir: Path
    sm_arch: str = "sm_86"
    output_dir: Path = Path("results")
    warmup: int = 5
    iterations: int = 100
    batch_sizes: list[int] = field(default_factory=lambda: [1])
    correctness: bool = False
    profiler: str = "none"
    tracy_build_dir: Path | None = None
    gpu_index: int | None = None
    ctl_extra_flags: list[str] = field(default_factory=list)

    @property
    def env(self) -> dict[str, str]:
        """Environment variables for subprocess calls."""
        import os

        env = dict(os.environ)
        if self.gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        return env

    @property
    def compile_env(self) -> dict[str, str]:
        """Clean environment for iree-compile (avoids torch library conflicts)."""
        env = dict(self.env)
        env.pop("LD_PRELOAD", None)
        ld_path = env.get("LD_LIBRARY_PATH", "")
        cleaned = ":".join(
            p for p in ld_path.split(":")
            if "torch" not in p and "nvidia" not in p and "triton" not in p
        )
        if cleaned:
            env["LD_LIBRARY_PATH"] = cleaned
        else:
            env.pop("LD_LIBRARY_PATH", None)
        return env

    @property
    def iree_compile(self) -> str:
        return _find_tool(self.build_dir, "iree-compile")

    @property
    def iree_benchmark_module(self) -> str:
        return _find_tool(self.build_dir, "iree-benchmark-module")

    @property
    def iree_run_module(self) -> str:
        return _find_tool(self.build_dir, "iree-run-module")


def _find_tool(build_dir: Path, tool_name: str) -> str:
    import shutil

    candidates = [
        build_dir / "tools" / tool_name,
        build_dir / "bin" / tool_name,
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_mode & 0o111:
            return str(c)
    found = shutil.which(tool_name)
    if found:
        return found
    raise FileNotFoundError(f"Could not find {tool_name}")
