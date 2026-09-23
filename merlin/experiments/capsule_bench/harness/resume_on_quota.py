"""Compatibility CLI; interruption policy is experiments-owned."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("merlin_experiments.phase1.recovery", run_name="__main__")
