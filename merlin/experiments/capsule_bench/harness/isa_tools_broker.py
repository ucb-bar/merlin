"""Native CLI compatibility launcher for the package-owned ISA tool broker."""


def _native_context():
    import _common as C

    return C.CONTEXT


if __name__ == "__main__":
    from merlin_experiments.phase1.brokers.isa_tools import main

    raise SystemExit(main(context=_native_context))
