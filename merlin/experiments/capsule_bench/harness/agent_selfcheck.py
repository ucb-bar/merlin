"""Native CLI compatibility for the trusted functional feedback worker."""

from merlin_experiments.phase1.feedback.selfcheck import main


def _native_context():
    import _common

    return _common.CONTEXT


if __name__ == "__main__":
    raise SystemExit(main(context=_native_context))
