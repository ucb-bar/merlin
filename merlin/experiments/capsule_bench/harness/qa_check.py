"""Native CLI compatibility; the QA implementation belongs to merlin-experiments."""

from merlin_experiments.phase1.feedback.qa import main


def _native_context():
    import _common

    return _common.CONTEXT


if __name__ == "__main__":
    raise SystemExit(main(context=_native_context))
