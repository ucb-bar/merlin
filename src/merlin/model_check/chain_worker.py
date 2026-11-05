"""One worker of a bundle's INT8 reference pool: ``python -m merlin.model_check.chain_worker <request> <k>``.

The request (written by :func:`merlin.model_check.references.build` with ``workers`` > 1) names the
cases and the pool's directory; worker ``k`` runs every case it claims, in order, in this one process,
sharing one :class:`merlin.model_check.chain.ChainCache`, with its own scratch.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    from merlin.model_check import references as REF

    REF.run_worker(argv[0], int(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
