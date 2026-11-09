#!/usr/bin/env python3
"""Compatibility CLI for installed Merlin Phase 0 derivation."""

from merlin_experiments.phase0.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
