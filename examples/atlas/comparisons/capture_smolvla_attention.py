#!/usr/bin/env python3
"""Merlin C graph attention recapture entrypoint for the Atlas M2 trajectory.

The authoritative continuation is the delivered C graph. This adapter explicitly
selects that backend; it does not claim to traverse the Torch model.
"""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).absolute().parents[4] / 'capture_smolvla_attention.py'),run_name='__main__')
