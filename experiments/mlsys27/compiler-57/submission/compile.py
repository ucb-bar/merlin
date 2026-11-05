#!/usr/bin/env python3
"""CLI for the self-contained xDSL Gemmini compiler."""
import argparse
import json
import sys
from pathlib import Path
from frontend import parse_source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify-diagnostics',action='store_true')
    p.add_argument('--convert-iface-to-gemmini',action='store_true')
    p.add_argument('--emit-command-buffer')
    p.add_argument('--emit-target-artifact',action='store_true')
    p.add_argument('input')
    a=p.parse_args()
    text=Path(a.input).read_text()
    cb=parse_source(text)
    if a.verify_diagnostics:
        return 0
    try:
        from backend import compile_buffer
        lowered=compile_buffer(cb)
    except ImportError as e:
        raise RuntimeError('Device compiler integration is incomplete: '+str(e)) from e
    cb.pop('_source_text',None)
    if a.emit_command_buffer:
        Path(a.emit_command_buffer).write_text(json.dumps(cb,indent=2))
    print(lowered)
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(type(e).__name__+': '+str(e),file=sys.stderr)
        sys.exit(1)
