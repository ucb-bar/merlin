#!/usr/bin/env python3
import sys, json

flags = [a for a in sys.argv[1:] if a.startswith("-")]
pos = [a for a in sys.argv[1:] if not a.startswith("-")]
emit = [f for f in flags if f.startswith("--emit-command-buffer=")]
if "--convert-gemmini-to-llvm-rocc" in flags:
    print("module {}"); sys.exit(0)
if emit:
    out = emit[0].split("=", 1)[1]
    open(out,"w").write("this is not json !!!")
    sys.exit(0)
if "--convert-iface-to-gemmini" in flags:
    print("module {}"); sys.exit(0)
sys.exit(0)
