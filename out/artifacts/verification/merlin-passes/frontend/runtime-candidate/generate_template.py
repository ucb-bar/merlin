#!/usr/bin/env python3
"""Regenerate raw Lean AST data for review from an authoritative isolated-cut export.

This is a data transcription tool. Checker booleans and soundness facts are recomputed in
Lean. The maintained FrontendMaskTemplate additionally factors the count prefix for proof
composition; regenerating that implementation requires review and a successful Lean build.
"""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Transcribe authoritative cut IR into reviewable Lean AST data; no proof facts are generated."
)
parser.add_argument("input", type=Path, help="authoritative exported isolated SmolVLA cut JSON")
parser.add_argument("output", type=Path, help="write a raw AST candidate for review, not the proof implementation")
args = parser.parse_args()
path = args.input
j = json.loads(path.read_text())
fn = j["module"]["ops"][0]
block = fn["regions"][0]["blocks"][0]
ops = block["ops"]
if len(block["args"]) != 3 or len(ops) != 26:
    raise ValueError("expected the isolated three-input, 26-operation captured cut")
rename = {value["id"]: new for value, new in zip(block["args"], (50, 51, 52), strict=True)}
result_ids = [60, 61, 40, 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, None, 62, 63, 41, 42, 15, 16, 17, None, 21]
for o, identifier in zip(ops, result_ids):
    if identifier is not None:
        rename[o["results"][0]["id"]] = identifier
for op, newargs, newresults in [(ops[5], [300, 301], [302]), (ops[8], [303, 304], [305])]:
    b = op["regions"][0]["blocks"][0]
    for v, new in zip(b["args"], newargs):
        rename[v["id"]] = new
    rename[b["ops"][0]["results"][0]["id"]] = newresults[0]
for index, base, out in [(15, 100, [13, 14]), (23, 200, [18, 19])]:
    o = ops[index]
    for v, new in zip(o["results"], out):
        rename[v["id"]] = new
    b = o["regions"][0]["blocks"][0]
    for v, new in zip(b["args"], range(base, base + 3)):
        rename[v["id"]] = new
    rename[b["ops"][0]["results"][0]["id"]] = base + 3
    branch = b["ops"][1]
    for v, new in zip(branch["results"], [base + 4, base + 5]):
        rename[v["id"]] = new
    for nested, new in zip(branch["regions"][0]["blocks"][0]["ops"], range(base + 6, base + 9)):
        if nested["results"]:
            rename[nested["results"][0]["id"]] = new


def q(s):
    return json.dumps(s, ensure_ascii=False)


def ls(values):
    return "[" + ", ".join(values) + "]"


def ty(t):
    k = t["kind"]
    if k == "int":
        return "(.int " + str(t["width"]) + ")"
    if k == "index":
        return ".index"
    if k == "tensor" and t.get("encoding") is None:
        return (
            "(.tensor "
            + ls("none" if n is None else "(some " + str(n) + ")" for n in t["shape"])
            + " "
            + ty(t["element"])
            + " none)"
        )
    if k == "function":
        return "(.function " + ls(map(ty, t["inputs"])) + " " + ls(map(ty, t["results"])) + ")"
    raise ValueError(t)


def affine(e):
    k = e["kind"]
    return "(." + k + " " + str(e["pos"] if k in ("dim", "symbol") else e["value"]) + ")"


def attr(a):
    k = a["kind"]
    if k == "string":
        return "(.str " + q(a["value"]) + ")"
    if k == "int":
        return "(.int " + str(a["value"]) + " " + ty(a["type"]) + ")"
    if k == "enum":
        return "(.enum " + q(a["dialect"]) + " " + q(a["name"]) + " " + q(a["value"]) + ")"
    if k == "type":
        return "(.type " + ty(a["value"]) + ")"
    if k == "dense_array":
        return "(.denseArray " + ty(a["element"]) + " " + ls(str(v) for v in a["values"]) + ")"
    if k == "array":
        return "(.array " + ls(map(attr, a["elements"])) + ")"
    if k == "affine_map":
        return (
            "(.affineMap {numDims := "
            + str(a["dims"])
            + ", numSyms := "
            + str(a["symbols"])
            + ", results := "
            + ls(map(affine, a["results"]))
            + "})"
        )
    raise ValueError(a)


def attrs(a):
    return ls("(" + q(k) + ", " + attr(v) + ")" for k, v in a.items())


def value(v):
    return "{id := " + str(rename[v["id"]]) + ", ty := " + ty(v["type"]) + "}"


def region(r):
    return (
        "(.mk " + ls("(.mk " + ls(map(value, b["args"])) + " " + ls(map(op, b["ops"])) + ")" for b in r["blocks"]) + ")"
    )


def op(o):
    return (
        "(.mk "
        + q(o["name"])
        + " "
        + ls(map(value, o["results"]))
        + " "
        + ls(str(rename[v]) for v in o["operands"])
        + " "
        + attrs(o["attrs"])
        + " "
        + attrs(o["props"])
        + " "
        + ls(map(region, o["regions"]))
        + " "
        + ls(str(i) for i in o["successors"])
        + ")"
    )


names = [
    "maskCollapse0",
    "maskExpand0",
    "valuesCollapse",
    "maskFlatten0",
    "countEmpty",
    "countMap",
    "countZero",
    "countSplat",
    "countReduce",
    "countExtract",
    "countCast",
    "gatherLower",
    "gatherStep",
    "gatherUpper",
    "compactEmpty",
    "gatherLoop",
    "maskCollapse1",
    "maskExpand1",
    "seedCollapse",
    "maskFlatten1",
    "scatterLower",
    "scatterStep",
    "scatterUpper",
    "scatterLoop",
    "resultExpand",
    "returned",
]
text = """/- Concrete SmolVLA mask-selection cut, transcribed from the authoritative MLIR export.
The checker compares decoded programs to this AST; this file contributes data, not proof facts.
SSA identifiers are renamed to stable proof-local names. Input/output shapes remain captured.
This is raw regenerated data; the maintained template factors the same AST into named fragments.
-/
import MerlinVerify.Rules.FrontendMaskCountProgram
import MerlinVerify.Rules.FrontendMaskAlias

namespace MerlinVerify.Rules.FrontendMaskTemplate
open MerlinVerify

"""
for name, o in zip(names, ops):
    text += "def " + name + " : Op :=\n  " + op(o) + "\n\n"
text += "def operations : List Op := " + ls(names) + "\n\n"
text += (
    'def subject : Module := {ops := [.mk "func.func" [] [] '
    + attrs(fn["attrs"])
    + " "
    + attrs(fn["props"])
    + " [.mk [.mk "
    + ls(map(value, block["args"]))
    + " operations]] []]}\n\n"
)
text += "end MerlinVerify.Rules.FrontendMaskTemplate\n"
args.output.write_text(text)
