"""Actual exact-integer rewrites; representation and complete graph boundaries."""

from __future__ import annotations

from merlin.common.paths import repo_root
from merlin.verify.lean_graph_state_integer import intake

REFERENCE = repo_root().parent / "merlin-graph-proof-intake"


def standalone_source(width=8):
    t = f"i{width}"
    max_signed = (1 << (width - 1)) - 1
    # i1 has no signed literal +1; the bit-pattern -1 is the native one word.
    one = 1 if width > 1 else -1
    two = 2 if width > 2 else -1
    return f"""builtin.module {{
  func.func private @watch({t}, {t}, {t})
  func.func @main(%x: {t}) -> {t} {{
    %zero = arith.constant 0 : {t}
    %one = arith.constant {one} : {t}
    %max = arith.constant {max_signed} : {t}
    %two = arith.constant {two} : {t}
    %fold = arith.addi %max, %two : {t}
    %chain = arith.subi %fold, %one : {t}
    %product = arith.muli %chain, %two : {t}
    %n0 = arith.addi %zero, %x : {t}
    %n1 = arith.addi %n0, %zero : {t}
    %n2 = arith.subi %n1, %zero : {t}
    %n3 = arith.muli %one, %n2 : {t}
    %n4 = arith.muli %n3, %one : {t}
    func.call @watch(%fold, %chain, %product) : ({t}, {t}, {t}) -> ()
    func.return %n4 : {t}
  }}
  func.func @untouched(%x: {t}) -> {t} {{ func.return %x : {t} }}
}}"""


def chain_source(width=32, *, lower=0, upper=12, stride=1):
    t = f"i{width}"
    return f"""builtin.module {{
  func.func private @provider({t}, index) -> {t}
  func.func private @watch({t}, index)
  func.func private @final({t})
  func.func @main(%x: {t}, %bias: {t}) -> {t} {{
    %lb = arith.constant {lower} : index
    %ub = arith.constant {upper} : index
    %st = arith.constant {stride} : index
    %zero = arith.constant 0 : {t}
    %one = arith.constant 1 : {t}
    %a = arith.constant -3 : {t}
    %b = arith.constant 3 : {t}
    %foldedzero = arith.addi %a, %b : {t}
    %later = arith.addi %x, %foldedzero : {t}
    %dead = arith.subi %x, %bias : {t}
    %out = scf.for %iv = %lb to %ub step %st iter_args(%acc = %later) -> ({t}) {{
      %invariant = arith.addi %bias, %bias : {t}
      %duplicate = arith.addi %bias, %bias : {t}
      %neutral = arith.muli %acc, %one : {t}
      %sum = arith.addi %neutral, %duplicate : {t}
      %sum2 = arith.addi %sum, %invariant : {t}
      func.call @watch(%sum2, %iv) : ({t}, index) -> ()
      %v = func.call @provider(%sum2, %iv) : ({t}, index) -> {t}
      %next = arith.subi %v, %zero : {t}
      scf.yield %next : {t}
    }}
    func.call @final(%out) : ({t}) -> ()
    func.return %out : {t}
  }}
  func.func @untouched(%x: {t}) -> {t} {{ func.return %x : {t} }}
}}"""


def nested_source(width=16, *, empty=False):
    t = f"i{width}"
    return f"""builtin.module {{
  func.func private @provider({t}, {t}, index) -> ({t}, {t})
  func.func private @watch({t}, {t})
  func.func private @other({t}, {t})
  func.func @main(%x: {t}, %bias: {t}) -> ({t}, {t}) {{
    %lb = arith.constant 0 : index
    %ub = arith.constant {0 if empty else 3} : index
    %innerub = arith.constant 2 : index
    %st = arith.constant 1 : index
    %zero = arith.constant 0 : {t}
    %one = arith.constant 1 : {t}
    %a = arith.constant -3 : {t}
    %b = arith.constant 3 : {t}
    %fold = arith.addi %a, %b : {t}
    %out:2 = scf.for %iv = %lb to %ub step %st iter_args(%first = %x, %second = %bias) -> ({t}, {t}) {{
      %neutral = arith.addi %first, %fold : {t}
      %inner:2 = scf.for %j = %lb to %innerub step %st iter_args(%left = %neutral, %right = %second) -> ({t}, {t}) {{
        %capture = arith.muli %bias, %one : {t}
        %keep = arith.subi %left, %zero : {t}
        func.call @watch(%keep, %capture) : ({t}, {t}) -> ()
        %v:2 = func.call @provider(%keep, %right, %j) : ({t}, {t}, index) -> ({t}, {t})
        scf.yield %v#0, %v#1 : {t}, {t}
      }}
      scf.yield %inner#0, %inner#1 : {t}, {t}
    }}
    func.call @watch(%out#0, %out#1) : ({t}, {t}) -> ()
    func.return %out#1, %out#0 : {t}, {t}
  }}
  func.func @untouched(%x: {t}) -> {t} {{ func.return %x : {t} }}
}}"""


def produce(tmp_path, text, *, chain=False, settings=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "input.mlir"
    source.write_text(text)
    return intake(REFERENCE, tmp_path / "producer", source=source, chain=chain, settings=settings)


def identity_source(kind):
    if kind == "flagged":
        text = """builtin.module {func.func @main(%x:i8)->i8 {
          %z=arith.constant 0:i8
          %v=arith.addi %x,%z overflow<nsw>:i8
          func.return %v:i8
        }}"""
    elif kind == "index":
        text = """builtin.module {func.func @main(%x:index)->index {
          %z=arith.constant 0:index
          %v=arith.addi %x,%z:index
          func.return %v:index
        }}"""
    else:
        text = """builtin.module {func.func @main(%x:f32)->f32 {
          %z=arith.constant 0.0:f32
          %v=arith.addf %x,%z:f32
          func.return %v:f32
        }}"""
    return text
