# Extend one compiler pass

You are given ONE Python module from a RISC-V vector (RVV) compiler and one machine-checked promise it
currently fails. Return the complete new module source.

## The promise

- **Divergence axis**: `{axis}`
- **What the emitted code must acquire**: `{intended_facet}`
- **Our current value on that axis**: `{ours}`
- **The expert's value on that axis**: `{expert}`
- **The change the router asks for**: {change}

The axis value is not something you assert. It is *lifted from the emitted assembly* after your pass
runs, by a decoder that reads the instruction stream. Your pass is accepted only if that lift reports
the promised value.

## Measured evidence

{evidence}

## The module

The current source is in `current_pass.py` in your working directory. It is the file you are rewriting.

## Hard rules

1. **Return the COMPLETE new module** in a single fenced ```python block. It replaces the file whole.
   Nothing outside the block is read.
2. **Change only this module.** It will be placed in an overlay where every other module resolves to
   the real checkout. A change that needs a second file edited cannot be applied and will be refused.
3. **Numerics must not move.** The build is run against a golden and must stay bit-exact. A pass that
   changes results is refused however fast it is.
4. **The frozen baseline must not move.** With an empty feature set the compiler must still lower
   byte-identically. Every recorded measurement in this project is read against that control, so a
   pass that perturbs it invalidates all of them. Keep your change behind the feature that already
   gates this pass.
5. **Never name a model.** A pass that can tell which model it is compiling is overfit by
   construction — the whole point is that a lever found on one model transfers to the next. Model
   names in comments recording a measurement are fine; a model name the code can compare or dispatch
   on is rejected automatically.
6. **Do not reach for the answer.** Reading a golden, asserting your own gate verdict, mutating the
   feature registry, or monkeypatching are all detected structurally and rejected before anything is
   built.
7. **Fail closed.** Where you cannot derive a fact, raise or leave the IR untouched. Never substitute
   a default and never silently skip — a pass that quietly does nothing reports success and costs more
   than one that errors.
8. **Derive, never hardcode.** No opcode, no funct value, no mesh dimension, no memory base, no
   register field, no target name. Every such fact comes from the target's own description at run
   time. This project plugs in arbitrary hardware targets; a literal here defeats that.

## What to aim at

Say briefly, in a comment at the top, what you changed and why it should move the axis. Then write the
code. Prefer the smallest change that acquires the facet: the gate does not reward size, and a large
rewrite is more likely to move numerics or the frozen baseline and be refused for a reason unrelated to
your idea.
