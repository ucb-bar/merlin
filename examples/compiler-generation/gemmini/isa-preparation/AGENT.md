# Public instruction data preparation

This target example selects and checks public declaration data using generic
Merlin extractors and generators. Keep target assumptions explicit here; do not
add target constants to shared compiler tooling. Preserve original hardware
facts, source pins and unknown semantics. Generated facts and check-only dialects
belong under the task output root. They are not an author backend seed and do not
establish command acceptance, numerical correctness or scheduling legality.
