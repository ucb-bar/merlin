"""Build, run and judge complete quantized models for a device submission.

``python -m merlin.model_check`` compiles every program of a model once, replays each device kernel on
frozen boundary inputs, runs the whole chain per input on a functional ISA model, and judges the outputs
against an FP and an INT8 reference under the ratio rule. The contract (CLI, numerics policy, census,
report) is ``docs/reference/model_check.md``; the report schema is
``merlin/schemas/model_check_report.schema.yaml``.

Models, programs, chains and judged outputs are data (``examples/<target>/numerics/models.yaml``);
nothing here names a model or a target.
"""

CONTRACT_VERSION = "1.5"
REPORT_SCHEMA = "merlin.model_check.report.v1"
