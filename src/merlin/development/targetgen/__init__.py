"""TargetGen pipeline: ingest -> extract -> plan -> generate -> validate.

Onboards a hardware target from its own sources: RTL facts and capability derivation, capsule corpora,
grading and oracles, tier policy, and publishing the generated compiler package.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
