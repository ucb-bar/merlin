"""Explicit capability selections over the frozen fresh10 environment.

These selections are independent of historical EL numbers. Source membership is
verified separately; a capability record alone never grants an implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

BASE_REVISION = "f9d7fb5c3d6fa2e5168e982953dd0b1d9b480076"
NAMES = (
    "fresh10-tooling-v1/infra",
    "fresh10-tooling-v1/scheduling",
    "fresh10-tooling-v1/no-reference-kernels",
)
#: From-scratch (unseeded) fresh10 arms that read exactly the coverage study's pinned raw originals
#: (COVERAGE_RAW_REFERENCE_SHA256, the same check as cca-coverage-v1/raw) and no reference map. Like
#: the coverage arms they stage the current reviewed full compiler and library, never the baseline
#: projection, which reproduces its library only on the baseline's own export.
RAW_ORIGINALS_NAMES = (
    "fresh10-tooling-v1/full-ref",
    "fresh10-tooling-v1/full-ref-bank-all",
    "fresh10-tooling-v1/infra-originals",
)
#: Raw-originals arms that withhold scheduling or transformation sources. The current reviewed
#: library has no reviewed per-capability source removal (only the baseline projection has one),
#: so these refuse preparation instead of staging sources their capability record withholds.
UNPREPARABLE_CURRENT_LIBRARY = frozenset((RAW_ORIGINALS_NAMES[2],))
#: The reviewed baseline source/library projection (fresh10_profile_data.json ``profiles`` key).
PROJECTIONS = {NAMES[0]: "A", NAMES[1]: "B", NAMES[2]: "C"}
#: Seeded continuations of one frozen fresh10 compiler for the CCA coverage study. They keep the
#: fresh10 author tools, services and prompt composition, stage the current reviewed compiler
#: sources and library instead of the baseline projection, and read one pinned raw reference set.
COVERAGE_NAMES = ("cca-coverage-v1/raw",)
#: Bank arms of the same study: exactly one precomputed CCA bank (one ``reference_subsets`` condition
#: payload, read-only as ``cca-evidence``) plus only the raw originals that bank was prepared from
#: (its ``bank.json`` ``raw`` list, read-only as ``reference-kernels``); other models' originals and
#: other banks are never granted. Data, not code: each entry maps a profile name to the SHA256 of that
#: condition's ``bank.json``, which transitively pins every bank file, the admitted unit roster and
#: the raw grant. Enforcement reads only the pinned bank, never these names.
#: Historical operator reference maps (REFERENCE_MAP.json of earlier reference arms), by exact bytes.
#: No coverage arm receives one under any alias; a bank arm's only map-like data is the inventory the
#: host derives from its verified bank. Any file grant carrying a reference-map schema also refuses.
WITHHELD_REFERENCE_MAP_SHA256 = frozenset((
    "51568c1de0842ab143d2fad503a146a9a41056f08c21c3ae73fdf7dbbbf65720",
    "5134c637c5d154008e7c877090a72868009f82655d4b2d23e58c8efa4c173ef7",
    "d4feeebaa6fc7eb6fe103c049c39c92237126c59fca5d5fa51afef60f709eb46",
    "9d7d56c59072493f2f1aa4b3a2471811e21a5da62e1d548e900201e275bfcfd3",
))
COVERAGE_BANK_PREFIX = "cca-coverage-v1/bank-"
#: Single-source banks from the v3 expert reference corpus (upstream Exo only, each whole model
#: alone, and all originals together); each pin is one prepared condition's bank.json, from the
#: products expert-kernel-subsets-v3-exo-upstream and expert-kernel-subsets-v3-single-<source>.
COVERAGE_BANKS: dict[str, str] = {
    "cca-coverage-v1/bank-cca-exo": "cb3380f20cad7405067666faf9800a5416220d7dd63433bb7f0e2d7dab57b0b2",
    "cca-coverage-v1/bank-cca-resnet50": "6ef300a6fed4cd9b1f585744ec24498ff24ccc875aaa8878a81b11ef4d4e0a33",
    "cca-coverage-v1/bank-cca-tinyllama": "4e5c773062cb3edb7ec7fad21a109e4abd05ac5cd1d7e15c7fb5c9ac6e9016f6",
    "cca-coverage-v1/bank-cca-smolvla": "cdcdf1ad6e6f155799084b67df2d1157643961073697b736ec070a3a10120e4d",
    "cca-coverage-v1/bank-cca-all": "8c5ce71524744c4e0c6236a21462f7b196acf82de362d2087d7367ca15d1265a",
}
#: The from-scratch bank arm serves the bank prepared from every original; its raw grant (that bank's
#: raw list plus the originals manifest) is therefore exactly the 23 pinned originals.
RAW_ORIGINALS_BANKS: dict[str, str] = {
    RAW_ORIGINALS_NAMES[1]: COVERAGE_BANKS["cca-coverage-v1/bank-cca-all"],
}
#: The Raw arm's reference grant, by exact bytes (a bank arm's is its bank's raw list, a subset of
#: these): every file of the expert
#: reference database's originals/ (v2): the three whole-model executables, the upstream Exo Gemmini apps,
#: generated public reference and licenses, the supplied Exo LLM source and the originals manifest.
#: Derived analyses (kernel bodies, mappings, indexes, CCA products, maps) are never part of it.
COVERAGE_RAW_REFERENCE_SHA256 = (
    # originals/models/resnet50.elf
    "97decefb6ff3852d01ce69d30f60d97f7f13bb6ce86815081899da66b87765fd",
    # originals/models/smolvla.elf
    "e598857dd6001d177868d34dac598a333a4c393dc9202734e6587456b121196b",
    # originals/models/tinyllama.elf
    "f7a70f3a9ad2791956a808ae5aa212043614307e8ed5d5ab22328af41534de65",
    # originals/supplied-source/exo_llm.c
    "d177c24a8099161e5388eee0e33ac534916bb87e96373f1b1d1f8dc9e1456109",
    # originals/supplied-source/exo_llm.h
    "793fcd421365cf0f0a2850f8007a4aa340bef86313d09710db269f8b8d55190f",
    # originals/upstream-exo/LICENSE.md
    "1d182f2d26230d4369b8062d97eb961920b418a96fff45570f662b304f9afe26",
    # originals/upstream-exo/apps/gemmini/src/exo/conv.py
    "11fb72594bcc3a4c5f06e03b024e1e943e8b7be75f97b1f709d0f2ee79814c71",
    # originals/upstream-exo/apps/gemmini/src/exo/matmul.py
    "76e5f8163916ff6c411f402e99fcd37f150d6ff838c7b92ab1329d19de558347",
    # originals/upstream-exo/licenses/gemmini-rocc-tests-LICENSE.txt
    "c3b204bde93ad5054b0797fdbbb23a36e68d25672dc2d5ed39c3f69299457fac",
    # originals/upstream-exo/licenses/riscv-test-env-LICENSE.txt
    "43b7309213293b4323d65301a90e05472579abcd51feb260a850354bcb49ff8f",
    # originals/upstream-exo/licenses/riscv-tests-LICENSE.txt
    "43b7309213293b4323d65301a90e05472579abcd51feb260a850354bcb49ff8f",
    # originals/upstream-exo/licenses/rocc-software-LICENSE.txt
    "b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1",
    # originals/upstream-exo/public-reference/elf/conv_3_input58_valid.elf
    "bbd0f851063a1e09446fbfc7770a1095d1dcfc6e21e946c776d101963040efad",
    # originals/upstream-exo/public-reference/elf/matmul_512x512x512.elf
    "92f44d811042a49b51616e864609be17251aa0a418facd1cabe2a57a7ae76771",
    # originals/upstream-exo/public-reference/generated/gemmini_lib.c
    "d8e458b7eab4477f89f211d42e3eaeaa1272b690dee95b6f85176885f147a88c",
    # originals/upstream-exo/public-reference/generated/gemmini_lib.h
    "c0321ca4dca631cc9dac180408e0cd573cd432e365207041c3e8c01a5f9b4098",
    # originals/upstream-exo/public-reference/headers/gemmini.h
    "06988403de641bbf22fea84e76debd863ab3185498f2b48ceb4b475d4af0914e",
    # originals/upstream-exo/public-reference/headers/gemmini_params.h
    "d6db18f8edbfeb854780286d8c2afb81ae3b9103224522478a9047d5550272e5",
    # originals/upstream-exo/public-reference/platform/gemm_acc_malloc.c
    "ae1131ccaa4ab215b974b2bd898f0908cbebe7a2d20080d19c9ba67c21433cab",
    # originals/upstream-exo/public-reference/platform/gemm_acc_malloc.h
    "e314b0a86f410a7d99d1cec3cbbc101e3f731cc2a8fe9ec6ad80a3467eaebd0f",
    # originals/upstream-exo/public-reference/platform/gemm_malloc.c
    "5c39bf34f6a0f21f8a0433fce382e884b8d4829178af26487694b518400a9f29",
    # originals/upstream-exo/public-reference/platform/gemm_malloc.h
    "efb76a29ce5f2a90b95cb133149fb9ac87f491c2a847998406ffff6b7f48a01f",
    # originals/manifest.json
    "1f6420ad01c4d1f27eea25a390dc88f4a066a6abac69b1a15b87aa7e92994dde",
)
#: The originals manifest (last pin above). It names every original, so a bank arm receives it only
#: when its bank was prepared from every original; that arm's grant is then exactly the Raw arm's.
COVERAGE_ORIGINALS_MANIFEST_SHA256 = COVERAGE_RAW_REFERENCE_SHA256[-1]
TOOLS = ("merlin_infra", "xdsl_kit", "isa_tools", "rtl_generators", "rtl_facts")
SERVICES = frozenset(("reference_analysis", "automatic_kernel_learning", "scheduling_semantics",
                      "transform_library", "catalog_advice", "raw_kernel_references",
                      "performance_diagnostics", "compiler_integration"))


@dataclass(frozen=True)
class Capabilities:
    name: str
    scheduling_semantics: bool
    transform_library: bool
    raw_kernel_references: bool
    #: A coverage continuation: seeded, current sources, pinned raw references only.
    coverage: bool = False
    #: A coverage bank arm: SHA256 of the one admitted precomputed CCA bank's bank.json.
    bank_sha256: str | None = None
    #: A from-scratch arm whose raw grant is exactly the pinned originals; never seeded, never a map.
    raw_originals: bool = False

    def allows(self, service):
        if service not in SERVICES:
            raise ValueError("unknown fresh10 author capability: " + str(service))
        if service == "reference_analysis":
            # Only a bank arm, and only over its pinned precomputed bank (reference_bank enforces it).
            return self.bank_sha256 is not None
        if service == "automatic_kernel_learning":
            return False
        if service in {"performance_diagnostics", "compiler_integration"}:
            return True
        if service == "catalog_advice":
            return self.transform_library
        return getattr(self, service)

    def record(self):
        return {name: self.allows(name) for name in sorted(SERVICES)}


def _sha256(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def coverage_names():
    """Every coverage profile: the Raw arm and each prepared bank arm."""
    return (*COVERAGE_NAMES, *sorted(COVERAGE_BANKS))


def bank_pin(name):
    """The admitted bank.json SHA256 of a bank arm; None for every other profile."""
    if name not in COVERAGE_BANKS:
        return None
    pin = COVERAGE_BANKS[name]
    if not name.startswith(COVERAGE_BANK_PREFIX) or name == COVERAGE_BANK_PREFIX or not _sha256(pin):
        raise ValueError("coverage bank profile requires a bank-* name and one bank.json SHA256 pin")
    return pin


def resolve(name):
    if name == NAMES[0]:
        return Capabilities(name, False, False, True)
    if name == NAMES[1]:
        return Capabilities(name, True, False, True)
    if name == NAMES[2]:
        # User scope: remove reference material, retain general compiler passes.
        return Capabilities(name, True, True, False)
    if name == RAW_ORIGINALS_NAMES[0]:
        return Capabilities(name, True, True, True, raw_originals=True)
    if name in RAW_ORIGINALS_BANKS:
        # Precomputed bank service only: reference analysis exists solely through the pinned bank.
        pin = RAW_ORIGINALS_BANKS[name]
        if not _sha256(pin):
            raise ValueError("raw-originals bank profile requires one bank.json SHA256 pin")
        return Capabilities(name, True, True, True, bank_sha256=pin, raw_originals=True)
    if name == RAW_ORIGINALS_NAMES[2]:
        return Capabilities(name, False, False, True, raw_originals=True)
    if name in COVERAGE_NAMES:
        # The no-reference-kernels tooling plus the study's raw reference grant; CCA stays off.
        return Capabilities(name, True, True, True, coverage=True)
    if name in COVERAGE_BANKS:
        # The Raw arm's tooling, one pinned precomputed CCA bank and only that bank's raw originals.
        return Capabilities(name, True, True, True, coverage=True, bank_sha256=bank_pin(name))
    raise ValueError("unknown fresh10 tooling profile: " + str(name))


def require_preparable(name):
    capabilities = resolve(name)
    if name in UNPREPARABLE_CURRENT_LIBRARY:
        raise ValueError(name + " withholds scheduling/transformation sources, but the current reviewed "
                         "library has no reviewed source removal for them; it is not preparable")
    return capabilities


def broker_allowed(name, action):
    capabilities = resolve(name)
    if action == "reference-analysis":
        # Served only from the arm's pinned precomputed bank; never on-demand CCA of raw references.
        return capabilities.bank_sha256 is not None
    if action == "kernel-learning":
        return False
    if action in {"common-transform", "common-check"}:
        return capabilities.transform_library
    return True  # Ordinary numerical selfcheck and measurement admission stay common.
