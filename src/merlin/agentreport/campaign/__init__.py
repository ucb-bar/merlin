"""Campaign figure: one agentic compiler-construction campaign drawn as a single stitched run.

``manifest`` names the inputs, ``records`` reads them (read-only), ``figure`` computes the series and
draws them, ``cli`` writes a versioned product. Run ``python -m merlin.agentreport.campaign --help``.
"""

#: Product name (declared in ``merlin/contract/storage.yaml`` ``product_roots``) and figure version.
PRODUCT = "campaign-figure"
FIGURE_VERSION = 3
