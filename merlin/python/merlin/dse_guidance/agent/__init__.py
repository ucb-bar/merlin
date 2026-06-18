"""Agent slots for dse_guidance — LLM proposes, a deterministic gate disposes.

Mirrors the targetgen/agent runtime ("the LLM proposes into a typed slot; the deterministic gate
disposes"). The only agent here is a devil's-advocate critic over an emitted insight-mining run:
it proposes over-claims/gaps in the *interpretation* layer, and a deterministic citation gate
rejects any critique that does not quote a real artifact. Agents NEVER produce numbers — every
quantity stays bit-exact from the committed artifacts + the verifier.
"""
