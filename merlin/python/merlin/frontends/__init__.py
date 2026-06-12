"""Merlin frontends: ingest external IR into the core-dialect pipeline.

:mod:`linalg_mlir` parses linalg-on-tensors MLIR (as emitted by model2MLIR) with xDSL
and extracts a matmul inventory; :mod:`facts` lifts the inventory into contract-level
facts and drives the existing lowering pipeline with real model shapes.
"""
