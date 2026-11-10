# src/merlin/compile/analysis

The actual public analysis implementations live in `src/merlin/analysis`.
Contraction-shape observation lives in `compiler/ir/contraction_shapes.py`.
This package retains exact temporary module aliases for active import and pickle
cutover only; add no implementation or new consumer here. Fresh SDK26 source
uses canonical owners and explicit existing grants. Historical SDKs and pickle
replay retain their exact archived source environment.
