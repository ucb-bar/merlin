/- Runtime refusal checks for the combined static-source and initialized-execution checker.
The base is a simple total test model, not a claim about IEEE arithmetic. -/
import MerlinVerify.Check.PolicyW8A8Source

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules

def base : FloatModel where
  op _ _ _ := 0
  cmp _ _ _ _ := false
  convert _ _ _ _ := 0
  toInt _ _ _ _ := none
  ofInt _ _ _ _ := 0

def tensor (shape : List Nat) (elt : Ty) (value : Val) : RVal := .tensor ⟨shape, elt, fun _ => value⟩
def config (kind : W8A8SourceContract.Kind) : W8A8SourceContract.Config := ⟨kind, [2], 64⟩
def types (kind : W8A8SourceContract.Kind) : List (ValueId × Ty) :=
  [0, 1, 2].zip (W8A8SourceContract.operandTypes (config kind))
def op (kind : W8A8SourceContract.Kind) : Op :=
  .mk (W8A8SourceContract.name kind) [⟨3, W8A8SourceContract.resultType (config kind)⟩] [0, 1, 2] []
    ([("quant_min", .int (-128) (.int 64)), ("quant_max", .int 127 (.int 64))] ++
      if kind = .quantize then [("output_dtype", .str "int8")] else []) [] []
def scale : RVal := tensor [] (.float .f32) (.some (.flt .f32 0x3f800000#32))
def zero : RVal := tensor [] (.int 64) (.some (.bits 64 0))
def data (kind : W8A8SourceContract.Kind) : RVal :=
  if kind = .quantize then tensor [2] (.float .f32) (.some (.flt .f32 0))
  else tensor [2] (.int 8) (.some (.bits 8 128))
def check (kind : W8A8SourceContract.Kind) (args : List RVal) : Bool :=
  PolicyW8A8Source.check (types kind) (op kind) base args

#guard check .quantize [data .quantize, scale, zero]
#guard check .dequantize [data .dequantize, scale, zero]
#guard !check .dequantize [tensor [2] (.int 8) .poison, scale, zero]
#guard !check .dequantize [data .dequantize, tensor [] (.float .f32) .poison, zero]
#guard !check .dequantize [data .dequantize, scale, tensor [] (.int 64) .poison]
#guard !check .dequantize [tensor [2] (.int 8) (.some (.flt .f32 0)), scale, zero]
#guard !check .dequantize [data .dequantize, scale, tensor [1] (.int 64) (.some (.bits 64 0))]
#guard !check .dequantize [data .dequantize, scale]
#guard !check .quantize [data .dequantize, scale, zero]

def main : IO Unit := IO.println "source instance guards passed"
