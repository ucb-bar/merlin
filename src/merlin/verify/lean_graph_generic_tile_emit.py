# ruff: noqa: E501
"""Actual bounded TILE query over arbitrary data and partial external effects.

Source/target control occurrences use the explicit occurrence-aware projection;
ordinary events keep the same uninterpreted partial meaning.
"""

from .lean_graph_generic_prepare import program
from .lean_graph_generic_tile_common import footer, header


def emit(prepared):
    left, right = prepared["minimal"]
    lower, upper, step, tile, block = (
        prepared["bounds"][key] for key in ("lower", "upper", "step", "tile", "block_step")
    )
    source = left[3]["loop"]
    outer = right[4]["loop"]
    inner = outer["body"][2]["loop"]
    if any(loop["carries"] or loop["initial"] for loop in (source, outer, inner)):
        raise ValueError("bounded generic TILE excludes carries before Lean")
    if len(source["body"]) != 2 or source["body"][-1] != {"ret": []}:
        raise ValueError("bounded generic TILE requires one opaque zero-result event")
    event = source["body"][0]
    if "effect" not in event or event["outputs"] or len(event["arguments"]) != 2 or event["arguments"][0] != 0:
        raise ValueError("bounded generic TILE body must capture actual IV and one data value")
    data_slot = event["arguments"][1] - 4
    if data_slot < 0:
        raise ValueError("bounded generic TILE event data must be captured outside literal setup")
    returned_slot = left[-1]["ret"][0] - 3
    if returned_slot < 0:
        raise ValueError("bounded generic TILE returns one captured data value")
    source_factored = program(left).replace(program(source["body"]), "Bsource", 1)
    outer_factored = program(outer["body"]).replace(program(inner["body"]), "Binner", 1)
    target_factored = program(right).replace(program(outer["body"]), "Bouter", 1)
    text, names = header(prepared)
    count = len(prepared["captures"])
    scope_left = "[" + ",".join(f"(.var {index})" for index in [0, 5, 6, 7, *range(8, 8 + count)]) + "]"
    scope_right = "[" + ",".join(f"(.var {index})" for index in range(8 + count)) + "]"
    returned = f"(data[{returned_slot}]?.getD default)"
    text += f"""@[irreducible] def Bsource : Program := {program(source["body"])}
@[irreducible] def Binner : Program := {program(inner["body"])}
@[irreducible] def Bouter : Program := {outer_factored}
def MinimalSource : Program := {program(left)}
def MinimalTarget : Program := {program(right)}
def FactoredSource : Program := {source_factored}
def FactoredTarget : Program := {target_factored}
theorem sourceFactored : MinimalSource = FactoredSource := by
  unfold MinimalSource FactoredSource Bsource
  rfl
theorem targetFactored : MinimalTarget = FactoredTarget := by
  unfold MinimalTarget FactoredTarget Bouter Binner
  rfl
def Esource (data : List Value) : List Value :=
  [codec.encode {step},codec.encode {upper},codec.encode {lower}]++data
def Etarget (data : List Value) : List Value :=
  [codec.encode {block},codec.encode {step},codec.encode {upper},codec.encode {lower}]++data
def Einner (data : List Value) (outer : Int) : List Value :=
  [codec.encode (min (outer+{block}) {upper}),codec.encode (outer+{block}),codec.encode outer]++Etarget codec data
def O (data : List Value) (index : Int) (carry : List Value) (world : World) : Outcome Value World Error :=
  evalProgram (m codec default primitive external) Bsource (codec.encode index::carry++Esource codec data) world
def I (data : List Value) (outer index : Int) (carry : List Value) (world : World) : Outcome Value World Error :=
  evalProgram (m codec default primitive external) Binner (codec.encode index::carry++Einner codec data outer) world
def U (data : List Value) (index : Int) (carry : List Value) (world : World) : Outcome Value World Error :=
  evalProgram (m codec default primitive external) Bouter (codec.encode index::carry++Etarget codec data) world
def InnerLeft : List Expr := {scope_left}
def InnerRight : List Expr := {scope_right}
theorem innerScopeChecked : Scope.check registry InnerLeft InnerRight Bsource Binner = true := by
  unfold Bsource Binner
  decide
theorem inner_same (data : List Value) (outer index : Int) (carry : List Value) (world : World)
    (arity : carry.length = 0) :
    I codec default primitive external data outer index carry world =
      O codec default primitive external data index carry world := by
  have empty : carry = [] := List.length_eq_zero_iff.mp arity
  subst carry
  simp only [I,O,Bsource,Binner,Esource,Einner,Etarget,evalProgram,reads,m_default,List.cons_append,
    List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some]
theorem inner_function (data : List Value) (outer : Int) :
    (fun index values state => evalProgram (m codec default primitive external) Binner
      (codec.encode index::(values++codec.encode (min (outer+{block}) {upper})::codec.encode (outer+{block})::codec.encode outer::codec.encode {block}::codec.encode {step}::codec.encode {upper}::codec.encode {lower}::data)) state) =
      I codec default primitive external data outer := by
  funext index values state
  simp only [I,Einner,Etarget,List.cons_append,List.nil_append] <;> rfl
"""
    simps = ",".join(
        [
            "U,Bouter,Etarget",
            *names,
            "m_default,m_index,m_iterations,IndexOp.eval,evalProgram,reads",
            "List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero",
            "List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind",
            "List.length_cons,List.length_nil,codec.roundTrip",
        ]
    )
    text += f"""theorem outer_form (data : List Value) (base : Int) (world : World) :
    U codec default primitive external data base [] world =
      (loopEval 0 (I codec default primitive external data base)
        (indexIterations base (min (base+{block}) {upper}) {step}) [] world).bind
          (fun result => .ok ([],result.2)) := by
  simp only [{simps}]
  rw [inner_function]
theorem outer_same (data : List Value) (base : Nat) (carry : List Value) (world : World)
    (arity : carry.length = 0) :
    U codec default primitive external data (Int.ofNat base) carry world =
      loopEval 0 (O codec default primitive external data)
        ((scfRange base (min (base+{block}) {upper}) {step}).map Int.ofNat) carry world := by
  have empty : carry = [] := List.length_eq_zero_iff.mp arity
  subst carry
  rw [outer_form]
  have replace := loopEval_congr 0
    (I codec default primitive external data (Int.ofNat base))
    (O codec default primitive external data)
    (fun index values state size => inner_same codec default primitive external data (Int.ofNat base) index values state size)
  rw [replace]
  have successor : Int.ofNat (base+{block}) = Int.ofNat base+{block} := by
    simpa only [Int.ofNat_eq_natCast,Int.cast_ofNat_Int] using Int.natCast_add base {block}
  have clipped : min (Int.ofNat base+{block}) ({upper} : Int) = Int.ofNat (min (base+{block}) {upper}) := by
    rw [← successor]
    simpa only [IndexOp.eval,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Int.ofNat_eq_natCast,Int.cast_ofNat_Int] using minui_natural (base+{block}) {upper}
  rw [clipped]
  have interval : indexIterations (Int.ofNat base) (Int.ofNat (min (base+{block}) {upper})) {step} =
      (scfRange base (min (base+{block}) {upper}) {step}).map Int.ofNat := by
    simpa only [Int.ofNat_eq_natCast,Int.cast_ofNat_Int] using naturalRange base (min (base+{block}) {upper}) {step} (by decide)
  rw [interval]
  cases run : loopEval 0 (O codec default primitive external data)
    ((scfRange base (min (base+{block}) {upper}) {step}).map Int.ofNat) [] world with
  | error failure => rfl
  | ok result =>
    have size := loopEval_arity 0 _ _ _ _ run
    rcases result with ⟨values,finalWorld⟩
    have empty : values = [] := List.length_eq_zero_iff.mp size
    simp only [Except.bind,empty]
theorem segmented (data : List Value) (world : World) :
    loopEval 0 (O codec default primitive external data) (indexIterations {lower} {upper} {step}) [] world =
      loopEval 0 (U codec default primitive external data) (indexIterations {lower} {upper} {block}) [] world := by
  have sourceRange : indexIterations {lower} {upper} {step} = (scfRange {lower} {upper} {step}).map Int.ofNat := by
    simpa only [Int.ofNat_eq_natCast,Int.cast_ofNat_Int] using naturalRange {lower} {upper} {step} (by decide)
  have targetRange : indexIterations {lower} {upper} {block} = (scfRange {lower} {upper} {block}).map Int.ofNat := by
    simpa only [Int.ofNat_eq_natCast,Int.cast_ofNat_Int] using naturalRange {lower} {upper} {block} (by decide)
  rw [sourceRange,targetRange]
  exact segment_loop 0 _ _ {lower} {upper} {step} {tile} (by decide) (by decide)
    (fun index values state size => outer_same codec default primitive external data index values state size) [] world
theorem source_function (data : List Value) :
    (fun index values state => evalProgram (m codec default primitive external) Bsource
      (codec.encode index::(values++codec.encode {step}::codec.encode {upper}::codec.encode {lower}::data)) state) =
      O codec default primitive external data := by
  funext index values state
  simp only [O,Esource,List.cons_append,List.nil_append] <;> rfl
theorem outer_function (data : List Value) :
    (fun index values state => evalProgram (m codec default primitive external) Bouter
      (codec.encode index::(values++codec.encode {block}::codec.encode {step}::codec.encode {upper}::codec.encode {lower}::data)) state) =
      U codec default primitive external data := by
  funext index values state
  simp only [U,Etarget,List.cons_append,List.nil_append] <;> rfl
"""
    form_simps = ",".join(
        [
            "FactoredSource,FactoredTarget",
            *names,
            "m_default,m_index,m_iterations,IndexOp.eval,evalProgram,reads",
            "List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero",
            "List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind",
            "List.length_cons,List.length_nil,codec.roundTrip",
        ]
    )
    for label, callback, stride in (("source", "O", step), ("target", "U", block)):
        factored = "sourceFactored" if label == "source" else "targetFactored"
        fold = "source_function" if label == "source" else "outer_function"
        text += f"""theorem {label}_form (data : List Value) (world : World) :
    evalProgram (m codec default primitive external) Minimal{label.title()} data world =
      (loopEval 0 ({callback} codec default primitive external data) (indexIterations {lower} {upper} {stride}) [] world).bind
        (fun result => .ok ([{returned}],result.2)) := by
  rw [{factored}]
  simp only [{form_simps}]
  rw [{fold}]
  cases run : loopEval 0 ({callback} codec default primitive external data)
    (indexIterations {lower} {upper} {stride}) [] world with
  | error failure => rfl
  | ok result =>
    have size := loopEval_arity 0 _ _ _ _ run
    have empty : result.1 = [] := List.length_eq_zero_iff.mp size
    simp only [Except.bind,empty,List.nil_append,reads,List.map_cons,List.map_nil,
      List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some]
"""
    text += """theorem minimal (environment : List Value) (world : World) :
    evalProgram (m codec default primitive external) MinimalSource environment world =
      evalProgram (m codec default primitive external) MinimalTarget environment world := by
  rw [source_form,target_form,segmented]
"""
    return text + footer(prepared)
