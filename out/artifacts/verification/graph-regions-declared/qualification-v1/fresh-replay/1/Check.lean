import Merlin.GraphRegionsDeclaredBridge

import Merlin.GraphRegionsDeclaredRaw

import Merlin.GraphRegionsDeclaredProjection

open Merlin MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

def actualBefore : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [] (.int 32) none)] [(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [] (.int 32) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [] (.int 32) none)]))),("sym_name",(.str "forward"))] [] [(.mk [(.mk [⟨0,(.tensor [(some 2),(some 23)] (.int 16) none)⟩,⟨1,(.tensor [(some 2),(some 23)] (.int 16) none)⟩,⟨2,(.tensor [(some 2),(some 23)] (.int 16) none)⟩,⟨3,(.tensor [(some 2),(some 23)] (.int 16) none)⟩,⟨4,(.tensor [] (.int 32) none)⟩] [(.mk "arith.constant" [⟨5,(.int 32)⟩] [] [("value",(.int (1:Int) (.int 32)))] [] [] []),(.mk "tensor.empty" [⟨6,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [] [] [] [] []),(.mk "tensor.empty" [⟨7,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [] [] [] [] []),(.mk "tensor.empty" [⟨8,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [] [] [] [] []),(.mk "linalg.generic" [⟨9,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [2,0,3,6] [("indexing_maps",(.array [(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩)])),("iterator_types",(.array [(.enum "linalg" "iterator_type" "parallel"),(.enum "linalg" "iterator_type" "parallel")])),("operandSegmentSizes",(.denseArray (.int 32) [(3:Int),(1:Int)]))] [] [(.mk [(.mk [⟨10,(.int 16)⟩,⟨11,(.int 16)⟩,⟨12,(.int 16)⟩,⟨13,(.int 16)⟩] [(.mk "arith.muli" [⟨14,(.int 16)⟩] [10,11] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "arith.addi" [⟨15,(.int 16)⟩] [14,12] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "linalg.yield" [] [15] [] [] [] [])])])] []),(.mk "linalg.generic" [⟨16,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [9,1,7] [("indexing_maps",(.array [(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩)])),("iterator_types",(.array [(.enum "linalg" "iterator_type" "parallel"),(.enum "linalg" "iterator_type" "parallel")])),("operandSegmentSizes",(.denseArray (.int 32) [(2:Int),(1:Int)]))] [] [(.mk [(.mk [⟨17,(.int 16)⟩,⟨18,(.int 16)⟩,⟨19,(.int 16)⟩] [(.mk "arith.addi" [⟨20,(.int 16)⟩] [17,18] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "linalg.yield" [] [20] [] [] [] [])])])] []),(.mk "linalg.generic" [⟨21,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [9,1,8] [("indexing_maps",(.array [(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩)])),("iterator_types",(.array [(.enum "linalg" "iterator_type" "parallel"),(.enum "linalg" "iterator_type" "parallel")])),("operandSegmentSizes",(.denseArray (.int 32) [(2:Int),(1:Int)]))] [] [(.mk [(.mk [⟨22,(.int 16)⟩,⟨23,(.int 16)⟩,⟨24,(.int 16)⟩] [(.mk "arith.muli" [⟨25,(.int 16)⟩] [22,23] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "linalg.yield" [] [25] [] [] [] [])])])] []),(.mk "tensor.empty" [⟨26,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [] [] [] [] []),(.mk "linalg.generic" [⟨27,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [16,21,4,26] [("indexing_maps",(.array [(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩)])),("iterator_types",(.array [(.enum "linalg" "iterator_type" "parallel"),(.enum "linalg" "iterator_type" "parallel")])),("operandSegmentSizes",(.denseArray (.int 32) [(3:Int),(1:Int)]))] [] [(.mk [(.mk [⟨28,(.int 16)⟩,⟨29,(.int 16)⟩,⟨30,(.int 32)⟩,⟨31,(.int 16)⟩] [(.mk "arith.addi" [⟨32,(.int 16)⟩] [28,29] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "arith.trunci" [⟨33,(.int 16)⟩] [30] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "arith.addi" [⟨34,(.int 16)⟩] [32,33] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "linalg.yield" [] [34] [] [] [] [])])])] []),(.mk "tensor.extract" [⟨35,(.int 32)⟩] [4] [] [] [] []),(.mk "arith.addi" [⟨36,(.int 32)⟩] [35,5] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "tensor.empty" [⟨37,(.tensor [] (.int 32) none)⟩] [] [] [] [] []),(.mk "tensor.insert" [⟨38,(.tensor [] (.int 32) none)⟩] [36,37] [] [] [] []),(.mk "func.return" [] [16,21,27,38,2,0,1,3,4] [] [] [] [])])])] [])],[],[]⟩

def actualRetained : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [] (.int 32) none)] [(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [] (.int 32) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [(some 2),(some 23)] (.int 16) none),(.tensor [] (.int 32) none)]))),("sym_name",(.str "forward"))] [] [(.mk [(.mk [⟨0,(.tensor [(some 2),(some 23)] (.int 16) none)⟩,⟨1,(.tensor [(some 2),(some 23)] (.int 16) none)⟩,⟨2,(.tensor [(some 2),(some 23)] (.int 16) none)⟩,⟨3,(.tensor [(some 2),(some 23)] (.int 16) none)⟩,⟨4,(.tensor [] (.int 32) none)⟩] [(.mk "arith.constant" [⟨5,(.int 32)⟩] [] [("value",(.int (1:Int) (.int 32)))] [] [] []),(.mk "tensor.empty" [⟨6,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [] [] [] [] []),(.mk "tensor.empty" [⟨7,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [] [] [] [] []),(.mk "tensor.empty" [⟨8,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [] [] [] [] []),(.mk "linalg.generic" [⟨9,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [2,0,3,6] [("indexing_maps",(.array [(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩)])),("iterator_types",(.array [(.enum "linalg" "iterator_type" "parallel"),(.enum "linalg" "iterator_type" "parallel")])),("operandSegmentSizes",(.denseArray (.int 32) [(3:Int),(1:Int)]))] [] [(.mk [(.mk [⟨10,(.int 16)⟩,⟨11,(.int 16)⟩,⟨12,(.int 16)⟩,⟨13,(.int 16)⟩] [(.mk "arith.muli" [⟨14,(.int 16)⟩] [10,11] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "arith.addi" [⟨15,(.int 16)⟩] [14,12] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "linalg.yield" [] [15] [] [] [] [])])])] []),(.mk "linalg.generic" [⟨16,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [9,1,7] [("indexing_maps",(.array [(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩)])),("iterator_types",(.array [(.enum "linalg" "iterator_type" "parallel"),(.enum "linalg" "iterator_type" "parallel")])),("operandSegmentSizes",(.denseArray (.int 32) [(2:Int),(1:Int)]))] [] [(.mk [(.mk [⟨17,(.int 16)⟩,⟨18,(.int 16)⟩,⟨19,(.int 16)⟩] [(.mk "arith.addi" [⟨20,(.int 16)⟩] [17,18] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "linalg.yield" [] [20] [] [] [] [])])])] []),(.mk "linalg.generic" [⟨21,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [9,1,8] [("indexing_maps",(.array [(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩)])),("iterator_types",(.array [(.enum "linalg" "iterator_type" "parallel"),(.enum "linalg" "iterator_type" "parallel")])),("operandSegmentSizes",(.denseArray (.int 32) [(2:Int),(1:Int)]))] [] [(.mk [(.mk [⟨22,(.int 16)⟩,⟨23,(.int 16)⟩,⟨24,(.int 16)⟩] [(.mk "arith.muli" [⟨25,(.int 16)⟩] [22,23] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "linalg.yield" [] [25] [] [] [] [])])])] []),(.mk "tensor.empty" [⟨26,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [] [] [] [] []),(.mk "linalg.generic" [⟨27,(.tensor [(some 2),(some 23)] (.int 16) none)⟩] [16,21,4,26] [("indexing_maps",(.array [(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩),(.affineMap ⟨2,0,[]⟩),(.affineMap ⟨2,0,[(.dim 0),(.dim 1)]⟩)])),("iterator_types",(.array [(.enum "linalg" "iterator_type" "parallel"),(.enum "linalg" "iterator_type" "parallel")])),("operandSegmentSizes",(.denseArray (.int 32) [(3:Int),(1:Int)]))] [] [(.mk [(.mk [⟨28,(.int 16)⟩,⟨29,(.int 16)⟩,⟨30,(.int 32)⟩,⟨31,(.int 16)⟩] [(.mk "arith.addi" [⟨32,(.int 16)⟩] [28,29] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "arith.trunci" [⟨33,(.int 16)⟩] [30] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "arith.addi" [⟨34,(.int 16)⟩] [32,33] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "linalg.yield" [] [34] [] [] [] [])])])] []),(.mk "tensor.extract" [⟨35,(.int 32)⟩] [4] [] [] [] []),(.mk "arith.addi" [⟨36,(.int 32)⟩] [35,5] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "tensor.empty" [⟨37,(.tensor [] (.int 32) none)⟩] [] [] [] [] []),(.mk "tensor.insert" [⟨38,(.tensor [] (.int 32) none)⟩] [36,37] [] [] [] []),(.mk "func.return" [] [16,21,27,38,2,0,1,3,4] [] [] [] [])])])] [])],[],[]⟩

def stage0 : ScalarStage := ⟨4,"('produce', 2, 2, 3, 4, 0)",[2,0,3],9,["tensor[2,23]i16","tensor[2,23]i16","tensor[2,23]i16"],"tensor[2,23]i16",⟨[16,16,16],[⟨(.binary 16 .mul),[0,1]⟩,⟨(.binary 16 .add),[3,2]⟩],4,16⟩⟩

def program0 : Program := ⟨[16,16,16],[⟨(.binary 16 .mul),[0,1]⟩,⟨(.binary 16 .add),[3,2]⟩],4,16⟩

def scalar0_source : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.int 16),(.int 16),(.int 16)] [(.int 16)]))),("sym_name",(.str "source"))] [] [(.mk [(.mk [⟨0,(.int 16)⟩,⟨1,(.int 16)⟩,⟨2,(.int 16)⟩] [(.mk "arith.muli" [⟨3,(.int 16)⟩] [0,1] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "arith.addi" [⟨4,(.int 16)⟩] [3,2] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "func.return" [] [4] [] [] [] [])])])] [])],[],[]⟩

def scalar0_adapter : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.int 16),(.int 16),(.int 16)] [(.int 16)]))),("sym_name",(.str "forward"))] [] [(.mk [(.mk [⟨0,(.int 16)⟩,⟨1,(.int 16)⟩,⟨2,(.int 16)⟩] [(.mk "arith.muli" [⟨3,(.int 16)⟩] [0,1] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "arith.addi" [⟨4,(.int 16)⟩] [3,2] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "func.return" [] [4] [] [] [] [])])])] [])],[],[]⟩

def scalar0_selected : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.int 16),(.int 16),(.int 16)] [(.int 16)]))),("sym_name",(.str "forward"))] [] [(.mk [(.mk [⟨0,(.int 16)⟩,⟨1,(.int 16)⟩,⟨2,(.int 16)⟩] [(.mk "arith.muli" [⟨3,(.int 16)⟩] [0,1] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "arith.addi" [⟨4,(.int 16)⟩] [3,2] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "func.return" [] [4] [] [] [] [])])])] [])],[],[]⟩

theorem scalar0_adapter_accepted : checkScalar scalar0_source scalar0_adapter "source" "forward" none none program0 = true := by
  apply checkScalar_iff.mpr
  exact ⟨rfl, rfl, by decide⟩

theorem scalar0_selected_accepted : checkScalar scalar0_source scalar0_selected "source" "forward" none none program0 = true := by
  apply checkScalar_iff.mpr
  exact ⟨rfl, rfl, by decide⟩

def bound0 : BoundCall := ⟨"produce","('produce', 2, 2, 3, 4, 0)",[("n",46),("a",2),("ao",0),("b",3),("bo",0),("c",4),("co",0),("dst",0),("out",0),("width",2)],[⟨1,0,92⟩,⟨2,0,92⟩,⟨3,0,92⟩],⟨0,0,92⟩,["little","little","little","little"],[2,23],16,"synchronous",[],none,[]⟩

def schema0 : CallSchema := ⟨"produce","('produce', 2, 2, 3, 4, 0)",["n","a","ao","b","bo","c","co","dst","out","width"],[32,32,32,32,32,32,32,32,32,32],[(0,⟨0,0,512⟩),(1,⟨0,0,512⟩),(2,⟨1,0,512⟩),(3,⟨2,0,512⟩),(4,⟨3,0,512⟩),(5,⟨4,0,512⟩),(6,⟨5,0,512⟩),(7,⟨6,0,512⟩)],[("a","ao"),("b","bo"),("c","co")],"dst","out","n","width",["little","little","little","little"],[2,23]⟩

def stage1 : ScalarStage := ⟨5,"('add', 2, 0, 5, 0, 6)",[9,1],16,["tensor[2,23]i16","tensor[2,23]i16"],"tensor[2,23]i16",⟨[16,16],[⟨(.binary 16 .add),[0,1]⟩],2,16⟩⟩

def program1 : Program := ⟨[16,16],[⟨(.binary 16 .add),[0,1]⟩],2,16⟩

def scalar1_source : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.int 16),(.int 16)] [(.int 16)]))),("sym_name",(.str "source"))] [] [(.mk [(.mk [⟨0,(.int 16)⟩,⟨1,(.int 16)⟩] [(.mk "arith.addi" [⟨2,(.int 16)⟩] [0,1] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "func.return" [] [2] [] [] [] [])])])] [])],[],[]⟩

def scalar1_adapter : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.int 16),(.int 16)] [(.int 16)]))),("sym_name",(.str "forward"))] [] [(.mk [(.mk [⟨0,(.int 16)⟩,⟨1,(.int 16)⟩] [(.mk "arith.addi" [⟨2,(.int 16)⟩] [0,1] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "func.return" [] [2] [] [] [] [])])])] [])],[],[]⟩

def scalar1_selected : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.int 16),(.int 16)] [(.int 16)]))),("sym_name",(.str "forward"))] [] [(.mk [(.mk [⟨0,(.int 16)⟩,⟨1,(.int 16)⟩] [(.mk "arith.addi" [⟨2,(.int 16)⟩] [0,1] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "func.return" [] [2] [] [] [] [])])])] [])],[],[]⟩

theorem scalar1_adapter_accepted : checkScalar scalar1_source scalar1_adapter "source" "forward" none none program1 = true := by
  apply checkScalar_iff.mpr
  exact ⟨rfl, rfl, by decide⟩

theorem scalar1_selected_accepted : checkScalar scalar1_source scalar1_selected "source" "forward" none none program1 = true := by
  apply checkScalar_iff.mpr
  exact ⟨rfl, rfl, by decide⟩

def bound1 : BoundCall := ⟨"add","('add', 2, 0, 5, 0, 6)",[("n",46),("a",0),("ao",0),("b",5),("bo",0),("c",0),("co",0),("dst",6),("out",0),("width",2)],[⟨0,0,92⟩,⟨4,0,92⟩],⟨5,0,92⟩,["little","little","little"],[2,23],16,"synchronous",[],none,[]⟩

def schema1 : CallSchema := ⟨"add","('add', 2, 0, 5, 0, 6)",["n","a","ao","b","bo","c","co","dst","out","width"],[32,32,32,32,32,32,32,32,32,32],[(0,⟨0,0,512⟩),(1,⟨0,0,512⟩),(2,⟨1,0,512⟩),(3,⟨2,0,512⟩),(4,⟨3,0,512⟩),(5,⟨4,0,512⟩),(6,⟨5,0,512⟩),(7,⟨6,0,512⟩)],[("a","ao"),("b","bo")],"dst","out","n","width",["little","little","little"],[2,23]⟩

def stage2 : ScalarStage := ⟨6,"('multiply', 2, 1, 5, 0, 7)",[9,1],21,["tensor[2,23]i16","tensor[2,23]i16"],"tensor[2,23]i16",⟨[16,16],[⟨(.binary 16 .mul),[0,1]⟩],2,16⟩⟩

def program2 : Program := ⟨[16,16],[⟨(.binary 16 .mul),[0,1]⟩],2,16⟩

def scalar2_source : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.int 16),(.int 16)] [(.int 16)]))),("sym_name",(.str "source"))] [] [(.mk [(.mk [⟨0,(.int 16)⟩,⟨1,(.int 16)⟩] [(.mk "arith.muli" [⟨2,(.int 16)⟩] [0,1] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "func.return" [] [2] [] [] [] [])])])] [])],[],[]⟩

def scalar2_adapter : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.int 16),(.int 16)] [(.int 16)]))),("sym_name",(.str "forward"))] [] [(.mk [(.mk [⟨0,(.int 16)⟩,⟨1,(.int 16)⟩] [(.mk "arith.muli" [⟨2,(.int 16)⟩] [0,1] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "func.return" [] [2] [] [] [] [])])])] [])],[],[]⟩

def scalar2_selected : Module := ⟨[(.mk "func.func" [] [] [("function_type",(.type (.function [(.int 16),(.int 16)] [(.int 16)]))),("sym_name",(.str "forward"))] [] [(.mk [(.mk [⟨0,(.int 16)⟩,⟨1,(.int 16)⟩] [(.mk "arith.muli" [⟨2,(.int 16)⟩] [0,1] [("overflowFlags",(.enum "arith" "overflow" "none"))] [] [] []),(.mk "func.return" [] [2] [] [] [] [])])])] [])],[],[]⟩

theorem scalar2_adapter_accepted : checkScalar scalar2_source scalar2_adapter "source" "forward" none none program2 = true := by
  apply checkScalar_iff.mpr
  exact ⟨rfl, rfl, by decide⟩

theorem scalar2_selected_accepted : checkScalar scalar2_source scalar2_selected "source" "forward" none none program2 = true := by
  apply checkScalar_iff.mpr
  exact ⟨rfl, rfl, by decide⟩

def bound2 : BoundCall := ⟨"multiply","('multiply', 2, 1, 5, 0, 7)",[("n",46),("a",1),("ao",0),("b",5),("bo",0),("c",0),("co",0),("dst",7),("out",0),("width",2)],[⟨0,0,92⟩,⟨4,0,92⟩],⟨6,0,92⟩,["little","little","little"],[2,23],16,"synchronous",[],none,[]⟩

def schema2 : CallSchema := ⟨"multiply","('multiply', 2, 1, 5, 0, 7)",["n","a","ao","b","bo","c","co","dst","out","width"],[32,32,32,32,32,32,32,32,32,32],[(0,⟨0,0,512⟩),(1,⟨0,0,512⟩),(2,⟨1,0,512⟩),(3,⟨2,0,512⟩),(4,⟨3,0,512⟩),(5,⟨4,0,512⟩),(6,⟨5,0,512⟩),(7,⟨6,0,512⟩)],[("a","ao"),("b","bo")],"dst","out","n","width",["little","little","little"],[2,23]⟩

def sites : List GenericSite := [⟨stage0,[2,23],[10,11,12,14,15],6,13⟩,⟨stage1,[2,23],[17,18,20],7,19⟩,⟨stage2,[2,23],[22,23,25],8,24⟩]

def items : List CallStage := [⟨stage0,bound0⟩,⟨stage1,bound1⟩,⟨stage2,bound2⟩]

def initialKnown : List Nat := [2,0,3,1]

def addresses : Nat → Window := fun slot => if slot = 2 then ⟨1,0,92⟩ else if slot = 0 then ⟨2,0,92⟩ else if slot = 3 then ⟨3,0,92⟩ else if slot = 9 then ⟨0,0,92⟩ else if slot = 1 then ⟨4,0,92⟩ else if slot = 16 then ⟨5,0,92⟩ else if slot = 21 then ⟨6,0,92⟩ else ⟨0,0,0⟩

def actualKernel : RawKernel := ⟨"retained_producer",[⟨"input0",[2,23],"i16","read"⟩,⟨"input1",[2,23],"i16","read"⟩,⟨"input2",[2,23],"i16","read"⟩,⟨"input3",[2,23],"i16","read"⟩,⟨"output0",[2,23],"i16","write"⟩,⟨"output1",[2,23],"i16","write"⟩],[],[⟨"produce",[("n",46),("a",2),("ao",0),("b",3),("bo",0),("c",4),("co",0),("dst",0),("out",0),("width",2)],[],[],none,[],[],none⟩,⟨"add",[("n",46),("a",0),("ao",0),("b",5),("bo",0),("c",0),("co",0),("dst",6),("out",0),("width",2)],[],[],none,[],[],none⟩,⟨"multiply",[("n",46),("a",1),("ao",0),("b",5),("bo",0),("c",0),("co",0),("dst",7),("out",0),("width",2)],[],[],none,[],[],none⟩]⟩

def originalKernel : RawKernel := ⟨"retained_producer",[⟨"input0",[2,23],"i16","read"⟩,⟨"input1",[2,23],"i16","read"⟩,⟨"input2",[2,23],"i16","read"⟩,⟨"input3",[2,23],"i16","read"⟩,⟨"output0",[2,23],"i16","write"⟩,⟨"output1",[2,23],"i16","write"⟩],[],[⟨"produce",[("n",46),("a",2),("ao",0),("b",3),("bo",0),("c",4),("co",0),("dst",0),("out",1024),("width",2)],[],[],none,[],[],none⟩,⟨"add",[("n",46),("a",0),("ao",1024),("b",5),("bo",0),("c",0),("co",0),("dst",6),("out",0),("width",2)],[],[],none,[],[],none⟩,⟨"multiply",[("n",46),("a",1),("ao",1024),("b",5),("bo",0),("c",0),("co",0),("dst",7),("out",0),("width",2)],[],[],none,[],[],none⟩]⟩

def actualGraph : Graph := (projectModule actualBefore sites).getD ⟨[],[],[],[]⟩

theorem actual_kernel_decoded : checkKernel actualKernel originalKernel.name originalKernel.arguments [schema0,schema1,schema2] [bound0,bound1,bound2] = true := by decide

theorem actual_calls_valid : checkSequence 46 addresses initialKnown items = true := by decide

theorem actual_generic_binding : SourceBinding actualBefore actualRetained sites actualGraph := by
  refine ⟨rfl, rfl, ?_⟩
  constructor
  · simp [SitesMatch, findSite, sites, stage0, stage1, stage2, checkGeneric, genericOperation,
      genericWellFormed, renamedOperations, renameSlot, identityMap,
      typedIds, typeTag, Op.results, Program.Valid, instructionsValid, finalWidths,
      Primitive.Valid, Primitive.inputs, Primitive.width, Primitive.name, Primitive.attrs]
    all_goals repeat first | apply And.intro | rfl | decide
  · intro site member
    simp only [sites, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl <;>
      refine ⟨_, rfl, rfl, by decide⟩

#print axioms actual_kernel_decoded

#print axioms actual_calls_valid

#print axioms actual_generic_binding

def actualPlan : Plan := ⟨[2,0,3,9,1,16,21],[⟨stage0.ordinal,stage0.role,[0,1,2],[3],stage0.inputTypes,[stage0.outputType]⟩,⟨stage1.ordinal,stage1.role,[3,4],[5],stage1.inputTypes,[stage1.outputType]⟩,⟨stage2.ordinal,stage2.role,[3,4],[6],stage2.inputTypes,[stage2.outputType]⟩],[0,1,2,4],[5,6],[(-1,3),(-1,3),(-1,3),(0,2),(-1,3),(1,3),(2,3)]⟩

def actualFused : FusedBody := ⟨items.map (fun item => item.stage.node),initialKnown,[stage1.output,stage2.output]⟩

theorem actual_realization : checkRealization actualGraph actualGraph 4 3 [stage0.ordinal,stage1.ordinal,stage2.ordinal] actualPlan actualFused = true := by decide

theorem actual_boundary : items.map (fun item => item.stage.node) = (actualGraph.nodes.drop 4).take 3 := by rfl

theorem actual_nodes_unique : ((items.map CallStage.stage).map ScalarStage.node).Nodup := by decide

theorem actual_output_coverage : ∀ slot ∈ liveOuts (items.map (fun item => item.stage.node)) (actualGraph.nodes.drop 7) actualGraph.observations, slot ∈ finalKnown initialKnown items := by decide

theorem actual_complete_observers {α M : Type} [Inhabited α]
    (primitive : Primitive → List α → α)
    (fallback : ∀ node, TypedAction node (Tensor 46 α) (fun _ _ => True))
    (read : M → Window → Option (Tensor 46 α)) (invoke : BoundCall → M → Option M)
    (contracts : ∀ item ∈ items, InstructionContract read invoke primitive item.call item.stage.program)
    (pack : StateEffects.State (Tensor 46 α) → M)
    (initialized : ∀ entry, Refines read addresses initialKnown entry (pack entry))
    (native : StateEffects.Action
      ⟨liveIns ((actualGraph.nodes.drop 4).take 3),
        liveOuts ((actualGraph.nodes.drop 4).take 3)
          (actualGraph.nodes.drop 7) actualGraph.observations⟩
      (Tensor 46 α))
    (publication : ∀ entry final, runCalls invoke items (pack entry) = some final →
      ∀ slot ∈ liveOuts ((actualGraph.nodes.drop 4).take 3)
          (actualGraph.nodes.drop 7) actualGraph.observations,
        read final (addresses slot) = some (native.step entry slot))
    (initial : StateEffects.State (Tensor 46 α)) :
    SourceBinding actualBefore actualRetained sites actualGraph ∧
    checkKernel actualKernel originalKernel.name originalKernel.arguments
      [schema0,schema1,schema2] [bound0,bound1,bound2] = true ∧
    (∀ arguments, ScalarExecution primitive scalar0_source arguments (program0.evaluate primitive arguments) ∧
      ScalarExecution primitive scalar0_selected arguments (program0.evaluate primitive arguments)) ∧
    (∀ arguments, ScalarExecution primitive scalar1_source arguments (program1.evaluate primitive arguments) ∧
      ScalarExecution primitive scalar1_selected arguments (program1.evaluate primitive arguments)) ∧
    (∀ arguments, ScalarExecution primitive scalar2_source arguments (program2.evaluate primitive arguments) ∧
      ScalarExecution primitive scalar2_selected arguments (program2.evaluate primitive arguments)) ∧
    (let meaning := registry primitive fallback (items.map CallStage.stage)
     actualGraph.observations.map (executeNodes meaning (actualGraph.nodes.drop 7)
       (native.step (executeNodes meaning (actualGraph.nodes.take 4) initial))) =
       actualGraph.observations.map (executeNodes meaning actualGraph.nodes initial)) := by
  refine ⟨actual_generic_binding, actual_kernel_decoded,
    (fun arguments => checkScalar_sound scalar0_selected_accepted primitive arguments),
    (fun arguments => checkScalar_sound scalar1_selected_accepted primitive arguments),
    (fun arguments => checkScalar_sound scalar2_selected_accepted primitive arguments), ?_⟩
  exact checked_calls_preserve_observers primitive fallback actual_realization items actual_boundary
    addresses initialKnown
    (checkSequence_sound actual_calls_valid) actual_nodes_unique read invoke contracts pack initialized
    actual_output_coverage native publication initial
#print axioms actual_complete_observers

-- A mathematical nonzero initialized witness. Its total interpretation is
-- illustrative; no target primitive fidelity follows from this computation.
def witnessPrimitive (primitive : Primitive) (values : List Nat) : Nat :=
  match primitive with
  | .constant _ literal => literal.toNat
  | .binary _ .add => values[0]?.getD 0 + values[1]?.getD 0
  | .binary _ .sub => values[0]?.getD 0 - values[1]?.getD 0
  | .binary _ .mul => values[0]?.getD 0 * values[1]?.getD 0
  | .cast .. => values[0]?.getD 0
def witnessInitial : StateEffects.State (Tensor 46 Nat) := fun slot _ =>
  if slot = stage0.inputs[0]?.getD 0 then 3 else
  if slot = stage0.inputs[1]?.getD 0 then 5 else
  if slot = stage0.inputs[2]?.getD 0 then 7 else
  if slot = stage1.inputs[1]?.getD 0 then 11 else 1
theorem actual_nonzero_witness : ∀ index : Fin 46,
    executeStages witnessPrimitive (items.map CallStage.stage) witnessInitial stage1.output index = 33 ∧
    executeStages witnessPrimitive (items.map CallStage.stage) witnessInitial stage2.output index = 242 := by
  intro index
  simp [executeStages, ScalarStage.step, tensorEvaluate, Program.evaluate,
    Merlin.GraphRegionsDeclared.run, Merlin.GraphRegionsDeclared.bind,
    witnessPrimitive, witnessInitial, items, stage0, stage1, stage2]
#print axioms actual_nonzero_witness

