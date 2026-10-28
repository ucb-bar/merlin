builtin.module {
  func.func @row_moments(%x: tensor<32x32xf32>) -> (tensor<32x1xf32>, tensor<32x1xf32>) {
    %354 = "tensor.empty"() : () -> tensor<32x1xf32>
    %355 = "tensor.empty"() : () -> tensor<32x1xf32>
    %356, %357 = "linalg.generic"(%354, %355) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 2>}> ({
    ^bb10(%358: f32, %359: f32):
      %360 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %361 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %362 = "arith.constant"() <{value = 0 : index}> : () -> index
      %363 = "arith.constant"() <{value = 4 : index}> : () -> index
      %364 = "arith.constant"() <{value = 1 : index}> : () -> index
      %365, %366, %367, %368, %369, %370, %371, %372, %373, %374, %375, %376, %377, %378, %379, %380 = "scf.for"(%362, %363, %364, %361, %361, %361, %361, %361, %361, %361, %361, %361, %361, %361, %361, %361, %361, %361, %361) ({
      ^bb11(%381: index, %382: f32, %383: f32, %384: f32, %385: f32, %386: f32, %387: f32, %388: f32, %389: f32, %390: f32, %391: f32, %392: f32, %393: f32, %394: f32, %395: f32, %396: f32, %397: f32):
        %398 = "arith.constant"() <{value = 1 : index}> : () -> index
        %399 = "arith.addi"(%381, %398) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %400 = "arith.index_cast"(%399) : (index) -> i64
        %401 = "arith.sitofp"(%400) : (i64) -> f32
        %402 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
        %403 = "arith.divf"(%402, %401) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %404 = "arith.constant"() <{value = 8 : index}> : () -> index
        %405 = "arith.muli"(%381, %404) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %406 = "arith.constant"() <{value = 0 : index}> : () -> index
        %407 = "arith.addi"(%405, %406) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %408 = "tensor.extract"(%x, %360, %407) : (tensor<32x32xf32>, index, index) -> f32
        %409 = "arith.subf"(%408, %382) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %410 = "math.fma"(%409, %403, %382) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %411 = "arith.subf"(%408, %410) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %412 = "math.fma"(%409, %411, %390) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %413 = "arith.constant"() <{value = 8 : index}> : () -> index
        %414 = "arith.muli"(%381, %413) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %415 = "arith.constant"() <{value = 1 : index}> : () -> index
        %416 = "arith.addi"(%414, %415) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %417 = "tensor.extract"(%x, %360, %416) : (tensor<32x32xf32>, index, index) -> f32
        %418 = "arith.subf"(%417, %383) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %419 = "math.fma"(%418, %403, %383) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %420 = "arith.subf"(%417, %419) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %421 = "math.fma"(%418, %420, %391) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %422 = "arith.constant"() <{value = 8 : index}> : () -> index
        %423 = "arith.muli"(%381, %422) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %424 = "arith.constant"() <{value = 2 : index}> : () -> index
        %425 = "arith.addi"(%423, %424) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %426 = "tensor.extract"(%x, %360, %425) : (tensor<32x32xf32>, index, index) -> f32
        %427 = "arith.subf"(%426, %384) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %428 = "math.fma"(%427, %403, %384) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %429 = "arith.subf"(%426, %428) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %430 = "math.fma"(%427, %429, %392) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %431 = "arith.constant"() <{value = 8 : index}> : () -> index
        %432 = "arith.muli"(%381, %431) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %433 = "arith.constant"() <{value = 3 : index}> : () -> index
        %434 = "arith.addi"(%432, %433) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %435 = "tensor.extract"(%x, %360, %434) : (tensor<32x32xf32>, index, index) -> f32
        %436 = "arith.subf"(%435, %385) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %437 = "math.fma"(%436, %403, %385) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %438 = "arith.subf"(%435, %437) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %439 = "math.fma"(%436, %438, %393) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %440 = "arith.constant"() <{value = 8 : index}> : () -> index
        %441 = "arith.muli"(%381, %440) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %442 = "arith.constant"() <{value = 4 : index}> : () -> index
        %443 = "arith.addi"(%441, %442) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %444 = "tensor.extract"(%x, %360, %443) : (tensor<32x32xf32>, index, index) -> f32
        %445 = "arith.subf"(%444, %386) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %446 = "math.fma"(%445, %403, %386) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %447 = "arith.subf"(%444, %446) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %448 = "math.fma"(%445, %447, %394) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %449 = "arith.constant"() <{value = 8 : index}> : () -> index
        %450 = "arith.muli"(%381, %449) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %451 = "arith.constant"() <{value = 5 : index}> : () -> index
        %452 = "arith.addi"(%450, %451) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %453 = "tensor.extract"(%x, %360, %452) : (tensor<32x32xf32>, index, index) -> f32
        %454 = "arith.subf"(%453, %387) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %455 = "math.fma"(%454, %403, %387) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %456 = "arith.subf"(%453, %455) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %457 = "math.fma"(%454, %456, %395) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %458 = "arith.constant"() <{value = 8 : index}> : () -> index
        %459 = "arith.muli"(%381, %458) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %460 = "arith.constant"() <{value = 6 : index}> : () -> index
        %461 = "arith.addi"(%459, %460) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %462 = "tensor.extract"(%x, %360, %461) : (tensor<32x32xf32>, index, index) -> f32
        %463 = "arith.subf"(%462, %388) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %464 = "math.fma"(%463, %403, %388) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %465 = "arith.subf"(%462, %464) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %466 = "math.fma"(%463, %465, %396) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %467 = "arith.constant"() <{value = 8 : index}> : () -> index
        %468 = "arith.muli"(%381, %467) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %469 = "arith.constant"() <{value = 7 : index}> : () -> index
        %470 = "arith.addi"(%468, %469) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %471 = "tensor.extract"(%x, %360, %470) : (tensor<32x32xf32>, index, index) -> f32
        %472 = "arith.subf"(%471, %389) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %473 = "math.fma"(%472, %403, %389) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        %474 = "arith.subf"(%471, %473) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %475 = "math.fma"(%472, %474, %397) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
        "scf.yield"(%410, %419, %428, %437, %446, %455, %464, %473, %412, %421, %430, %439, %448, %457, %466, %475) : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> ()
      }) : (index, index, index, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
      %476 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %477 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %478 = "arith.divf"(%476, %477) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %479 = "arith.subf"(%365, %361) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %480 = "arith.mulf"(%478, %479) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %481 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %482 = "arith.mulf"(%479, %481) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %483 = "arith.addf"(%361, %480) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %484 = "arith.addf"(%361, %373) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %485 = "math.fma"(%482, %480, %484) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %486 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %487 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %488 = "arith.divf"(%486, %487) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %489 = "arith.subf"(%366, %361) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %490 = "arith.mulf"(%488, %489) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %491 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %492 = "arith.mulf"(%489, %491) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %493 = "arith.addf"(%361, %490) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %494 = "arith.addf"(%361, %374) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %495 = "math.fma"(%492, %490, %494) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %496 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %497 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %498 = "arith.divf"(%496, %497) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %499 = "arith.subf"(%367, %361) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %500 = "arith.mulf"(%498, %499) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %501 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %502 = "arith.mulf"(%499, %501) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %503 = "arith.addf"(%361, %500) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %504 = "arith.addf"(%361, %375) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %505 = "math.fma"(%502, %500, %504) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %506 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %507 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %508 = "arith.divf"(%506, %507) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %509 = "arith.subf"(%368, %361) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %510 = "arith.mulf"(%508, %509) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %511 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %512 = "arith.mulf"(%509, %511) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %513 = "arith.addf"(%361, %510) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %514 = "arith.addf"(%361, %376) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %515 = "math.fma"(%512, %510, %514) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %516 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %517 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %518 = "arith.divf"(%516, %517) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %519 = "arith.subf"(%369, %361) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %520 = "arith.mulf"(%518, %519) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %521 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %522 = "arith.mulf"(%519, %521) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %523 = "arith.addf"(%361, %520) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %524 = "arith.addf"(%361, %377) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %525 = "math.fma"(%522, %520, %524) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %526 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %527 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %528 = "arith.divf"(%526, %527) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %529 = "arith.subf"(%370, %361) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %530 = "arith.mulf"(%528, %529) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %531 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %532 = "arith.mulf"(%529, %531) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %533 = "arith.addf"(%361, %530) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %534 = "arith.addf"(%361, %378) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %535 = "math.fma"(%532, %530, %534) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %536 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %537 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %538 = "arith.divf"(%536, %537) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %539 = "arith.subf"(%371, %361) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %540 = "arith.mulf"(%538, %539) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %541 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %542 = "arith.mulf"(%539, %541) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %543 = "arith.addf"(%361, %540) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %544 = "arith.addf"(%361, %379) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %545 = "math.fma"(%542, %540, %544) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %546 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %547 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %548 = "arith.divf"(%546, %547) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %549 = "arith.subf"(%372, %361) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %550 = "arith.mulf"(%548, %549) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %551 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %552 = "arith.mulf"(%549, %551) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %553 = "arith.addf"(%361, %550) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %554 = "arith.addf"(%361, %380) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %555 = "math.fma"(%552, %550, %554) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %556 = "arith.constant"() <{value = 32 : index}> : () -> index
      %557 = "arith.constant"() <{value = 32 : index}> : () -> index
      %558 = "arith.constant"() <{value = 1 : index}> : () -> index
      %559, %560 = "scf.for"(%556, %557, %558, %361, %361) ({
      ^bb12(%561: index, %562: f32, %563: f32):
        %564 = "tensor.extract"(%x, %360, %561) : (tensor<32x32xf32>, index, index) -> f32
        %565 = "arith.subf"(%564, %562) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %566 = "arith.constant"() <{value = 32 : index}> : () -> index
        %567 = "arith.subi"(%561, %566) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %568 = "arith.constant"() <{value = 1 : index}> : () -> index
        %569 = "arith.addi"(%567, %568) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %570 = "arith.index_cast"(%569) : (index) -> i64
        %571 = "arith.sitofp"(%570) : (i64) -> f32
        %572 = "arith.divf"(%565, %571) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %573 = "arith.addf"(%562, %572) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %574 = "arith.subf"(%564, %573) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %575 = "arith.mulf"(%565, %574) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %576 = "arith.addf"(%563, %575) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "scf.yield"(%573, %576) : (f32, f32) -> ()
      }) : (index, index, index, f32, f32) -> (f32, f32)
      %577 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %578 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %579 = "arith.divf"(%577, %578) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %580 = "arith.subf"(%483, %559) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %581 = "math.fma"(%579, %580, %559) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %582 = "arith.mulf"(%580, %580) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %583 = "arith.mulf"(%582, %579) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %584 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %585 = "math.fma"(%583, %584, %485) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %586 = "arith.addf"(%560, %585) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %587 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %588 = "arith.constant"() <{value = 8.000000e+00 : f32}> : () -> f32
      %589 = "arith.divf"(%587, %588) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %590 = "arith.subf"(%493, %581) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %591 = "math.fma"(%589, %590, %581) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %592 = "arith.mulf"(%590, %590) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %593 = "arith.mulf"(%592, %589) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %594 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %595 = "math.fma"(%593, %594, %495) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %596 = "arith.addf"(%586, %595) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %597 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %598 = "arith.constant"() <{value = 1.200000e+01 : f32}> : () -> f32
      %599 = "arith.divf"(%597, %598) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %600 = "arith.subf"(%503, %591) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %601 = "math.fma"(%599, %600, %591) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %602 = "arith.mulf"(%600, %600) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %603 = "arith.mulf"(%602, %599) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %604 = "arith.constant"() <{value = 8.000000e+00 : f32}> : () -> f32
      %605 = "math.fma"(%603, %604, %505) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %606 = "arith.addf"(%596, %605) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %607 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %608 = "arith.constant"() <{value = 1.600000e+01 : f32}> : () -> f32
      %609 = "arith.divf"(%607, %608) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %610 = "arith.subf"(%513, %601) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %611 = "math.fma"(%609, %610, %601) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %612 = "arith.mulf"(%610, %610) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %613 = "arith.mulf"(%612, %609) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %614 = "arith.constant"() <{value = 1.200000e+01 : f32}> : () -> f32
      %615 = "math.fma"(%613, %614, %515) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %616 = "arith.addf"(%606, %615) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %617 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %618 = "arith.constant"() <{value = 2.000000e+01 : f32}> : () -> f32
      %619 = "arith.divf"(%617, %618) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %620 = "arith.subf"(%523, %611) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %621 = "math.fma"(%619, %620, %611) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %622 = "arith.mulf"(%620, %620) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %623 = "arith.mulf"(%622, %619) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %624 = "arith.constant"() <{value = 1.600000e+01 : f32}> : () -> f32
      %625 = "math.fma"(%623, %624, %525) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %626 = "arith.addf"(%616, %625) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %627 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %628 = "arith.constant"() <{value = 2.400000e+01 : f32}> : () -> f32
      %629 = "arith.divf"(%627, %628) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %630 = "arith.subf"(%533, %621) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %631 = "math.fma"(%629, %630, %621) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %632 = "arith.mulf"(%630, %630) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %633 = "arith.mulf"(%632, %629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %634 = "arith.constant"() <{value = 2.000000e+01 : f32}> : () -> f32
      %635 = "math.fma"(%633, %634, %535) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %636 = "arith.addf"(%626, %635) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %637 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %638 = "arith.constant"() <{value = 2.800000e+01 : f32}> : () -> f32
      %639 = "arith.divf"(%637, %638) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %640 = "arith.subf"(%543, %631) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %641 = "math.fma"(%639, %640, %631) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %642 = "arith.mulf"(%640, %640) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %643 = "arith.mulf"(%642, %639) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %644 = "arith.constant"() <{value = 2.400000e+01 : f32}> : () -> f32
      %645 = "math.fma"(%643, %644, %545) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %646 = "arith.addf"(%636, %645) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %647 = "arith.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
      %648 = "arith.constant"() <{value = 3.200000e+01 : f32}> : () -> f32
      %649 = "arith.divf"(%647, %648) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %650 = "arith.subf"(%553, %641) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %651 = "math.fma"(%649, %650, %641) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %652 = "arith.mulf"(%650, %650) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %653 = "arith.mulf"(%652, %649) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %654 = "arith.constant"() <{value = 2.800000e+01 : f32}> : () -> f32
      %655 = "math.fma"(%653, %654, %555) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
      %656 = "arith.addf"(%646, %655) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %657 = "arith.constant"() <{value = 3.200000e+01 : f32}> : () -> f32
      %658 = "arith.divf"(%656, %657) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%651, %658) : (f32, f32) -> ()
    }) {prov.aten = "aten.native_layer_norm.default"} : (tensor<32x1xf32>, tensor<32x1xf32>) -> (tensor<32x1xf32>, tensor<32x1xf32>)
    func.return %356, %357 : tensor<32x1xf32>, tensor<32x1xf32>
  }
}
