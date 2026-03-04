if (auto op = dyn_cast<::merlin::gemmini::ComputeAccumulated_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_compute_accumulated,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::ComputePreloaded_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_compute_preloaded,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::ConfigEX_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_config_ex,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::ConfigNorm_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_config_norm,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::ConfigSt_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_config_st,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::ConifgLd_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_config_ld,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::Flush_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_flush,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopConvWsConfig1_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_conv_ws_config1,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopConvWsConfig2_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_conv_ws_config2,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopConvWsConfig3_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_conv_ws_config3,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopConvWsConfig4_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_conv_ws_config4,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopConvWsConfig5_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_conv_ws_config5,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopConvWsConfig6_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_conv_ws_config6,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopConvWs_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_conv_ws,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopWsConfigAddrsAB_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_ws_config_addrs_ab,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopWsConfigAddrsDC_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_ws_config_addrs_dc,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopWsConfigBounds_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_ws_config_bounds,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopWsConfigStridesAB_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_ws_config_strides_ab,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopWsConfigStridesDC_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_ws_config_strides_dc,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::LoopWs_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_loop_ws,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::Mvin2_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_mvin2,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::Mvin3_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_mvin3,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::Mvin_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_mvin,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::Mvout_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_mvout,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::merlin::gemmini::Preload_IntrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::riscv_preload,0,{},{},{},{});
    (void) inst;
    
  return success();
}
