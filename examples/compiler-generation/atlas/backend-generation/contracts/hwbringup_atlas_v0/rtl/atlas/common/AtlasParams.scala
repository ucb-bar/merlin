package atlas.common

case class AtlasParams(
  ipt:        InnerProductTreeParams  = InnerProductTreeParams(),
  sa:         SystolicArrayParams     = SystolicArrayParams(),
  vpu:        VpuParams               = VpuParams(),
  dma:        DmaParams               = DmaParams(),
  mreg:       MregParams              = MregParams(),
  vmem:       VmemParams              = VmemParams()
)