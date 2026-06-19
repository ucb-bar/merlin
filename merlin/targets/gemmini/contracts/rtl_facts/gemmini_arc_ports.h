// generated from gemmini.state.json model @Gemmini — do not edit
#pragma once
#include <stdint.h>
#include <string.h>
#define ARC_STATE_BYTES 363289
extern void Gemmini_eval(void*);
extern void Gemmini_clock(void*);
extern void Gemmini_passthrough(void*);

#define P_clock_OFF 0  // 1b input
#define P_clock_BITS 1
#define P_reset_OFF 1  // 1b input
#define P_reset_BITS 1
#define P_auto_spad_id_out_a_ready_OFF 2  // 1b input
#define P_auto_spad_id_out_a_ready_BITS 1
#define P_auto_spad_id_out_d_valid_OFF 3  // 1b input
#define P_auto_spad_id_out_d_valid_BITS 1
#define P_auto_spad_id_out_d_bits_opcode_OFF 4  // 3b input
#define P_auto_spad_id_out_d_bits_opcode_BITS 3
#define P_auto_spad_id_out_d_bits_param_OFF 5  // 2b input
#define P_auto_spad_id_out_d_bits_param_BITS 2
#define P_auto_spad_id_out_d_bits_size_OFF 6  // 4b input
#define P_auto_spad_id_out_d_bits_size_BITS 4
#define P_auto_spad_id_out_d_bits_source_OFF 7  // 5b input
#define P_auto_spad_id_out_d_bits_source_BITS 5
#define P_auto_spad_id_out_d_bits_sink_OFF 8  // 4b input
#define P_auto_spad_id_out_d_bits_sink_BITS 4
#define P_auto_spad_id_out_d_bits_denied_OFF 9  // 1b input
#define P_auto_spad_id_out_d_bits_denied_BITS 1
#define P_auto_spad_id_out_d_bits_data_OFF 16  // 128b input
#define P_auto_spad_id_out_d_bits_data_BITS 128
#define P_auto_spad_id_out_d_bits_corrupt_OFF 32  // 1b input
#define P_auto_spad_id_out_d_bits_corrupt_BITS 1
#define P_io_cmd_valid_OFF 33  // 1b input
#define P_io_cmd_valid_BITS 1
#define P_io_cmd_bits_inst_funct_OFF 34  // 7b input
#define P_io_cmd_bits_inst_funct_BITS 7
#define P_io_cmd_bits_inst_rs2_OFF 35  // 5b input
#define P_io_cmd_bits_inst_rs2_BITS 5
#define P_io_cmd_bits_inst_rs1_OFF 36  // 5b input
#define P_io_cmd_bits_inst_rs1_BITS 5
#define P_io_cmd_bits_inst_xd_OFF 37  // 1b input
#define P_io_cmd_bits_inst_xd_BITS 1
#define P_io_cmd_bits_inst_xs1_OFF 38  // 1b input
#define P_io_cmd_bits_inst_xs1_BITS 1
#define P_io_cmd_bits_inst_xs2_OFF 39  // 1b input
#define P_io_cmd_bits_inst_xs2_BITS 1
#define P_io_cmd_bits_inst_rd_OFF 40  // 5b input
#define P_io_cmd_bits_inst_rd_BITS 5
#define P_io_cmd_bits_inst_opcode_OFF 41  // 7b input
#define P_io_cmd_bits_inst_opcode_BITS 7
#define P_io_cmd_bits_rs1_OFF 48  // 64b input
#define P_io_cmd_bits_rs1_BITS 64
#define P_io_cmd_bits_rs2_OFF 56  // 64b input
#define P_io_cmd_bits_rs2_BITS 64
#define P_io_cmd_bits_status_debug_OFF 64  // 1b input
#define P_io_cmd_bits_status_debug_BITS 1
#define P_io_cmd_bits_status_cease_OFF 65  // 1b input
#define P_io_cmd_bits_status_cease_BITS 1
#define P_io_cmd_bits_status_wfi_OFF 66  // 1b input
#define P_io_cmd_bits_status_wfi_BITS 1
#define P_io_cmd_bits_status_isa_OFF 68  // 32b input
#define P_io_cmd_bits_status_isa_BITS 32
#define P_io_cmd_bits_status_dprv_OFF 72  // 2b input
#define P_io_cmd_bits_status_dprv_BITS 2
#define P_io_cmd_bits_status_dv_OFF 73  // 1b input
#define P_io_cmd_bits_status_dv_BITS 1
#define P_io_cmd_bits_status_prv_OFF 74  // 2b input
#define P_io_cmd_bits_status_prv_BITS 2
#define P_io_cmd_bits_status_v_OFF 75  // 1b input
#define P_io_cmd_bits_status_v_BITS 1
#define P_io_cmd_bits_status_sd_OFF 76  // 1b input
#define P_io_cmd_bits_status_sd_BITS 1
#define P_io_cmd_bits_status_zero2_OFF 80  // 23b input
#define P_io_cmd_bits_status_zero2_BITS 23
#define P_io_cmd_bits_status_mpv_OFF 83  // 1b input
#define P_io_cmd_bits_status_mpv_BITS 1
#define P_io_cmd_bits_status_gva_OFF 84  // 1b input
#define P_io_cmd_bits_status_gva_BITS 1
#define P_io_cmd_bits_status_mbe_OFF 85  // 1b input
#define P_io_cmd_bits_status_mbe_BITS 1
#define P_io_cmd_bits_status_sbe_OFF 86  // 1b input
#define P_io_cmd_bits_status_sbe_BITS 1
#define P_io_cmd_bits_status_sxl_OFF 87  // 2b input
#define P_io_cmd_bits_status_sxl_BITS 2
#define P_io_cmd_bits_status_uxl_OFF 88  // 2b input
#define P_io_cmd_bits_status_uxl_BITS 2
#define P_io_cmd_bits_status_sd_rv32_OFF 89  // 1b input
#define P_io_cmd_bits_status_sd_rv32_BITS 1
#define P_io_cmd_bits_status_zero1_OFF 90  // 8b input
#define P_io_cmd_bits_status_zero1_BITS 8
#define P_io_cmd_bits_status_tsr_OFF 91  // 1b input
#define P_io_cmd_bits_status_tsr_BITS 1
#define P_io_cmd_bits_status_tw_OFF 92  // 1b input
#define P_io_cmd_bits_status_tw_BITS 1
#define P_io_cmd_bits_status_tvm_OFF 93  // 1b input
#define P_io_cmd_bits_status_tvm_BITS 1
#define P_io_cmd_bits_status_mxr_OFF 94  // 1b input
#define P_io_cmd_bits_status_mxr_BITS 1
#define P_io_cmd_bits_status_sum_OFF 95  // 1b input
#define P_io_cmd_bits_status_sum_BITS 1
#define P_io_cmd_bits_status_mprv_OFF 96  // 1b input
#define P_io_cmd_bits_status_mprv_BITS 1
#define P_io_cmd_bits_status_xs_OFF 97  // 2b input
#define P_io_cmd_bits_status_xs_BITS 2
#define P_io_cmd_bits_status_fs_OFF 98  // 2b input
#define P_io_cmd_bits_status_fs_BITS 2
#define P_io_cmd_bits_status_mpp_OFF 99  // 2b input
#define P_io_cmd_bits_status_mpp_BITS 2
#define P_io_cmd_bits_status_vs_OFF 100  // 2b input
#define P_io_cmd_bits_status_vs_BITS 2
#define P_io_cmd_bits_status_spp_OFF 101  // 1b input
#define P_io_cmd_bits_status_spp_BITS 1
#define P_io_cmd_bits_status_mpie_OFF 102  // 1b input
#define P_io_cmd_bits_status_mpie_BITS 1
#define P_io_cmd_bits_status_ube_OFF 103  // 1b input
#define P_io_cmd_bits_status_ube_BITS 1
#define P_io_cmd_bits_status_spie_OFF 104  // 1b input
#define P_io_cmd_bits_status_spie_BITS 1
#define P_io_cmd_bits_status_upie_OFF 105  // 1b input
#define P_io_cmd_bits_status_upie_BITS 1
#define P_io_cmd_bits_status_mie_OFF 106  // 1b input
#define P_io_cmd_bits_status_mie_BITS 1
#define P_io_cmd_bits_status_hie_OFF 107  // 1b input
#define P_io_cmd_bits_status_hie_BITS 1
#define P_io_cmd_bits_status_sie_OFF 108  // 1b input
#define P_io_cmd_bits_status_sie_BITS 1
#define P_io_cmd_bits_status_uie_OFF 109  // 1b input
#define P_io_cmd_bits_status_uie_BITS 1
#define P_io_resp_ready_OFF 110  // 1b input
#define P_io_resp_ready_BITS 1
#define P_io_ptw_0_req_ready_OFF 111  // 1b input
#define P_io_ptw_0_req_ready_BITS 1
#define P_io_ptw_0_resp_valid_OFF 112  // 1b input
#define P_io_ptw_0_resp_valid_BITS 1
#define P_io_ptw_0_resp_bits_ae_ptw_OFF 113  // 1b input
#define P_io_ptw_0_resp_bits_ae_ptw_BITS 1
#define P_io_ptw_0_resp_bits_ae_final_OFF 114  // 1b input
#define P_io_ptw_0_resp_bits_ae_final_BITS 1
#define P_io_ptw_0_resp_bits_pf_OFF 115  // 1b input
#define P_io_ptw_0_resp_bits_pf_BITS 1
#define P_io_ptw_0_resp_bits_gf_OFF 116  // 1b input
#define P_io_ptw_0_resp_bits_gf_BITS 1
#define P_io_ptw_0_resp_bits_hr_OFF 117  // 1b input
#define P_io_ptw_0_resp_bits_hr_BITS 1
#define P_io_ptw_0_resp_bits_hw_OFF 118  // 1b input
#define P_io_ptw_0_resp_bits_hw_BITS 1
#define P_io_ptw_0_resp_bits_hx_OFF 119  // 1b input
#define P_io_ptw_0_resp_bits_hx_BITS 1
#define P_io_ptw_0_resp_bits_pte_ppn_OFF 120  // 44b input
#define P_io_ptw_0_resp_bits_pte_ppn_BITS 44
#define P_io_ptw_0_resp_bits_pte_d_OFF 126  // 1b input
#define P_io_ptw_0_resp_bits_pte_d_BITS 1
#define P_io_ptw_0_resp_bits_pte_a_OFF 127  // 1b input
#define P_io_ptw_0_resp_bits_pte_a_BITS 1
#define P_io_ptw_0_resp_bits_pte_g_OFF 128  // 1b input
#define P_io_ptw_0_resp_bits_pte_g_BITS 1
#define P_io_ptw_0_resp_bits_pte_u_OFF 129  // 1b input
#define P_io_ptw_0_resp_bits_pte_u_BITS 1
#define P_io_ptw_0_resp_bits_pte_x_OFF 130  // 1b input
#define P_io_ptw_0_resp_bits_pte_x_BITS 1
#define P_io_ptw_0_resp_bits_pte_w_OFF 131  // 1b input
#define P_io_ptw_0_resp_bits_pte_w_BITS 1
#define P_io_ptw_0_resp_bits_pte_r_OFF 132  // 1b input
#define P_io_ptw_0_resp_bits_pte_r_BITS 1
#define P_io_ptw_0_resp_bits_pte_v_OFF 133  // 1b input
#define P_io_ptw_0_resp_bits_pte_v_BITS 1
#define P_io_ptw_0_resp_bits_level_OFF 134  // 2b input
#define P_io_ptw_0_resp_bits_level_BITS 2
#define P_io_ptw_0_resp_bits_homogeneous_OFF 135  // 1b input
#define P_io_ptw_0_resp_bits_homogeneous_BITS 1
#define P_io_ptw_0_ptbr_mode_OFF 136  // 4b input
#define P_io_ptw_0_ptbr_mode_BITS 4
#define P_auto_spad_id_out_a_valid_OFF 137  // 1b output
#define P_auto_spad_id_out_a_valid_BITS 1
#define P_auto_spad_id_out_a_bits_opcode_OFF 138  // 3b output
#define P_auto_spad_id_out_a_bits_opcode_BITS 3
#define P_auto_spad_id_out_a_bits_param_OFF 139  // 3b output
#define P_auto_spad_id_out_a_bits_param_BITS 3
#define P_auto_spad_id_out_a_bits_size_OFF 140  // 4b output
#define P_auto_spad_id_out_a_bits_size_BITS 4
#define P_auto_spad_id_out_a_bits_source_OFF 141  // 5b output
#define P_auto_spad_id_out_a_bits_source_BITS 5
#define P_auto_spad_id_out_a_bits_address_OFF 144  // 32b output
#define P_auto_spad_id_out_a_bits_address_BITS 32
#define P_auto_spad_id_out_a_bits_mask_OFF 148  // 16b output
#define P_auto_spad_id_out_a_bits_mask_BITS 16
#define P_auto_spad_id_out_a_bits_data_OFF 160  // 128b output
#define P_auto_spad_id_out_a_bits_data_BITS 128
#define P_auto_spad_id_out_a_bits_corrupt_OFF 176  // 1b output
#define P_auto_spad_id_out_a_bits_corrupt_BITS 1
#define P_auto_spad_id_out_d_ready_OFF 177  // 1b output
#define P_auto_spad_id_out_d_ready_BITS 1
#define P_io_cmd_ready_OFF 178  // 1b output
#define P_io_cmd_ready_BITS 1
#define P_io_resp_valid_OFF 179  // 1b output
#define P_io_resp_valid_BITS 1
#define P_io_resp_bits_rd_OFF 180  // 5b output
#define P_io_resp_bits_rd_BITS 5
#define P_io_resp_bits_data_OFF 184  // 64b output
#define P_io_resp_bits_data_BITS 64
#define P_io_busy_OFF 192  // 1b output
#define P_io_busy_BITS 1
#define P_io_interrupt_OFF 193  // 1b output
#define P_io_interrupt_BITS 1
#define P_io_ptw_0_req_valid_OFF 194  // 1b output
#define P_io_ptw_0_req_valid_BITS 1
#define P_io_ptw_0_req_bits_bits_addr_OFF 196  // 27b output
#define P_io_ptw_0_req_bits_bits_addr_BITS 27
#define P_io_ptw_0_req_bits_bits_need_gpa_OFF 200  // 1b output
#define P_io_ptw_0_req_bits_bits_need_gpa_BITS 1

static inline uint64_t arc_get(const void* st, int off, int bits){
  uint64_t v=0; int nb=(bits+7)/8; memcpy(&v, (const uint8_t*)st+off, nb>8?8:nb);
  if(bits<64) v &= ((uint64_t)1<<bits)-1; return v; }
static inline void arc_set(void* st, int off, int bits, uint64_t val){
  int nb=(bits+7)/8; if(bits<64) val &= ((uint64_t)1<<bits)-1;
  memcpy((uint8_t*)st+off, &val, nb>8?8:nb); }
static inline void arc_get128(const void* st,int off,uint64_t out[2]){memcpy(out,(const uint8_t*)st+off,16);}
static inline void arc_set128(void* st,int off,const uint64_t in[2]){memcpy((uint8_t*)st+off,in,16);}
