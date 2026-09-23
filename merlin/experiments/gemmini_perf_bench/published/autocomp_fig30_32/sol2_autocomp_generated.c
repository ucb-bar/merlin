/* AutoComp's OWN generated GEMM kernel for 12544x64x256, transcribed from the paper.
 *
 * Source: "Autocomp: LLM-Driven Code Optimization for Tensor Accelerators" (arXiv:2505.18574),
 * Appendix D.1, Figures 30, 31 and 32 -- one listing split across three figure pages, captioned
 * "Example of an optimized version of the same GEMM from Fig. 28, generated using Autocomp.
 * Achieves 93% compute utilization."
 *
 * The paper prints this listing as text; there is no machine-readable copy of it in the published
 * artifact (autocomp/sols/exo/ ships only the gemmini_baseline, exo_baseline and exo_opt arms).
 * This file is that printed listing reassembled into compilable C. Only three classes of edit were
 * made, all of them typographic, and none of them touches an instruction, an address or an extent:
 *
 *   1. The PDF text layer merges a banner comment and the statement that follows it onto one line
 *      (e.g. "// ----------config_st (64) ;"). Each such line is split back into its comment and
 *      its statement.
 *   2. The PDF text layer inserts spaces inside tokens ("1.0 f", "1U << 31", "0 x40000000",
 *      "int_fast32_t"). These are closed up.
 *   3. Each unrolled segment's opening brace is printed at the end of its banner comment
 *      ("// --- Unrolled segment "seg" ------{"), which would put it inside a // comment. It is
 *      restored as a real block scope on its own line -- which is also what makes the listing
 *      compile at all, since every segment redeclares block_id / sub_offset / A_seg_addr.
 *
 * The function was renamed test -> solution: this bench's published-kernel renderer
 * (render_autocomp_matmul_layer) calls solution(A, B, C), exactly as it does for the three arms the
 * artifact does publish. The signature, the argument order and the extents are the paper's.
 *
 * Correctness is NOT asserted by this comment. The kernel is admitted only if the digest the program
 * prints equals the off-device exact-integer oracle for the same operands; a mis-transcription
 * computes a different function and is refused.
 */

void solution(int8_t A[12544][256], int8_t B[256][64], int8_t C[12544][64]) {
  // -------------------------------------------------------------------------
  // Tile and matrix constants
  // -------------------------------------------------------------------------
  const uint32_t tile_dim = 16;          // tile dimension
  const uint32_t tile_offset = tile_dim; // each row in a tile is tile_dim elements
  // For double buffering, we use two accumulator base addresses separated by 4 tile rows.
  const uint32_t acc_base0 = 1U << 31;
  const uint32_t acc_base1 = (1U << 31) + (4 * tile_dim);
  const uint32_t new_B_base = 8192;
  const uint32_t A_tile_base0 = 2048;
  const uint32_t A_tile_base1 = 6144;

  // -------------------------------------------------------------------------
  // Gemmini configuration
  // -------------------------------------------------------------------------
  config_st(64);
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld(64, 1.0f, tile_dim, 2);
  config_ld(256, 1.0f, tile_dim, 1);
  config_ld(0, 1.0f, 0, 0);

  // -------------------------------------------------------------------------
  // Load the entire weight matrix B into the scratchpad once.
  // -------------------------------------------------------------------------
  for (int br = 0; br < 256; br += tile_dim) {
    mvin3(&B[br][0],
          new_B_base + (br / tile_dim) * (tile_dim * 4),
          tile_dim * 4,
          tile_dim);
  }

  // -------------------------------------------------------------------------
  // Begin double-buffered accumulator computation.
  //
  // cur_acc_base: where the current tile is computed.
  // prev_acc_base: holds the result of the previously computed tile.
  // -------------------------------------------------------------------------
  uint32_t cur_acc_base = acc_base0;
  uint32_t prev_acc_base = 0; // Will be set after the first iteration.

  // Outer loop over tiles (784 tiles along A's first dimension)
  for (int i = 0; i < 784; i++) {
    // Determine which scratchpad region holds A for this tile.
    // Alternating between two buffers allows overlapping the load of the next tile.
    uint32_t current_A_buffer = (i % 2 == 0) ? A_tile_base0 : A_tile_base1;
    uint32_t next_A_buffer = (i % 2 == 0) ? A_tile_base1 : A_tile_base0;

    // ------------------------------------------------------------------------
    // For the first iteration load the A tile into scratchpad.
    // For subsequent iterations, write the previous tile from the accumulator.
    // ------------------------------------------------------------------------
    if (i == 0) {
      for (int b = 0; b < 4; b++) {
        mvin2(&A[tile_dim * i][64 * b],
              current_A_buffer + b * (tile_dim * 4),
              tile_dim * 4,
              tile_dim);
      }
    } else {
      for (int j_in_o = 0; j_in_o < 4; j_in_o++) {
        uint32_t j_off = j_in_o * tile_dim;
        mvout(&C[tile_dim * (i - 1)][tile_dim * j_in_o],
              prev_acc_base + j_off,
              tile_dim,
              tile_dim);
      }
    }

    // -------------------------------------------------------------------------
    // Instead of explicitly zeroing-out the accumulator tile via extended mvin,
    // we fuse the accumulator zeroing into the compute stream.
    //
    // For each accumulator sub-tile (indexed by j_in_o), the very first compute
    // call is issued in overwrite mode (i.e. the accumulator address is used as-is)
    // and then all subsequent compute calls for that sub-tile are issued in accumulate mode.
    //
    // We use an array (first_compute) to track whether a given sub-tile has been updated.
    // -------------------------------------------------------------------------
    int first_compute[4] = {1, 1, 1, 1};

    // -------------------------------------------------------------------------
    // Compute the current tile.
    // The complete computation is divided into 16 segments.
    // Loop unrolling by a factor of 4 is applied to the segments loop.
    // -------------------------------------------------------------------------
    for (int seg = 0; seg < 16; seg += 4) {
      for (int j_in_o = 0; j_in_o < 4; j_in_o++) {
        uint32_t j_off = j_in_o * tile_dim;
        uint32_t acc_addr; // will hold the computed accumulator address for preload

        // --- Unrolled segment "seg" ------------------------------------------------
        {
          uint32_t block_id = seg / 4;
          uint32_t sub_offset = (seg % 4) * tile_dim;
          uint32_t A_seg_addr = current_A_buffer + block_id * (tile_dim * 4) + sub_offset;
          // If this is the very first compute for this sub-tile, use overwrite mode
          if (first_compute[j_in_o]) {
            acc_addr = cur_acc_base + j_off;
            first_compute[j_in_o] = 0;
          } else {
            acc_addr = (cur_acc_base + j_off) | 0x40000000;
          }
          preload(new_B_base + seg * (tile_dim * 4) + j_off,
                  acc_addr,
                  tile_dim, tile_dim,
                  tile_dim, tile_dim);
          compute_preloaded(A_seg_addr,
                            ~(uint32_t)0,
                            tile_dim, tile_dim,
                            tile_dim, tile_dim);
        }

        // --- Unrolled segment "seg+1" ------------------------------------------------
        {
          uint32_t block_id = (seg + 1) / 4;
          uint32_t sub_offset = ((seg + 1) % 4) * tile_dim;
          uint32_t A_seg_addr = current_A_buffer + block_id * (tile_dim * 4) + sub_offset;
          preload(new_B_base + (seg + 1) * (tile_dim * 4) + j_off,
                  (cur_acc_base + j_off) | 0x40000000,
                  tile_dim, tile_dim,
                  tile_dim, tile_dim);
          compute_preloaded(A_seg_addr,
                            ~(uint32_t)0,
                            tile_dim, tile_dim,
                            tile_dim, tile_dim);
        }

        // --- Unrolled segment "seg+2" ------------------------------------------------
        {
          uint32_t block_id = (seg + 2) / 4;
          uint32_t sub_offset = ((seg + 2) % 4) * tile_dim;
          uint32_t A_seg_addr = current_A_buffer + block_id * (tile_dim * 4) + sub_offset;
          preload(new_B_base + (seg + 2) * (tile_dim * 4) + j_off,
                  (cur_acc_base + j_off) | 0x40000000,
                  tile_dim, tile_dim,
                  tile_dim, tile_dim);
          compute_preloaded(A_seg_addr,
                            ~(uint32_t)0,
                            tile_dim, tile_dim,
                            tile_dim, tile_dim);
        }

        // --- Unrolled segment "seg+3" ------------------------------------------------
        {
          uint32_t block_id = (seg + 3) / 4;
          uint32_t sub_offset = ((seg + 3) % 4) * tile_dim;
          uint32_t A_seg_addr = current_A_buffer + block_id * (tile_dim * 4) + sub_offset;
          preload(new_B_base + (seg + 3) * (tile_dim * 4) + j_off,
                  (cur_acc_base + j_off) | 0x40000000,
                  tile_dim, tile_dim,
                  tile_dim, tile_dim);
          compute_preloaded(A_seg_addr,
                            ~(uint32_t)0,
                            tile_dim, tile_dim,
                            tile_dim, tile_dim);
        }
      } // end inner loop over j_in_o

      // -------------------------------------------------------------------------
      // For seg==0 (i.e. the first unrolled iteration), launch prefetching of the next A tile.
      // This overlaps memory-access with computation.
      // -------------------------------------------------------------------------
      if (seg == 0 && i < 783) {
        for (int b = 0; b < 4; b++) {
          mvin2(&A[tile_dim * (i + 1)][64 * b],
                next_A_buffer + b * (tile_dim * 4),
                tile_dim * 4,
                tile_dim);
        }
      }
    } // end segments loop

    // -------------------------------------------------------------------------
    // Swap accumulator buffers.
    // The tile computed in this iteration (in cur_acc_base) becomes the previous tile,
    // so it must be written back in the next iteration.
    // -------------------------------------------------------------------------
    prev_acc_base = cur_acc_base;
    cur_acc_base = (cur_acc_base == acc_base0) ? acc_base1 : acc_base0;
  } // end outer tile loop

  // -------------------------------------------------------------------------
  // Write back the final computed tile (tile index 783) from the accumulator.
  // -------------------------------------------------------------------------
  for (int j_in_o = 0; j_in_o < 4; j_in_o++) {
    uint32_t j_off = j_in_o * tile_dim;
    mvout(&C[tile_dim * (784 - 1)][tile_dim * j_in_o],
          prev_acc_base + j_off,
          tile_dim,
          tile_dim);
  }

  fence();
}
