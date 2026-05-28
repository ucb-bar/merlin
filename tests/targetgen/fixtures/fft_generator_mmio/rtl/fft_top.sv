// Synthetic fixture.
module fft_top (
    input  logic         clk,
    input  logic         rst,
    input  logic [31:0]  axi_awaddr,
    output logic         axi_awready
);
  // RegField-style mmio register
  always_ff @(posedge clk) begin
    axi_awready <= 1'b1;
  end
endmodule

interface fft_axi_if;
endinterface
