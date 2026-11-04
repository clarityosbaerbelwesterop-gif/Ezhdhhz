// ROUGE TITAN-X1-D0: 16-lane INT8 MAC tile.
module rtl_mac_array #(
    parameter int LANES = 16,
    parameter int DATA_W = 8,
    parameter int ACC_W = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         enable,
    input  logic                         clear_acc,
    input  logic [LANES*DATA_W-1:0]       a_flat,
    input  logic [LANES*DATA_W-1:0]       b_flat,
    output logic [LANES*ACC_W-1:0]        accum_flat
);
    logic signed [DATA_W-1:0] a_lane [LANES];
    logic signed [DATA_W-1:0] b_lane [LANES];
    logic signed [2*DATA_W-1:0] product [LANES];
    logic signed [ACC_W-1:0] accum [LANES];

    for (genvar i = 0; i < LANES; i++) begin : g_lanes
        assign a_lane[i] = $signed(a_flat[i*DATA_W +: DATA_W]);
        assign b_lane[i] = $signed(b_flat[i*DATA_W +: DATA_W]);
        assign product[i] = a_lane[i] * b_lane[i];
        assign accum_flat[i*ACC_W +: ACC_W] = accum[i];

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                accum[i] <= '0;
            else if (clear_acc)
                accum[i] <= '0;
            else if (enable)
                accum[i] <= accum[i] + {{(ACC_W-2*DATA_W){product[i][2*DATA_W-1]}}, product[i]};
        end
    end
endmodule
