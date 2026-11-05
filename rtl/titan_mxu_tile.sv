// ROUGE TITAN-X1: matrix-engine (MXU) tile, outer-product dataflow.
//
// Computes C[ROWS x COLS] (+)= A[ROWS x K] * B[K x COLS] by streaming one k per
// cycle: a_col = A[:, k], b_row = B[k, :]. Every processing element (i, j) adds
// a_col[i] * b_row[j] to its accumulator. Inputs are signed DATA_W-bit integers
// (INT8: DATA_W=8, INT4: DATA_W=4); accumulators are ACC_W-bit two's complement
// and wrap on overflow, exactly as specified for MATMUL in docs/ISA.md.
//
// Control priority per cycle: clear > load_acc > step.
module titan_mxu_tile #(
    parameter int ROWS   = 4,
    parameter int COLS   = 4,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          clear,
    input  logic                          load_acc,
    input  logic [ROWS*COLS*ACC_W-1:0]    acc_in_flat,
    input  logic                          step,
    input  logic [ROWS*DATA_W-1:0]        a_col,
    input  logic [COLS*DATA_W-1:0]        b_row,
    output logic [ROWS*COLS*ACC_W-1:0]    acc_flat
);
    localparam int PW = 2 * DATA_W;

    for (genvar i = 0; i < ROWS; i++) begin : g_row
        for (genvar j = 0; j < COLS; j++) begin : g_col
            localparam int IDX = i * COLS + j;
            logic signed [DATA_W-1:0] a_v, b_v;
            logic signed [PW-1:0]     prod;
            logic signed [ACC_W-1:0]  acc;

            assign a_v  = $signed(a_col[i*DATA_W +: DATA_W]);
            assign b_v  = $signed(b_row[j*DATA_W +: DATA_W]);
            assign prod = a_v * b_v;
            assign acc_flat[IDX*ACC_W +: ACC_W] = acc;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    acc <= '0;
                else if (clear)
                    acc <= '0;
                else if (load_acc)
                    acc <= acc_in_flat[IDX*ACC_W +: ACC_W];
                else if (step)
                    acc <= acc + {{(ACC_W-PW){prod[PW-1]}}, prod};
            end
        end
    end
endmodule
