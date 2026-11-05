// Randomized self-checking test of rtl_mac_array against an in-testbench reference model.
// Random operands, random enable/clear (including both asserted), 2000 cycles, wrap-around semantics.
// Verified with Verilator 5.020 and Icarus Verilog 12.
module tb_rtl_mac_array_random;
    localparam int LANES = 16;
    localparam int DATA_W = 8;
    localparam int ACC_W = 32;
    localparam int CYCLES = 2000;

    logic clk = 0;
    logic rst_n = 0;
    logic enable = 0;
    logic clear_acc = 0;
    logic [LANES*DATA_W-1:0] a_flat = '0;
    logic [LANES*DATA_W-1:0] b_flat = '0;
    logic [LANES*ACC_W-1:0] accum_flat;

    logic signed [ACC_W-1:0] model [LANES];
    logic signed [2*DATA_W-1:0] prod;
    logic [LANES*DATA_W-1:0] a_next, b_next;
    logic [31:0] lfsr = 32'hACE1_2468;
    int errors = 0;

    rtl_mac_array #(.LANES(LANES), .DATA_W(DATA_W), .ACC_W(ACC_W)) dut (.*);

    /* verilator lint_off BLKSEQ */
    always #5 clk = ~clk;
    /* verilator lint_on BLKSEQ */

    task automatic tick;
        @(posedge clk);
        #1;
    endtask

    task automatic next_rand;   // xorshift32
        lfsr = lfsr ^ (lfsr << 13);
        lfsr = lfsr ^ (lfsr >> 17);
        lfsr = lfsr ^ (lfsr << 5);
    endtask

    task automatic check(input int cyc);
        for (int i = 0; i < LANES; i++) begin
            // product in a 2*DATA_W context (a size cast alone would evaluate the multiply at DATA_W bits)
            prod = $signed(a_flat[i*DATA_W +: DATA_W]) * $signed(b_flat[i*DATA_W +: DATA_W]);
            if (clear_acc)
                model[i] = '0;
            else if (enable)
                model[i] = model[i] + ACC_W'(prod);
            if ($signed(accum_flat[i*ACC_W +: ACC_W]) !== model[i]) begin
                if (errors < 3)
                    $display("cycle %0d lane %0d: dut %0d model %0d", cyc, i,
                             $signed(accum_flat[i*ACC_W +: ACC_W]), model[i]);
                errors++;
            end
        end
    endtask

    initial begin
        for (int i = 0; i < LANES; i++) model[i] = '0;
        rst_n = 0;
        tick();
        rst_n = 1;
        tick();
        for (int c = 0; c < CYCLES; c++) begin
            next_rand();
            enable = lfsr[0] | lfsr[1];
            clear_acc = (lfsr[7:2] == 6'd0);
            for (int i = 0; i < LANES; i++) begin
                next_rand();
                a_next[i*DATA_W +: DATA_W] = lfsr[7:0];
                b_next[i*DATA_W +: DATA_W] = lfsr[15:8];
            end
            // Whole-vector assignment: with Verilator 5.020, loop part-select writes to a_flat/b_flat
            // were not propagated into the DUT's continuous assigns before the next clock edge.
            a_flat = a_next;
            b_flat = b_next;
            tick();
            check(c);
        end
        if (errors != 0) $fatal(1, "random MAC test: %0d mismatches", errors);
        $display("PASS: %0d random cycles x %0d lanes match the reference model", CYCLES, LANES);
        $finish;
    end
endmodule
