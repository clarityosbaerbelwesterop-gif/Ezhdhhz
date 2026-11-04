module tb_rtl_mac_array;
    localparam int LANES = 16;
    localparam int DATA_W = 8;
    localparam int ACC_W = 32;

    logic clk = 0;
    logic rst_n = 0;
    logic enable = 0;
    logic clear_acc = 0;
    logic [LANES*DATA_W-1:0] a_flat = '0;
    logic [LANES*DATA_W-1:0] b_flat = '0;
    logic [LANES*ACC_W-1:0] accum_flat;

    rtl_mac_array #(.LANES(LANES), .DATA_W(DATA_W), .ACC_W(ACC_W)) dut (.*);

    /* verilator lint_off BLKSEQ */
    always #5 clk = ~clk;  // testbench clock generator
    /* verilator lint_on BLKSEQ */

    task automatic tick;
        @(posedge clk);
        #1;
    endtask

    initial begin
        rst_n = 0;
        tick();
        rst_n = 1;
        tick();

        for (int i = 0; i < LANES; i++) begin
            a_flat[i*DATA_W +: DATA_W] = 8'(i+1);
            b_flat[i*DATA_W +: DATA_W] = 8'(2);
        end
        enable = 1;
        tick();
        enable = 0;

        for (int i = 0; i < LANES; i++) begin
            logic signed [ACC_W-1:0] got;
            got = $signed(accum_flat[i*ACC_W +: ACC_W]);
            if (got !== ACC_W'((i+1)*2))
                $fatal(1, "lane %0d: expected %0d got %0d", i, (i+1)*2, got);
        end

        enable = 1;
        tick();
        enable = 0;
        for (int i = 0; i < LANES; i++) begin
            logic signed [ACC_W-1:0] got;
            got = $signed(accum_flat[i*ACC_W +: ACC_W]);
            if (got !== ACC_W'((i+1)*4))
                $fatal(1, "accum lane %0d: expected %0d got %0d", i, (i+1)*4, got);
        end

        clear_acc = 1;
        tick();
        clear_acc = 0;
        for (int i = 0; i < LANES; i++) begin
            if (accum_flat[i*ACC_W +: ACC_W] !== '0)
                $fatal(1, "clear failed at lane %0d", i);
        end

        a_flat = '0;
        b_flat = '0;
        a_flat[0 +: DATA_W] = -8'sd3;
        b_flat[0 +: DATA_W] =  8'sd7;
        enable = 1;
        tick();
        enable = 0;
        if ($signed(accum_flat[0 +: ACC_W]) !== -32'sd21)
            $fatal(1, "signed multiplication failed");

        $display("PASS: all 16 MAC lanes, accumulation, clear and signed arithmetic");
        $finish;
    end
endmodule
