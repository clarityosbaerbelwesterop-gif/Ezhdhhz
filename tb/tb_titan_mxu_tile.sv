// Testbench for titan_mxu_tile.
//
// Mode 1 (co-simulation): +vec=<file> +out=<file>
//   <file> holds 32-bit hex words (one per line):
//     n_jobs
//     per job: K, has_init, [ROWS*COLS init accumulators], K * (ROWS a-values, COLS b-values)
//   Values are sign-extended 32-bit words. Results (ROWS*COLS words per job) go to +out.
// Mode 2 (no +vec): built-in directed self-check, used by `make test`.
module tb_titan_mxu_tile;
    parameter int ROWS   = 4;
    parameter int COLS   = 4;
    parameter int DATA_W = 8;
    parameter int ACC_W  = 32;
    localparam int MAXW = 1 << 22;

    logic clk = 0;
    logic rst_n = 0;
    logic clear = 0, load_acc = 0, step = 0;
    logic [ROWS*COLS*ACC_W-1:0] acc_in_flat = '0;
    logic [ROWS*DATA_W-1:0] a_col = '0;
    logic [COLS*DATA_W-1:0] b_row = '0;
    logic [ROWS*COLS*ACC_W-1:0] acc_flat;

    titan_mxu_tile #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W), .ACC_W(ACC_W)) dut (.*);

    /* verilator lint_off BLKSEQ */
    always #5 clk = ~clk;
    /* verilator lint_on BLKSEQ */

    logic [31:0] mem [MAXW];
    // Stimulus vectors are built here and assigned as a whole (with Verilator 5.020, part-select
    // writes to DUT inputs were not always propagated into the DUT's continuous assigns).
    logic [ROWS*DATA_W-1:0] a_next;
    logic [COLS*DATA_W-1:0] b_next;
    logic [ROWS*COLS*ACC_W-1:0] acc_next;
    string vec_path, out_path;

    task automatic tick;
        @(posedge clk);
        #1;
    endtask

    function automatic logic signed [ACC_W-1:0] acc_at(int idx);
        return $signed(acc_flat[idx*ACC_W +: ACC_W]);
    endfunction

    task automatic directed_check;
        // A[i][k] = (i%4)+1-k, B[k][j] = (j%4)-2+k over K=3 (in range for INT4 and INT8);
        // also checks load_acc, clear priority and wrap-around.
        int unsigned errors = 0;
        clear = 1; tick(); clear = 0;
        for (int k = 0; k < 3; k++) begin
            for (int i = 0; i < ROWS; i++) a_next[i*DATA_W +: DATA_W] = DATA_W'((i % 4) + 1 - k);
            for (int j = 0; j < COLS; j++) b_next[j*DATA_W +: DATA_W] = DATA_W'((j % 4) - 2 + k);
            a_col = a_next;
            b_row = b_next;
            step = 1; tick(); step = 0;
        end
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) begin
                int exp_v = 0;
                for (int k = 0; k < 3; k++) exp_v += ((i % 4) + 1 - k) * ((j % 4) - 2 + k);
                if (acc_at(i*COLS + j) !== ACC_W'(exp_v)) begin
                    $display("mismatch (%0d,%0d): got %0d expected %0d", i, j, acc_at(i*COLS + j), exp_v);
                    errors++;
                end
            end
        // wrap-around: preload 2^31-1 and add 1*1
        acc_next = '0;
        acc_next[0 +: ACC_W] = {1'b0, {(ACC_W-1){1'b1}}};
        acc_in_flat = acc_next;
        load_acc = 1; tick(); load_acc = 0;
        a_next = '0; b_next = '0;
        a_next[0 +: DATA_W] = DATA_W'(1);
        b_next[0 +: DATA_W] = DATA_W'(1);
        a_col = a_next;
        b_row = b_next;
        step = 1; tick(); step = 0;
        if (acc_at(0) !== {1'b1, {(ACC_W-1){1'b0}}}) begin
            $display("wrap-around failed: %0d", acc_at(0));
            errors++;
        end
        clear = 1; load_acc = 1; step = 1; tick(); clear = 0; load_acc = 0; step = 0;
        for (int i = 0; i < ROWS*COLS; i++)
            if (acc_at(i) !== '0) errors++;
        if (errors != 0) $fatal(1, "titan_mxu_tile directed check: %0d errors", errors);
        $display("PASS: titan_mxu_tile %0dx%0d INT%0d directed check (outer products, load, wrap, clear priority)",
                 ROWS, COLS, DATA_W);
    endtask

    task automatic run_vectors;
        int fd, p, n_jobs, kk, has_init;
        $readmemh(vec_path, mem);
        fd = $fopen(out_path, "w");
        if (fd == 0) $fatal(1, "cannot open %s", out_path);
        p = 0;
        n_jobs = int'(mem[p]); p++;
        for (int job = 0; job < n_jobs; job++) begin
            kk = int'(mem[p]); p++;
            has_init = int'(mem[p]); p++;
            if (has_init != 0) begin
                for (int e = 0; e < ROWS*COLS; e++) begin
                    acc_next[e*ACC_W +: ACC_W] = ACC_W'(mem[p]); p++;
                end
                acc_in_flat = acc_next;
                load_acc = 1; tick(); load_acc = 0;
            end else begin
                clear = 1; tick(); clear = 0;
            end
            for (int k = 0; k < kk; k++) begin
                for (int i = 0; i < ROWS; i++) begin a_next[i*DATA_W +: DATA_W] = DATA_W'(mem[p]); p++; end
                for (int j = 0; j < COLS; j++) begin b_next[j*DATA_W +: DATA_W] = DATA_W'(mem[p]); p++; end
                a_col = a_next;
                b_row = b_next;
                step = 1; tick(); step = 0;
            end
            for (int e = 0; e < ROWS*COLS; e++) $fwrite(fd, "%08x\n", 32'(acc_flat[e*ACC_W +: ACC_W]));
        end
        $fclose(fd);
        $display("PASS: %0d jobs executed", n_jobs);
    endtask

    initial begin
        rst_n = 0; tick(); rst_n = 1; tick();
        if ($value$plusargs("vec=%s", vec_path) && $value$plusargs("out=%s", out_path))
            run_vectors();
        else
            directed_check();
        $finish;
    end
endmodule
