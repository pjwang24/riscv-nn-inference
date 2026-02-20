`include "const.vh"

// =============================================================================
// MatmulAccelerator — 4x4 Block Outer Product Engine (Phase 2)
//
// MMIO register map (active when dcache_addr[31] == 1):
//   0x80000000  CTRL/STATUS
//       W: [0]=start [4]=err_clr
//       R: [0]=busy [1]=done [2]=full [3]=empty [4]=error
//   0x80000004  W_ADDR   (base of packed B vectors)
//   0x80000008  X_ADDR   (base of packed A vectors)
//   0x8000000C  M_DIM    (valid rows 1..4)
//   0x80000010  N_DIM    (valid cols 1..4)
//   0x80000014  K_DIM    (number of elements, not bytes)
//   0x80000018 - 0x80000054 RESULTS (16 words) row-major c[0][0]..c[3][3]
//   0x80000058  X_STRIDE (byte jump after a row of blocks)
//   0x8000005C  K_ROW_LEN(number of 128-bit blocks per row)
//
// Notes:
// - Results base moved to 0x18 to avoid overlap with K_DIM at 0x14.
// - Assumes FIFO.sv is the REGISTERED-on-pop version (dout valid after pop).
// - Assumes dma_rdata is valid one cycle after dma_re pulse.
// =============================================================================

module MatmulAccelerator (
    input  wire         clk,
    input  wire         reset,

    // --- MMIO interface ---
    input  wire [31:0]  mmio_addr,
    input  wire [31:0]  mmio_wdata,
    input  wire [3:0]   mmio_we,
    input  wire         mmio_re,
    output reg  [31:0]  mmio_rdata,

    // --- DMA interface ---
    output reg  [31:0]  dma_addr,
    output reg          dma_re,
    input  wire [127:0] dma_rdata,

    // --- Control ---
    output wire         accel_busy
);

    // ---- Shadow config ----
    reg [31:0] shadow_w_addr, shadow_x_addr;
    reg [31:0] shadow_m_dim, shadow_n_dim, shadow_k_dim;
    reg [31:0] shadow_x_stride, shadow_k_row_len;

    // ---- Active command ----
    reg [31:0] active_ctrl;
    reg [31:0] active_w_addr, active_x_addr;
    reg [31:0] active_m_dim, active_n_dim, active_k_dim;
    reg [31:0] active_x_stride, active_k_row_len;

    // ---- FIFO ----
    wire [255:0] fifo_din, fifo_dout;
    wire         fifo_full, fifo_empty;
    wire         fifo_push;
    reg          fifo_pop;
    reg          error_sticky;

    // Packet: [0]=CTRL [1]=W [2]=X [3]=X_STRIDE [4]=M [5]=N [6]=K [7]=K_ROW_LEN
    assign fifo_din = {
        shadow_k_row_len,
        shadow_k_dim,
        shadow_n_dim,
        shadow_m_dim,
        shadow_x_stride,
        shadow_x_addr,
        shadow_w_addr,
        mmio_wdata
    };

    // ---- MMIO decode ----
    wire       mmio_wr    = (|mmio_we);
    wire [7:0] reg_offset = mmio_addr[7:0];
    assign fifo_push = mmio_wr && (reg_offset == 8'h00) && mmio_wdata[0];

    FIFO #(.WIDTH(256), .DEPTH(4)) cmd_fifo (
        .clk   (clk),
        .reset (reset),
        .push  (fifo_push && !fifo_full),
        .pop   (fifo_pop),
        .din   (fifo_din),
        .dout  (fifo_dout),
        .full  (fifo_full),
        .empty (fifo_empty),
        .count ()
    );

    // ---- Config writes ----
    always @(posedge clk) begin
        if (reset) begin
            shadow_w_addr <= 0; shadow_x_addr <= 0;
            shadow_m_dim  <= 0; shadow_n_dim  <= 0; shadow_k_dim <= 0;
            error_sticky  <= 0;
        end else if (mmio_wr) begin
            case (reg_offset)
                8'h04: shadow_w_addr <= mmio_wdata;
                8'h08: shadow_x_addr <= mmio_wdata;
                8'h0C: shadow_m_dim  <= mmio_wdata;
                8'h10: shadow_n_dim  <= mmio_wdata;
                8'h14: shadow_k_dim  <= mmio_wdata;
                8'h58: shadow_x_stride  <= mmio_wdata;
                8'h5C: shadow_k_row_len <= mmio_wdata;
                8'h00: begin
                    if (mmio_wdata[4]) error_sticky <= 1'b0;       // err_clr
                    if (mmio_wdata[0] && fifo_full) error_sticky <= 1'b1; // start when full
                end
            endcase
        end
    end

    // ---- Data regs ----
    reg [127:0] reg_A, reg_B;
    reg [127:0] A_buf [0:1], B_buf [0:1];
    reg         buf_valid [0:1];
    reg         use_idx, fill_idx;
    reg         cmd_active;
    reg signed [31:0] c [0:3][0:3];

    reg [31:0] k_cnt, k_limit;
    reg [31:0] pref_k_cnt;
    reg [31:0] pref_addr_a, pref_addr_b;
    reg [31:0] pref_k_row_cnt;
    wire       dma_re_from_fill;
    wire       dma_resp_valid;
    wire       produce_pulse;
    wire       consume_pulse;
`ifndef SYNTHESIS
    reg        reset_check_pending, latch_check_pending;
    reg        compute_wait_prev;
    reg        fill_idle_stall_prev;
    reg [31:0] last_dma_addr_q;
    reg [31:0] produced_blocks, consumed_blocks;
    reg        last_req_is_a_q;
    reg [31:0] accel_busy_cycles;
    reg [31:0] compute_cycles;
    reg [31:0] compute_stall_cycles;
    reg [31:0] fill_cycles;
    reg [31:0] dma_req_count;
    reg [1:0]  max_occupancy;
    reg [31:0] occupancy0_cycles;
    reg [31:0] occupancy1_cycles;
    reg [31:0] occupancy2_cycles;
    reg [31:0] cmd_seq;
    reg signed [31:0] c_prev [0:3][0:3];
`endif

    // ---- States ----
    localparam S_IDLE        = 4'd0;
    localparam S_POP         = 4'd1;
    localparam S_LATCH       = 4'd2;
    localparam S_COMPUTE     = 4'd3;
    localparam S_DONE        = 4'd4;
    localparam F_IDLE        = 2'd0;
    localparam F_WAIT_A      = 2'd1;
    localparam F_WAIT_B      = 2'd2;

    reg [3:0] state;
    reg [1:0] fill_state;
    reg       startup_prefill_done;

    // Busy: running OR queued work exists
    assign accel_busy = (state != S_IDLE) || !fifo_empty;

    // Done bit: true when totally idle and queue empty
    wire done_bit = (state == S_IDLE) && fifo_empty;

    // ---- Helpers: packed int8 access ----
    function automatic signed [7:0] get_a(input int step, input int idx);
        int lo;
        begin lo = step*32 + idx*8; get_a = $signed(reg_A[lo +: 8]); end
    endfunction
    function automatic signed [7:0] get_b(input int step, input int idx);
        int lo;
        begin lo = step*32 + idx*8; get_b = $signed(reg_B[lo +: 8]); end
    endfunction

    // ceil(K/4)
    function automatic [31:0] ceil_div4(input [31:0] K);
        begin ceil_div4 = (K + 32'd3) >> 2; end
    endfunction

    // remaining elements in this block = K - 4*k_cnt
    function automatic int rem_k(input int K, input int blk);
        begin rem_k = K - (blk * 4); end
    endfunction

    integer i, j;
    assign dma_re_from_fill   = dma_re;
    // DMA model presents response one cycle after request pulse in this implementation.
    assign dma_resp_valid     = dma_re;
    assign produce_pulse      = (fill_state == F_WAIT_B) && dma_resp_valid;
    assign consume_pulse      = (state == S_COMPUTE) && startup_prefill_done &&
                                (k_cnt < k_limit) && buf_valid[use_idx];

    always @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE;
            fill_state <= F_IDLE;
            fifo_pop <= 0;
            dma_re <= 0;
            startup_prefill_done <= 1'b0;
            k_cnt <= 0; k_limit <= 0;
            pref_k_cnt <= 0;
            pref_addr_a <= 0; pref_addr_b <= 0;
            pref_k_row_cnt <= 0;
            active_ctrl <= 0;
            active_w_addr <= 0; active_x_addr <= 0;
            active_m_dim <= 0; active_n_dim <= 0; active_k_dim <= 0;
            active_x_stride <= 0; active_k_row_len <= 0;
            reg_A <= 0; reg_B <= 0;
            A_buf[0] <= 128'd0; A_buf[1] <= 128'd0;
            B_buf[0] <= 128'd0; B_buf[1] <= 128'd0;
            buf_valid[0] <= 1'b0; buf_valid[1] <= 1'b0;
            use_idx <= 1'b0; fill_idx <= 1'b0;
            cmd_active <= 1'b0;
            for (i=0;i<4;i=i+1) for (j=0;j<4;j=j+1) c[i][j] <= '0;
        end else begin
            fifo_pop <= 0;
            dma_re   <= 0;

            case (state)
                S_IDLE: begin
                    if (!fifo_empty) begin
                        fifo_pop <= 1'b1;
                        state    <= S_POP;
                    end
                end

                S_POP: begin
                    // fifo_dout now holds popped entry (registered-on-pop FIFO)
                    state <= S_LATCH;
                end

                S_LATCH: begin
                    active_ctrl   <= fifo_dout[31:0];
                    active_w_addr <= fifo_dout[63:32];
                    active_x_addr <= fifo_dout[95:64];
                    active_m_dim  <= fifo_dout[159:128];
                    active_n_dim  <= fifo_dout[191:160];
                    active_k_dim  <= fifo_dout[223:192];
                    active_x_stride  <= fifo_dout[127:96];
                    active_k_row_len <= fifo_dout[255:224];
                    cmd_active <= 1'b1;

                    // STEP 2 hygiene: new command clears prefetch buffer bookkeeping.
                    A_buf[0] <= 128'd0; A_buf[1] <= 128'd0;
                    B_buf[0] <= 128'd0; B_buf[1] <= 128'd0;
                    buf_valid[0] <= 1'b0; buf_valid[1] <= 1'b0;
                    use_idx <= 1'b0; fill_idx <= 1'b0;

                    // clear accumulators once per command
                    for (i=0;i<4;i=i+1) for (j=0;j<4;j=j+1) c[i][j] <= '0;

                    // setup loop
                    k_cnt     <= 0;
                    k_limit   <= ceil_div4(fifo_dout[223:192]);

                    pref_k_cnt <= 0;
                    pref_k_row_cnt <= 0;
                    pref_addr_a <= fifo_dout[95:64];
                    pref_addr_b <= fifo_dout[63:32];
                    fill_state <= F_IDLE;
                    startup_prefill_done <= 1'b0;

                    state    <= S_COMPUTE;
                end

                S_COMPUTE: begin
                    if (k_cnt >= k_limit) begin
                        state <= S_DONE;
                    end else if (!startup_prefill_done) begin
                        if (((k_limit <= 1) && (buf_valid[0] || buf_valid[1])) ||
                            ((k_limit > 1) && (buf_valid[0] && buf_valid[1]))) begin
                            startup_prefill_done <= 1'b1;
                        end
                        state <= S_COMPUTE;
                    end else if (buf_valid[use_idx]) begin
                        int rem;
                        rem = rem_k(active_k_dim, k_cnt);

                        reg_A <= A_buf[use_idx];
                        reg_B <= B_buf[use_idx];
                        buf_valid[use_idx] <= 1'b0;
                        use_idx <= ~use_idx;

                        for (i=0;i<4;i=i+1) begin
                            for (j=0;j<4;j=j+1) begin
                                if ((i < active_m_dim) && (j < active_n_dim)) begin
                                    c[i][j] <= c[i][j]
                                        + ((rem > 0) ? (get_a(0,i) * get_b(0,j)) : 0)
                                        + ((rem > 1) ? (get_a(1,i) * get_b(1,j)) : 0)
                                        + ((rem > 2) ? (get_a(2,i) * get_b(2,j)) : 0)
                                        + ((rem > 3) ? (get_a(3,i) * get_b(3,j)) : 0);
                                end
                            end
                        end

                        if (k_cnt + 1 >= k_limit) begin
                            state <= S_DONE;
                        end else begin
                            k_cnt <= k_cnt + 1;
                            state <= S_COMPUTE;
                        end
                    end else begin
                        // Prefetch lagged; hold compute state and keep accumulators unchanged.
                        state <= S_COMPUTE;
                    end
                end

                S_DONE: begin
                    cmd_active <= 1'b0;
                    fill_state <= F_IDLE;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase

            // STEP 3 fill engine:
            // fill slot with A then B using a single outstanding DMA request.
            // Gated off during S_LATCH/S_DONE to prevent cross-command response bleed-through.
            if (cmd_active && (state != S_LATCH) && (state != S_DONE)) begin
                case (fill_state)
                    F_IDLE: begin
                        if ((pref_k_cnt < k_limit) && !buf_valid[fill_idx]) begin
                            dma_addr <= pref_addr_a;
                            dma_re <= 1'b1;
                            fill_state <= F_WAIT_A;
`ifndef SYNTHESIS
                            last_dma_addr_q <= pref_addr_a;
                            last_req_is_a_q <= 1'b1;
`endif
                        end
                    end

                    F_WAIT_A: begin
                        if (dma_resp_valid) begin
                            A_buf[fill_idx] <= dma_rdata;
                            dma_addr <= pref_addr_b;
                            dma_re <= 1'b1;
                            fill_state <= F_WAIT_B;
`ifndef SYNTHESIS
                            last_dma_addr_q <= pref_addr_b;
                            last_req_is_a_q <= 1'b0;
`endif
                        end
                    end

                    F_WAIT_B: begin
                        if (dma_resp_valid) begin
                            logic [31:0] next_pref_addr_a;
                            logic [31:0] next_pref_addr_b;
                            logic [31:0] next_pref_k_row_cnt;
                            logic [0:0]  next_fill_slot;

                            B_buf[fill_idx] <= dma_rdata;
                            buf_valid[fill_idx] <= 1'b1;
                            pref_k_cnt <= pref_k_cnt + 1;

                            next_pref_addr_b = pref_addr_b + 32'd16;
                            next_pref_k_row_cnt = pref_k_row_cnt;
                            if (active_k_row_len > 0 && (pref_k_row_cnt + 1 >= active_k_row_len)) begin
                                next_pref_k_row_cnt = 0;
                                next_pref_addr_a = pref_addr_a + active_x_stride;
                            end else begin
                                next_pref_k_row_cnt = pref_k_row_cnt + 32'd1;
                                next_pref_addr_a = pref_addr_a + 32'd16;
                            end

                            pref_addr_b <= next_pref_addr_b;
                            pref_addr_a <= next_pref_addr_a;
                            pref_k_row_cnt <= next_pref_k_row_cnt;

                            next_fill_slot = ~fill_idx;
                            fill_idx <= next_fill_slot;

                            // Launch next A immediately to remove block-boundary bubble.
                            if ((pref_k_cnt + 1 < k_limit) && !buf_valid[next_fill_slot]) begin
                                dma_addr <= next_pref_addr_a;
                                dma_re <= 1'b1;
                                fill_state <= F_WAIT_A;
`ifndef SYNTHESIS
                                last_dma_addr_q <= next_pref_addr_a;
                                last_req_is_a_q <= 1'b1;
`endif
                            end else begin
                                fill_state <= F_IDLE;
                            end
                        end
                    end

                    default: begin
                        fill_state <= F_IDLE;
                    end
                endcase
            end else begin
                fill_state <= F_IDLE;
            end
        end
    end

`ifndef SYNTHESIS
    // STEP 2: One-cycle delayed checks for synchronous reset and command latch hygiene.
    always @(posedge clk) begin
        if (reset) begin
            reset_check_pending <= 1'b1;
            latch_check_pending <= 1'b0;
            compute_wait_prev <= 1'b0;
            fill_idle_stall_prev <= 1'b0;
            last_dma_addr_q <= 32'd0;
            produced_blocks <= 32'd0;
            consumed_blocks <= 32'd0;
            last_req_is_a_q <= 1'b0;
            accel_busy_cycles <= 32'd0;
            compute_cycles <= 32'd0;
            compute_stall_cycles <= 32'd0;
            fill_cycles <= 32'd0;
            dma_req_count <= 32'd0;
            max_occupancy <= 2'd0;
            occupancy0_cycles <= 32'd0;
            occupancy1_cycles <= 32'd0;
            occupancy2_cycles <= 32'd0;
            cmd_seq <= 32'd0;
            for (i=0;i<4;i=i+1) begin
                for (j=0;j<4;j=j+1) begin
                    c_prev[i][j] <= '0;
                end
            end
        end else begin
            if (reset_check_pending) begin
                if (buf_valid[0] || buf_valid[1] || use_idx || fill_idx || cmd_active) begin
                    $fatal(1, "Matmul STEP2 reset hygiene failed");
                end
            end

            if (latch_check_pending) begin
                if (buf_valid[0] || buf_valid[1]) begin
                    $fatal(1, "Matmul STEP2 latch hygiene failed: buf_valid not cleared");
                end
                if (use_idx || fill_idx) begin
                    $fatal(1, "Matmul STEP2 latch hygiene failed: indices not reset");
                end
                if (!cmd_active) begin
                    $fatal(1, "Matmul STEP2 latch hygiene failed: cmd_active not set");
                end
            end

            if (fill_idx !== 1'b0 && fill_idx !== 1'b1) begin
                $fatal(1, "Matmul STEP3 invariant failed: fill_idx not 1-bit");
            end
            if (use_idx !== 1'b0 && use_idx !== 1'b1) begin
                $fatal(1, "Matmul STEP3 invariant failed: use_idx not 1-bit");
            end
            if (buf_valid[fill_idx] && dma_re_from_fill) begin
                $fatal(1, "Matmul STEP3 invariant failed: fill into occupied slot");
            end
            if (!cmd_active) begin
                if (buf_valid[0] || buf_valid[1]) begin
                    $fatal(1, "Matmul STEP3 invariant failed: stale valid buffer without cmd");
                end
                if (fill_state != F_IDLE || dma_re_from_fill) begin
                    $fatal(1, "Matmul STEP3 invariant failed: fill active without cmd");
                end
            end

            if (reset_check_pending && dma_re_from_fill) begin
                $fatal(1, "Matmul STEP3 invariant failed: dma_re_from_fill right after reset");
            end
            if (dma_resp_valid && (fill_state != F_WAIT_A) && (fill_state != F_WAIT_B)) begin
                $fatal(1, "Matmul STEP3 invariant failed: DMA response captured outside wait state");
            end
            if ((fill_state == F_WAIT_A) && dma_resp_valid && !last_req_is_a_q) begin
                $fatal(1, "Matmul STEP3 invariant failed: expected A response tag");
            end
            if ((fill_state == F_WAIT_B) && dma_resp_valid && last_req_is_a_q) begin
                $fatal(1, "Matmul STEP3 invariant failed: expected B response tag");
            end
            if ((fill_state == F_WAIT_A) && dma_resp_valid && (last_dma_addr_q != pref_addr_a)) begin
                $fatal(1, "Matmul STEP3 invariant failed: A response addr mismatch");
            end
            if ((fill_state == F_WAIT_B) && dma_resp_valid && (last_dma_addr_q != pref_addr_b)) begin
                $fatal(1, "Matmul STEP3 invariant failed: B response addr mismatch");
            end
            if ((state == S_LATCH) && dma_resp_valid) begin
                $fatal(1, "Matmul STEP3 invariant failed: response observed during S_LATCH");
            end

            if (state == S_LATCH) begin
                cmd_seq <= cmd_seq + 32'd1;
                produced_blocks <= 32'd0;
                consumed_blocks <= 32'd0;
                accel_busy_cycles <= 32'd0;
                compute_cycles <= 32'd0;
                compute_stall_cycles <= 32'd0;
                fill_cycles <= 32'd0;
                dma_req_count <= 32'd0;
                max_occupancy <= 2'd0;
                occupancy0_cycles <= 32'd0;
                occupancy1_cycles <= 32'd0;
                occupancy2_cycles <= 32'd0;
            end else begin
                if (produce_pulse) produced_blocks <= produced_blocks + 32'd1;
                if (consume_pulse) consumed_blocks <= consumed_blocks + 32'd1;

                if (cmd_active || !fifo_empty || (state != S_IDLE)) begin
                    accel_busy_cycles <= accel_busy_cycles + 32'd1;
                end
                if (state == S_COMPUTE && startup_prefill_done &&
                    (k_cnt < k_limit) && buf_valid[use_idx]) begin
                    compute_cycles <= compute_cycles + 32'd1;
                end
                if (state == S_COMPUTE && (k_cnt < k_limit) && !buf_valid[use_idx]) begin
                    compute_stall_cycles <= compute_stall_cycles + 32'd1;
                    if (consume_pulse) begin
                        $fatal(1, "Matmul STEP4.5 invariant failed: consume during compute stall");
                    end
                    if ((pref_k_cnt < k_limit) && !buf_valid[fill_idx] && (fill_state == F_IDLE)) begin
                        if (fill_idle_stall_prev) begin
                            $fatal(1, "Matmul STEP4.5 invariant failed: fill idle while compute stalled");
                        end
                        fill_idle_stall_prev <= 1'b1;
                    end else begin
                        fill_idle_stall_prev <= 1'b0;
                    end
                end else begin
                    fill_idle_stall_prev <= 1'b0;
                end
                if (fill_state != F_IDLE) begin
                    fill_cycles <= fill_cycles + 32'd1;
                end
                if (dma_re_from_fill) begin
                    dma_req_count <= dma_req_count + 32'd1;
                end

                if (buf_valid[0] && buf_valid[1]) begin
                    occupancy2_cycles <= occupancy2_cycles + 32'd1;
                    if (max_occupancy < 2'd2) max_occupancy <= 2'd2;
                end else if (buf_valid[0] || buf_valid[1]) begin
                    occupancy1_cycles <= occupancy1_cycles + 32'd1;
                    if (max_occupancy < 2'd1) max_occupancy <= 2'd1;
                end else begin
                    occupancy0_cycles <= occupancy0_cycles + 32'd1;
                end
            end
            if (state == S_DONE) begin
                if (produced_blocks != consumed_blocks) begin
                    $fatal(1, "Matmul STEP4 invariant failed: produced/consumed mismatch cmd=%0d k_limit=%0d produced=%0d consumed=%0d pref_k_cnt=%0d",
                           cmd_seq, k_limit, produced_blocks, consumed_blocks, pref_k_cnt);
                end
                if (consumed_blocks != k_limit) begin
                    $fatal(1, "Matmul STEP4 invariant failed: consumed count != k_limit cmd=%0d k_limit=%0d consumed=%0d",
                           cmd_seq, k_limit, consumed_blocks);
                end
                $display("ACCEL_PERF cmd=%0d k_limit=%0d busy=%0d compute=%0d stall=%0d fill=%0d dma_req=%0d produced=%0d consumed=%0d occ0=%0d occ1=%0d occ2=%0d occ_max=%0d",
                         cmd_seq, k_limit, accel_busy_cycles, compute_cycles, compute_stall_cycles,
                         fill_cycles, dma_req_count, produced_blocks, consumed_blocks,
                         occupancy0_cycles, occupancy1_cycles, occupancy2_cycles, max_occupancy);
            end

            if (compute_wait_prev) begin
                for (i=0;i<4;i=i+1) begin
                    for (j=0;j<4;j=j+1) begin
                        if (c[i][j] !== c_prev[i][j]) begin
                            $fatal(1, "Matmul STEP4 invariant failed: c changed while compute stalled");
                        end
                    end
                end
            end
            compute_wait_prev <= (state == S_COMPUTE) && !buf_valid[use_idx];
            for (i=0;i<4;i=i+1) begin
                for (j=0;j<4;j=j+1) begin
                    c_prev[i][j] <= c[i][j];
                end
            end

            reset_check_pending <= 1'b0;
            latch_check_pending <= (state == S_LATCH);
        end
    end
`endif

    // ---- MMIO Read ----
    reg [7:0] reg_offset_d;
    always @(posedge clk) begin
        if (reset) reg_offset_d <= 8'd0;
        else if (mmio_re) reg_offset_d <= reg_offset;
    end

    // STATUS bits: [0]=busy [1]=done [2]=full [3]=empty [4]=error
    always @(*) begin
        case (reg_offset_d)
            8'h00: mmio_rdata = {27'd0, error_sticky, fifo_empty, fifo_full, done_bit, accel_busy};

            // Results start at 0x18 now
            8'h18: mmio_rdata = c[0][0];
            8'h1C: mmio_rdata = c[0][1];
            8'h20: mmio_rdata = c[0][2];
            8'h24: mmio_rdata = c[0][3];

            8'h28: mmio_rdata = c[1][0];
            8'h2C: mmio_rdata = c[1][1];
            8'h30: mmio_rdata = c[1][2];
            8'h34: mmio_rdata = c[1][3];

            8'h38: mmio_rdata = c[2][0];
            8'h3C: mmio_rdata = c[2][1];
            8'h40: mmio_rdata = c[2][2];
            8'h44: mmio_rdata = c[2][3];

            8'h48: mmio_rdata = c[3][0];
            8'h4C: mmio_rdata = c[3][1];
            8'h50: mmio_rdata = c[3][2];
            8'h54: mmio_rdata = c[3][3];

            default: mmio_rdata = 32'd0;
        endcase
    end

endmodule
