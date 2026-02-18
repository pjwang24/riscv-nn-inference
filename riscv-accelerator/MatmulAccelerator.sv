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

    // ---- Active command ----
    reg [31:0] active_ctrl;
    reg [31:0] active_w_addr, active_x_addr;
    reg [31:0] active_m_dim, active_n_dim, active_k_dim;

    // ---- FIFO ----
    wire [255:0] fifo_din, fifo_dout;
    wire         fifo_full, fifo_empty;
    wire         fifo_push;
    reg          fifo_pop;
    reg          error_sticky;

    // Packet: [0]=CTRL [1]=W [2]=X [3]=RSVD [4]=M [5]=N [6]=K [7]=RSVD
    assign fifo_din = {
        32'd0,
        shadow_k_dim,
        shadow_n_dim,
        shadow_m_dim,
        32'd0,
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
                8'h00: begin
                    if (mmio_wdata[4]) error_sticky <= 1'b0;       // err_clr
                    if (mmio_wdata[0] && fifo_full) error_sticky <= 1'b1; // start when full
                end
            endcase
        end
    end

    // ---- Data regs ----
    reg [127:0] reg_A, reg_B;
    reg signed [31:0] c [0:3][0:3];

    reg [31:0] curr_addr_a, curr_addr_b;
    reg [31:0] k_cnt, k_limit;

    // ---- States ----
    localparam S_IDLE        = 4'd0;
    localparam S_POP         = 4'd1;
    localparam S_LATCH       = 4'd2;
    localparam S_LOAD_A      = 4'd3;
    localparam S_LOAD_A_WAIT = 4'd4;
    localparam S_LOAD_B      = 4'd5;
    localparam S_LOAD_B_WAIT = 4'd6;
    localparam S_COMPUTE     = 4'd7;
    localparam S_DONE        = 4'd8;

    reg [3:0] state;

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

    always @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE;
            fifo_pop <= 0;
            dma_re <= 0;
            k_cnt <= 0; k_limit <= 0;
            curr_addr_a <= 0; curr_addr_b <= 0;
            active_ctrl <= 0;
            active_w_addr <= 0; active_x_addr <= 0;
            active_m_dim <= 0; active_n_dim <= 0; active_k_dim <= 0;
            reg_A <= 0; reg_B <= 0;
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

                    // clear accumulators once per command
                    for (i=0;i<4;i=i+1) for (j=0;j<4;j=j+1) c[i][j] <= '0;

                    // setup loop
                    k_cnt   <= 0;
                    k_limit <= ceil_div4(fifo_dout[223:192]);

                    curr_addr_a <= fifo_dout[95:64];
                    curr_addr_b <= fifo_dout[63:32];

                    // kick first A load
                    dma_addr <= fifo_dout[95:64];
                    dma_re   <= 1'b1;
                    state    <= S_LOAD_A;
                end

                S_LOAD_A: begin
                    state <= S_LOAD_A_WAIT;
                end

                S_LOAD_A_WAIT: begin
                    reg_A <= dma_rdata;
                    dma_addr <= curr_addr_b;
                    dma_re   <= 1'b1;
                    state    <= S_LOAD_B;
                end

                S_LOAD_B: begin
                    state <= S_LOAD_B_WAIT;
                end

                S_LOAD_B_WAIT: begin
                    reg_B <= dma_rdata;
                    state <= S_COMPUTE;
                end

                S_COMPUTE: begin
                    int rem;
                    rem = rem_k(active_k_dim, k_cnt);

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
                        k_cnt       <= k_cnt + 1;
                        curr_addr_a <= curr_addr_a + 16;
                        curr_addr_b <= curr_addr_b + 16;

                        dma_addr <= curr_addr_a + 16;
                        dma_re   <= 1'b1;
                        state    <= S_LOAD_A;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

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
