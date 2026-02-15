`include "const.vh"

// =============================================================================
// MatmulAccelerator — 4-lane DMA-based INT8 dot-product accelerator
//
// MMIO register map (active when dcache_addr[31] == 1):
//   0x80000000  CTRL/STATUS  W: [0]=start [1]=clear   R: [0]=busy [1]=done
//   0x80000004  W_ADDR       W: base byte address of weight matrix
//   0x80000008  X_ADDR       W: base byte address of input vector
//   0x8000000C  M_DIM        W: number of output rows to process (≤4)
//   0x80000010  N_DIM        W: dot-product length (must be multiple of 4)
//   0x80000014  RESULT0      R: lane 0 accumulated result (INT32)
//   0x80000018  RESULT1      R: lane 1 accumulated result
//   0x8000001C  RESULT2      R: lane 2 accumulated result
//   0x80000020  RESULT3      R: lane 3 accumulated result
//
// Operation:
//   1. CPU writes W_ADDR, X_ADDR, M_DIM, N_DIM
//   2. CPU writes CTRL=1 (start)
//   3. Accelerator takes over dcache port, reads weights+inputs via DMA
//   4. CPU polls STATUS until done bit is set
//   5. CPU reads RESULT0–RESULT3
// =============================================================================

module MatmulAccelerator (
    input wire        clk,
    input wire        reset,

    // --- MMIO interface (directly from CPU dcache signals) ---
    input wire [31:0] mmio_addr,    // CPU dcache_addr when addr[31]==1
    input wire [31:0] mmio_wdata,   // CPU dcache_din
    input wire [3:0]  mmio_we,      // CPU dcache_we (byte enables)
    input wire        mmio_re,      // CPU dcache_re
    output reg [31:0] mmio_rdata,   // read data back to CPU

    // --- DMA interface (directly drives dcache port when active) ---
    output reg [31:0] dma_addr,     // address to read from data memory
    output reg        dma_re,       // read enable for data memory
    input wire [31:0] dma_rdata,    // data from data memory

    // --- Control ---
    output wire       accel_busy    // 1 when accelerator owns dcache port
);

    // ---- Configuration registers ----
    reg [31:0] cfg_w_addr;   // base byte address of weights
    reg [31:0] cfg_x_addr;   // base byte address of input
    reg [31:0] cfg_m_dim;    // rows to compute (1–4)
    reg [31:0] cfg_n_dim;    // dot-product length

    // ---- State machine ----
    localparam S_IDLE        = 3'd0;
    localparam S_LOAD_W      = 3'd1;  // DMA read weights into lane SRAMs
    localparam S_LOAD_W_WAIT = 3'd2;  // wait for read response
    localparam S_COMPUTE     = 3'd3;  // DMA read input + MAC
    localparam S_COMP_WAIT   = 3'd4;  // wait for read response
    localparam S_DONE        = 3'd5;

    reg [2:0]  state;
    reg        done_flag;

    // ---- DMA counters ----
    reg [31:0] w_word_cnt;    // which word we're loading for weights
    reg [31:0] n_words;       // cfg_n_dim / 4
    reg [1:0]  lane_cnt;      // which lane (0–3) during weight load
    reg [31:0] x_word_cnt;    // which word during compute phase

    // ---- 4 MAC lanes ----
    // Each lane has: weight SRAM (up to 256 words = 1024 bytes), accumulator
    localparam W_SRAM_DEPTH = 256; // supports N up to 1024

    reg [31:0] w_sram_0 [0:W_SRAM_DEPTH-1];
    reg [31:0] w_sram_1 [0:W_SRAM_DEPTH-1];
    reg [31:0] w_sram_2 [0:W_SRAM_DEPTH-1];
    reg [31:0] w_sram_3 [0:W_SRAM_DEPTH-1];

    reg signed [31:0] acc_0, acc_1, acc_2, acc_3;

    // ---- Busy signal ----
    assign accel_busy = (state != S_IDLE) && (state != S_DONE);

    // ---- MMIO write handling ----
    wire mmio_wr = (|mmio_we);
    wire [7:0] reg_offset = mmio_addr[7:0];

    // Registered read offset for 1-cycle read latency (breaks comb loop)
    reg [7:0] reg_offset_d;

    // ---- Config register writes (accepted in ALL states) ----
    always @(posedge clk) begin
        if (reset) begin
            cfg_w_addr <= 0;
            cfg_x_addr <= 0;
            cfg_m_dim  <= 0;
            cfg_n_dim  <= 0;
        end else if (mmio_wr) begin
            case (reg_offset)
                8'h04: cfg_w_addr <= mmio_wdata;
                8'h08: cfg_x_addr <= mmio_wdata;
                8'h0C: cfg_m_dim  <= mmio_wdata;
                8'h10: cfg_n_dim  <= mmio_wdata;
            endcase
        end
    end

    // ---- State machine ----
    always @(posedge clk) begin
        if (reset) begin
            state      <= S_IDLE;
            done_flag  <= 0;
            dma_re     <= 0;
            dma_addr   <= 0;
            acc_0      <= 0;
            acc_1      <= 0;
            acc_2      <= 0;
            acc_3      <= 0;
            w_word_cnt <= 0;
            lane_cnt   <= 0;
            x_word_cnt <= 0;
            n_words    <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    dma_re <= 0;
                    if (mmio_wr && reg_offset == 8'h00) begin
                        if (mmio_wdata[1]) begin
                            // Clear accumulators
                            acc_0 <= 0;
                            acc_1 <= 0;
                            acc_2 <= 0;
                            acc_3 <= 0;
                        end
                        if (mmio_wdata[0]) begin
                            // Start
                            state      <= S_LOAD_W;
                            done_flag  <= 0;
                            n_words    <= cfg_n_dim >> 2;
                            w_word_cnt <= 0;
                            lane_cnt   <= 0;
                            acc_0      <= 0;
                            acc_1      <= 0;
                            acc_2      <= 0;
                            acc_3      <= 0;
                            // Start first weight read
                            dma_addr   <= cfg_w_addr;
                            dma_re     <= 1;
                        end
                    end
                end

                // ---- Weight Loading Phase ----
                // Read N/4 words per lane × M lanes (up to 4)
                S_LOAD_W: begin
                    // We issued a read last cycle, now wait for response
                    dma_re <= 0;
                    state  <= S_LOAD_W_WAIT;
                end

                S_LOAD_W_WAIT: begin
                    // Response arrives this cycle (1-cycle memory)
                    // Store into appropriate lane SRAM
                    case (lane_cnt)
                        2'd0: w_sram_0[w_word_cnt] <= dma_rdata;
                        2'd1: w_sram_1[w_word_cnt] <= dma_rdata;
                        2'd2: w_sram_2[w_word_cnt] <= dma_rdata;
                        2'd3: w_sram_3[w_word_cnt] <= dma_rdata;
                    endcase

                    // Advance to next word
                    if (w_word_cnt + 1 == n_words) begin
                        // Done with this lane
                        w_word_cnt <= 0;
                        if (lane_cnt + 1 == cfg_m_dim[1:0] ||
                            lane_cnt == 2'd3) begin
                            // All lanes loaded, start compute
                            state      <= S_COMPUTE;
                            x_word_cnt <= 0;
                            dma_addr   <= cfg_x_addr;
                            dma_re     <= 1;
                        end else begin
                            // Next lane
                            lane_cnt   <= lane_cnt + 1;
                            dma_addr   <= cfg_w_addr + ((lane_cnt + 1) * cfg_n_dim);
                            dma_re     <= 1;
                            state      <= S_LOAD_W;
                        end
                    end else begin
                        // Next word in same lane
                        w_word_cnt <= w_word_cnt + 1;
                        dma_addr   <= cfg_w_addr + (lane_cnt * cfg_n_dim) + ((w_word_cnt + 1) << 2);
                        dma_re     <= 1;
                        state      <= S_LOAD_W;
                    end
                end

                // ---- Compute Phase ----
                // Read input words, MAC with stored weights
                S_COMPUTE: begin
                    dma_re <= 0;
                    state  <= S_COMP_WAIT;
                end

                S_COMP_WAIT: begin
                    // Input word arrived — do 4 MACs per lane
                    // Unpack input bytes (little-endian)
                    // dma_rdata = [byte3, byte2, byte1, byte0]

                    // Read weights from SRAMs for this position
                    // w_sram_X[x_word_cnt] was stored during load phase

                    // Perform packed INT8 dot product for each active lane
                    acc_0 <= acc_0 + packed_dot(w_sram_0[x_word_cnt], dma_rdata);
                    if (cfg_m_dim > 1)
                        acc_1 <= acc_1 + packed_dot(w_sram_1[x_word_cnt], dma_rdata);
                    if (cfg_m_dim > 2)
                        acc_2 <= acc_2 + packed_dot(w_sram_2[x_word_cnt], dma_rdata);
                    if (cfg_m_dim > 3)
                        acc_3 <= acc_3 + packed_dot(w_sram_3[x_word_cnt], dma_rdata);

                    if (x_word_cnt + 1 == n_words) begin
                        // All input words processed
                        state     <= S_DONE;
                        done_flag <= 1;
                        dma_re    <= 0;
                    end else begin
                        x_word_cnt <= x_word_cnt + 1;
                        dma_addr   <= cfg_x_addr + ((x_word_cnt + 1) << 2);
                        dma_re     <= 1;
                        state      <= S_COMPUTE;
                    end
                end

                S_DONE: begin
                    dma_re <= 0;
                    // Stay in DONE until CPU reads status or starts new op
                    if (mmio_wr && reg_offset == 8'h00) begin
                        if (mmio_wdata[0]) begin
                            // New start
                            state      <= S_LOAD_W;
                            done_flag  <= 0;
                            n_words    <= cfg_n_dim >> 2;
                            w_word_cnt <= 0;
                            lane_cnt   <= 0;
                            acc_0      <= 0;
                            acc_1      <= 0;
                            acc_2      <= 0;
                            acc_3      <= 0;
                            dma_addr   <= cfg_w_addr;
                            dma_re     <= 1;
                        end else begin
                            state     <= S_IDLE;
                            done_flag <= 0;
                        end
                    end
                end
            endcase
        end
    end

    // ---- Register read offset (1-cycle latency like memory) ----
    always @(posedge clk) begin
        if (reset)
            reg_offset_d <= 0;
        else if (mmio_re)
            reg_offset_d <= reg_offset;
    end

    // ---- MMIO read handling (registered, uses delayed offset) ----
    always @(*) begin
        case (reg_offset_d)
            8'h00:   mmio_rdata = {30'd0, done_flag, accel_busy};
            8'h14:   mmio_rdata = acc_0;
            8'h18:   mmio_rdata = acc_1;
            8'h1C:   mmio_rdata = acc_2;
            8'h20:   mmio_rdata = acc_3;
            default: mmio_rdata = 32'd0;
        endcase
    end

    // ---- Packed INT8 dot product function ----
    // Takes two 32-bit words, each containing 4 packed INT8 values
    // Returns sum of 4 pairwise INT8 products
    function signed [31:0] packed_dot;
        input [31:0] a;
        input [31:0] b;
        reg signed [7:0] a0, a1, a2, a3;
        reg signed [7:0] b0, b1, b2, b3;
        begin
            a0 = $signed(a[7:0]);
            a1 = $signed(a[15:8]);
            a2 = $signed(a[23:16]);
            a3 = $signed(a[31:24]);
            b0 = $signed(b[7:0]);
            b1 = $signed(b[15:8]);
            b2 = $signed(b[23:16]);
            b3 = $signed(b[31:24]);
            packed_dot = (a0 * b0) + (a1 * b1) + (a2 * b2) + (a3 * b3);
        end
    endfunction

endmodule
