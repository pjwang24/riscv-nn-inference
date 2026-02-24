`include "const.vh"

module MatmulAcceleratorSystolic (
    input  wire         clk,
    input  wire         reset,
    input  wire [31:0]  mmio_addr,
    input  wire [31:0]  mmio_wdata,
    input  wire [3:0]   mmio_we,
    input  wire         mmio_re,
    output reg  [31:0]  mmio_rdata,
    output reg  [31:0]  dma_addr,
    output reg          dma_re,
    input  wire         dma_req_ready,
    input  wire         dma_resp_valid,
    input  wire [255:0] dma_rdata,
    output wire         accel_busy
);

    localparam integer N = 4;

    reg [31:0] ctrl_status;
    reg [31:0] shadow_w_addr, shadow_x_addr;
    reg [31:0] shadow_m_dim, shadow_n_dim, shadow_k_dim;
    reg [31:0] shadow_x_stride, shadow_k_row_len;

    reg busy_r;
    reg done_r;
    reg [31:0] run_count;
    reg [31:0] k_limit;

    logic clear_acc;
    logic en;
    logic signed [15:0] a_left [N];
    logic signed [15:0] b_top [N];
    logic signed [31:0] c_out [N][N];

    integer i, j;

    SystolicArray #(
        .N(N),
        .DATA_W(16),
        .ACC_W(32)
    ) u_systolic (
        .clk      (clk),
        .rst      (reset),
        .clear_acc(clear_acc),
        .en       (en),
        .a_left   (a_left),
        .b_top    (b_top),
        .c_out    (c_out)
    );

    assign accel_busy = busy_r;
    wire mmio_wr = (|mmio_we);
    wire [7:0] reg_offset = mmio_addr[7:0];

    always @(*) begin
        for (i = 0; i < N; i = i + 1) begin
            // Placeholder feed. Full DMA-fed skew controller comes next phase.
            a_left[i] = 16'sd0;
            b_top[i] = 16'sd0;
        end

        clear_acc = 1'b0;
        en = busy_r;
        dma_addr = 32'd0;
        dma_re = 1'b0;
    end

    always @(posedge clk) begin
        if (reset) begin
            shadow_w_addr <= 32'd0;
            shadow_x_addr <= 32'd0;
            shadow_m_dim <= 32'd0;
            shadow_n_dim <= 32'd0;
            shadow_k_dim <= 32'd0;
            shadow_x_stride <= 32'd0;
            shadow_k_row_len <= 32'd0;
            busy_r <= 1'b0;
            done_r <= 1'b0;
            run_count <= 32'd0;
            k_limit <= 32'd0;
        end else begin
            if (mmio_wr) begin
                case (reg_offset)
                    8'h04: shadow_w_addr <= mmio_wdata;
                    8'h08: shadow_x_addr <= mmio_wdata;
                    8'h0C: shadow_m_dim <= mmio_wdata;
                    8'h10: shadow_n_dim <= mmio_wdata;
                    8'h14: shadow_k_dim <= mmio_wdata;
                    8'h58: shadow_x_stride <= mmio_wdata;
                    8'h5C: shadow_k_row_len <= mmio_wdata;
                    8'h00: begin
                        if (mmio_wdata[0] && !busy_r) begin
                            busy_r <= 1'b1;
                            done_r <= 1'b0;
                            run_count <= 32'd0;
                            k_limit <= (shadow_k_dim + 32'd3) >> 2;
                        end
                    end
                    default: begin
                    end
                endcase
            end

            if (busy_r) begin
                run_count <= run_count + 32'd1;
                // Conservative completion model for now. Prevents deadlock if selected.
                if (run_count >= (k_limit + 32'd8)) begin
                    busy_r <= 1'b0;
                    done_r <= 1'b1;
                end
            end
        end
    end

    reg [7:0] reg_offset_d;
    always @(posedge clk) begin
        if (reset) reg_offset_d <= 8'd0;
        else if (mmio_re) reg_offset_d <= reg_offset;
    end

    always @(*) begin
        case (reg_offset_d)
            8'h00: mmio_rdata = {27'd0, 1'b0, 1'b1, 1'b0, done_r, busy_r};
            8'h18: mmio_rdata = c_out[0][0];
            8'h1C: mmio_rdata = c_out[0][1];
            8'h20: mmio_rdata = c_out[0][2];
            8'h24: mmio_rdata = c_out[0][3];
            8'h28: mmio_rdata = c_out[1][0];
            8'h2C: mmio_rdata = c_out[1][1];
            8'h30: mmio_rdata = c_out[1][2];
            8'h34: mmio_rdata = c_out[1][3];
            8'h38: mmio_rdata = c_out[2][0];
            8'h3C: mmio_rdata = c_out[2][1];
            8'h40: mmio_rdata = c_out[2][2];
            8'h44: mmio_rdata = c_out[2][3];
            8'h48: mmio_rdata = c_out[3][0];
            8'h4C: mmio_rdata = c_out[3][1];
            8'h50: mmio_rdata = c_out[3][2];
            8'h54: mmio_rdata = c_out[3][3];
            default: mmio_rdata = 32'd0;
        endcase
    end

    // Tie off currently unused DMA signals to avoid lint confusion in this phase.
    wire _unused_dma = dma_req_ready | dma_resp_valid | ^dma_rdata;
    wire _unused_cfg = ^shadow_w_addr ^ ^shadow_x_addr ^ ^shadow_m_dim ^
                       ^shadow_n_dim ^ ^shadow_x_stride ^ ^shadow_k_row_len ^
                       _unused_dma;

endmodule
