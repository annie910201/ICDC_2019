
`timescale 1ns/10ps

module  CONV(clk,reset,busy,ready,iaddr,idata,cwr,caddr_wr,cdata_wr,crd,caddr_rd,cdata_rd,csel);
input						clk;
input						reset;
output	reg					busy;	
input						ready;	
		
output	reg 		[11:0] 	iaddr;
input 				[19:0]	idata;	

output	reg 				cwr;
output	reg 		[11:0]	caddr_wr;
output  reg signed 	[19:0]	cdata_wr;

output	reg 				crd;
output	reg  		[11:0]	caddr_rd;
input				[19:0] 	cdata_rd;

output	reg 		[2:0] 	csel;
reg signed [19:0] buffer[0:8];
reg 				[3:0] 	current_state;
reg 				[3:0] 	next_state;
reg 				[5:0]	center_x;
reg 				[5:0]	center_y;
reg 				[3:0]	counter;
reg 	signed			[44:0] 	sum ;// 9 * idata(20 bits) * kernel(20 bits) + bias(20 bits) * 9 = 2^20 * 2^20 * 9 + 2^20 * 9 ~= 2^40 * 9 ~= 2^40 * 16 = 2^44
reg write_over;
// reg 	signed			[19:0]	max_L0;
reg pooling_over;
reg flatten_over;
wire signed [20:0] tmp_res;
wire [5:0] x_left;
wire [5:0] x_right;
wire [5:0] y_up;
wire [5:0] y_buttom;
parameter INITIAL = 0;
parameter READ_K0 = 1;
parameter CONVOLUTION_K0 = 2;
parameter RELU_K0 = 3;
parameter WR_L0_K0 = 4;
parameter RD_L0_K0 = 5;
parameter WR_L1_K0 = 6;

parameter READ_K1 = 7;
parameter CONVOLUTION_K1 = 8;
parameter RELU_K1 = 9;
parameter WR_L0_K1 = 10;
parameter RD_L0_K1 = 11;
parameter WR_L1_K1 = 12;
parameter FLATTEN = 13;
parameter WR_L2 = 14;
parameter FINISH = 15;

parameter signed K0_0 = 20'h0A89E;
parameter signed K0_1 = 20'h092D5;
parameter signed K0_2 = 20'h06D43;
parameter signed K0_3 = 20'h01004;
parameter signed K0_4 = 20'hF8F71;
parameter signed K0_5 = 20'hF6E54;
parameter signed K0_6 = 20'hFA6D7;
parameter signed K0_7 = 20'hFC834;
parameter signed K0_8 = 20'hFAC19;
parameter bias_0 = 20'h01310;

parameter signed K1_0 = 20'hFDB55;
parameter signed K1_1 = 20'h02992;
parameter signed K1_2 = 20'hFC994;
parameter signed K1_3 = 20'h050FD;
parameter signed K1_4 = 20'h02F20;
parameter signed K1_5 = 20'h0202D;
parameter signed K1_6 = 20'h03BD7;
parameter signed K1_7 = 20'hFD369;
parameter signed K1_8 = 20'h05E68;
parameter bias_1 = 20'hF7295;

assign tmp_res = sum[35:15] + 20'd1;
assign x_left = center_x - 1;
assign x_right = center_x + 1;
assign y_up = center_y - 1;
assign y_buttom = center_y + 1;
reg [19:0] tmp1_mul;
reg  [19:0] tmp2_mul;
wire [44:0] mul;
assign mul = tmp1_mul * tmp2_mul;

reg flatten_0or1;
reg [44:0] tmp1_add;
reg  [44:0] tmp2_add;
wire [44:0] add;
assign add = tmp1_add + tmp2_add;

// state register
always @(posedge clk) begin
	if(reset)
		current_state = INITIAL;
	else
		current_state = next_state;
end

// next state logic 
always @(posedge clk) begin
case(current_state)
INITIAL:
	next_state = (ready)?READ_K0: INITIAL;
READ_K0:
	next_state = (counter == 4'd9)?CONVOLUTION_K0: READ_K0;
CONVOLUTION_K0:
	next_state = (counter == 4'd10)?RELU_K0: CONVOLUTION_K0;
RELU_K0:
	next_state = WR_L0_K0;
WR_L0_K0:
	next_state = (write_over)?RD_L0_K0:READ_K0;
RD_L0_K0:
	next_state = (counter == 4'd4)? WR_L1_K0:RD_L0_K0;
WR_L1_K0:
	next_state = (pooling_over)?READ_K1:RD_L0_K0;

READ_K1:
	next_state = (counter == 4'd9)?CONVOLUTION_K1: READ_K1;
CONVOLUTION_K1:
	next_state = (counter == 4'd10)?RELU_K1: CONVOLUTION_K1;
RELU_K1:
	next_state = WR_L0_K1;
WR_L0_K1:
	next_state = (write_over)?RD_L0_K1:READ_K1;
RD_L0_K1:
	next_state = (counter == 4'd4)? WR_L1_K1:RD_L0_K1;
WR_L1_K1:
	next_state = (pooling_over)?FLATTEN:RD_L0_K1;
FLATTEN:
	next_state = WR_L2;
WR_L2:
	next_state = (flatten_over)?FINISH:FLATTEN;
FINISH:
	next_state = FINISH;
endcase
end

/* output */
// busy
always @(posedge clk)begin
	if(reset)
		busy <= 0;
	else if(ready)
		busy <= 1;
	else if(next_state == FINISH)
		busy <= 0;
end

// iaddr
always @(posedge clk)begin
	if(reset)
		iaddr <= 0;
	else if(current_state == READ_K0 || current_state == READ_K1)
	begin
		case (counter)
			4'd0:
				iaddr <= {y_up, x_left};
			4'd1:
				iaddr <= {y_up, center_x};
			4'd2:
				iaddr <= {y_up, x_right};
			4'd3:
				iaddr <= {center_y, x_left};
			4'd4:
				iaddr <= {center_y, center_x};
			4'd5:
				iaddr <= {center_y, x_right};
			4'd6:
				iaddr <= {y_buttom, x_left};
			4'd7:
				iaddr <= {y_buttom, center_x};
			4'd8:
				iaddr <= {y_buttom, x_right};
		endcase
	end
end
// cwr
always @(posedge clk)begin
	if(reset)
		cwr <= 0;
	else if(next_state == WR_L0_K0 || next_state == WR_L0_K1 || next_state == WR_L1_K0 || next_state == WR_L1_K1)
		cwr <= 1;
	else 
		cwr <= 0;
end

// caddr_wr
always @(posedge clk)begin
	if(reset)
		caddr_wr <= 0;
	else if(next_state == WR_L0_K0 || next_state == WR_L0_K1 )
		caddr_wr <= {center_y, center_x};
	else if(next_state == WR_L1_K0 || next_state == WR_L1_K1)
		caddr_wr <= {center_y[5:1], center_x[5:1]};
	else if(next_state == WR_L2)
		caddr_wr <= caddr_wr + 1;
	else 
		caddr_wr <= 0;
end
// cdata_wr
always @(posedge clk)begin
	if(reset)
		cdata_wr <= 0;
	else if(next_state == WR_L0_K0 || next_state == WR_L0_K1 )
		cdata_wr <= (sum[35])?0:tmp_res[20:1];
	else if(next_state == RD_L0_K0 || next_state == RD_L0_K1)
	begin
		if(counter == 4'd1)
			cdata_wr <= cdata_rd;
		else if(cdata_rd > cdata_wr)
			cdata_wr <= cdata_rd;
	end
	else if(next_state == WR_L2)
		cdata_wr <= cdata_rd;
end
// crd
always @(posedge clk)begin
	if(reset)
		crd <= 0;
	else if(next_state == RD_L0_K0 || next_state == RD_L0_K1)
		crd <= 1;
	else 
		crd <= 0;
end
// caddr_rd
always @(posedge clk)begin
	if(reset)
		caddr_rd <= 0;
	else if(next_state == RD_L0_K0 || next_state == RD_L0_K1)
	begin
		case (counter)
			4'd0:
				caddr_rd <= {center_y, center_x};
			4'd1:
				caddr_rd <= {center_y, x_right};
			4'd2:
				caddr_rd <= {y_buttom, center_x};
			4'd3:
				caddr_rd <= {y_buttom, x_right};
		endcase
	end
	else if(next_state == FLATTEN)
		caddr_rd <= {center_y, center_x};
end
// csel
always @(posedge clk)begin
	if(reset)
		csel <= 0;
	else if(next_state == WR_L0_K0)
		csel <= 3'b001;
	else if(next_state == WR_L0_K1)
		csel <= 3'b010;
	else if(next_state == RD_L0_K0)
		csel <= 3'b001;
	else if(next_state == RD_L0_K1)
		csel <= 3'b010;
	else if(next_state == WR_L1_K0)
		csel <= 3'b011;
	else if(next_state == WR_L1_K1)
		csel <= 3'b100;
	else if(next_state == FLATTEN && !flatten_0or1)
		csel <= 3'b011;
	else if(next_state == FLATTEN && flatten_0or1)
		csel <= 3'b100;
	else if(next_state == WR_L2)
		csel <= 3'b101; 
end

/* datapath */ 
// counter
always @(posedge clk)begin
if(reset)
	counter <= 0;
else if(current_state == READ_K0 && counter == 4'd9)
begin
	counter <= 0;
	$display("%d %d", counter, current_state);
end
else if((current_state == CONVOLUTION_K0 || current_state == CONVOLUTION_K1) && counter == 4'd10)
	counter <= 0;
else if((current_state == RD_L0_K0 || current_state == RD_L0_K1) && counter == 4'd4)
	counter <= 0;
else
	counter <= counter + 1;


end

// buffer and tmp1 and tmp2
always @(posedge clk) begin
	if(reset)
	begin
		buffer[0] <= 0;
		buffer[1] <= 0;
		buffer[2] <= 0;
		buffer[3] <= 0;
		buffer[4] <= 0;
		buffer[5] <= 0;
		buffer[6] <= 0;
		buffer[7] <= 0;
		buffer[8] <= 0;
		tmp1_add <= 0;
		tmp1_mul <= 0;
		tmp2_add <= 0;
		tmp2_mul <= 0;
	end
	else if(current_state == READ_K0 || current_state == READ_K1)
		case (counter)
			4'd1:
				buffer[0] <= (center_y!=0 && center_x !=0) ? idata : 0;
			4'd2:
				buffer[1] <= (center_y!=0)?idata : 0;
			4'd3:
				buffer[2] <= (center_x!=6'd63 && center_y!=0)?idata:0;
			4'd4:
				buffer[3] <= (center_x!=0)?idata:0;
			4'd5:
				buffer[4] <= idata;
			4'd6:
				buffer[5] <= (center_x!=6'd63)?idata:0;
			4'd7:
				buffer[6] <= (center_x!=0 && center_y != 6'd63)? idata:0;
			4'd8:
				buffer[7] <= (center_y!=6'd63)?idata:0;
			4'd9:
				buffer[8] <= (center_y!=6'd63 && center_x!= 6'd63)?idata:0;
		endcase
	else if(current_state == CONVOLUTION_K0)
		case (counter)
			4'd0:
			begin
				tmp1_mul <= buffer[0];
				tmp2_mul <= K0_0;
			end
			4'd1:
			begin
				sum <= mul;
				tmp1_mul <= buffer[1];
				tmp2_mul <= K0_1;
			end
			4'd2:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				tmp1_mul <= buffer[2];
				tmp2_mul <= K0_2;
			end
			4'd3:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[3];
				tmp2_mul <= K0_3;
			end
			4'd4:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[4];
				tmp2_mul <= K0_4;
			end
			4'd5:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[5];
				tmp2_mul <= K0_5;
			end
			4'd6:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[6];
				tmp2_mul <= K0_6;
			end
			4'd7:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[7];
				tmp2_mul <= K0_7;
			end
			4'd8:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[8];
				tmp2_mul <= K0_8;
			end
			4'd9:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
			end
			4'd10:
				sum <= add + {bias_0, 16'b0};
		endcase
	else if(current_state == CONVOLUTION_K1)
		case (counter)
			4'd0:
			begin
				tmp1_mul <= buffer[0];
				tmp2_mul <= K1_0;
			end
			4'd1:
			begin
				sum <= mul;
				tmp1_mul <= buffer[1];
				tmp2_mul <= K1_1;
			end
			4'd2:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				tmp1_mul <= buffer[2];
				tmp2_mul <= K1_2;
			end
			4'd3:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[3];
				tmp2_mul <= K1_3;
			end
			4'd4:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[4];
				tmp2_mul <= K1_4;
			end
			4'd5:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[5];
				tmp2_mul <= K1_5;
			end
			4'd6:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[6];
				tmp2_mul <= K1_6;
			end
			4'd7:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[7];
				tmp2_mul <= K1_7;
			end
			4'd8:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
				tmp1_mul <= buffer[8];
				tmp2_mul <= K1_8;
			end
			4'd9:
			begin
				tmp1_add <= sum;
				tmp2_add <= mul;
				sum <= add;
			end
			4'd10:
				sum <= add + {bias_1, 16'b0};
		endcase
end

// center_x and center_y and write_over and pooling_over and flatten_over
always @(posedge clk) begin
	if(reset)
	begin
		center_x <= 0;
		center_y <= 0;
		write_over <= 0;
		pooling_over <= 0;
		flatten_over <= 0;
	end
	else if(current_state == WR_L0_K0 || current_state == WR_L0_K1)
	begin
		if(center_x == 6'd63 && center_y == 6'd63)
		begin
			write_over <= 1;
			center_x <= 0;
			center_y <= 0;
		end
		else if(center_x == 6'd63)
		begin
			center_x <= 0;
			center_y <= center_y + 1;
		end
		else 
			center_x <= center_x + 1;
	end
	else if(current_state == WR_L1_K0 || current_state == WR_L1_K1)
	begin
		if(center_x == 6'd62 && center_y == 6'd62)
		begin
			pooling_over <= 1;
			center_x <= 0;
			center_y <= 0;
		end
		else if(center_x == 6'd62)
		begin
			center_x <= 0;
			center_y <= center_y + 2;
		end
		else 
			center_x <= center_x + 2;
	end
	else if(current_state == WR_L2)
	begin
		if(center_x == 6'd63 && center_y == 6'd63 && flatten_0or1)
			flatten_over <= 1;
		else if(center_x == 6'd63 && flatten_0or1)
		begin
			center_x <= 0;
			center_y <= center_y + 1;
		end
		else if(flatten_0or1)
			center_x <= center_x + 1;
	end
end
always @(posedge clk) begin
	if(reset)
		flatten_0or1 <= 0;
	else if(current_state == WR_L2)
		flatten_0or1 <= flatten_0or1+1;
end
endmodule




