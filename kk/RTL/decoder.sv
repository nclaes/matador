`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/09/06 21:49:10
// Design Name: 
// Module Name: decoder
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module org_adder #(
    parameter CLAUSE_NUM,
    parameter CLASS_NUM,
    parameter WEIGHT_LENGTH
    )
    (
    input clk,
    input valid,
    input [CLAUSE_NUM - 1:0] clause[CLASS_NUM],
    output signed [WEIGHT_LENGTH - 1:0] class_sums [CLASS_NUM],
    output reg adder_done
    );
    
    wire [CLASS_NUM - 1:0] done;
    generate
    genvar i;
        for (i = 0; i < CLASS_NUM; i = i + 1) begin
            decoder #(
            .CLAUSE_NUM(CLAUSE_NUM),
            .WEIGHT_LENGTH(WEIGHT_LENGTH)
            )
            dec(
            .clk(clk),
            .valid(valid),
            .clause(clause[i]),
            .class_sum(class_sums[i]),
            .adder_done(done[i])
            );
            
        end
    endgenerate
    always@ (posedge clk) begin
        if (done == {CLASS_NUM{1'b1}}) begin
            adder_done = 1;
        end
        else if (done == {CLASS_NUM{1'b0}}) begin
            adder_done = 0;
        end
    end
endmodule

module decoder  #(
    parameter CLAUSE_NUM,
    parameter WEIGHT_LENGTH
    )
    (
    input clk,
    input valid,
    input [CLAUSE_NUM - 1:0] clause,
    output logic signed [WEIGHT_LENGTH - 1:0] class_sum,
    output reg adder_done
    );
    
    wire [CLAUSE_NUM/2 - 1:0] pos_clause;
    wire [CLAUSE_NUM/2 - 1:0] neg_clause;
    reg valid_reg;
    
    integer sum_int;
    
    initial begin
        class_sum =  {WEIGHT_LENGTH{1'b0}};
        adder_done = 0;
    end
    
    generate
    genvar i;
        for (i = 0; i < CLAUSE_NUM/2; i = i + 1) begin
            assign pos_clause[i] = clause[2 * i];
            assign neg_clause[i] = clause[2 * i + 1];
        end
    endgenerate 
    
    integer j;
    always@ (posedge clk) begin
        valid_reg <= valid;
        if (valid && !valid_reg) begin
            sum_int = 0;
            for (j = 0; j < CLAUSE_NUM/2; j = j + 1) begin
                if (pos_clause[j] && !neg_clause[j]) begin
                    sum_int = sum_int + 1;
                end
                else if (!pos_clause[j] && neg_clause[j]) begin
                    sum_int = sum_int - 1;
                end
            end
            adder_done = 1;
            class_sum = sum_int;
        end
        else if (!valid) begin
            adder_done = 0;
        end
    end
    
endmodule
