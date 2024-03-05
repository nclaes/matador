`timescale 1ns / 1ps

module axi_stream_tb();
    
    parameter C_S00_AXIS_DATA_WIDTH = 64; 
    parameter C_M00_AXIS_DATA_WIDTH = 64; 
    parameter NUM_PACKETS = 13;
    parameter DATAPOINTS = 10;
    
    reg clock;
    reg areset;
    
    reg s00_axis_tready;
    reg m00_axis_tready;
    reg s00_last;
    reg m00_last; 
    reg m00_valid; 
    reg s00_valid; 
    reg [C_S00_AXIS_DATA_WIDTH-1:0] payload; 
    reg [C_M00_AXIS_DATA_WIDTH-1:0] recieved;
    reg [63:0] input_data [0:253]; 
    
    wire [7:0] bytes; 
    int input_counter = 0; 
    
    axis_wrapper_top  
        DUT(
            .s00_axis_aclk(clock),
            .s00_axis_aresetn(areset),
            .s00_axis_tready(s00_axis_tready),                  
            .s00_axis_tdata(payload),
            .s00_axis_tstrb(bytes),
            .s00_axis_tlast(s00_last),
            .s00_axis_tvalid(s00_valid),
            .m00_axis_aclk(clock),
            .m00_axis_aresetn(areset),
            .m00_axis_tready(m00_axis_tready),
            .m00_axis_tdata(recieved),                                                     
            .m00_axis_tlast(m00_last),                          
            .m00_axis_tvalid(m00_valid)                        
        );
 
    initial begin                                              
        #0 clock = 0;
        forever begin
            #5 clock = ~clock;
        end 
    end
    
    initial begin     
        $readmemb("test_data_copy_paste_10_examples.mem", input_data);
        #5 s00_last        <= 1'b0;
        #5 areset          <= 1'b0;
        #5 areset          <= 1'b1;
        #5 s00_valid       <= 1'b0;
        m00_axis_tready     <= 1'b1;
        #5 payload         <= input_data[0];
        s00_valid           <= 1'b1;
    end
    
    always @(posedge clock) begin  
        if(s00_axis_tready & s00_valid) begin
            if(input_counter == DATAPOINTS*NUM_PACKETS-2) begin 
               s00_last     <= 1'b1;  
            end
            if(s00_last == 1'b1) begin 
                s00_last    <= 1'b0;
                s00_valid   <= 1'b0;
            end
            input_counter   = input_counter + 1;
            payload         = input_data[input_counter];
        end 
    end

endmodule
