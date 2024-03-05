`timescale 1ns / 1ps

module axi_stream_tb();
    
    parameter C_S00_AXIS_DATA_WIDTH = 64; 
    parameter C_M00_AXIS_DATA_WIDTH = 64; 
    parameter NUM_PACKETS = 13;
    parameter DATAPOINTS = 100;
    
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
    reg [63:0] input_data [0:DATAPOINTS * NUM_PACKETS - 1]; 
    
    wire [7:0] bytes; 
    int input_counter = 0; 
    
    reg last_triggered; 
    
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
        last_triggered = 1'b0;                                          
        #0 clock = 0;
        forever begin
            #20 clock = ~clock;
        end 
    end
    
    initial begin     
        $readmemb("test_data_copy_paste_100_new_examples.mem", input_data);
        areset <= 1'b1;
        #20 s00_last        <= 1'b0;
        #20 areset          <= 1'b0;
        #20 areset          <= 1'b1;
        #20 s00_valid       <= 1'b0;
        m00_axis_tready     <= 1'b1;
        #20 payload         <= input_data[0];
        s00_valid           <= 1'b1;
    end
    integer cnt = 0;
    always @(posedge clock) begin  
        if(s00_axis_tready && s00_valid) begin
            if(input_counter == DATAPOINTS*NUM_PACKETS-2) begin 
               s00_last     <= 1'b1;  
            end
            else if (input_counter == 160) begin
                s00_valid = 0;
            end
            
            if(s00_last == 1'b1) begin 
                s00_last    <= 1'b0;
                s00_valid   <= 1'b0;
            end
            input_counter   = input_counter + 1;
            payload         = input_data[input_counter];
        end 
        else if (!s00_valid) begin
            if (cnt < 12) begin
                cnt = cnt + 1;
            end
            else if (cnt ==12) begin
                s00_valid = 1;
                cnt = 0;
            end
        end
        
        if(m00_last) begin 
            last_triggered = 1'b1;
        end 
        else if(last_triggered) begin 
            input_counter = 0; 
            s00_last    <= 1'b0;
            s00_valid   <= 1'b1;
            last_triggered = 1'b0;
            payload         <= input_data[0];
        end
    end

endmodule