

create_project -force Matador_BD Matador_BD -part xc7z020clg400-1
set_property  ip_repo_paths  Matador_ip [current_project]
update_ip_catalog

create_bd_design "Matador_System"

update_compile_order -fileset sources_1

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
endgroup

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

startgroup
set_property -dict [list CONFIG.c_m_axi_s2mm_data_width.VALUE_SRC USER CONFIG.c_s_axis_s2mm_tdata_width.VALUE_SRC USER] [get_bd_cells axi_dma_0]
set_property -dict [list \
  CONFIG.c_include_sg {0} \
  CONFIG.c_m_axi_mm2s_data_width {64} \
  CONFIG.c_m_axi_s2mm_data_width {64} \
  CONFIG.c_m_axis_mm2s_tdata_width {64} \
  CONFIG.c_s_axis_s2mm_tdata_width {64} \
] [get_bd_cells axi_dma_0]
endgroup

startgroup
set_property -dict [list \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {50} \
  CONFIG.PCW_USE_S_AXI_HP0 {1} \
  CONFIG.PCW_USE_S_AXI_HP2 {1} \
] [get_bd_cells processing_system7_0]
endgroup

startgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/axi_dma_0/M_AXI_MM2S} Slave {/processing_system7_0/S_AXI_HP0} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/axi_dma_0/M_AXI_S2MM} Slave {/processing_system7_0/S_AXI_HP2} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins processing_system7_0/S_AXI_HP2]
endgroup

set_property  ip_repo_paths  /home/tousif/MATADOR/dd/Matador_ip [current_project]
update_ip_catalog

startgroup
create_bd_cell -type ip -vlnv user.org:user:axis_wrapper_top:1.0 axis_wrapper_top_0
endgroup
connect_bd_intf_net [get_bd_intf_pins axis_wrapper_top_0/s00_axis] [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]
connect_bd_intf_net [get_bd_intf_pins axis_wrapper_top_0/m00_axis] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
startgroup
endgroup



startgroup
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (50 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axis_wrapper_top_0/m00_axis_aclk]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (50 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axis_wrapper_top_0/s00_axis_aclk]
endgroup

set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {axi_dma_0_M_AXIS_MM2S axis_wrapper_top_0_m00_axis}]

startgroup
apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
                                                          [get_bd_intf_nets axi_dma_0_M_AXIS_MM2S] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/processing_system7_0/FCLK_CLK0" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                          [get_bd_intf_nets axis_wrapper_top_0_m00_axis] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/processing_system7_0/FCLK_CLK0" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                         ]
endgroup

validate_bd_design
make_wrapper -files [get_files Matador_BD/Matador_BD.srcs/sources_1/bd/Matador_System/Matador_System.bd] -top

add_files -norecurse Matador_BD/Matador_BD.gen/sources_1/bd/Matador_System/hdl/Matador_System_wrapper.v
launch_runs impl_1 -to_step write_bitstream -jobs 2

puts "Generating bitstream. It will take some time!"
wait_on_run impl_1
puts "Implementation done!"
open_hw_manager



