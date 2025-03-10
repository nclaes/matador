import numpy as  np 
import argparse
import os
import json 
from stages_print import *

def write_bd_tcl(rtl_config, run_dir, abs_run_dir):
    # create the bd file 
    bw = rtl_config["BusWidth"]
    print(synth_impl, "Selected BusWidth:", rtl_config["BusWidth"])

    system_file = run_dir + "/Matador_System_Create.tcl"
    with open(system_file, "w") as f:
        print("create_project -force Matador_BD Matador_BD -part xc7z020clg400-1", file=f)
        print("set_property  ip_repo_paths  %s [current_project]" %(run_dir + "/Matador_ip"), file=f)
        print("set_property board_part www.digilentinc.com:pynq-z1:part0:1.0 [current_project]", file=f)
        print("update_ip_catalog", file=f)
        print("""
create_bd_design "Matador_System"

update_compile_order -fileset sources_1

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup
set_property CONFIG.PCW_UART0_PERIPHERAL_ENABLE {1} [get_bd_cells processing_system7_0]
set_property -dict [list \
  CONFIG.PCW_UIPARAM_DDR_BUS_WIDTH {16 Bit} \
  CONFIG.PCW_UIPARAM_DDR_FREQ_MHZ {525} \
] [get_bd_cells processing_system7_0]              
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
endgroup

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
              
startgroup
set_property -dict [list CONFIG.c_m_axi_s2mm_data_width.VALUE_SRC USER CONFIG.c_s_axis_s2mm_tdata_width.VALUE_SRC USER] [get_bd_cells axi_dma_0]
set_property -dict [list \\
    CONFIG.c_include_sg {0} \\""", file=f)
        print("\tCONFIG.c_m_axi_mm2s_data_width {%s} %s" %(bw, "\\"), file=f)
        print("\tCONFIG.c_m_axi_s2mm_data_width {%s} \\" %(bw), file=f)
        print("\tCONFIG.c_m_axis_mm2s_tdata_width {%s}\\" %(bw), file=f)
        print("\tCONFIG.c_m_axis_mm2s_tdata_width {%s}\\" %(bw), file=f)
        print("\tCONFIG.c_s_axis_s2mm_tdata_width {%s}\\" %(bw), file=f)
        print("] [get_bd_cells axi_dma_0]", file=f)
        print("endgroup", file=f)
        print("""
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
endgroup""", file=f)
        print("set_property  ip_repo_paths {%s} [current_project]" %(abs_run_dir + "/Matador_ip"), file=f)
        print("""
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

    """, file=f)
        print("# write_hw_platform -fixed -include_bit -force -file %s/Matador_System_wrapper.xsa" %(abs_run_dir), file=f)

# write_hw_platform -fixed -include_bit -force -file /home/tousif/Desktop/Matador_Docker/Sport/Matador_BD/Matador_System_wrapper.xsa


# read in the directory
def main():

    os.system("apt-get install libtinfo5")  # A cheap fix - need to create a proper solution later

    config_file = open("accel_config.json")
    config = json.load(config_file)

    run_dir = config["Output_Directory"]
    vivado_path= config["Vivado"] 

    # source_cmd = vivado_path.replace("/bin/vivado", "")
    # source_cmd = "source " + source_cmd + "/settings64.sh"
    # os.system(source_cmd)
    # print(synth_impl, "Sourcing: ", source_cmd)

    abs_run_dir = os.path.abspath(run_dir)

    print(synth_impl, "Writing the System Configuration File")
    isExist = os.path.exists(run_dir +"/gen_RTL.json")
    if isExist:
        print(synth_impl, "Found RTL config - reading in params...")
        rtl_config_file = open(run_dir +"/gen_RTL.json")
        rtl_config = json.load(rtl_config_file)

        write_bd_tcl(rtl_config, run_dir, abs_run_dir)

    else: 
        print(synth_impl, "The RTL configuration json was not found")
        print(synth_impl, "Generate RTL files and check json exists")
        exit()

    # TO DO remove the accel_config.json 

    abs_path_matador_ip_tcl = os.path.abspath("Matador_ip_create.tcl")
    abs_path_matador_bd_tcl = os.path.abspath(run_dir + "/Matador_System_Create.tcl")

    print("\t\t\t******** PACKAGING IP ******** ")  
    os.chdir(abs_run_dir)
    os.system(vivado_path + " -mode batch  -source "+ abs_path_matador_ip_tcl)
    # os.system(vivado_path + " -mode batch -source "+ abs_path_matador_ip_tcl)

    print(" ")
    print("\t\t\t******** GENERATING BITSTREAM ******** ")  
    os.system(vivado_path + " -source "+ abs_path_matador_bd_tcl)


if __name__ == '__main__':
    main()
