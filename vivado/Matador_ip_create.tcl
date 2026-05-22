create_project -f Matador_ip Matador_ip -part  xc7z020clg400-1
file mkdir Matador_ip/Matador_ip.srcs/sources_1/new

# add the generated RTL to the file 
file copy -force RTL/TM_top.sv Matador_ip/Matador_ip.srcs/sources_1/new/TM_top.sv
file copy -force RTL/TM_Hard_Coded_Clause_Blocks.sv Matador_ip/Matador_ip.srcs/sources_1/new/TM_Hard_Coded_Clause_Blocks.sv
file copy -force RTL/axis_wrapper.sv Matador_ip/Matador_ip.srcs/sources_1/new/axis_wrapper.sv
file copy -force RTL/decoder.sv Matador_ip/Matador_ip.srcs/sources_1/new/decoder.sv
file copy -force RTL/TM_argmax.sv Matador_ip/Matador_ip.srcs/sources_1/new/TM_argmax.sv
file copy -force RTL/AXI_Interface.sv Matador_ip/Matador_ip.srcs/sources_1/new/AXI_Interface.sv
file copy -force RTL/HCB_top.sv Matador_ip/Matador_ip.srcs/sources_1/new/HCB_top.sv
# file copy -force RTL/hard_coded_weight.sv Matador_ip/Matador_ip.srcs/sources_1/new/hard_coded_weight.sv

# read in each of the files 
read_verilog Matador_ip/Matador_ip.srcs/sources_1/new/TM_top.sv
read_verilog Matador_ip/Matador_ip.srcs/sources_1/new/TM_Hard_Coded_Clause_Blocks.sv
read_verilog Matador_ip/Matador_ip.srcs/sources_1/new/axis_wrapper.sv
read_verilog Matador_ip/Matador_ip.srcs/sources_1/new/decoder.sv
read_verilog Matador_ip/Matador_ip.srcs/sources_1/new/TM_argmax.sv
read_verilog Matador_ip/Matador_ip.srcs/sources_1/new/AXI_Interface.sv
read_verilog Matador_ip/Matador_ip.srcs/sources_1/new/HCB_top.sv
# read_verilog Matador_ip/Matador_ip.srcs/sources_1/new/hard_coded_weight.sv

# find the hierarchy
update_compile_order -fileset sources_1
ipx::package_project -root_dir Matador_ip/Matador_ip.srcs/sources_1/new -vendor user.org -library user -taxonomy /UserIP
ipx::open_ipxact_file Matador_ip/Matador_ip.srcs/sources_1/new/component.xml
ipx::merge_project_changes hdl_parameters [ipx::current_core]
ipx::merge_project_changes ports [ipx::current_core]
set_property core_revision 1 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
update_ip_catalog 




