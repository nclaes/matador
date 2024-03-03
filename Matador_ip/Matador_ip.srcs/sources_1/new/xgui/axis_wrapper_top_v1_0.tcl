# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "CLASS_NUM" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CLAUSE_NUM" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_M00_AXIS_TDATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_S00_AXIS_TDATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FEATURE_NUM" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PACKETS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PACKETS_NUM" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STAGE_NUM" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WEIGHT_LENGTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WIDTH" -parent ${Page_0}


}

proc update_PARAM_VALUE.CLASS_NUM { PARAM_VALUE.CLASS_NUM } {
	# Procedure called to update CLASS_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLASS_NUM { PARAM_VALUE.CLASS_NUM } {
	# Procedure called to validate CLASS_NUM
	return true
}

proc update_PARAM_VALUE.CLAUSE_NUM { PARAM_VALUE.CLAUSE_NUM } {
	# Procedure called to update CLAUSE_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLAUSE_NUM { PARAM_VALUE.CLAUSE_NUM } {
	# Procedure called to validate CLAUSE_NUM
	return true
}

proc update_PARAM_VALUE.C_M00_AXIS_TDATA_WIDTH { PARAM_VALUE.C_M00_AXIS_TDATA_WIDTH } {
	# Procedure called to update C_M00_AXIS_TDATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_M00_AXIS_TDATA_WIDTH { PARAM_VALUE.C_M00_AXIS_TDATA_WIDTH } {
	# Procedure called to validate C_M00_AXIS_TDATA_WIDTH
	return true
}

proc update_PARAM_VALUE.C_S00_AXIS_TDATA_WIDTH { PARAM_VALUE.C_S00_AXIS_TDATA_WIDTH } {
	# Procedure called to update C_S00_AXIS_TDATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_S00_AXIS_TDATA_WIDTH { PARAM_VALUE.C_S00_AXIS_TDATA_WIDTH } {
	# Procedure called to validate C_S00_AXIS_TDATA_WIDTH
	return true
}

proc update_PARAM_VALUE.DEPTH { PARAM_VALUE.DEPTH } {
	# Procedure called to update DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DEPTH { PARAM_VALUE.DEPTH } {
	# Procedure called to validate DEPTH
	return true
}

proc update_PARAM_VALUE.FEATURE_NUM { PARAM_VALUE.FEATURE_NUM } {
	# Procedure called to update FEATURE_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FEATURE_NUM { PARAM_VALUE.FEATURE_NUM } {
	# Procedure called to validate FEATURE_NUM
	return true
}

proc update_PARAM_VALUE.PACKETS { PARAM_VALUE.PACKETS } {
	# Procedure called to update PACKETS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PACKETS { PARAM_VALUE.PACKETS } {
	# Procedure called to validate PACKETS
	return true
}

proc update_PARAM_VALUE.PACKETS_NUM { PARAM_VALUE.PACKETS_NUM } {
	# Procedure called to update PACKETS_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PACKETS_NUM { PARAM_VALUE.PACKETS_NUM } {
	# Procedure called to validate PACKETS_NUM
	return true
}

proc update_PARAM_VALUE.STAGE_NUM { PARAM_VALUE.STAGE_NUM } {
	# Procedure called to update STAGE_NUM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STAGE_NUM { PARAM_VALUE.STAGE_NUM } {
	# Procedure called to validate STAGE_NUM
	return true
}

proc update_PARAM_VALUE.WEIGHT_LENGTH { PARAM_VALUE.WEIGHT_LENGTH } {
	# Procedure called to update WEIGHT_LENGTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WEIGHT_LENGTH { PARAM_VALUE.WEIGHT_LENGTH } {
	# Procedure called to validate WEIGHT_LENGTH
	return true
}

proc update_PARAM_VALUE.WIDTH { PARAM_VALUE.WIDTH } {
	# Procedure called to update WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WIDTH { PARAM_VALUE.WIDTH } {
	# Procedure called to validate WIDTH
	return true
}


proc update_MODELPARAM_VALUE.DEPTH { MODELPARAM_VALUE.DEPTH PARAM_VALUE.DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DEPTH}] ${MODELPARAM_VALUE.DEPTH}
}

proc update_MODELPARAM_VALUE.WIDTH { MODELPARAM_VALUE.WIDTH PARAM_VALUE.WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WIDTH}] ${MODELPARAM_VALUE.WIDTH}
}

proc update_MODELPARAM_VALUE.PACKETS { MODELPARAM_VALUE.PACKETS PARAM_VALUE.PACKETS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PACKETS}] ${MODELPARAM_VALUE.PACKETS}
}

proc update_MODELPARAM_VALUE.C_S00_AXIS_TDATA_WIDTH { MODELPARAM_VALUE.C_S00_AXIS_TDATA_WIDTH PARAM_VALUE.C_S00_AXIS_TDATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_S00_AXIS_TDATA_WIDTH}] ${MODELPARAM_VALUE.C_S00_AXIS_TDATA_WIDTH}
}

proc update_MODELPARAM_VALUE.C_M00_AXIS_TDATA_WIDTH { MODELPARAM_VALUE.C_M00_AXIS_TDATA_WIDTH PARAM_VALUE.C_M00_AXIS_TDATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_M00_AXIS_TDATA_WIDTH}] ${MODELPARAM_VALUE.C_M00_AXIS_TDATA_WIDTH}
}

proc update_MODELPARAM_VALUE.STAGE_NUM { MODELPARAM_VALUE.STAGE_NUM PARAM_VALUE.STAGE_NUM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STAGE_NUM}] ${MODELPARAM_VALUE.STAGE_NUM}
}

proc update_MODELPARAM_VALUE.CLAUSE_NUM { MODELPARAM_VALUE.CLAUSE_NUM PARAM_VALUE.CLAUSE_NUM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLAUSE_NUM}] ${MODELPARAM_VALUE.CLAUSE_NUM}
}

proc update_MODELPARAM_VALUE.CLASS_NUM { MODELPARAM_VALUE.CLASS_NUM PARAM_VALUE.CLASS_NUM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLASS_NUM}] ${MODELPARAM_VALUE.CLASS_NUM}
}

proc update_MODELPARAM_VALUE.WEIGHT_LENGTH { MODELPARAM_VALUE.WEIGHT_LENGTH PARAM_VALUE.WEIGHT_LENGTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WEIGHT_LENGTH}] ${MODELPARAM_VALUE.WEIGHT_LENGTH}
}

proc update_MODELPARAM_VALUE.FEATURE_NUM { MODELPARAM_VALUE.FEATURE_NUM PARAM_VALUE.FEATURE_NUM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FEATURE_NUM}] ${MODELPARAM_VALUE.FEATURE_NUM}
}

proc update_MODELPARAM_VALUE.PACKETS_NUM { MODELPARAM_VALUE.PACKETS_NUM PARAM_VALUE.PACKETS_NUM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PACKETS_NUM}] ${MODELPARAM_VALUE.PACKETS_NUM}
}

