import numpy as  np 
import argparse
import os
import json 


# read in the directory
def main():


    config_file = open("accel_config.json")
    config = json.load(config_file)

    run_dir = config["Output_Directory"]
    vivado_path= config["Vivado"] 
    abs_run_dir = os.path.abspath(run_dir)

    TM_info_file = abs_run_dir + "/training_config.json"
    

    abs_path_matador_ip_tcl = os.path.abspath("Matador_ip_create.tcl")
    abs_path_matador_bd_tcl = os.path.abspath("Matador_bd_create.tcl")

    print("\t\t\t******** PACKAGING IP ******** ")  
    os.chdir(abs_run_dir)
    os.system(vivado_path + " -mode batch -source "+ abs_path_matador_ip_tcl)

    print(" ")
    print("\t\t\t******** GENERATING BITSTREAM ******** ")  
    os.system(vivado_path + " -mode batch -source "+ abs_path_matador_bd_tcl)


if __name__ == '__main__':
    main()
