import numpy as  np 
import argparse
import math
import os
import json 
from stages_print import *
from datetime import datetime
import subprocess

def tcl_create(config, output_dir, stream_data, datapoints, output_dir_old):
   
    sources = output_dir+ "/src"
    # if the src exists then re-write it 
    isExist = os.path.exists(sources)
    if not isExist:
        os.makedirs(sources)
    # move all the templates to this directoy
    copy = "cp " + "utils/deploy_templates/lscript.ld " + sources  
    os.popen(copy) 
    copy = "cp " + "utils/deploy_templates/Matador.c " + sources  
    os.popen(copy)
    copy = "cp " + "utils/deploy_templates/platform.c " + sources  
    os.popen(copy)
    copy = "cp " + "utils/deploy_templates/platform.h " + sources  
    os.popen(copy)
    copy = "cp " + "utils/deploy_templates/platform_config.h " + sources  
    os.popen(copy)
    copy = "cp " + "utils/deploy_templates/Xilinx.spec " + sources  
    os.popen(copy)
    
    header_file = sources + "/instructions_and_data.h"

    hfile = open(header_file, "w")
    hfile.write("""

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xaxidma.h"
#include "xparameters.h"
#include "xdebug.h"
#include "sleep.h"
#include "xtime_l.h"

#if defined(XPAR_UARTNS550_0_BASEADDR)
#include "xuartns550_l.h"       /* to use uartns550 */
#endif

char space[5] = "\t";

/******************** Constant Definitions **********************************/

/*
 * Device hardware build related constants.
 */

#ifndef SDT
#define DMA_DEV_ID		XPAR_AXIDMA_0_DEVICE_ID

#ifdef XPAR_AXI_7SDDR_0_S_AXI_BASEADDR
#define DDR_BASE_ADDR		XPAR_AXI_7SDDR_0_S_AXI_BASEADDR
#elif defined (XPAR_MIG7SERIES_0_BASEADDR)
#define DDR_BASE_ADDR	XPAR_MIG7SERIES_0_BASEADDR
#elif defined (XPAR_MIG_0_C0_DDR4_MEMORY_MAP_BASEADDR)
#define DDR_BASE_ADDR	XPAR_MIG_0_C0_DDR4_MEMORY_MAP_BASEADDR
#elif defined (XPAR_PSU_DDR_0_S_AXI_BASEADDR)
#define DDR_BASE_ADDR	XPAR_PSU_DDR_0_S_AXI_BASEADDR
#endif

#else

#ifdef XPAR_MEM0_BASEADDRESS
#define DDR_BASE_ADDR		XPAR_MEM0_BASEADDRESS
#endif
#endif

#define NUMBER_OF_TRANSFERS	1
#define POLL_TIMEOUT_COUNTER    90000000U

#ifndef DDR_BASE_ADDR
#warning CHECK FOR THE VALID DDR ADDRESS IN XPARAMETERS.H, \
DEFAULT SET TO 0x01000000

#define MEM_BASE_ADDR		0x01000000
#else
#define MEM_BASE_ADDR		(DDR_BASE_ADDR + 0x1000000)
#endif\n""")
	
    # assign memory offset to the outputs from the training data

    base_addr = int(0x01000000)
    number_of_test_data_packets = len(stream_data)

    print(bare_metal, "Number of Testing Data Packets\t\t:", number_of_test_data_packets)

    rx_base_addr 		= base_addr + 640
    rx_base_addr_test 	= rx_base_addr + 640 +  (number_of_test_data_packets)*64

    print(bare_metal, "Memory offset from data Base Address for Testing Data\t:", hex(rx_base_addr_test))
    print(bare_metal, "This will be the start of the Data RX_Buffer for Testing Data")

    hfile.write("#define RX_TESTING_DATA_BUFFER_BASE		(MEM_BASE_ADDR + " + str(hex(rx_base_addr_test)) + ")\n")

    hfile.write("#define NUMBER_OF_TESTING_DATA_PKTS "+ str(number_of_test_data_packets)+ "\n")
    hfile.write("#define NUMBER_Of_TESTING_DATAPOINTS " + str(datapoints) + "\n")

    hfile.write("#define MAX_TESTING_PKT_LEN		((NUMBER_OF_TESTING_DATA_PKTS*64)/8)\n")
    hfile.write("#define MAX_TESTING_DATAPOINT_LEN ((NUMBER_Of_TESTING_DATAPOINTS*64)/8)\n")

    # write the testing data stream array
    hfile.write("// Inference Data Stream\n")
    hfile.write("//Number of data packets: " + str(number_of_test_data_packets)+ "\n")
    hfile.write("unsigned long long Inference_BufferPtr[] ={")
    for i in range(number_of_test_data_packets):
        hfile.write(str(stream_data[i][0])+"llU")
        if i < (number_of_test_data_packets-1):
            hfile.write(", ")
    hfile.write("};\n")

    tcl_file = output_dir + "platform.tcl"
    tcl= open(tcl_file, "w")

    tcl.write("""

# Usage: To re-create this platform project launch xsct with below options.
# xsct /home/tousif/Desktop/mignon_ai/Bare_Metal_Fifo_Test/platform.tcl
# 
# OR launch xsct and run below command.
# source /home/tousif/Desktop/mignon_ai/Bare_Metal_Fifo_Test/platform.tcl
# 
# To create the platform in a different location, modify the -out option of "platform create" command.
# -out option specifies the output directory of the platform project.

set tcl_ "Starting Bare Metal Automation"

puts $tcl_
puts "Running Auto-generated TCL script"
puts "For errors see workspace /.metadata/.log"
""")

    project_name = output_dir.replace("-", "")
    
    isExist = os.path.exists(project_name+ "_app")
    if not isExist:
        os.makedirs(project_name+ "_app")
    project_name = output_dir.split("/")[-2]

    tcl.write("set xsa \"" + str(output_dir_old + "/m_wrapper/matador_wrapper.xsa")+ "\"\n")
    tcl.write("set processor  \"" + str("ps7_cortexa9_0") + "\"\n")
    tcl.write("set workspace  \"" + str(output_dir)+ "\"\n")
    tcl.write("set x \"" + str(project_name)+ "\"\n")#
    tcl.write("set x_platform \"" + str(project_name)+ "_platform\"\n")
    tcl.write("set elf_ \"/Debug/" + str(project_name)+ ".elf\"\n")
    tcl.write("set pth \"" + sources + "\"\n")
    tcl.write("set dir_ \"" + output_dir + "\"\n")
    tcl.write("set bitstream \"/_ide/bitstream/" + "matador_wrapper.bit" + "\"\n")  
    tcl.write("""

set os "standalone"   
set ps7_init "/_ide/psinit/ps7_init.tcl"

puts "Setting Workspace as: $workspace"
setws $workspace
platform create -name $x_platform  -hw $xsa -proc {ps7_cortexa9_0} -os {standalone} -out $workspace

platform write
platform generate -domains 
platform active $x_platform
domain active {standalone_domain}
bsp reload
platform generate
platform active $x_platform
platform generate


# Build application project 

platform active $x_platform

app create -name $x -platform $x_platform -os {standalone} -lang C -template {Empty Application}
importsources -name $x -path $pth
# app config -name $x build-config debug
app build -name $x

connect -url tcp:127.0.0.1:3121
targets -set -nocase -filter {name =~"ARM Cortex-A9 MPCore #0"} 
rst -system
after 3000
fpga -file $dir_/$x$bitstream
targets -set -nocase -filter {name =~"ARM Cortex-A9 MPCore #0"} 
loadhw -hw $xsa
configparams force-mem-access 1
targets -set -nocase -filter {name =~"ARM Cortex-A9 MPCore #0"} 
source $dir_/$x$ps7_init
ps7_init
ps7_post_config
configparams force-mem-access 0
dow $dir_/$x$elf_
con
disconnect
exit
	""")	
    tcl_cmd = "/tools/Xilinx/Vitis/2023.2/bin/xsct" + " " + tcl_file
    subprocess.Popen(tcl_cmd, shell=True)



def convert_data_to_numpy(AXI_bus_size, train_file, classes, output_dir, stream_file_name):

  File_data = np.loadtxt(train_file, dtype=int)
  # Remove the class cols 
  classes_ =   File_data[:, -1]
  File_data = File_data[:, :-1]

  number_of_data_points = File_data.shape[0]

  print(bare_metal,"Number of examples in the file: ", File_data.shape[0])
  print(bare_metal, "Number of features:\t\t", File_data[0].shape[0])
  print(bare_metal, "Packets:\t\t\t", File_data[0].shape[0]/AXI_bus_size)

  num_extra_packets = math.ceil(File_data[0].shape[0]/AXI_bus_size) - math.floor(File_data[0].shape[0]/AXI_bus_size)

  print(bare_metal, "Number of Extra packet(s):\t", num_extra_packets)
  print(bare_metal, "-------------------------------------")
  packets_required = math.ceil(File_data[0].shape[0]/AXI_bus_size)
  #   if(num_extra_packets == 0):
  # now we are always adding one extra packet    

  print(bare_metal, "Number of packets required:\t", packets_required )
  print(bare_metal, "-------------------------------------")
  # Put the data into 32 int packets 
  mnist_packets = []
  packet_32 = []
  packet_counter = 0 

  for j in range(File_data.shape[0]):
      for i in range(File_data[j].shape[0]):
          if packet_counter <= AXI_bus_size-1: 
              packet_32.append(File_data[j][i])
          else: 
              mnist_packets.append(packet_32)
              packet_32 = []
              # print(i)
              packet_counter = 0
              packet_32.append(File_data[j][i])

          packet_counter += 1

      # If datapoint is complete and packet is not fully filled...
      # Fill the remainder with zeros 
      if(len(packet_32) != 0):
          # print("Half filled packet: ", len(packet_32))
          remainder_zero_fill = AXI_bus_size -len(packet_32)   
          for l in range(remainder_zero_fill):
              packet_32.append(0)
          mnist_packets.append(packet_32)

          packet_counter = 0
          packet_32 = []
      # we need to add one additional packet 
      #if(num_extra_packets == 0):
    #   add_packet = []
    #   for i in range(AXI_bus_size):
    #     add_packet.append(0)
    #   mnist_packets.append(add_packet)
        #   packets_required += 1
        #   print("adjusted pakcets: ", packets_required)

  # convert the list of lists to a numpy array 
#   print(len(mnist_packets))
  mnist_packets_np_1 = np.array(mnist_packets)
  mnist_packets_np_bits =np.packbits(mnist_packets_np_1, axis=-1,  bitorder='little')
  if(AXI_bus_size > 64):
    print("	[RTL_gen]	Bus size is greater than maximum!")
  else:	
    mnist_packets_np_bits.dtype = np.uint64

  print(bare_metal, "Total Number of packets: ", mnist_packets_np_bits.shape[0])
  stream_file = output_dir + stream_file_name
  np.savetxt(stream_file, mnist_packets_np_bits)

  return mnist_packets_np_bits, number_of_data_points


print(bare_metal, "Starting Deployment Flow")
os.system("gnome-terminal -e 'minicom -c on -s'")

# look for the test data - found in RTL config 
# look for the xsa 
parser = argparse.ArgumentParser(description="Matador Deployment")

# to train the output directory for the results must be specified. 
parser.add_argument("-output_dir", help="results directory to store the training data", type=str)
args = parser.parse_args()

output_dir = args.output_dir
print(bare_metal, "Output Directory: ", output_dir)

output_dir_old = output_dir

config_file = str(output_dir) + "/gen_RTL.json"
# check if the RTL json exists 
isExist = os.path.exists(config_file)

if isExist: 

    current_dateTime = str(datetime.now())
    date_time = current_dateTime.replace(" ", "_")
    date_time = date_time.replace(".", "_")
    date_time = date_time.replace(":", "_")
    output_name = date_time
    
    output_dir = output_dir + "_" + output_name + '/'

    print(bare_metal, "Reading in the config file")
    config_file = open(config_file)
    config = json.load(config_file)
    features = int(config["Features"])
    classes  = int(config["Classes"])
    test_data = config["Test_Data"]
    bus_width = int(config["BusWidth"])

    stream_file_name = "test_data_stream_file"

    if(not os.path.exists(output_dir)):
      os.makedirs(output_dir)
    
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    # convert data 
    stream_data, datapoints = convert_data_to_numpy(bus_width, test_data, classes, output_dir, stream_file_name)
    tcl_create(config, output_dir, stream_data, datapoints, output_dir_old)


else: 
    print(bare_metal, "[Error] - RTL config json not found")