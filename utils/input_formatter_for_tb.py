import numpy as np 
import math
# this script takes the MNIST boolean inputs and packages them into 
# packets of 32, this can then be fed through AXI stream from the 
# processor to the accelerator 

mnist_file = "mnist_test.txt"
File_data = np.loadtxt(mnist_file, dtype=int)

number_of_examples = 100
# Remove the class cols 
cols = File_data[:, -1]

f = open("expected_answers.txt", "w")
for i in range(cols.shape[0]):
	f.write(str(cols[i])+"\n")
f.close()


print(cols)
exit

File_data = File_data[:, :-1]


AXI_bus_size = 64


print("Number of features:\t\t", File_data[0].shape[0])
print(File_data[0])

print("Packets:\t\t\t", File_data[0].shape[0]/AXI_bus_size)
print("Number of Extra packet(s):\t", math.ceil(File_data[0].shape[0]/AXI_bus_size) - math.floor(File_data[0].shape[0]/AXI_bus_size))
print("-------------------------------------")
print("Number of packets required:\t", math.ceil(File_data[0].shape[0]/AXI_bus_size))
print("-------------------------------------")

# Put the data into 32 int packets 

mnist_packets = []
packet_32 = []
packet_counter = 0 

for j in range(number_of_examples):
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

	# for l in range((math.ceil(File_data[0].shape[0]/32) - math.floor(File_data[0].shape[0]/32))):
	# 	mnist_packets.append(zero_list) 

# convert the list of lists to a numpy array 
mnist_packets_np_1 = np.array(mnist_packets)
mnist_packets_np  = np.fliplr(mnist_packets_np_1)
# print(mnist_packets_np)
# now convert each 32 number array to its equivalent binary 
# this will be the 32 bit packet sent each time 

mnist_packets_np_bits =np.packbits(mnist_packets_np_1, axis=-1,  bitorder='little')
mnist_packets_np_bits.dtype = np.uint64
# print(mnist_packets_np_bits)
# now the mnist data is in 32 bit packets -- we can write the testbench 
tb_file_test = "test_data_copy_paste_100_new_examples.mem"

f = open(tb_file_test, "w")

for i in range(mnist_packets_np.shape[0]):
	if i %13 == 0:
		print("------")
	print(mnist_packets_np_bits[i])
	print(''.join(map(str, mnist_packets_np[i])), file=f)
