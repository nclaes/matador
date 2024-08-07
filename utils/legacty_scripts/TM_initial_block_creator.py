import numpy as np 
from math import log
from math import ceil
import json

# NOTE - this needs to be made available to the user later

# number_of_states 	= 400
# clauses 			= 500
# classes 			= 10 
# features 			= 784

config_file = open("gen_RTL.json")
config = json.load(config_file)

output_dir = config["Output_Directory"]
TA_file = config["TAs"]
Weights_file = config["Weights"]

number_of_states = int(config["No_States"])
classes = int(config["Classes"])
features = int(config["Features"])
clauses  = int(config["Clauses"])

# "/home/tousif/MATADOR/TAs_500_8_800.txt"
# Weights_file  = "/home/tousif/LASAGNE/New_TM/Working_CoTM/weights.txt"

TA_data_raw = np.loadtxt(TA_file, dtype=int)

include_counter = 0 

for i in range(TA_data_raw.shape[0]): 
	if TA_data_raw[i] > 128:
		TA_data_raw[i] = 1
		include_counter += 1
	else: 
		TA_data_raw[i] = 0

print("\tTA actions: ", TA_data_raw)
print("\tNumber of includes: ", include_counter)

# Re-arrange to 3d array 
TA_data = np.reshape(TA_data_raw, (clauses,features*2))
print("\tTA matrix shape: ", TA_data.shape)
# Transpose the data to give Features by Clauses 
TA_data = TA_data.transpose()
# print(TA_data.shape)

file = output_dir+"/Hard_Coded_TAs.txt"

print("\tThe data has been written to: ", file)
np.savetxt(file, TA_data, fmt="%d", delimiter=" ")

# now load the weights to find the largest weight values#
# Weights = []
# W_file = open(Weights_file, "r")
 
# data = W_file.read()
# data_into_list = data.split("\n")
# Weights = [int(i) for i in data_into_list]

# Weights = np.array(Weights)
# Weights = np.reshape(Weights, (classes,clauses))
# print(Weights.shape)

# max_positive = 0 
# max_negative = 0 

# max_positive_current = 0 
# max_negative_current = 0  

# for i in range(Weights.shape[0]):
# 	# classes
# 	for j in range(Weights.shape[1]):
# 		#clauses
# 		if Weights[i][j] > 0: 
# 			max_positive_current += Weights[i][j]
# 		else:
# 			max_negative_current += Weights[i][j]

# 	# print("Class: ", i, " Maxpos: ", max_positive_current)
# 	# print("Class: ", i, " Maxneg: ", max_negative_current)

# 	if max_positive_current > max_positive: 
# 		max_positive = max_positive_current

# 	if max_negative_current < max_negative: 
# 		max_negative = max_negative_current

# 	max_negative_current = 0 
# 	max_positive_current = 0 

# print("Max Positive Weight: " , max_positive)
# print("Max Negative Weight: ", max_negative)

# max_pos = abs(max_positive)
# max_neg = abs(max_negative) 

# bits 	= 0 

# if max_pos > max_neg: 
# 	abs_w = max_pos
# 	bits = ceil(log(abs_w, 2))
# else: 
# 	abs_w = max_neg 
# 	bits = ceil(log(abs_w, 2)) + 1

# print("bits required: ",  bits)