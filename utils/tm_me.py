# Coalesced Tsetlin Machine derived from first principles (https://arxiv.org/pdf/2108.07594.pdf) 
# Tousif Rahman and Gang Mao Version 2023 

import numpy as np
import ctypes as C
import itertools
import os
import csv
import random
#from print_banner import *
import time
import sys
import argparse
import json 

toolbar_width = 40

this_dir, this_filename = os.path.split(__file__)
_lib = np.ctypeslib.load_library('libTM', os.path.join(this_dir))  

class CTsetlinMachine(C.Structure):
    None

tm_pointer = C.POINTER(CTsetlinMachine)

array_1d_uint = np.ctypeslib.ndpointer(
    dtype=np.uint32,
    ndim=1,
    flags='CONTIGUOUS')

_lib.CreateCoalescedTsetlinMachine.restype  = tm_pointer  
_lib.CreateCoalescedTsetlinMachine.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int ,C.c_double, C.c_int] 

_lib.tm_initialize.restype  = None                      
_lib.tm_initialize.argtypes = [tm_pointer] 

_lib.tm_ta_state.restype    = C.c_int                    
_lib.tm_ta_state.argtypes   = [tm_pointer, C.c_int, C.c_int, C.c_int]

_lib.tm_get_clause_weight.restype   = C.c_int
_lib.tm_get_clause_weight.argtypes  = [tm_pointer, C.c_int, C.c_int]

_lib.tm_update.restype  = None                    
_lib.tm_update.argtypes = [tm_pointer, array_1d_uint, C.c_int ,C.c_int, C.c_float]

_lib.get_rands_used.restype  = C.c_int 
_lib.get_rands_used.argtypes = [tm_pointer]

_lib.rands_reset.restype    = C.c_int 
_lib.rands_reset.argtypes   = [tm_pointer]

_lib.tm_load_TAs_load_Weights.restype   = None
_lib.tm_load_TAs_load_Weights.argtypes  = [tm_pointer, C.c_int, C.c_int, array_1d_uint, array_1d_uint]

_lib.tm_analyse_inference.restype   = None
_lib.tm_analyse_inference.argtypes  = [tm_pointer, array_1d_uint]

_lib.tm_score.restype   = C.c_int                    
_lib.tm_score.argtypes  = [tm_pointer, array_1d_uint, C.c_int]

def read_data(train_file, test_file):

    X_train_raw = np.loadtxt(train_file, dtype=int)
    X_test_raw = np.loadtxt(test_file, dtype=int)
    
    X_train = X_train_raw[:,0:-1]
    y_train = X_train_raw[:,-1]

    X_test = X_test_raw[:,0:-1]
    y_test = X_test_raw[:,-1]

    return X_train, y_train, X_test, y_test

class CoalescedTsetlinMachine():
    def __init__(self, number_of_clauses, number_of_features, number_of_classes,
        T, s ,boost_true_positive_feedback=1, number_of_states=8):
        
        self.number_of_classes  = number_of_classes 
        self.number_of_clauses  = number_of_clauses  
        self.number_of_states   = number_of_states
        self.number_of_features = number_of_features
        self.T                  = int(T)
        self.s                  = s
        self.boost_true_positive_feedback = boost_true_positive_feedback
        self.tm = _lib.CreateCoalescedTsetlinMachine(self.number_of_classes, self.number_of_clauses, self.number_of_features, self.T, self.s ,self.boost_true_positive_feedback)

    def get_weights(self, clauses,  s, T, output_dir):
        weights = []
        for i in range(self.number_of_classes):
            for j in range(self.number_of_clauses):
                weights.append(str(_lib.tm_get_clause_weight(self.tm, i, j)))

        file_name = output_dir + "/"+ "weigths_" + str(clauses) + "_" + str(int(s)) + "_" + str(T) + ".txt"
        with open(file_name, 'w') as fp:
            fp.write('\n'.join(weights))

    def get_states(self, print_states, no_states, clauses, s, T, output_dir):
        TAs = []
        for i in range(self.number_of_clauses): 
            for j in range(self.number_of_features):
                for k in range(2):
                    TAs.append(str(_lib.tm_ta_state(self.tm, i,j,k)))
                    if(print_states):
                        print("TA[",k,"]", "Feature[", j, "]", "Clause[",i,"] = ", (int((_lib.tm_ta_state(self.tm, i,j,k)))))
        
        if(print_states == 0):
            file_name = output_dir + "/"+ "TAs_" + str(clauses) + "_" + str(int(s)) + "_" + str(T) + ".txt"
            with open(file_name,'w') as TA_file:
                TA_file.write(' '.join(TAs))
            TA_file.close()

    def load_TAs_load_Weights(self, TAs, Weights):
        TAs = np.array(TAs).astype(np.uint32)
        Weights =np.array(Weights).astype(np.uint32)
        
        self.number_of_TAs       = TAs.shape[0]
        self.number_of_Weights   = Weights.shape[0]

        _lib.tm_load_TAs_load_Weights(self.tm, self.number_of_TAs, self.number_of_Weights, TAs, Weights)

    def analyse_inference_(self, test_data, exp_test_outputs ,number_of_examples):
        for i in range(number_of_examples):
            print("Datapoint: ", i, " Expecting: ", exp_test_outputs[i])
            _lib.tm_analyse_inference(self.tm, test_data[i].astype(np.uint32))

    def tm_fit(self, X, y, X_test, y_test, epochs, save_best_acc, include_, no_states, incremental=True):
        number_of_examples  = X.shape[0]
        best_acc            = 0
        rand_count          = 0

        for e in range(epochs):
            # sys.stdout.write("[%s]" % (" " * toolbar_width))
            sys.stdout.flush()
            # sys.stdout.write("\b" * (toolbar_width+1))
            shuffled_index = np.arange(X.shape[0])
            np.random.shuffle(shuffled_index)
            progress_bar_print_size = shuffled_index.shape[0]/toolbar_width
            errors  = 0
            counter = 0

            for i in shuffled_index:
                _lib.tm_update(self.tm, X[i].astype(np.uint32), 1,  y[i].astype(np.int32), float(self.s))
 
            for i in shuffled_index:
                max_class = 0
                max_class_sum = _lib.tm_score(self.tm, X[i].astype(np.uint32), 0) 
                for k in range(1,self.number_of_classes):               
                    class_sum = _lib.tm_score(self.tm, X[i].astype(np.uint32), k)
                    if max_class_sum < class_sum:
                        max_class = k
                        max_class_sum = class_sum

                if max_class != y[i]:
                    errors += 1

                # if(counter%progress_bar_print_size == 0):
                #     sys.stdout.write("▶")
                #     sys.stdout.flush()
                # counter +=1 
                    
            acc = 1 - (errors/number_of_examples)    
            sys.stdout.write("\tEpoch: "+ str(e)+ "  \t\tTraining Accuracy: "+ str(acc*100) + "%\n") 
            # _lib.get_rands_used(self.tm)
            # _lib.rands_reset(self.tm)
            
            if(include_):
                include_count = 0
                for clauses in range(self.number_of_clauses): 
                        for features in range(self.number_of_features):
                            for literals in range(2):
                                if(_lib.tm_ta_state(self.tm, clauses, features, literals)>no_states):
                                    include_count += 1
                print("\tNumber of Includes: ", include_count)

            if(save_best_acc):
                if(acc > best_acc):
                    best_acc = acc
                    TA_states = []
                    for clauses in range(self.number_of_clauses): 
                        for features in range(self.number_of_features):
                            for literals in range(2):
                                TA_states.append(str(_lib.tm_ta_state(self.tm, clauses, features, literals)) + " ")

                    TAFile=open('TAs_.txt','w')
                    TAFile.writelines(TA_states)
                    TAFile.close()

                    weights = []
                    for k in range(self.number_of_classes):
                        for j in range(self.number_of_clauses):
                            weights.append(str(_lib.tm_get_clause_weight(self.tm, k, j))+ " ")

                    WFile=open('weights_.txt','w')
                    WFile.writelines(weights)
                    WFile.close()

                    num_test_examples = X_test.shape[0]
                    errors = 0
                    for i in range(num_test_examples):
                        max_class = 0
                        max_class_sum = _lib.tm_score(self.tm, X_test[i].astype(np.uint32), 0)  
                        for k in range(1,self.number_of_classes):               
                            class_sum = _lib.tm_score(self.tm, X_test[i].astype(np.uint32), k)
                            if max_class_sum < class_sum:
                                max_class = k
                                max_class_sum = class_sum
                        if max_class != y_test[i]:
                            errors += 1
                    test_acc = 1 - (errors/num_test_examples)
                    print("\tTesting acc:", test_acc) 

        print("\tTraining Complete")    

    def tm_evaluate(self, X, y):
        number_of_examples  = X.shape[0]
        errors              = 0
        for i in range(number_of_examples):
            max_class = 0
            max_class_sum = _lib.tm_score(self.tm, X[i].astype(np.uint32), 0)  
            for k in range(1,self.number_of_classes):               
                class_sum = _lib.tm_score(self.tm, X[i].astype(np.uint32), k)
                if max_class_sum < class_sum:
                    max_class = k
                    max_class_sum = class_sum
            if max_class != y[i]:
                errors += 1
        acc = 1 - (errors/number_of_examples)
        print("\tTesting acc:", acc)    

def main():


    config_file = open("training_config.json")
    config = json.load(config_file)

    # parser = argparse.ArgumentParser(description="MATADOR CoTM Training")
    # parser.add_argument("-dir", help="results directory to store the training data", type=str)
    # parser.add_argument("-threshold", help="Provide the value of \"Threshold\" used while training the TM to obtain the TA file", type=int)
    # parser.add_argument("-s", help="Provide the value of \"Threshold\" used while training the TM to obtain the TA file", type=float)
    # parser.add_argument("-clauses", help="Provide the number of clauses used while training the TM", type=int)
    # parser.add_argument("-classes", help="Provide the number of classes in the undertaken classification problem", type=int)
    # parser.add_argument("-features", help="Provide the number of features in the undertaken classification problem", type=int)       
    # parser.add_argument("-test_filename", help="Provide the name of the test file", type=str)
    # parser.add_argument("-train_filename", help="Provide the name of the test file", type=str)
    # parser.add_argument("-inf", help="inference only?", type=int)
    # parser.add_argument("-epochs", help="number of epochs needed", type=int)

    # args = parser.parse_args()

    #====================================================
    output_dir      = str(config["Output_Directory"])
    clauses         = int(config["Clauses"])
    classes         = int(config["Classes"])
    s               = float(config["s_value"])
    T               = int(config["T_value"])
    epochs          = int(config["epochs"])
    train_file      = str(config["training_data"])
    test_file       = str(config["test_data"])
    features        = int(config["features"])

    # Update the TM parameters here
    # INFEERENCE ONLY
    inf_only        = 0 # Load trained models 
    # number_of_examples      = 10  # Numnber of infs to do
    print_states            = 0
    # TRAINING + INFERENCE
    no_states       = 400 
    save_best_acc   = 1
    include_        = 1
    #====================================================
    
    # print_banner()
    print("\tLoading Data")
    X_train, y_train, X_test, y_test = read_data(train_file, test_file)
    print("\tDone!")
    print("\t--------------------")
    print("\t Output Directory: ", output_dir)
    print("\t Clauses:  ", clauses)
    print("\t Classes:  ", classes)
    print("\t s value:  ", s)
    print("\t T value:  ", T)
    print("\t epochs :  ", epochs)
    print("\t--------------------")
    print("\tTraining may take a long time... ")

    if(inf_only):
        TAs = []
        # TA_file = open("TAs_500_8_800.txt", "r")
        data = TA_file.read()
        data_into_list = data.split(" ")
        TAs = [int(i) for i in data_into_list]
        
        Weights = []
        # W_file = open("wegiths_500_8_800.txt", "r")
        data = W_file.read()
        data_into_list = data.split("\n")
        Weights = [int(i) for i in data_into_list]

        print_inf_only()
        tm = CoalescedTsetlinMachine(clauses, features, classes, T, s, boost_true_positive_feedback=1)
        tm.load_TAs_load_Weights(TAs, Weights)
        print("\tModel Loaded")
        tm.analyse_inference_(X_test, y_test, number_of_examples)
        # tm.get_states(print_states, no_states)


    else:
        tm = CoalescedTsetlinMachine(clauses, features, classes, T, s, boost_true_positive_feedback=1)
        tm.tm_fit(X_train, y_train, X_test, y_test, epochs, save_best_acc, include_, no_states)
        tm.tm_evaluate(X_test, y_test)
        tm.get_weights(clauses, s, T, output_dir)
        tm.get_states(print_states, no_states, clauses, s, T, output_dir)

if __name__ == "__main__":
    main()
