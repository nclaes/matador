import logging
import argparse
import json
import numpy as np
# from TMU.tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
# from TMU.tmu.util.cuda_profiler import CudaProfiler

from stages_print import *

_LOGGER = logging.getLogger(__name__)

def prep_data(confg):
    # put the data into dict
    # The data is expected to be in space seperated files as seen with the
    # example data provided in MATADOR_ROOT_DIR/data  
    train_data  = np.genfromtxt(config["training_data"], delimiter=" ")
    test_data   = np.genfromtxt(config["test_data"], delimiter=" ")

    x_train = train_data[:,:-1]
    y_train = train_data[:,-1]
    del train_data

    x_test = test_data[:,:-1]
    y_test = test_data[:,-1]
    del test_data

    return dict(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

def checkconfig(config, config_err):

    if config["TM"] == "Select Tsetlin Machine Type":
        print(train_model_, "[Error] The type of TM was not specified")
        config_err = 1
    if config["Clauses"] == "":
        print(train_model_, "[Error] The Clauses field was empty")
        config_err = 1

    if config["Classes"] == "":
        print(train_model_, "[Error] The Classes field was empty")
        config_err = 1

    if config["s_value"] == "":
        print(train_model_, "[Error] The s_value field was empty")
        config_err = 1

    if config["T_value"] == "":
        print(train_model_, "[Error] The T_value field was empty")
        config_err = 1

    if config["T_value"] == "":
        print(train_model_, "[Error] The T_value field was empty")
        config_err = 1
    
    if config["features"] == "":
        print(train_model_, "[Error] The features field was empty")
        config_err = 1
    
    if config["epochs"] == "":
        print(train_model_, "[Error] The epochs field was empty")
        config_err = 1
    
    if config["max_included_literals"] == "":
        print(train_model_, "[Error] The max_included_literals field was empty or incorrect")
        config_err = 1  

    if config["training_data"] == "":
        print(train_model_, "[Error] The training_data field was empty")
        config_err = 1
    
    if config["test_data"] == "":
        print(train_model_, "[Error] The testing_data field was empty")
        config_err = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matador Training")

    # to train the output directory for the results must be specified. 
    parser.add_argument("-output_dir", help="results directory to store the training data", type=str)
    args = parser.parse_args()

    output_dir = args.output_dir
    config_file = output_dir + "/training_config.json"
    config_file = open(config_file)
    config = json.load(config_file)

    # check the cofig is correct 
    config_err = 0
    checkconfig(config, config_err)

    if(config_err):
        print(train_model_,"There were errors in the training config json")
        print(train_model_, "You need to run the training config again")
        exit()

    print(train_model_, "Checking data...")
    # The data needs to go into some kind of dictionary with the x_train
    # y_train, x_test and y_test defined
    data = prep_data(config)
    print(data)

    if(config["TM"] == "Tsetlin Machine: Vanilla "):
        tm = TMClassifier(
            # Note - for now, the normal TM is not supporting the weighted clauses 
            # and type III feedback is not being used 
            type_iii_feedback=False,
            number_of_clauses=int(config["Clauses"]),
            T=int(config["T_value"]),
            s=float(config["s_value"]),
            max_included_literals=int(config["max_included_literals"]),
            weighted_clauses=False,
            seed=42,
        )

        # _LOGGER.info(f"Running {TMClassifier} for {int(config["epochs"])}")
        for epoch in range(int(config["epochs"])):
            benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
            with benchmark_total:
                benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
                with benchmark1:
                    res = tm.fit(
                        data["x_train"],
                        data["y_train"],
                        metrics=["update_p"],
                    )
                benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
                with benchmark2:
                    result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()

                _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                            f"Testing Time: {benchmark2.elapsed():.2f}s")
                
        # after the training is over the TM states can be extracted - this is the vanilla TM - no weights 
        TM_file_name = "TM_TA_states_Clauses_" + config["Clauses"] + "_s_value_" + str(int(float(config["s_value"]))) + "_T_value_" + config["T_value"] + "_epochs_" + config["epochs"] + "_max_literals_" + config["max_included_literals"]

        TM_file = open(args.output_dir + "/" + TM_file_name, "w")
        clauses_2 = int(int(config["Clauses"]) /2)
        print("clauses/2: ", clauses_2)

        for i in range(int(config["Classes"])):
            for j in range(int(config["Clauses"])):
                # this is for handling the clause polarity
                TAs = []
                for k in range(int(config["features"])*2):
                    ta = tm.get_ta_action(j, k, i, polarity=0)
                    TAs.append(int(ta))
                # print(TAs)
                # now we need to rewrite the TAs in the the right order
                for TA_ in range(int(config["features"])):
                    TM_file.write(str(int(TAs[TA_])) + " " + str(int(TAs[int(config["features"]) + TA_])) + " ")

                # for k in range(int(config["features"]*2)):
                #     ta = tm.get_ta_action(j, k, i, polarity=1)
                #     TM_file.write(str(int(ta)) + " ")


