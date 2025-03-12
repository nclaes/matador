### <img src="/banner.png" width=900/>
Automated ASIC and FPGA design for Tsetlin Machine Acclerators. The repo is organized according to the table below: 

| Repo Branch      | Description |
| -------------------------------|----------------------------------------------------------------------------------- |
| ```main```               | Used to give an overview of each Matador flow branch.   |
|```development```             | Branch for automated FPGA accelerator design. |
|```4adrian```             | Non-GUI version allowing developers to use the core MATADOR functions. |


## JSON file based usage (`4adrian` branch only)

This version uses a JSON file to perform the MATADOR flow. Please review `MATADOR_NO_GUI.json` for how to use this branch. The codebase should be a little more readable now. 

Developers should focus on `MATADOR_main.py`

### Readining in TM models from TMU 
The `TAs.txt` and `weights.txt` files must be written in a specific format. This code snippet shows how the TAs are extracted from the 
Vanilla TM model from TMU

```
        for i in range(int(config["Classes"])):
            for j in range(clauses_2):
                TAs = []
                for k in range(int(config["features"])*2):
                    ta = tm.get_ta_action(j, k, the_class=i, polarity=0)
                    TAs.append(int(ta))

                for TA_x_ in range(int(config["features"])):
                    TM_file.write(str(int(TAs[TA_x_])) + " " + str(int(TAs[int(config["features"]) + TA_x_])) + " ")

                TAs = []
                for k in range(int(config["features"])*2):
                    ta = tm.get_ta_action(j, k, the_class=i, polarity=1)
                    TAs.append(int(ta))

                for TA_x_ in range(int(config["features"])):
                    TM_file.write(str(int(TAs[TA_x_])) + " " + str(int(TAs[int(config["features"]) + TA_x_])) + " ")
    

```
Note for Coalesced TM: It is essential that the weights are extracted in the same order as the clauses. 
By default TMU stores the clauses like this: `All the Features` then `All the Complements`. But MATADOR expects alternating feature and complement. 


Contributors: Tousif Rahman, Gang Mao, Sidharth Maheshwari, Marcos Sartori, Shengyu Duan, Bob Pattison, Adrian Wheeldon



