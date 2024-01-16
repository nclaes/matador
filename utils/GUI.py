# import customtkinter
import os
from pathlib import Path
from PIL import ImageTk, Image
from functools import partial

import tkinter as tk
from tkPDFViewer import tkPDFViewer as pdf

from tkinter.ttk import *
from tkinter import Label
import sys
import subprocess
import threading 
import webbrowser as wb

from stages_print import *

# Directories for installation paths for vivado and the output dir where the data will
# be stored 
vivado_path = ""
output_dir = ""

Matador_home_path = Path.cwd() 

class Redirect():

    def __init__(self, widget, autoscroll=True):
        self.widget = widget
        self.autoscroll = autoscroll

    def write(self, text):
        self.widget.insert('end', text)
        if self.autoscroll:
            self.widget.see("end")  # autoscroll
        
    def flush(self):
       self.widget.insert('end', " ")

# --- functions ---
def show_entry_fields(e1, e2):
    print("TAs File Location: %s\nWeights File Location: %s" % (e1.get(), e2.get()))

def run_Train():
    threading.Thread(target=train_CoTM).start()

def run_Manual():
    threading.Thread(target=Matador_manual).start()

def run_CA():
    threading.Thread(target=create_accel).start()    

def run_About():
    threading.Thread(target=about).start()

def Matador_manual():
    # open a new window of a pdf reader 
    print("  ")
    print("  Matador Manual is open ")
    print("  It will be destroyed when this window is destroyed")
    print(" ")

    os.system("evince images/Matador_Manual.pdf")


def CoTM_train_py(clauses, classes, train_data, test_data, s_value, T_value, epochs, features, TM, newWindow):
    
    global output_dir
    print("\t\t\tOutput Directory: ", output_dir)

    clauses_    = clauses.get()
    classes_    = classes.get()
    train_data_ = train_data.get()
    test_data_  = test_data.get()
    s_value_    = s_value.get()
    T_value_    = T_value.get()
    epochs_     = epochs.get()
    features_   = features.get()
    TM          = TM.get()
    newWindow.destroy()
    # print("Selected specs for the TM:   ", clauses_, " ",classes_," ", train_data_, " ", test_data_, " ", s_value_ )
    
    if(output_dir == ""):
        print(train_model_, "The output directory has not been specified -- please run the Setup to configure")
        # sleep(10)
        exit()

    print("")
    print(train_model_, " Training Start")
    print(train_model_, "Selected TM: ", TM)


    f = open(output_dir+"/training_config.json", "w")
    f.write("{\n")
    f.write("   \"Output_Directory\":\"" + str(output_dir) + "\",\n")
    f.write("")
    f.write("   \"Clauses\"     : \"" + str(clauses_) + "\",\n")
    f.write("   \"Classes\"     : \"" + str(classes_) + "\",\n")
    f.write("   \"s_value\"     : \"" + str(s_value_) + "\",\n")
    f.write("   \"T_value\"     : \"" + str(T_value_) + "\",\n")
    f.write("   \"epochs\"     : \"" + str(epochs_) + "\",\n")
    f.write("   \"states\"     : \"" + str(256) + "\",\n")
    f.write("   \"features\"   : \"" + str(features_) + "\",\n")
    f.write("   \"training_data\"      : \"" + str(train_data_) + "\",\n")
    f.write("   \"test_data\"          : \"" + str(train_data_) + "\"\n")
    f.write("}\n")

    f.close()

    f = open("training_config.json", "w")
    f.write("{\n")
    f.write("   \"Output_Directory\":\"" + str(output_dir) + "\",\n")
    f.write("   \"Clauses\"     : \"" + str(clauses_) + "\",\n")
    f.write("   \"Classes\"     : \"" + str(classes_) + "\",\n")
    f.write("   \"s_value\"     : \"" + str(s_value_) + "\",\n")
    f.write("   \"T_value\"     : \"" + str(T_value_) + "\",\n")
    f.write("   \"epochs\"     : \"" + str(epochs_) + "\",\n")
    f.write("   \"states\"     : \"" + str(256) + "\",\n")
    f.write("   \"features\"   : \"" + str(features_) + "\",\n")
    f.write("   \"training_data\"      : \"" + str(train_data_) + "\",\n")
    f.write("   \"test_data\"          : \"" + str(train_data_) + "\"\n")
    f.write("}\n")

    f.close()


def train_CoTM():
    # first check if the user has specified the output directory
    print("  ----------------------------------------------------------------------------")
    print("                          Entering Training Mode                              ")
    print("  ----------------------------------------------------------------------------")
    print(train_model_, "Training Mode now incorporates TMU codebase for training")
    print(train_model_, "Enter the data as required from the pop-up window")
    print(train_model_, "TMU               : https://github.com/cair/tmu")
    print(train_model_, "Vanilla TM        : https://arxiv.org/abs/1804.01508")
    print(train_model_, "Coalesced TM      : https://arxiv.org/abs/2108.07594")
    print(train_model_, "Lit Budgeting Info: https://arxiv.org/abs/2301.08190")
    # create a new window to take the user inputs for the CoTM training 
    newWindow = tk.Toplevel(root)
    newWindow.resizable(0,0)
    newWindow.title("Train Model")
    newWindow.geometry("285x340")
    p1 = ImageTk.PhotoImage(file = 'images/image.png')
    newWindow.iconphoto(False, p1)    
    s = Style()
    s.configure("TMenubutton", background="gray", foreground="white", font='Terminal 9', width=28, bd=5)

    #Create a dropdown Menu    
    menu = tk.StringVar()
    #menu.configure(font='Terminal 9')
    menu.set(" Select Tsetlin Machine Type ")
    drop_ = OptionMenu(newWindow, menu, "Select Tsetlin Machine Type", "Tsetlin Machine: Vanilla ", "Tsetlin Machine: Coalesced").grid(row=0, columnspan=2, pady=8, padx=20, sticky=tk.W)

    Label(newWindow, 
             text="Clauses", font='Terminal 9').grid(row=1)
    Label(newWindow, 
             text="Classes", font='Terminal 9').grid(row=2)
    Label(newWindow, 
             text="Train data", font='Terminal 9').grid(row=3)
    Label(newWindow, 
             text="Test data", font='Terminal 9').grid(row=4)
    Label(newWindow, 
             text="s value", font='Terminal 9').grid(row=5)
    Label(newWindow, 
             text="T value", font='Terminal 9').grid(row=6)
    Label(newWindow, 
             text="Epochs", font='Terminal 9').grid(row=7)

    Label(newWindow, 
             text="Features", font='Terminal 9').grid(row=8)
             
             
    Label(newWindow, 
             text="Lit Budget\n(y/n)", font='Terminal 9').grid(row=9, rowspan=2)

    clauses = tk.Entry(newWindow, bd=3)
    classes = tk.Entry(newWindow, bd=3)
    train_data = tk.Entry(newWindow , bd=3)
    test_data  =  tk.Entry(newWindow, bd=3)
    s_value =  tk.Entry(newWindow, bd=3)
    T_value = tk.Entry(newWindow, bd=3)
    epochs =  tk.Entry(newWindow, bd=3)
    features = tk.Entry(newWindow, bd=3)
    lit_budg = tk.Entry(newWindow, bd=3)

    clauses.grid(row=1, column=1)
    classes.grid(row=2, column=1)
    train_data.grid(row=3, column=1)
    test_data.grid(row=4, column=1)
    s_value.grid(row=5, column=1)
    T_value.grid(row=6, column=1)
    features.grid(row=8, column=1)
    epochs.grid(row=7, column=1)
    lit_budg.grid(row=9, column=1)
    

    callable_print_collect_tm_training = partial(CoTM_train_py, clauses, classes, train_data, test_data, s_value, T_value, epochs, features, menu, newWindow)

    var = tk.IntVar()
    done_button = tk.Button(newWindow, text='Start Training', font='Terminal 9', command=callable_print_collect_tm_training, bg="black", fg = "white", width=14).grid(row=11, column=1, sticky=tk.W, pady=3, padx=30)

    root.wait_window(newWindow)
    exec_py = "python3 utils/tm_me.py"     
    # subprocess.check_call(exec_py.split(), stdout=sys.stdout , stderr=subprocess.STDOUT)
    p = subprocess.Popen(exec_py, shell=True, stdout=subprocess.PIPE, bufsize=1, text=True)
    while True:
        nextline = p.stdout.readline()
        sys.stdout.write(nextline)
        if nextline == '' and p.poll() is not None:
            break
    #     # sys.stdout.write(nextline)
    #     sys.stdout.flush()

    # output = p.communicate()[0]
    # exitCode = p.returncode
    # while p.poll() is None:
    #     msg = p.stdout.readline().strip() # read a line from the process output
    #     if msg:
    #         print(msg)

    print(train_model_, "CoTM Training End")
    print("")
    print("  ----------------------------------------------------------------------------")
    print("                           Exiting Training Mode                              ")
    print("  ----------------------------------------------------------------------------")


def write_setup_config(vivado, output_dir_user, newWindow):
    # create the directory that has been specified
    global vivado_path 
    vivado_path= vivado.get()
    global output_dir 
    output_dir= output_dir_user.get()
    if(output_dir == ""):
        print(setup_, "Empty Directory given")
        print(setup_, "Error in Setup - exiting")
        newWindow.destroy()
    elif(vivado_path == ""):
        print(setup_, "Empty Vivado path given")
        print(setup_, "Error in Setup - exiting") 
        newWindow.destroy()
    else:
        isExist = os.path.exists(output_dir)
        if not isExist:
            print(setup_, "Creating User Directory: ", output_dir)
            print(setup_, "Existing files will be re-written")
            # Create a new directory because it does not exist
            os.makedirs(output_dir)
        # checking Vivado path 
        print(setup_, "Checking for Vivado bin path...")
        isExist = os.path.exists(vivado_path    )
        if not isExist:
            print(setup_, "The Vivado path provided does not exist")
            print(setup_, "Error in Setup - exiting")

        else:
            print(setup_, "Vivado bin path found")
            print(setup_, "Writing setup config.json")
            f = open(output_dir+"/config.json", "w")
            f.write("{\n")
            f.write("   \"vivado_path\": \"" + str(vivado_path) + "\",\n")
            f.write("   \"Matador_home_path\": \"" + str(Matador_home_path) + "\"\n")
            f.write("}")
            f.close()
            print(setup_, "Setup complete...")
            print("  ----------------------------------------------------------------------------")
            print("                            Exiting Setup Mode                                ")
            print("  ----------------------------------------------------------------------------")
        newWindow.destroy()

def run_Setup():
    newWindow = tk.Toplevel(root)
    newWindow.title("Setup")
    newWindow.geometry("350x100")
    p1 = ImageTk.PhotoImage(file = 'images/image.png')
    newWindow.iconphoto(False, p1) 
    Label(newWindow, 
             text="Vivado bin path", font='Terminal 9').grid(row=0)
    Label(newWindow, 
             text="Output Directory", font='Terminal 9').grid(row=1)

    e1 = tk.Entry(newWindow)
    e2 = tk.Entry(newWindow)

    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    print("")
    print("  ----------------------------------------------------------------------------")
    print("                           Entering Setup Mode                                ")
    print("  ----------------------------------------------------------------------------")
    print("")

    callable_print_collect = partial(write_setup_config, e1, e2, newWindow)

    done_button = tk.Button(newWindow, text='Done', font='Terminal 9', command=callable_print_collect, bg="black", fg = "white").grid(row=3, column=1, sticky=tk.W, pady=4) 
  
    
def run_UEM():
    threading.Thread(target=uem).start() 
    

def run_RTL_gen():
    threading.Thread(target=RTL_gen).start()

def RTL_gen():
    print(gen_RTL,"Generating RTL")
    # exec_py = "python3 utils/TM_initial_block_creator.py"       
    # # subprocess.check_call(exec_py.split(), stdout=sys.stdout , stderr=subprocess.STDOUT)
    # p = subprocess.Popen(exec_py, shell=True, stdout=subprocess.PIPE, bufsize=1, text=True)
    # while True:
    #     nextline = p.stdout.readline()
    #     sys.stdout.write(nextline)
    #     if nextline == '' and p.poll() is not None:
    #         break

    exec_py = "python3 utils/RTL_gen.py"
    p = subprocess.Popen(exec_py, shell=True, stdout=subprocess.PIPE, bufsize=1, text=True)
    while True:
        nextline = p.stdout.readline()
        sys.stdout.write(nextline)
        if nextline == '' and p.poll() is not None:
            break



def config_existing_model(TM, TAs, Weights, Clauses, Classes, BW, F, No_s,  newWindow, adds):

    global output_dir
    # if(output_dir == ""):
    #     print(" The output directory has not been specified -- please run the Setup to configure")
    #     newWindow.destroy()
    # else:
    TM          = TM.get()
    TAs_        = TAs.get()
    Weights_    = Weights.get()
    Clauses_    = Clauses.get()
    Classes_    = Classes.get()
    BusWidth_   = BW.get()
    Features_   = F.get()
    No_s_       = No_s.get()
    adds_       = adds.get()
    newWindow.destroy()

    f = open("gen_RTL.json", "w")
    f.write("{\n")
    f.write("   \"TM\":\"" + str(TM) + "\",\n")
    f.write("   \"Output_Directory\":\"" + str(output_dir) + "\",\n")
    f.write("   \"TAs\"          : \""   + str(TAs_) + "\",\n")
    f.write("   \"Weights\"      : \""   + str(Weights_) + "\",\n")
    f.write("   \"Classes\"      : \""   + str(Classes_) + "\",\n")
    f.write("   \"Clauses\"      : \""   + str(Clauses_) + "\",\n")
    f.write("   \"BusWidth\"     : \""   + str(BusWidth_) + "\",\n")
    f.write("   \"Features\"     : \""   + str(Features_) + "\",\n")
    f.write("   \"No_States\"    : \""   + str(No_s_) + "\",\n")
    f.write("   \"Adder_Stages\" : \""   + str(adds_) + "\"\n")
    f.write("}\n")
    f.close()

    run_RTL_gen()
    

def uem():
    # Use existing model 
    global output_dir
    print("  Output Directory: ", output_dir)
    print("  ")
    print("  Converting Existing Model ")
    print("  Converting requires existing TA file and or Weights file depending on the TM")
    print(" ")
    print("  ----------------------------------------------------------------------------")
    print("                               Conversion Help                                ")
    print("  ----------------------------------------------------------------------------")
    print("     [Convert]  Specify the TM type: [vanilla, coal]")
    print("     [Convert]  Fill out the TM model specs - use the training.json if created")
    print("     [Convert]  Bus Width: [32, 64]")
    print("     [Convert]  Adder Stages: number of pipeline stages in the adder (coal TM)")
    print("     [Convert]  No States: Maximum TA state number")
    print("  ----------------------------------------------------------------------------")
    print(" ")

    newWindow = tk.Toplevel(root)
    newWindow.title("Convert Existing Model")
    newWindow.geometry("350x300")
    p1 = ImageTk.PhotoImage(file = 'images/image.png')
    newWindow.iconphoto(False, p1)  
    
    Label(newWindow, 
        text="TM type", font='Terminal 9').grid(row=0)
    Label(newWindow, 
             text="TA file", font='Terminal 9').grid(row=1)
    Label(newWindow, 
             text="Weights file", font='Terminal 9').grid(row=2)
    Label(newWindow, 
             text="Clauses ", font='Terminal 9').grid(row=3)
    Label(newWindow, 
             text="Classes", font='Terminal 9').grid(row=4)
    Label(newWindow, 
             text="Bus Width", font='Terminal 9').grid(row=5)
    Label(newWindow, 
             text="Features", font='Terminal 9').grid(row=6)
    Label(newWindow, 
             text="No States", font='Terminal 9').grid(row=7)
    Label(newWindow, 
             text="Adder Stages", font='Terminal 9').grid(row=8)

    TM = tk.Entry(newWindow)
    TA = tk.Entry(newWindow)
    W = tk.Entry(newWindow)
    Clauses = tk.Entry(newWindow)
    Classes = tk.Entry(newWindow)
    BW      = tk.Entry(newWindow)
    F       = tk.Entry(newWindow)
    No_s    = tk.Entry(newWindow)
    adds    = tk.Entry(newWindow)

    TM.grid(row=0, column=1)
    TA.grid(row=1, column=1)
    W.grid(row=2, column=1)
    Clauses.grid(row=3, column= 1)
    Classes.grid(row=4, column=1)
    BW.grid(row=5, column=1)
    F.grid(row=6, column=1)
    No_s.grid(row=7, column=1)
    adds.grid(row=8, column=1)

    callable_print_collect_uem = partial(config_existing_model,TM, TA, W, Clauses, Classes, BW, F, No_s, newWindow, adds)

    done_button = tk.Button(newWindow, text='Done', font='Terminal 9', command=callable_print_collect_uem, bg="black", fg = "white").grid(row=9, column=1, sticky=tk.W, pady=4) 

def create_accel():
    # create the accelerator
    # first check for if the directory has been specified 
    global output_dir
    print("\t\tChecking Output Directory: ", output_dir)

    global vivado_dir
    print("\t\tChecking Vivado Path: ", vivado_path)


    print("\t\tChecking for model type...")

    # Open training config.json - if it doesn't exist it must be created 

    if os.path.isdir(output_dir+"/training_config.json"):
        if os.path.isdir(output_dir+"/RTL"):
            print("\t\tRTL directory found")

            f = open("accel_config.json", "w")
            f.write("{\n")
            f.write("   \"Output_Directory\":\"" + str(output_dir) + "\",\n")
            f.write("   \"Vivado\"     : \"" + str(vivado_path) + "\"\n")
            f.write("}\n")
            f.close()

            exec_py = "python3 utils/Create_Bitstream.py"       
            # subprocess.check_call(exec_py.split(), stdout=sys.stdout , stderr=subprocess.STDOUT)
            p = subprocess.Popen(exec_py, shell=True, stdout=subprocess.PIPE, bufsize=1, text=True)
            while True:
                nextline = p.stdout.readline()
                sys.stdout.write(nextline)
                if nextline == '' and p.poll() is not None:
                    break

            print("Now this can be deployed on Pynq")
        else:
            print("RTL directory not found - please use the Setup option to set this")

    else:
        print("\t\tError - training_config not found - please check")

def run_Pynq():
    threading.Thread(target=pynq).start() 

def pynq():
    wb.open("192.168.2.99",new=2)


def ole():
    print("""
        -------------------------------------------------------------
                                ,           ,
                              ((__-^^-,-^^-__))
                               `-_---' `---_-'
                                <__|o` 'o|__>
                                   \  `  /
                                    ): :(
                                    :o_o:
                                     "-"   

                                    ole!!!
        ------------------------------------------------------------- 

        """)    



def about():
    print("""
        -------------------------------------------------------------
        MATADOR: autoMated dATa bAndwith Driven lOgic based infeRence

        Copyright (C) 2023

        Matador is a tool for training and translating Coalesced 
        Tsetlin Machines (CoTM) into FPGA accelerator systems. It co-
        verts learnt clause propositions into custom hard coded fast 
        inference circuits which are then streamed inference data 
        from the system's processor. The tool's configurable design 
        and debug options allow for rapid prototyping of CoTM applica-
        tions to meet target performance and resource requirements.


        Tousif Rahman, Gang Mao 

        ------------------------------------------------------------- 

        """)

# --- main ---    

root = tk.Tk()
root.title('Matador v1.2 (2023)')

root.geometry("840x445")
root.resizable(0,0)
root.eval('tk::PlaceWindow . center')
p1 = ImageTk.PhotoImage(file = 'images/m_icon.png')
root.iconphoto(False, p1)
# - rame with Text and Scrollbar -
frame1 = tk.Frame(root, height=15)
# frame1.config(background='black')
frame1.pack(side = "top", expand=True, fill='both')
raw_image = Image.open("images/m_final.png")
# img = raw_image.zoom(25) #with 250, I ended up running out of memory
# img = img.subsample(32)
resized_image = raw_image.resize((140,142), Image.LANCZOS)
new_image = ImageTk.PhotoImage(resized_image)
# resized_image = raw_image.thumbnail((100,110), Image.ANTIALIAS)

panel =tk.Frame(root) 
panel.pack(side = "left", anchor="nw", pady=5)


frame = tk.Frame(root)
frame.pack(expand=True, fill='both')

text = tk.Text(frame ,bd=4, width=82)
text.pack(side='left' ,fill='y', expand=True)

scrollbar = tk.Scrollbar(frame , bd=2 )
scrollbar.pack(side='right', anchor="ne", fill='y',padx=5)

text['yscrollcommand'] = scrollbar.set
scrollbar['command'] = text.yview

# - rest -
button = tk.Button(panel, image=new_image, width =140, relief="flat", command=ole)
button.pack(pady=1)

button = tk.Button(panel, text='' , font='Terminal 8' , width = 14, relief="flat")
button.pack(pady=4)

button = tk.Button(panel, text='Train Model' , font='Terminal 9' , width = 14, command=run_Train)
button.pack(pady=1)

button = tk.Button(panel, text='Generate RTL' ,  font='Terminal 9' , width = 14, command=run_UEM)
button.pack(pady=1)

button = tk.Button(panel, text='Synth + Impl', font='Terminal 9', width = 14, command=run_CA)
button.pack(pady=1)

button = tk.Button(panel, text='About', width = 14, font='Terminal 9' ,command=run_About)
button.pack(pady=1)

button = tk.Button(panel, text='PYNQ', bg='pink',  font='Terminal 9' , width = 14, command=run_Pynq)
button.pack(pady=1)

button = tk.Button(panel, text='Manual', bg="gray", fg = "white", font='Terminal 9' , width = 14, command=run_Manual)
button.pack(pady=1)

button = tk.Button(panel, text='Setup', font='Terminal 9' , bg="black", fg = "white", width =14, command=run_Setup)
button.pack(pady=10)

frame2 = tk.Frame(root, height=30)
frame2.pack(side = "bottom", expand=True, fill='both')

old_stdout = sys.stdout    
sys.stdout = Redirect(text)

print("                                                                            ")
print("  Matador: autoMated dATa bAndwidth Driven lOgic based infeRence             ")
print("                                                                            ")
# print("  Tousif Rahman, Gang Mao, Sidharth Maheshwari                              ")                                  
print('  Copyright (C) 2023                                                        ')
print('                                                                            ')
print('  Permission to use, copy, modify, and/or distribute this software for any  ')
print('  purpose with or without fee is hereby granted, provided that the above    ')
print('  copyright notice and this permission notice appear in all copies.         ')
print('                                                                            ')
print('  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES  ')
print("  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF          ")
print('  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR   ')
print('  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ')
print('  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN     ')
print('  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF   ')
print('  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.            ')

root.mainloop()
# - after close window -
sys.stdout = old_stdout
# TO DO -- kill all background processes associated with the tool 

