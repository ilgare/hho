# Code for the paper
#
# Reconsidering the Harris' Hawks Optimization Algorithm
#
# by
#
# Kemal Ilgar Eroglu
#
# 11/17/2024
#
# Author: Kemal Ilgar Eroğlu
#
# This is the test comparing HHO against RHO using the benchmark
# suite functions with their origins shifted to (60,59,62,...)

import numpy as np
import pandas as pd
import random
import hho
import rho
import benchmarks_v3 as benchmarks
import time
import platform
from datetime import datetime

num_features = 30

num_trials = 1

# 0-based indices of the first and last functions to be tested
# e.g. 0 and 12 for F1 -- F13
start_fun=0
end_fun=12

# Use the first 23 test functions
#num_funcs= start_fun - end_fun

# Define function names, initialize the lists of average values
fnames = ['']*23
for i in range(0,23):
    fnames[i] = f"F{i+1}"

hho_avgs = [0]*23
rho_avgs = [0]*23

# We loop over the test functions
for i in range(start_fun,end_fun+1):

    # Get the function object 
    objective = getattr(benchmarks,fnames[i])

    resval = 0
    resval_s = 0

    details = benchmarks.getFunctionDetails(fnames[i])

    l_bound = details[1]
    u_bound = details[2]

    for nt in range(0,num_trials):

        print(f"\nFunction {fnames[i]}\tTrial {nt+1}/{num_trials} lb={l_bound} ub={u_bound}\n")

        hho_N = 30
        hho_T = 500
        
        hho_obj = hho.HHO(obj_func=objective, N=hho_N, T=hho_T, lb=l_bound, ub=u_bound, dim=num_features)

        r_energy, r_location, CNVG = hho_obj.optimize()

        #print(f"\nValue={r_energy}\nPosition{r_location}\n")

        resval += r_energy
        ####### End of HHO block

        # Initialize and run RHO
        
        rho_obj = rho.RHO(obj_func=objective, N=hho_N, T=hho_T, lb=l_bound, ub=u_bound, dim=num_features)

        r_energy_s, r_location_s, CNVG_s = rho_obj.optimize()

        #print(f"\nValue={r_energy_s}\nPosition={r_location_s}\n")

        resval_s += r_energy_s
        ####### End of RHO block

    # After the trial loop is over, compute averages
    hho_avgs[i] = resval / num_trials
    rho_avgs[i] = resval_s / num_trials
    # Name of the output file, with a timestamp
    now = datetime.now()
    output_fname=f"output_{num_features}D_{num_trials}_trials_v3_"+ now.strftime("%d-%m-%y_%H:%M:%S")+".txt"

with open(output_fname, 'w') as f:

    f.write(f"Using function suite v3 F1--F13")
    
    f.write(f"\nNumber of trials = {num_trials}")

    f.write(f"\nFeature number (dimension)={num_features}\n")
    
    # The table with the averages
    df_dict = {}

    df_dict['Optimizer'] = ['HHO','RHO']

    for i in range(start_fun,end_fun+1):
         df_dict[fnames[i]] = [hho_avgs[i], rho_avgs[i]]

    df = pd.DataFrame(df_dict)

    output_txt = df.to_markdown(index=False, tablefmt='plain', colalign=['center']*len(df.columns))

    f.write(output_txt)
    print(output_txt)

    

    f.write(f"\n\nSystem:\n{platform.platform()}\n")
