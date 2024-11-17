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
# This is the code used to generate Figures 1 and 2, showinf the history
# of generated points (hawks) by HHO and NHO.

import numpy as np
import pandas as pd
import random
import hho_wh
import nho_wh
#import benchmarks_orig as benchmarks
import benchmarks_v2 as benchmarks
import time
import platform
from datetime import datetime
from matplotlib import pyplot as plt

num_features = 2

num_trials = 1

# 0-based indices of the first and last functions to be tested
# e.g. 0 and 12 for F1 -- F13
start_fun=3
end_fun=3

# Use the first 23 test functions
#num_funcs= start_fun - end_fun

# Define function names, initialize the lists of average values
fnames = ['']*23
for i in range(0,23):
    fnames[i] = f"F{i+1}"

hho_avgs = [0]*23
nho_avgs = [0]*23

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
        
        hho_obj = hho_wh.HHO(obj_func=objective, N=hho_N, T=hho_T, lb=l_bound, ub=u_bound, dim=num_features)

        r_energy, r_location, CNVG, hist, hist_rabbit = hho_obj.optimize()

        #print(f"\nValue={r_energy}\nPosition{r_location}\n")

        resval += r_energy
        ####### End of HHO block

        # Initialize and run NHO
        
        nho_obj = nho_wh.NHO(obj_func=objective, N=hho_N, T=hho_T, lb=l_bound, ub=u_bound, dim=num_features)

        r_energy_s, r_location_s, CNVG_s, hist_s, hist_rabbit_s = nho_obj.optimize()

        #print(f"\nValue={r_energy_s}\nPosition={r_location_s}\n")

        resval_s += r_energy_s
        #case_prop += cases/ np.sum(cases)
        ####### End of NHO block

    # After the trial loop is over, compute averages
    hho_avgs[i] = resval / num_trials
    nho_avgs[i] = resval_s / num_trials
    # Name of the output file, with a timestamp
    now = datetime.now()
    output_fname=f"output_{num_features}D_{num_trials}_trials_"+ now.strftime("%d-%m-%y_%H:%M:%S")+".txt"

with open(output_fname, 'w') as f:

    f.write(f"Using original function suite F1--F13")
    
    f.write(f"\nNumber of trials = {num_trials}")

    f.write(f"\nFeature number (dimension)={num_features}\n")
    # The table with the averages
    df_dict = {}

    df_dict['Optimizer'] = ['HHO','NHO']

    for i in range(start_fun,end_fun+1):
         df_dict[fnames[i]] = [hho_avgs[i], nho_avgs[i]]

    df = pd.DataFrame(df_dict)

    output_txt = df.to_markdown(index=False, tablefmt='plain', colalign=['center']*len(df.columns))

    f.write(output_txt)
    print(output_txt)

    

    f.write(f"\n\nSystem:\n{platform.platform()}\n")


output_fname=f"history_nho_"+ now.strftime("%d-%m-%y_%H:%M:%S")

output_fname= "rabbit_" + output_fname

#print(hist[0:10,:,:])


#
# This bit produces Figure 2.
#

#swarm = np.concatenate(hist[0:110,:,:], axis=0)
#
#plt.scatter(swarm[:,0],swarm[:,1])
#plt.grid(True)
#plt.show()
#

#
# This bit produces Figure 3.
#
fig, axs = plt.subplots(2,3)

axs = axs.flatten()

for k in range(0,6):
        axs[k].grid(True)

swarm1 = np.concatenate(hist[0:100,:,:], axis=0)
swarm2 = np.concatenate(hist[200:300,:,:], axis=0)
swarm3 = np.concatenate(hist[400:,:,:], axis=0)

axs[0].scatter(swarm1[:,-2],swarm1[:,-1])
axs[1].scatter(swarm2[:,-2],swarm2[:,-1])
axs[2].scatter(swarm3[:,-2],swarm3[:,-1])

swarm1s = np.concatenate(hist_s[0:100,:,:], axis=0)
swarm2s = np.concatenate(hist_s[200:300,:,:], axis=0)
swarm3s = np.concatenate(hist_s[400:,:,:], axis=0)

axs[0].set_title("HHO t=0-100")
axs[1].set_title("HHO t=200-300")
axs[2].set_title("HHO t=400-500")

axs[3].scatter(swarm1s[:,0],swarm1s[:,1])
axs[4].scatter(swarm2s[:,0],swarm2s[:,1])
axs[5].scatter(swarm3s[:,0],swarm3s[:,1])

axs[3].set_title("NHO t=0-100")
axs[4].set_title("NHO t=200-300")
axs[5].set_title("NHO t=400-500")

plt.tight_layout()
plt.show()
