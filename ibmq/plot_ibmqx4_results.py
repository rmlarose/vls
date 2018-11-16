"""
Script for reading in 2-variable linear systems solver data on ibmqx4
and plotting.
"""

# =============================================================================
# imports
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# constants
# =============================================================================

# number of shots used to run circuits
NSHOTS = 8192

# step in angles
STEP = 0.1

# =============================================================================
# read in data
# =============================================================================

# base path
fpath = "./ibmq-results/"

# date of results
date = "thurs-11-15/"

# all filenames
fnames = ["distribution{}{}.csv".format(x, y) for x in range(7) for y in range(10)][:-7]

# list to store data
data = []

# read in data
for fname in fnames:
    cur = pd.read_csv(fpath + date + fname)
    vals = cur.values
    data.append([vals[0, 1], vals[1, 1]])
    
# grab simulated data
sdata = np.loadtxt("sim-data.txt")
    
# =============================================================================
# process data
# =============================================================================

xs = np.arange(0, 2 * np.pi, STEP)
costs = np.array([s[1] / NSHOTS for s in data])

# =============================================================================
# plot the cost
# =============================================================================

plt.plot(xs, costs, "-o", label="ibmqx4")
plt.plot(sdata[:, 0], sdata[:, 1], "-o", label="simulator")

plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                   ["0", "pi / 2", "pi", "3 pi / 2", "2 pi"])
plt.title("2x2 Linear System", weight="bold", fontsize=14)
plt.ylabel("Cost (Local)", fontsize=12)
plt.xlabel("Ansatz Angle", fontsize=12)
plt.legend()
plt.grid()