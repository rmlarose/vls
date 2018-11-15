"""
Main script for running vls with pauli operators.

written by Ryan LaRose <rlarose@umich.edu>
at LANL 11-14-2018
"""

# =============================================================================
# imports
# =============================================================================

from time import time

import numpy as np

from vls_pauli import PauliSystem

# =============================================================================
# constants
# =============================================================================

# pauli operators making up A. each row corresponds to a term of paulis
Amat_ops = np.array([["X", "Z", "Z", "Y"],
                     ["Y", "I", "X", "Z"],
                     ["Z", "X", "Y", "Y"]])

#Amat_ops = np.array([["X", "Z", "Z", "Y"],
#                     ["Y", "I", "X", "Z"]])

# coefficients multiplying the terms of the pauli operators in A
Amat_coeffs = np.array([1. +0.3j, -0.4 - 1j, 2. + 4.2j])

#Amat_coeffs = np.array([1. +0.3j, -0.4 - 1j])

bvec_ops = np.array(["X", "Y", "Z", "X"])

# =============================================================================
# main script
# =============================================================================

# set a random seed
np.random.seed(seed=100)

# get a pauli system
system = PauliSystem(Amat_coeffs, Amat_ops, bvec_ops)

# compute it's matrix
print("Matrix of system:\n", system.matrix())

# show the circuit to compute the local cost function for two terms
print("Hadamard test for local cost function circuit:")
print(system.make_hadamard_test_circuit(
        system.ops[0], system.ops[1], 3, "real")
    )

# set some random angles
angles = np.zeros(48)

# compute each expectation and time it
start = time()
cost = system.cost(angles)
print("cost runtime =", time() - start, "seconds")
print(cost)