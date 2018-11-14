"""
Main script for running vls with pauli operators.

written by Ryan LaRose <rlarose@umich.edu>
at LANL 11-14-2018
"""

# =============================================================================
# imports
# =============================================================================

import numpy as np

import cirq

from vls_pauli import PauliSystem

# =============================================================================
# constants
# =============================================================================

# pauli operators making up A. each row corresponds to a term of paulis
Amat_ops = np.array([["X", "Z", "Z", "Y"],
                     ["Y", "I", "X", "Z"],
                     ["Z", "X", "Y", "Y"]])

# coefficients multiplying the terms of the pauli operators in A
Amat_coeffs = np.array([1. +0.3j, -0.4 - 1j, 2. + 4.2j])

bvec_ops = np.array(["X", "X", "Z", "X"])

# =============================================================================
# main script
# =============================================================================

A = PauliSystem(Amat_coeffs, Amat_ops, bvec_ops)

print(A.matrix())


print(A.make_matrix_circuit())
print(A.make_controlled_matrix_circuit())

print(A.make_vector_circuit())