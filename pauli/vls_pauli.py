"""
Classes and functions for running VLS where the matrix A is assumed to be a 
sum of pauli operators.

written by Ryan LaRose <rlarose@umich.edu>
at LANL 11-14-2018
"""

# =============================================================================
# imports
# =============================================================================

import numpy as np

from cirq import (Circuit, InsertStrategy, LineQubit, ops, Symbol)
from cirq.ops.controlled_gate import ControlledGate

# =============================================================================
# classes
# =============================================================================

class PauliSystem():
    """PauliSystem class.
    
    """
    # =========================================================================
    # matrices of pauli operators
    # =========================================================================
    
    # identity
    imat = np.array([[1., 0.],
                     [0., 1.]], dtype=np.complex64)

    # pauli x
    xmat = np.array([[0., 1.],
                     [1., 0.]], dtype=np.complex64)

    # pauli y
    ymat = np.array([[0., -1j],
                     [1j, 0.]], dtype=np.complex64)

    # pauli z
    zmat = np.array([[1., 0.],
                     [0., -1.]], dtype=np.complex64)

    # =========================================================================
    # basic methods
    # =========================================================================

    def __init__(self, coeffs, ops, vec_ops):
        # check inputs
        assert len(coeffs) == ops.shape[0]
        assert len(vec_ops) == ops.shape[1]

        # attributes
        self.coeffs = coeffs
        self.ops = ops
        self.vec_ops = vec_ops
        
        self.ansatz = Circuit()

    def num_qubits(self):
        """Returns the number of qubits the PauliMatrix acts on."""
        return self.ops.shape[1]
    
    def size(self):
        """Returns the dimensions of the PauliMatrix."""
        dim = 2**(self.num_qubits())
        return (dim, dim)

    def num_elements(self):
        """Returns the number of elements in the PauliMatrix."""
        return 2**(2 * self.num_qubits())

    # =========================================================================
    # methods for the matrix of the Pauli System
    # =========================================================================
    
    def matrix(self):
        """Returns a matrix representation of the matrix of the PauliSystem."""
        # allocate space
        mat = np.zeros(self.size(), dtype=np.complex64)
        
        # loop over all terms in operators
        for (ind, op_list) in enumerate(self.ops):
            
            # form the first kronecker product
            if len(op_list) == 1:
                term = self._key_to_mat(op_list[0])
            else:
                term = np.kron(self._key_to_mat(op_list[0]),
                               self._key_to_mat(op_list[1]))
            
            # form all other terms in kronecker product
            for k in range(2, len(op_list)):
                key = op_list[k]
                term = np.kron(term, self._key_to_mat(key))
            
            # add the term to the matrix
            mat += self.coeffs[ind] * term
            
        return mat

    def _key_to_mat(self, key:str):
        """Returns a pauli matrix corresponding to a string key."""
        key_mat = {"I": self.imat,
                   "X": self.xmat,
                   "Y": self.ymat,
                   "Z": self.zmat}

        return key_mat[key]

    # =========================================================================
    # methods for creating circuits
    # =========================================================================

    def _key_to_gate(self, key:str):
        """Returns a gate corresponding to a string key."""
        key_mat = {"X": ops.X,
                   "Y": ops.Y,
                   "Z": ops.Z}
    
        return key_mat[key]

    def _key_to_cgate(self, key:str):
        """Returns a controlled-gate corresponding to a string key."""
        key_mat = {"X": ControlledGate(ops.X),
                   "Y": ControlledGate(ops.Y),
                   "Z": ControlledGate(ops.Z)}

        return key_mat[key]

    def make_op_list_circuit(self, op_list):
        """Returns a quantum circuit implementing a list of Pauli ops."""
        # get a circuit
        circ = Circuit()

        # get some qubits
        qbits = [LineQubit(x) for x in range(self.num_qubits() + 1)]

        for (q, key) in enumerate(op_list):
            if key == "I": continue
            q += 1
            circ.append(
                self._key_to_gate(key)(qbits[q]),
                strategy=InsertStrategy.EARLIEST
                )

        return circ

    def make_controlled_op_list_circuit(self, op_list):
        """Returns a quantum circuit implementing a list of controlled Pauli
        operations.
        """
        # get a circuit
        circ = Circuit()

        # get some qubits
        qbits = [LineQubit(x) for x in range(self.num_qubits() + 1)]

        for (q, key) in enumerate(op_list):
            if key == "I": continue
            q += 1
            circ.append(
                self._key_to_cgate(key)(qbits[0], qbits[q]),
                strategy=InsertStrategy.EARLIEST
                )

        return circ

    def make_matrix_circuit(self):
        """Returns a quantum circuit implementing the matrix of the
        PauliSystem.
        """
        # get a circuit
        circ = Circuit()

        # loop over each term in the matrix expansion
        for op_list in self.ops:
            circ += self.make_op_list_circuit(op_list)

        return circ

    def make_controlled_matrix_circuit(self):
        """Returns a quantum circuit implementing the matrix of the
        PauliSystem.
        """
        # get a circuit
        circ = Circuit()

        # loop over each term in the matrix expansion
        for op_list in self.ops:
            circ += self.make_controlled_op_list_circuit(op_list)

        return circ

    def make_vector_circuit(self):
        """Returns a quantum circuit implementing the unitary U that prepares
        the solution vector b from the ground state.
        
        That is, |b> = U|0>.
        """
        # get a circuit
        circ = Circuit()

        # get some qubits
        qbits = [LineQubit(x) for x in range(self.num_qubits() + 1)]

        # loop over each pauli operator
        for (q, key) in enumerate(self.vec_ops):
            if key == "I": continue
            q += 1
            circ.append(
                self._key_to_gate(key)(qbits[q]),
                strategy=InsertStrategy.EARLIEST
                )

        return circ

    def make_controlled_vector_circuit(self):
        """Returns a quantum circuit implementing the controlled unitary U 
        that prepares the solution vector b from the ground state.
        
        That is, |b> = U|0>.
        """
        # get a circuit
        circ = Circuit()
        
        # get some qubits
        qbits = [LineQubit(x) for x in range(self.num_qubits() + 1)]
        
        # loop over each pauli operator
        for (q, key) in enumerate(self.vec_ops):
            if key == "I": continue
            q += 1
            circ.append(
                self._key_to_cgate(key)(qbits[0], qbits[q]),
                strategy=InsertStrategy.EARLIEST
                )

        return circ

    # =========================================================================
    # methods for circuit ansatze
    # =========================================================================

    def make_ansatz_circuit(self):
        self.get_symbol_list_for_layer()
        params = self._reshape_list_to_layer_format(self.symbol_list)
        sparams = self._reshape_list_to_layer_format(self.symbol_list_shifted)
        self.layer(params, sparams)

    def layer(self, params, shifted_params):
        """Implements a single layer of the diagonalizing unitary.

        input:
            params [type: list<list<float>>]
                parameters for the first layer of gates.
                len(params) must be n // 2 where n is the number of qubits
                in the state and // indicates floor division.

                the format of params is as follows:

                params = [rotations for gates in layer]

                where the rotations for the gates in the layer have the form

                rotations for gates in layer =
                    [x1, y1, z1],
                    [x2, y2, z2],
                    [x3, y3, z3],
                    [x4, y4, z4].

                Note that each gate consists of 12 parameters. 3 parameters
                for each rotation and 4 total rotations.

                The general form for a gate, which acts on two qubits,
                is shown below:

                    ----------------------------------------------------------
                    | --Rx(x1)--Ry(y1)--Rz(z1)--@--Rx(x3)--Ry(y3)--Rz(z3)--@ |
                G = |                           |                          | |
                    | --Rx(x2)--Ry(y2)--Rz(z2)--X--Rx(x4)--Ry(y4)--Rz(z4)--X |
                    ----------------------------------------------------------

            shifted_params [type: ]
                TODO: figure this out
                parameters for the second shifted layer of gates

            copy [type: int (0 or 1)]
                the copy of the state to apply the layer to

        modifies:
            self.unitary_circ
                appends the layer of operations to self.unitary_circ
        """        
        # for brevity
        n = self.num_qubits()

        # get some qubits
        qbits = [LineQubit(x) for x in range(n + 1)]

        # =====================================================================
        # helper functions for layer
        # =====================================================================

        def gate(qubits, params):
            """Helper function to append the two qubit gate
            ("G" in the VQSD paper figure).

            input:
                qubits [type: list<Qubits>]
                    qubits to be acted on. must have length 2.

                params [type: list<list<angles>>]
                    the parameters of the rotations in the gate.
                    len(params) must be equal to 12: 4 arbitrary rotations x
                    3 angles per arbitrary rotation.

                    the format of params must be

                    [[x1, y1, z1],
                     [x2, y2, z2],
                     [x3, y3, z3],
                     [x4, y4, z4]].

                    the general form of a gate, which acts on two qubits,
                    is shown below:

                    ----------------------------------------------------------
                    | --Rx(x1)--Ry(y1)--Rz(z1)--@--Rx(x3)--Ry(y3)--Rz(z3)--@ |
                G = |                           |                          | |
                    | --Rx(x2)--Ry(y2)--Rz(z2)--X--Rx(x4)--Ry(y4)--Rz(z4)--X |
                    ----------------------------------------------------------

            modifies:
                self.unitary_circ
                    appends a gate acting on the qubits to the unitary circ.
            """
            # rotation on 'top' qubit
            self.ansatz.append(
                self._rot(qubits[0], params[0]),
                strategy=InsertStrategy.EARLIEST
                )

            # rotation on 'bottom' qubit
            self.ansatz.append(
                self._rot(qubits[1], params[1]),
                strategy=InsertStrategy.EARLIEST
                )

            # cnot from 'top' to 'bottom' qubit
            self.ansatz.append(
                ops.CNOT(qubits[0], qubits[1]),
                strategy=InsertStrategy.EARLIEST
                )

            # second rotation on 'top' qubit
            self.ansatz.append(
                self._rot(qubits[0], params[2]),
                strategy=InsertStrategy.EARLIEST
                )

            # second rotation on 'bottom' qubit
            self.ansatz.append(
                self._rot(qubits[1], params[3]),
                strategy=InsertStrategy.EARLIEST
                )

            # second cnot from 'top' to 'bottom' qubit
            self.ansatz.append(
                ops.CNOT(qubits[0], qubits[1]),
                strategy=InsertStrategy.EARLIEST
                )

        # helper function for indexing loops
        stop = lambda n: n - 1 if n % 2 == 1 else n

        # =====================================================================
        # implement the layer
        # =====================================================================

        # TODO: speedup. combine two loops into one

        # unshifted gates on adjacent qubit pairs
        for ii in range(0, stop(n), 2):
            gate(qbits[ii : ii + 2], params[ii // 2])

        # shifted gates on adjacent qubits
        if n > 2:
            for ii in range(1, n, 2):
                gate([qbits[ii],
                      qbits[(ii + 1) % n]],
                     shifted_params[ii // 2])

    def _rot(self, qubit, params):
        """Helper function that returns an arbitrary rotation of the form
        R = Rz(params[2]) * Ry(params[1]) * Rx(params[0])
        on the qubit, e.g. R |qubit>.

        Note that order is reversed when put into the circuit. The circuit is:
        |qubit>---Rx(params[0])---Ry(params[1])---Rz(params[2])---
        """
        rx = ops.RotXGate(half_turns=params[0])
        ry = ops.RotYGate(half_turns=params[1])
        rz = ops.RotZGate(half_turns=params[2])

        yield (rx(qubit), ry(qubit), rz(qubit))

    def get_symbol_list_for_layer(self):
        """Creates a list of Symbol's for the circuit ansatz."""
        ind = 12 * (self.num_qubits() // 2)
        self.symbol_list = np.array(
            [Symbol(str(ii)) for ii in range(0, ind)]
            )
        self.symbol_list_shifted = np.array(
            [Symbol(str(ii)) for ii in range(ind, 2 * ind)]
            )

    def _reshape_list_to_layer_format(self, symlist):
        """Helper function that converts a linear array of angles (used to call
        the optimize function) into the format required by a layer.
        """
        return symlist.reshape(self.num_qubits() // 2, 4, 3)

    # =========================================================================
    # methods for computing the cost
    # =========================================================================
    
    def make_hadamard_test_circuit(self, ops1, ops2, j, mode):
        """Returns a Hadamard test circuit for the local cost function."""
        # get a circuit
        circ = Circuit()
        qbits = [LineQubit(x) for x in range(self.num_qubits() + 1)]
        
        # add hadamard gate on top register
        circ.append(
            ops.H(qbits[0]),
            strategy=InsertStrategy.EARLIEST
            )
        
        # add ansatz on bottom register
        self.make_ansatz_circuit()
        circ += self.ansatz
        
        # add controlled sigma_k term (corresponding to ops1)
        circ += self.make_controlled_op_list_circuit(ops1)
        
        # add u dagger term
        # TODO: this only adds U, make sure it adds U\dagger
        circ += self.make_vector_circuit()
        
        # add controlled sigma_z on the jth qubit
        circ.append(
            ops.CZ(qbits[0], qbits[j + 1]),
            strategy=InsertStrategy.EARLIEST
            )
        
        # add u term
        circ += self.make_vector_circuit()
        
        # add controlled sigma_kprime term (corresponding to ops2)
        # add controlled sigma_k term (corresponding to ops1)
        circ += self.make_controlled_op_list_circuit(ops2)
        
        # optional s gate for imag part
        if mode == "imag" or mode != "real":
            circ.append(
                ops.S(qbits[0]),
                strategy=InsertStrategy.EARLIEST
                )
        
        # add hadamard gate on top register
        circ.append(
            ops.H(qbits[0]),
            strategy=InsertStrategy.EARLIEST
            )
        
        # add measurement to top qubit
        circ.append(
            ops.measure(qbits[0]),
            strategy=InsertStrategy.EARLIEST
            )
        return circ
    
    def run_hadamard_test(self, ops1, ops2, j, mode):
        """Returns the real or imaginary part of the term
        
        <Q_{k, kprime}^{j}> := <0|V\dag A_k\dag U P_j U\dag A_kprime V|0\>
        
        using the Hadmard test.
        
        Args:
            ops1 [type: list<str>]
                first list of pauli operators (\sigma_k term in notes)
            
            ops2 [type: list<str>]
                second list of pauli operators (\sigma_kprime in notes)
            
            j [type: int]
                which qubit to perform controlled-Z on (between 1 and n)
            
            mode [type: str]
                "real" or "imag"
                "real" mode computes real part of <Q_{k, kprime}^{j}>
                
                "imag" mode computes imag part of <Q_{k, kprime}^{j}>
        
        Returns:
            rtype: complex
            real/imag part of expectation value <Q_{k, kprime}^{j}>
        """
        # get a hadmard test circuit
        
        # run it
        
