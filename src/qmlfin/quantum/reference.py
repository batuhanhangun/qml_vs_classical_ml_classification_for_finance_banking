"""PennyLane versions of the circuits, used to validate the simulator and to count resources."""

import numpy as np
import pennylane as qml


def zz_feature_map(x, reps):
    n = len(x)
    for _ in range(reps):
        for i in range(n):
            qml.Hadamard(i)
            qml.PhaseShift(2 * x[i], wires=i)
        for i in range(n):
            for j in range(i + 1, n):
                qml.CNOT([i, j])
                qml.PhaseShift(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT([i, j])


def variational(x, weights):
    n = len(x)
    qml.AngleEmbedding(x, wires=range(n), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(n))


def resources(circuit, *args, n_qubits):
    """Circuit depth and two-qubit gate count after decomposing templates into elementary gates."""

    @qml.qnode(qml.device("default.qubit", wires=n_qubits))
    def qnode():
        circuit(*args)
        return qml.state()

    tape = qml.workflow.construct_tape(qnode, level="device")()
    ops = tape.operations
    depth = qml.tape.QuantumScript(ops).graph.get_depth()
    return {
        "n_qubits": n_qubits,
        "circuit_depth": int(depth),
        "two_qubit_gates": int(sum(len(op.wires) == 2 for op in ops)),
    }
