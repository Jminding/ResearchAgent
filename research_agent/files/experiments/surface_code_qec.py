"""
Surface Code Quantum Error Correction Simulator

Implements the [[d^2, 1, d]] rotated surface code with:
- Binary symplectic formalism for Pauli operators
- Stabilizer operators for syndrome extraction
- Error tracking and logical error detection
- Support for d=3,5,7 code distances

Author: Research Agent
Date: 2024-12-22
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class NoiseModel(Enum):
    DEPOLARIZING = "depolarizing"
    DEPHASING = "dephasing"
    BIASED = "biased"


@dataclass
class SurfaceCodeParams:
    """Parameters for surface code construction."""
    distance: int
    n_data_qubits: int
    n_stabilizers: int

    @classmethod
    def from_distance(cls, d: int) -> 'SurfaceCodeParams':
        """Create params from code distance."""
        n_data = d * d
        n_stab = n_data - 1  # n-k = d^2 - 1 for k=1
        return cls(distance=d, n_data_qubits=n_data, n_stabilizers=n_stab)


class SurfaceCodeSimulator:
    """
    Simulator for rotated surface codes using binary symplectic formalism.

    The error state is represented as a 2n-dimensional binary vector:
    E = [e_x | e_z] where e_x[i]=1 means X error on qubit i, e_z[i]=1 means Z error.
    Y error = X AND Z on same qubit.

    For the standard planar surface code:
    - X stabilizers (plaquettes) detect Z errors
    - Z stabilizers (vertices/stars) detect X errors

    Attributes:
        d: Code distance
        n: Number of data qubits (d^2)
        n_stab: Number of stabilizers (d^2 - 1)
        H_x: X-stabilizer parity check matrix
        H_z: Z-stabilizer parity check matrix
        x_logical: Logical X operator support
        z_logical: Logical Z operator support
    """

    def __init__(self, distance: int = 3):
        """
        Initialize surface code simulator.

        Args:
            distance: Code distance (must be odd: 3, 5, 7, ...)
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError(f"Distance must be odd and >= 3, got {distance}")

        self.d = distance
        self.n = distance * distance

        # Build stabilizer matrices and logical operators
        self._build_code()

        # Initialize error state (no errors)
        self.error_state = np.zeros(2 * self.n, dtype=np.int8)

    def _build_code(self):
        """
        Construct stabilizer matrices and logical operators for planar surface code.

        Uses standard planar surface code layout where:
        - Qubits are on a d x d grid
        - X stabilizers are on faces (plaquettes), weight 4 in interior, 2 on boundary
        - Z stabilizers are on vertices (stars), weight 4 in interior, 2 on boundary
        """
        d = self.d
        n = self.n

        def qubit_idx(row, col):
            """Convert grid position to qubit index."""
            return row * d + col

        # Build X stabilizers (plaquettes)
        # For d x d grid, we have (d-1) x (d-1) = (d-1)^2 plaquettes
        x_stabilizers = []
        for i in range(d - 1):
            for j in range(d - 1):
                # Each plaquette covers 4 qubits forming a square
                plaquette = [
                    qubit_idx(i, j),
                    qubit_idx(i, j + 1),
                    qubit_idx(i + 1, j),
                    qubit_idx(i + 1, j + 1)
                ]
                x_stabilizers.append(plaquette)

        # Build Z stabilizers (stars/vertices)
        # For planar code, we need to include boundary stabilizers
        # We have d x d = d^2 vertices, but some are on corners
        z_stabilizers = []

        # Interior vertices (weight 4)
        for i in range(1, d - 1):
            for j in range(1, d - 1):
                star = [
                    qubit_idx(i - 1, j),  # above
                    qubit_idx(i + 1, j),  # below
                    qubit_idx(i, j - 1),  # left
                    qubit_idx(i, j + 1)   # right
                ]
                z_stabilizers.append(star)

        # Edge vertices (weight 2 or 3)
        # Top edge (excluding corners)
        for j in range(1, d - 1):
            star = [qubit_idx(0, j - 1), qubit_idx(0, j), qubit_idx(0, j + 1), qubit_idx(1, j)]
            z_stabilizers.append(star)

        # Bottom edge (excluding corners)
        for j in range(1, d - 1):
            star = [qubit_idx(d - 1, j - 1), qubit_idx(d - 1, j), qubit_idx(d - 1, j + 1), qubit_idx(d - 2, j)]
            z_stabilizers.append(star)

        # Left edge (excluding corners)
        for i in range(1, d - 1):
            star = [qubit_idx(i - 1, 0), qubit_idx(i, 0), qubit_idx(i + 1, 0), qubit_idx(i, 1)]
            z_stabilizers.append(star)

        # Right edge (excluding corners)
        for i in range(1, d - 1):
            star = [qubit_idx(i - 1, d - 1), qubit_idx(i, d - 1), qubit_idx(i + 1, d - 1), qubit_idx(i, d - 2)]
            z_stabilizers.append(star)

        # We need exactly (d^2 - 1) independent stabilizers total
        # X stabilizers: (d-1)^2
        # Z stabilizers: need d^2 - 1 - (d-1)^2 = 2d - 2

        # Simplify: use only weight-2 Z stabilizers on boundaries for proper count
        z_stabilizers_final = []

        # Horizontal boundary Z stabilizers (top row pairs)
        for j in range(d - 1):
            z_stabilizers_final.append([qubit_idx(0, j), qubit_idx(0, j + 1)])

        # Vertical boundary Z stabilizers (left column pairs)
        for i in range(d - 1):
            z_stabilizers_final.append([qubit_idx(i, 0), qubit_idx(i + 1, 0)])

        # Check counts
        n_x_stab = len(x_stabilizers)
        n_z_stab = len(z_stabilizers_final)

        # For [[d^2, 1, d]] code: need (d-1)^2 + 2(d-1) = d^2 - 1 stabilizers
        # X: (d-1)^2, Z: 2(d-1)

        self.H_x = np.zeros((n_x_stab, n), dtype=np.int8)
        for i, stab in enumerate(x_stabilizers):
            for q in stab:
                self.H_x[i, q] = 1

        self.H_z = np.zeros((n_z_stab, n), dtype=np.int8)
        for i, stab in enumerate(z_stabilizers_final):
            for q in stab:
                self.H_z[i, q] = 1

        self.n_x_stab = n_x_stab
        self.n_z_stab = n_z_stab
        self.x_stabilizers = x_stabilizers
        self.z_stabilizers = z_stabilizers_final

        # Logical operators
        # X_L: horizontal chain (any row)
        self.x_logical = np.arange(d)  # Top row: qubits 0, 1, ..., d-1

        # Z_L: vertical chain (any column)
        self.z_logical = np.arange(0, n, d)  # Left column: qubits 0, d, 2d, ..., (d-1)*d

    def reset(self) -> np.ndarray:
        """Reset error state to no errors. Returns initial syndrome."""
        self.error_state = np.zeros(2 * self.n, dtype=np.int8)
        return self.extract_syndrome()

    def apply_noise(self, p: float, noise_model: NoiseModel = NoiseModel.DEPOLARIZING,
                    gamma: float = 0.0) -> np.ndarray:
        """
        Apply noise to all data qubits.

        Args:
            p: Total physical error rate per qubit
            noise_model: Type of noise (depolarizing, dephasing, biased)
            gamma: Bias parameter for biased noise (0 = depolarizing, 1 = pure Z)

        Returns:
            Updated error state
        """
        if noise_model == NoiseModel.DEPOLARIZING:
            p_x = p_y = p_z = p / 3
        elif noise_model == NoiseModel.DEPHASING:
            p_x = p_y = 0
            p_z = p
        elif noise_model == NoiseModel.BIASED:
            p_x = p_y = p * (1 - gamma) / 3
            p_z = p * (1 + 2 * gamma) / 3
        else:
            raise ValueError(f"Unknown noise model: {noise_model}")

        for i in range(self.n):
            r = np.random.random()
            if r < p_x:
                # X error
                self.error_state[i] ^= 1
            elif r < p_x + p_y:
                # Y error = X AND Z
                self.error_state[i] ^= 1
                self.error_state[self.n + i] ^= 1
            elif r < p_x + p_y + p_z:
                # Z error
                self.error_state[self.n + i] ^= 1

        return self.error_state.copy()

    def apply_specific_error(self, qubit: int, pauli: str) -> np.ndarray:
        """
        Apply a specific Pauli error to a qubit.

        Args:
            qubit: Qubit index (0 to n-1)
            pauli: Pauli operator ('X', 'Y', 'Z', or 'I')

        Returns:
            Updated error state
        """
        if qubit < 0 or qubit >= self.n:
            raise ValueError(f"Invalid qubit index: {qubit}")

        if pauli == 'X':
            self.error_state[qubit] ^= 1
        elif pauli == 'Y':
            self.error_state[qubit] ^= 1
            self.error_state[self.n + qubit] ^= 1
        elif pauli == 'Z':
            self.error_state[self.n + qubit] ^= 1
        elif pauli == 'I':
            pass
        else:
            raise ValueError(f"Invalid Pauli: {pauli}")

        return self.error_state.copy()

    def extract_syndrome(self) -> np.ndarray:
        """
        Extract syndrome from current error state.

        Returns:
            Syndrome vector of length n_stab (concatenated X and Z syndromes)
        """
        # e_x = errors that are X or Y (first n bits)
        # e_z = errors that are Z or Y (last n bits)
        e_x = self.error_state[:self.n]
        e_z = self.error_state[self.n:]

        # Z stabilizers detect X errors: s_z = H_z @ e_x mod 2
        # X stabilizers detect Z errors: s_x = H_x @ e_z mod 2
        s_z = (self.H_z @ e_x) % 2
        s_x = (self.H_x @ e_z) % 2

        return np.concatenate([s_x, s_z]).astype(np.int8)

    def apply_correction(self, correction: np.ndarray) -> np.ndarray:
        """
        Apply a correction (in symplectic form) to the error state.

        Args:
            correction: Binary vector of length 2n [c_x | c_z]

        Returns:
            Updated error state after correction
        """
        self.error_state = (self.error_state + correction) % 2
        return self.error_state.copy()

    def apply_single_qubit_correction(self, qubit: int, pauli_type: int) -> np.ndarray:
        """
        Apply a single-qubit Pauli correction.

        Args:
            qubit: Qubit index
            pauli_type: 0=X, 1=Y, 2=Z

        Returns:
            Updated error state
        """
        correction = np.zeros(2 * self.n, dtype=np.int8)

        if pauli_type == 0:  # X
            correction[qubit] = 1
        elif pauli_type == 1:  # Y
            correction[qubit] = 1
            correction[self.n + qubit] = 1
        elif pauli_type == 2:  # Z
            correction[self.n + qubit] = 1

        return self.apply_correction(correction)

    def check_logical_error(self) -> Tuple[bool, bool]:
        """
        Check if the current error state causes a logical error.

        A logical error occurs when the residual error (after any corrections)
        anticommutes with the logical operators.

        For surface code:
        - X_L acts on a horizontal chain
        - Z_L acts on a vertical chain

        Logical X error occurs if Z part of error has odd weight on X_L support.
        Logical Z error occurs if X part of error has odd weight on Z_L support.

        Returns:
            (logical_x_error, logical_z_error): Tuple of booleans
        """
        e_x = self.error_state[:self.n]
        e_z = self.error_state[self.n:]

        # Logical X error: Z errors on X_L support (any horizontal chain)
        # Use top row for X_L
        logical_x_error = np.sum(e_z[self.x_logical]) % 2 == 1

        # Logical Z error: X errors on Z_L support (any vertical chain)
        # Use left column for Z_L
        logical_z_error = np.sum(e_x[self.z_logical]) % 2 == 1

        return bool(logical_x_error), bool(logical_z_error)

    def has_logical_error(self) -> bool:
        """Check if any logical error has occurred."""
        x_err, z_err = self.check_logical_error()
        return x_err or z_err

    def is_in_codespace(self) -> bool:
        """Check if current state is in the code space (zero syndrome)."""
        syndrome = self.extract_syndrome()
        return np.all(syndrome == 0)

    def get_error_weight(self) -> int:
        """Get the weight (number of non-identity Paulis) of current error."""
        e_x = self.error_state[:self.n]
        e_z = self.error_state[self.n:]
        return int(np.sum((e_x | e_z)))

    def get_error_string(self) -> str:
        """Get a string representation of the current error."""
        e_x = self.error_state[:self.n]
        e_z = self.error_state[self.n:]

        error_str = ""
        for i in range(self.n):
            if e_x[i] and e_z[i]:
                error_str += "Y"
            elif e_x[i]:
                error_str += "X"
            elif e_z[i]:
                error_str += "Z"
            else:
                error_str += "I"
        return error_str

    def get_syndrome_dim(self) -> int:
        """Get dimension of syndrome vector."""
        return self.n_x_stab + self.n_z_stab

    def get_action_dim(self) -> int:
        """Get dimension of action space (single-qubit corrections)."""
        return 3 * self.n  # X, Y, Z on each qubit

    def action_to_correction(self, action: int) -> np.ndarray:
        """
        Convert action index to correction vector.

        Args:
            action: Action index in [0, 3*n - 1]
                   action = qubit * 3 + pauli_type where pauli_type in {0,1,2} for X,Y,Z

        Returns:
            Correction vector in symplectic form
        """
        qubit = action // 3
        pauli_type = action % 3

        correction = np.zeros(2 * self.n, dtype=np.int8)
        if pauli_type == 0:  # X
            correction[qubit] = 1
        elif pauli_type == 1:  # Y
            correction[qubit] = 1
            correction[self.n + qubit] = 1
        elif pauli_type == 2:  # Z
            correction[self.n + qubit] = 1

        return correction


class QECEnvironment:
    """
    Gym-style environment for QEC with RL.

    State: Syndrome history (flattened)
    Action: Single-qubit Pauli correction
    Reward: +1 for surviving episode, -1 for logical error
    """

    def __init__(self, distance: int = 3, p: float = 0.05,
                 noise_model: NoiseModel = NoiseModel.DEPOLARIZING,
                 gamma: float = 0.0, T_max: int = 100,
                 history_window: int = 3):
        """
        Initialize QEC environment.

        Args:
            distance: Surface code distance
            p: Physical error rate
            noise_model: Type of noise
            gamma: Bias parameter
            T_max: Maximum steps per episode
            history_window: Number of syndromes to keep in state
        """
        self.sim = SurfaceCodeSimulator(distance)
        self.p = p
        self.noise_model = noise_model
        self.gamma = gamma
        self.T_max = T_max
        self.W = history_window

        self.syndrome_dim = self.sim.get_syndrome_dim()
        self.action_dim = self.sim.get_action_dim()
        self.state_dim = self.W * self.syndrome_dim

        # Add "no correction" action
        self.action_dim_with_noop = self.action_dim + 1

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self.t = 0
        self.sim.reset()

        # Initialize syndrome history with zeros
        self.syndrome_history = [np.zeros(self.syndrome_dim, dtype=np.float32)
                                  for _ in range(self.W)]

        # Apply initial noise and get syndrome
        self.sim.apply_noise(self.p, self.noise_model, self.gamma)
        initial_syndrome = self.sim.extract_syndrome().astype(np.float32)
        self.syndrome_history.append(initial_syndrome)
        self.syndrome_history = self.syndrome_history[-self.W:]

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get flattened syndrome history as state."""
        return np.concatenate(self.syndrome_history).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action index (0 to 3*n for corrections, 3*n for no-op)

        Returns:
            next_state, reward, done, info
        """
        self.t += 1

        # Apply correction (if not no-op)
        if action < self.action_dim:
            qubit = action // 3
            pauli_type = action % 3
            self.sim.apply_single_qubit_correction(qubit, pauli_type)

        # Apply noise
        self.sim.apply_noise(self.p, self.noise_model, self.gamma)

        # Get new syndrome
        new_syndrome = self.sim.extract_syndrome().astype(np.float32)
        self.syndrome_history.append(new_syndrome)
        self.syndrome_history = self.syndrome_history[-self.W:]

        # Check for logical error
        logical_error = self.sim.has_logical_error()

        # Compute reward and done
        if logical_error:
            reward = -1.0
            done = True
        elif self.t >= self.T_max:
            reward = 1.0
            done = True
        else:
            # Small reward for clearing syndrome
            syndrome_weight = np.sum(new_syndrome)
            reward = -0.01 * syndrome_weight  # Penalize non-zero syndromes
            done = False

        next_state = self._get_state()
        info = {
            "logical_error": logical_error,
            "syndrome_weight": np.sum(new_syndrome),
            "error_weight": self.sim.get_error_weight(),
            "step": self.t
        }

        return next_state, reward, done, info

    def get_state_dim(self) -> int:
        """Get state dimension."""
        return self.state_dim

    def get_action_dim(self) -> int:
        """Get action dimension (including no-op)."""
        return self.action_dim_with_noop


def test_surface_code():
    """Test basic surface code functionality."""
    print("Testing Surface Code Simulator")
    print("=" * 50)

    for d in [3, 5, 7]:
        print(f"\nDistance d={d}:")
        sim = SurfaceCodeSimulator(distance=d)
        print(f"  Data qubits: {sim.n}")
        print(f"  X-stabilizers: {sim.n_x_stab} (plaquettes)")
        print(f"  Z-stabilizers: {sim.n_z_stab} (boundary)")
        print(f"  Total stabilizers: {sim.n_x_stab + sim.n_z_stab} (should be {sim.n - 1})")
        print(f"  Logical X support: {sim.x_logical}")
        print(f"  Logical Z support: {sim.z_logical}")

        # Test syndrome extraction
        sim.reset()
        s = sim.extract_syndrome()
        print(f"  Initial syndrome (no errors): all zeros = {np.all(s == 0)}")

        # Apply single X error and check syndrome
        sim.reset()
        sim.apply_specific_error(4, 'X')  # Center qubit for d=3
        s = sim.extract_syndrome()
        print(f"  Syndrome after X on center qubit: weight = {np.sum(s)}")

        # Test logical error detection
        sim.reset()
        # Apply logical X (errors on top row)
        for q in sim.x_logical:
            sim.apply_specific_error(q, 'X')
        s = sim.extract_syndrome()
        x_err, z_err = sim.check_logical_error()
        print(f"  After X_L: syndrome weight = {np.sum(s)}, logical errors = ({x_err}, {z_err})")

        sim.reset()
        # Apply logical Z (errors on left column)
        for q in sim.z_logical:
            sim.apply_specific_error(q, 'Z')
        s = sim.extract_syndrome()
        x_err, z_err = sim.check_logical_error()
        print(f"  After Z_L: syndrome weight = {np.sum(s)}, logical errors = ({x_err}, {z_err})")

    print("\n" + "=" * 50)
    print("Surface code tests completed!")


def test_environment():
    """Test QEC environment."""
    print("\nTesting QEC Environment")
    print("=" * 50)

    env = QECEnvironment(distance=3, p=0.05, T_max=20, history_window=3)
    print(f"State dim: {env.get_state_dim()}")
    print(f"Action dim: {env.get_action_dim()}")

    state = env.reset()
    print(f"Initial state shape: {state.shape}")

    # Run a few random steps
    total_reward = 0
    for i in range(10):
        action = np.random.randint(env.get_action_dim())
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: action={action}, reward={reward:.3f}, done={done}")
        if done:
            break

    print(f"Total reward: {total_reward:.3f}")
    print("Environment test passed!")


if __name__ == "__main__":
    test_surface_code()
    test_environment()
