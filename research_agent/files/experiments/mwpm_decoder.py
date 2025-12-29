"""
Minimum Weight Perfect Matching (MWPM) Decoder for Surface Code

Implements MWPM decoder as a baseline for comparison with RL decoder.
Uses NetworkX for graph operations and matching.

Author: Research Agent
Date: 2024-12-22
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import networkx as nx
from collections import defaultdict


class MWPMDecoder:
    """
    MWPM Decoder for surface code.

    Constructs a matching graph from syndrome defects and finds
    minimum weight perfect matching to determine correction.
    """

    def __init__(self, distance: int = 3, p: float = 0.05):
        """
        Initialize MWPM decoder.

        Args:
            distance: Surface code distance
            p: Physical error probability (used for edge weights)
        """
        self.d = distance
        self.n = distance * distance
        self.p = p

        # Build code structure
        self._build_code_structure()

        # Precompute matching graph structure
        self._build_matching_graph()

    def _build_code_structure(self):
        """Build surface code structure similar to simulator."""
        d = self.d

        # Build stabilizers for the code
        if d == 3:
            # X stabilizers (detect Z errors)
            self.x_stabilizers = [
                [0, 1, 3, 4],
                [1, 2, 4, 5],
                [3, 4, 6, 7],
                [4, 5, 7, 8],
            ]
            # Z stabilizers (detect X errors)
            self.z_stabilizers = [
                [0, 1, 3],
                [1, 2, 4, 5],
                [3, 4, 6, 7],
                [5, 7, 8],
            ]
        else:
            self.x_stabilizers, self.z_stabilizers = self._build_stabilizers_general(d)

        self.n_x_stab = len(self.x_stabilizers)
        self.n_z_stab = len(self.z_stabilizers)

        # Build parity check matrices
        self.H_x = np.zeros((self.n_x_stab, self.n), dtype=np.int8)
        for i, stab in enumerate(self.x_stabilizers):
            for q in stab:
                self.H_x[i, q] = 1

        self.H_z = np.zeros((self.n_z_stab, self.n), dtype=np.int8)
        for i, stab in enumerate(self.z_stabilizers):
            for q in stab:
                self.H_z[i, q] = 1

        # Logical operators
        self.x_logical = np.arange(d)  # top row
        self.z_logical = np.arange(0, self.n, d)  # left column

    def _build_stabilizers_general(self, d: int) -> Tuple[List[List[int]], List[List[int]]]:
        """Build stabilizers for general distance."""
        x_stabilizers = []
        z_stabilizers = []

        def qubit_index(row, col):
            return row * d + col

        # X-stabilizers (plaquettes)
        for i in range(d - 1):
            for j in range(d - 1):
                stab = [
                    qubit_index(i, j),
                    qubit_index(i, j + 1),
                    qubit_index(i + 1, j),
                    qubit_index(i + 1, j + 1)
                ]
                x_stabilizers.append(stab)

        # Z-stabilizers (vertices)
        for i in range(d):
            for j in range(d):
                neighbors = []
                if i > 0:
                    neighbors.append(qubit_index(i - 1, j))
                if i < d - 1:
                    neighbors.append(qubit_index(i + 1, j))
                if j > 0:
                    neighbors.append(qubit_index(i, j - 1))
                if j < d - 1:
                    neighbors.append(qubit_index(i, j + 1))
                neighbors.append(qubit_index(i, j))
                if len(neighbors) >= 2:
                    z_stabilizers.append(sorted(list(set(neighbors))))

        # Trim to n-1 stabilizers
        total_needed = d * d - 1
        n_x = len(x_stabilizers)
        n_z = total_needed - n_x
        if n_z > len(z_stabilizers):
            n_z = len(z_stabilizers)
            n_x = total_needed - n_z
            x_stabilizers = x_stabilizers[:n_x]
        else:
            z_stabilizers = z_stabilizers[:n_z]

        return x_stabilizers, z_stabilizers

    def _build_matching_graph(self):
        """
        Build the matching graph structure for MWPM.

        For surface codes, we create separate matching graphs for
        X-type and Z-type errors.
        """
        d = self.d

        # Positions of stabilizers (for computing distances)
        # X stabilizers are at plaquette centers
        self.x_stab_positions = []
        for i in range(d - 1):
            for j in range(d - 1):
                self.x_stab_positions.append((i + 0.5, j + 0.5))

        # Z stabilizers are at vertices
        self.z_stab_positions = []
        for i in range(d):
            for j in range(d):
                self.z_stab_positions.append((i, j))

        # Trim to match actual number of stabilizers
        self.x_stab_positions = self.x_stab_positions[:self.n_x_stab]
        self.z_stab_positions = self.z_stab_positions[:self.n_z_stab]

        # Edge weight function based on error probability
        # Weight = -log(p/(1-p)) * distance (for small p)
        if self.p > 0 and self.p < 1:
            self.log_likelihood_ratio = -np.log(self.p / (1 - self.p))
        else:
            self.log_likelihood_ratio = 1.0

    def _manhattan_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Compute Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _build_defect_graph(self, defect_positions: List[int],
                            stab_positions: List[Tuple[float, float]],
                            include_boundary: bool = True) -> nx.Graph:
        """
        Build graph connecting syndrome defects for matching.

        Args:
            defect_positions: Indices of stabilizers with syndrome = 1
            stab_positions: Positions of stabilizers
            include_boundary: Whether to include virtual boundary nodes

        Returns:
            NetworkX graph with edge weights
        """
        G = nx.Graph()

        # Add real defect nodes
        for i, defect_idx in enumerate(defect_positions):
            G.add_node(i, pos=stab_positions[defect_idx], is_boundary=False)

        # Add boundary nodes if odd number of defects
        n_defects = len(defect_positions)
        if include_boundary and n_defects % 2 == 1:
            # Add a virtual boundary node
            G.add_node(n_defects, pos=(-1, -1), is_boundary=True)

        # Add edges between all pairs of defects
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i = nodes[i]
                node_j = nodes[j]

                if G.nodes[node_i]['is_boundary'] or G.nodes[node_j]['is_boundary']:
                    # Edge to boundary has weight based on distance to nearest boundary
                    real_node = node_i if not G.nodes[node_i]['is_boundary'] else node_j
                    pos = G.nodes[real_node]['pos']
                    # Distance to nearest boundary
                    boundary_dist = min(pos[0], pos[1], self.d - 1 - pos[0], self.d - 1 - pos[1])
                    weight = max(0.1, boundary_dist * self.log_likelihood_ratio)
                else:
                    # Weight based on Manhattan distance
                    pos_i = G.nodes[node_i]['pos']
                    pos_j = G.nodes[node_j]['pos']
                    dist = self._manhattan_distance(pos_i, pos_j)
                    weight = max(0.1, dist * self.log_likelihood_ratio)

                G.add_edge(node_i, node_j, weight=weight)

        return G

    def _find_mwpm(self, G: nx.Graph) -> List[Tuple[int, int]]:
        """
        Find minimum weight perfect matching in graph.

        Args:
            G: NetworkX graph with edge weights

        Returns:
            List of matched node pairs
        """
        if len(G.nodes()) == 0:
            return []

        if len(G.nodes()) == 1:
            # Single defect - should match to boundary
            return []

        if len(G.nodes()) % 2 == 1:
            # Odd number of nodes - shouldn't happen if boundary handling is correct
            return []

        try:
            # Use NetworkX's min_weight_matching
            matching = nx.min_weight_matching(G, weight='weight')
            return list(matching)
        except Exception:
            # Fallback: greedy matching
            return self._greedy_matching(G)

    def _greedy_matching(self, G: nx.Graph) -> List[Tuple[int, int]]:
        """Simple greedy matching as fallback."""
        matching = []
        unmatched = set(G.nodes())

        while len(unmatched) >= 2:
            # Find minimum weight edge among unmatched nodes
            min_weight = float('inf')
            best_edge = None

            for u in unmatched:
                for v in unmatched:
                    if u < v and G.has_edge(u, v):
                        weight = G[u][v]['weight']
                        if weight < min_weight:
                            min_weight = weight
                            best_edge = (u, v)

            if best_edge is None:
                break

            matching.append(best_edge)
            unmatched.remove(best_edge[0])
            unmatched.remove(best_edge[1])

        return matching

    def _matching_to_correction(self, matching: List[Tuple[int, int]],
                                 defect_positions: List[int],
                                 stab_positions: List[Tuple[float, float]],
                                 is_x_syndrome: bool) -> np.ndarray:
        """
        Convert matching to correction operator.

        Args:
            matching: List of matched pairs
            defect_positions: Original defect indices
            stab_positions: Stabilizer positions
            is_x_syndrome: True if decoding X syndrome (Z errors), False for Z syndrome (X errors)

        Returns:
            Correction vector in symplectic form
        """
        correction = np.zeros(2 * self.n, dtype=np.int8)
        d = self.d

        def qubit_index(row, col):
            return int(row) * d + int(col)

        for pair in matching:
            node_i, node_j = pair

            # Skip if either node is beyond the defect list (boundary node)
            if node_i >= len(defect_positions) or node_j >= len(defect_positions):
                # Match to boundary
                real_node = node_i if node_i < len(defect_positions) else node_j
                if real_node >= len(defect_positions):
                    continue

                # Find path from defect to nearest boundary
                defect_idx = defect_positions[real_node]
                if defect_idx < len(stab_positions):
                    pos = stab_positions[defect_idx]
                    # Apply correction along shortest path to boundary
                    row, col = int(pos[0] + 0.5), int(pos[1] + 0.5)
                    row = min(max(0, row), d - 1)
                    col = min(max(0, col), d - 1)
                    qubit = qubit_index(row, col)
                    if qubit < self.n:
                        if is_x_syndrome:
                            correction[self.n + qubit] = 1  # Z correction
                        else:
                            correction[qubit] = 1  # X correction
                continue

            # Get positions
            pos_i = stab_positions[defect_positions[node_i]]
            pos_j = stab_positions[defect_positions[node_j]]

            # Find qubits along shortest path between defects
            # Simplified: just apply correction on qubits in the region
            row_i, col_i = int(pos_i[0] + 0.5), int(pos_i[1] + 0.5)
            row_j, col_j = int(pos_j[0] + 0.5), int(pos_j[1] + 0.5)

            row_i = min(max(0, row_i), d - 1)
            col_i = min(max(0, col_i), d - 1)
            row_j = min(max(0, row_j), d - 1)
            col_j = min(max(0, col_j), d - 1)

            # Apply correction along path (horizontal then vertical)
            # This is a simplified version - real MWPM traces error chains
            min_row, max_row = min(row_i, row_j), max(row_i, row_j)
            min_col, max_col = min(col_i, col_j), max(col_i, col_j)

            # Apply on connecting qubits
            for r in range(min_row, max_row + 1):
                qubit = qubit_index(r, min_col)
                if qubit < self.n:
                    if is_x_syndrome:
                        correction[self.n + qubit] ^= 1  # Z correction
                    else:
                        correction[qubit] ^= 1  # X correction

            for c in range(min_col + 1, max_col + 1):
                qubit = qubit_index(max_row, c)
                if qubit < self.n:
                    if is_x_syndrome:
                        correction[self.n + qubit] ^= 1
                    else:
                        correction[qubit] ^= 1

        return correction

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode syndrome to produce correction.

        Args:
            syndrome: Binary syndrome vector [s_x | s_z]

        Returns:
            Correction vector in symplectic form [c_x | c_z]
        """
        # Split syndrome into X and Z parts
        s_x = syndrome[:self.n_x_stab]  # X syndrome (detects Z errors)
        s_z = syndrome[self.n_x_stab:]  # Z syndrome (detects X errors)

        correction = np.zeros(2 * self.n, dtype=np.int8)

        # Decode X syndrome (find Z correction)
        x_defects = np.where(s_x == 1)[0].tolist()
        if len(x_defects) > 0:
            G = self._build_defect_graph(x_defects, self.x_stab_positions)
            matching = self._find_mwpm(G)
            z_correction = self._matching_to_correction(
                matching, x_defects, self.x_stab_positions, is_x_syndrome=True
            )
            correction += z_correction

        # Decode Z syndrome (find X correction)
        z_defects = np.where(s_z == 1)[0].tolist()
        if len(z_defects) > 0:
            G = self._build_defect_graph(z_defects, self.z_stab_positions)
            matching = self._find_mwpm(G)
            x_correction = self._matching_to_correction(
                matching, z_defects, self.z_stab_positions, is_x_syndrome=False
            )
            correction += x_correction

        return correction % 2

    def decode_to_action(self, syndrome: np.ndarray) -> int:
        """
        Decode syndrome and return single-qubit action.

        For compatibility with RL interface, returns the first non-trivial
        correction as an action, or no-op if correction is identity.

        Args:
            syndrome: Binary syndrome vector

        Returns:
            Action index (qubit * 3 + pauli_type, or n*3 for no-op)
        """
        correction = self.decode(syndrome)

        # Find first non-zero correction
        for i in range(self.n):
            c_x = correction[i]
            c_z = correction[self.n + i]

            if c_x and c_z:
                return i * 3 + 1  # Y
            elif c_x:
                return i * 3 + 0  # X
            elif c_z:
                return i * 3 + 2  # Z

        # No correction needed
        return self.n * 3  # No-op


class SimpleLookupDecoder:
    """
    Simple lookup table decoder for small codes (d=3).

    Precomputes optimal corrections for all possible syndromes.
    """

    def __init__(self, distance: int = 3):
        """
        Initialize lookup decoder.

        Args:
            distance: Code distance (only d=3 fully supported)
        """
        self.d = distance
        self.n = distance * distance

        if distance == 3:
            self._build_lookup_table()
        else:
            # Fall back to MWPM for larger codes
            self.mwpm = MWPMDecoder(distance)

    def _build_lookup_table(self):
        """Build lookup table for d=3 code."""
        # For [[9,1,3]], syndrome space is 2^8 = 256
        # Precompute optimal single-qubit corrections

        # Stabilizers for d=3
        x_stabs = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]]
        z_stabs = [[0, 1, 3], [1, 2, 4, 5], [3, 4, 6, 7], [5, 7, 8]]

        # Build lookup: syndrome -> correction
        self.lookup = {}

        # For each possible single-qubit error
        for qubit in range(9):
            for pauli_type in range(3):  # X, Y, Z
                # Compute syndrome for this error
                syndrome = [0] * 8

                # X or Y error on qubit
                if pauli_type in [0, 1]:  # X or Y
                    for i, stab in enumerate(z_stabs):
                        if qubit in stab:
                            syndrome[4 + i] ^= 1

                # Z or Y error on qubit
                if pauli_type in [1, 2]:  # Y or Z
                    for i, stab in enumerate(x_stabs):
                        if qubit in stab:
                            syndrome[i] ^= 1

                syndrome_key = tuple(syndrome)
                if syndrome_key not in self.lookup:
                    self.lookup[syndrome_key] = (qubit, pauli_type)

        # Add zero syndrome -> no correction
        self.lookup[tuple([0] * 8)] = None

    def decode_to_action(self, syndrome: np.ndarray) -> int:
        """
        Decode syndrome using lookup table.

        Args:
            syndrome: Binary syndrome vector

        Returns:
            Action index
        """
        if self.d != 3:
            return self.mwpm.decode_to_action(syndrome)

        syndrome_key = tuple(syndrome.astype(int).tolist())

        if syndrome_key in self.lookup:
            result = self.lookup[syndrome_key]
            if result is None:
                return 27  # No-op
            qubit, pauli_type = result
            return qubit * 3 + pauli_type
        else:
            # Unknown syndrome - return no-op
            return 27


def test_mwpm_decoder():
    """Test MWPM decoder functionality."""
    print("Testing MWPM Decoder")
    print("=" * 50)

    for d in [3, 5]:
        print(f"\nDistance d={d}:")
        decoder = MWPMDecoder(distance=d, p=0.05)

        # Test with zero syndrome
        syndrome_dim = decoder.n_x_stab + decoder.n_z_stab
        zero_syndrome = np.zeros(syndrome_dim, dtype=np.int8)
        correction = decoder.decode(zero_syndrome)
        print(f"  Zero syndrome correction weight: {np.sum(correction)}")

        # Test with random syndrome
        for trial in range(3):
            random_syndrome = np.random.randint(0, 2, syndrome_dim, dtype=np.int8)
            # Ensure even parity for each type (required for valid syndrome)
            if np.sum(random_syndrome[:decoder.n_x_stab]) % 2 == 1:
                random_syndrome[0] ^= 1
            if np.sum(random_syndrome[decoder.n_x_stab:]) % 2 == 1:
                random_syndrome[decoder.n_x_stab] ^= 1

            correction = decoder.decode(random_syndrome)
            action = decoder.decode_to_action(random_syndrome)
            print(f"  Trial {trial+1}: syndrome weight={np.sum(random_syndrome)}, "
                  f"correction weight={np.sum(correction)}, action={action}")

    print("\n" + "=" * 50)
    print("MWPM decoder tests completed!")


def test_lookup_decoder():
    """Test lookup decoder for d=3."""
    print("\nTesting Lookup Decoder (d=3)")
    print("=" * 50)

    decoder = SimpleLookupDecoder(distance=3)
    print(f"Lookup table size: {len(decoder.lookup)}")

    # Test some specific syndromes
    test_syndromes = [
        [0, 0, 0, 0, 0, 0, 0, 0],  # No error
        [1, 0, 0, 0, 0, 0, 0, 0],  # Single defect
        [1, 1, 0, 0, 0, 0, 0, 0],  # Two adjacent defects
    ]

    for syndrome in test_syndromes:
        action = decoder.decode_to_action(np.array(syndrome, dtype=np.int8))
        if action == 27:
            print(f"  Syndrome {syndrome}: No correction")
        else:
            qubit = action // 3
            pauli = ['X', 'Y', 'Z'][action % 3]
            print(f"  Syndrome {syndrome}: {pauli}_{qubit}")

    print("Lookup decoder tests completed!")


if __name__ == "__main__":
    test_mwpm_decoder()
    test_lookup_decoder()
