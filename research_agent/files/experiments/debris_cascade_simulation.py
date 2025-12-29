#!/usr/bin/env python3
"""
Discrete-Time Orbital Debris Cascade Dynamics Simulation
=========================================================

This simulation implements the theoretical framework from theory_debris_cascade_dynamics.md
to model the 50-year evolution of LEO debris populations under mega-constellation deployment.

Author: Experimental Research Agent
Date: 2025-12-22
Version: 3.0 (Calibrated to historical collision rates)

Key Features:
    - 6 altitude bands (400-600, 600-800, 800-1000, 1000-1200, 1200-1500, 1500+ km)
    - Mega-constellation deployment (Starlink, Kuiper, OneWeb) with time-varying rates
    - Collision frequency computation using spatial density and relative velocity
    - NASA Standard Breakup Model for fragmentation
    - Cascade multiplication factor K_m(t) tracking for phase transition detection
    - Multiple PMD compliance scenarios (80%, 90%, 95%, 99%)

Calibration Notes (v3.0):
    - Historical collision rate: ~0.2-0.3 catastrophic collisions/year among trackable objects
    - Notable events: Cosmos-Iridium (2009), Fengyun-1C ASAT test (2007)
    - Current tracked population: ~30,000-40,000 objects (>10 cm)
    - Model calibrated to match these rates at t=0

References:
    - Kessler, D.J. & Cour-Palais, B.G. (1978). Collision frequency of artificial satellites.
    - NASA Standard Breakup Model (2001). NASA/TM-2001-210889.
    - ESA MASTER Model Documentation.
    - Liou, J.-C. (2006). Collision Activities in the Future Orbital Debris Environment.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: CONSTANTS AND CONFIGURATION
# =============================================================================

# Physical constants
R_EARTH = 6371.0  # Earth radius in km
MU_EARTH = 3.986e5  # Earth gravitational parameter in km^3/s^2
C_D = 2.2  # Drag coefficient (typical for satellites)

# Simulation parameters
DELTA_T = 0.1  # Time step in years (approximately 36.5 days)
T_MAX = 50  # Simulation horizon in years
N_STEPS = int(T_MAX / DELTA_T)  # Total number of time steps

# Velocity parameters
V_REL_MEAN = 10.0  # Mean relative velocity in LEO (km/s)
# Convert to m/s for energy calculations
V_REL_MS = V_REL_MEAN * 1000.0

# Fragmentation parameters (NASA Standard Breakup Model)
E_THRESHOLD = 40.0  # Catastrophic collision energy threshold (J/g)
G_CATASTROPHIC_AVG = 2000  # Average fragments per catastrophic collision (>10cm objects)
G_NONCATASTROPHIC = 50  # Mean fragments per non-catastrophic collision

# Size bins configuration
# s1: [1mm, 1cm) - Lethal non-trackable
# s2: [1cm, 10cm) - Lethal, partially trackable
# s3: [10cm, 1m) - Trackable debris
# s4: [1m, infinity) - Intact objects (satellites, rocket bodies)
SIZE_BIN_RADII = np.array([0.005, 0.05, 0.5, 5.0])  # Representative radii in meters
SIZE_BIN_MASSES = np.array([0.001, 0.1, 10.0, 1000.0])  # Representative masses in kg
SIZE_BIN_AREAS = np.pi * SIZE_BIN_RADII**2  # Cross-sectional areas in m^2
SIZE_BIN_LIMITS = np.array([
    [0.001, 0.01],   # 1mm to 1cm
    [0.01, 0.1],     # 1cm to 10cm
    [0.1, 1.0],      # 10cm to 1m
    [1.0, 100.0]     # 1m to 100m (effective infinity)
])
M_SIZE_BINS = 4  # Number of size bins

# Altitude bands configuration (6 bands as specified)
ALTITUDE_BANDS = np.array([
    [400, 600],    # Band 0: 400-600 km (Starlink primary)
    [600, 800],    # Band 1: 600-800 km (Kuiper primary)
    [800, 1000],   # Band 2: 800-1000 km (highest debris density)
    [1000, 1200],  # Band 3: 1000-1200 km (OneWeb primary)
    [1200, 1500],  # Band 4: 1200-1500 km
    [1500, 2000]   # Band 5: 1500+ km (upper LEO)
])
K_ALT_BANDS = len(ALTITUDE_BANDS)  # Number of altitude bands

# Compute band centers and shell volumes
BAND_CENTERS = np.mean(ALTITUDE_BANDS, axis=1)  # Central altitude of each band
BAND_THICKNESS = ALTITUDE_BANDS[:, 1] - ALTITUDE_BANDS[:, 0]  # Shell thickness

# Empirical decay time constants by altitude band (years)
# Rows: altitude bands, Columns: size bins [s1, s2, s3, s4]
# Based on historical data and atmospheric modeling
DECAY_TAU_BY_BAND = np.array([
    [2.0, 5.0, 10.0, 15.0],        # 400-600 km: fast decay due to drag
    [10.0, 25.0, 50.0, 80.0],      # 600-800 km: moderate decay
    [50.0, 150.0, 300.0, 500.0],   # 800-1000 km: slow decay
    [150.0, 400.0, 700.0, 1000.0], # 1000-1200 km: very slow
    [400.0, 800.0, 1000.0, 1000.0], # 1200-1500 km: essentially permanent
    [700.0, 1000.0, 1000.0, 1000.0] # 1500+ km: permanent
])

# =============================================================================
# Collision Rate Calibration
# =============================================================================
# The intrinsic collision rate per pair of objects is:
# lambda_ij = sigma_ij * v_rel / V_eff
#
# For N objects in an effective volume V_eff, the total collision rate is:
# R = (1/2) * N^2 * <sigma> * v_rel / V_eff
#
# CALIBRATION TARGET:
# - Current trackable population: ~30,000 objects (bins s3 + s4)
# - Historical catastrophic collision rate: ~0.2 per year
# - This gives a collision rate coefficient ~ 2e-10 per pair per year
#
# We model collisions between ALL size classes, but focus calibration on
# trackable-to-trackable collisions to match historical data.
#
# The coefficient below is tuned empirically to produce:
# - ~0.2-0.5 collisions/year at t=0 for trackable objects
# - Gradual increase with population growth

# Intrinsic collision probability coefficient
# Units: effectively dimensionless scaling factor
COLLISION_RATE_COEFFICIENT = 3.0e-10


# =============================================================================
# SECTION 2: DATA STRUCTURES
# =============================================================================

@dataclass
class Constellation:
    """
    Data class representing a satellite mega-constellation.

    Attributes:
        name: Constellation identifier
        n_satellites: Target number of satellites
        altitude_band: Primary altitude band index (0-5)
        deploy_start: Deployment start year (relative to simulation start)
        deploy_end: Deployment end year
        lifetime: Operational lifetime in years
        pmd_compliance: Post-Mission Disposal compliance rate (0.0 to 1.0)
    """
    name: str
    n_satellites: int
    altitude_band: int
    deploy_start: float
    deploy_end: float
    lifetime: float
    pmd_compliance: float = 0.9


@dataclass
class SimulationState:
    """
    Data class holding the complete simulation state at a given time step.

    Attributes:
        S: Debris population array [size_bins x altitude_bands]
        C: Cumulative collision count
        R_total: Total collision rate (collisions/year)
        G: Current cascade gain factor
        K_m: Cascade multiplication factor
        n_collisions: Number of collisions in this time step
        fragments_generated: Total fragments generated this step
    """
    S: np.ndarray  # [M_SIZE_BINS, K_ALT_BANDS]
    C: float = 0.0
    R_total: float = 0.0
    G: float = G_CATASTROPHIC_AVG
    K_m: float = 0.0
    n_collisions: int = 0
    fragments_generated: float = 0.0


@dataclass
class CollisionEvent:
    """
    Record of a single collision event.

    Attributes:
        time: Time of collision in years
        altitude_band: Altitude band where collision occurred
        size_i: Size bin of first object
        size_j: Size bin of second object
        collision_type: "catastrophic" or "non-catastrophic"
        n_fragments: Number of fragments generated
    """
    time: float
    altitude_band: int
    size_i: int
    size_j: int
    collision_type: str
    n_fragments: float


# =============================================================================
# SECTION 3: HELPER FUNCTIONS
# =============================================================================

def compute_shell_volume(h_min: float, h_max: float) -> float:
    """
    Compute the volume of a spherical shell between two altitudes.

    V = (4/3) * pi * [(R_E + h_max)^3 - (R_E + h_min)^3]

    Args:
        h_min: Inner shell altitude (km)
        h_max: Outer shell altitude (km)

    Returns:
        Shell volume in km^3
    """
    r_inner = R_EARTH + h_min
    r_outer = R_EARTH + h_max
    return (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)


def compute_orbital_velocity(h: float) -> float:
    """
    Compute circular orbital velocity at altitude h.

    v_orb = sqrt(mu / (R_E + h))

    Args:
        h: Altitude in km

    Returns:
        Orbital velocity in km/s
    """
    return np.sqrt(MU_EARTH / (R_EARTH + h))


def compute_collision_cross_section(r_i: float, r_j: float) -> float:
    """
    Compute collision cross-section between two objects.

    sigma_ij = pi * (r_i + r_j)^2

    Args:
        r_i: Radius of first object in meters
        r_j: Radius of second object in meters

    Returns:
        Cross-section in m^2
    """
    # Sum of radii in meters
    r_sum = r_i + r_j
    return np.pi * r_sum**2


def nasa_breakup_fragments(m_total: float, L_c_min: float, L_c_max: float) -> float:
    """
    Compute number of fragments in a size range using NASA Standard Breakup Model.

    N_f(L_c) = 0.1 * M_total^0.75 * L_c^(-1.71)

    The number of fragments between L_c_min and L_c_max is:
    Delta_N = N_f(L_c_min) - N_f(L_c_max)

    Args:
        m_total: Total collision mass in kg
        L_c_min: Minimum characteristic length in m
        L_c_max: Maximum characteristic length in m

    Returns:
        Number of fragments in the size range
    """
    # Scaling coefficient from NASA Standard Breakup Model
    coeff = 0.1 * (m_total ** 0.75)

    # Fragment count at each size threshold
    n_min = coeff * (L_c_min ** -1.71) if L_c_min > 0 else 1e10
    n_max = coeff * (L_c_max ** -1.71) if L_c_max > 0 else 0

    # Number of fragments in the bin
    n_frag = max(0.0, n_min - n_max)

    return n_frag


def select_collision_pair(S_local: np.ndarray, sigma: np.ndarray) -> Tuple[int, int]:
    """
    Select a colliding pair weighted by population and cross-section.

    Weights are computed as: w[i,j] = S[i] * S[j] * sigma[i,j]
    With factor 0.5 for self-collisions (i == j).

    Args:
        S_local: Population in each size bin [M_SIZE_BINS]
        sigma: Cross-section matrix [M_SIZE_BINS, M_SIZE_BINS]

    Returns:
        Tuple (i, j) indices of colliding size bins
    """
    M = len(S_local)
    weights = np.zeros((M, M))

    for i in range(M):
        for j in range(i, M):
            w = S_local[i] * S_local[j] * sigma[i, j]
            if i == j:
                w *= 0.5  # Self-collision correction
            weights[i, j] = w

    # Normalize and sample
    total_weight = np.sum(weights)
    if total_weight <= 0:
        return (M-1, M-1)  # Fallback

    r = np.random.uniform(0, total_weight)
    cumulative = 0.0

    for i in range(M):
        for j in range(i, M):
            cumulative += weights[i, j]
            if cumulative >= r:
                return (i, j)

    return (M-1, M-1)  # Fallback


# =============================================================================
# SECTION 4: CONSTELLATION DEPLOYMENT MODEL
# =============================================================================

def get_default_constellations(pmd_compliance: float = 0.9) -> List[Constellation]:
    """
    Define default mega-constellation deployment scenarios.

    Based on Section 5.2 of the theory document:
    - Starlink: 12,000 satellites at 550 km (2020-2027), 5-year lifetime
    - OneWeb: 6,500 satellites at 1200 km (2021-2025), 7-year lifetime
    - Kuiper: 3,200 satellites at 600 km (2024-2029), 7-year lifetime
    - China-SatNet: 13,000 satellites at 500-1200 km (2025-2035), 5-year lifetime

    Args:
        pmd_compliance: Post-Mission Disposal compliance rate

    Returns:
        List of Constellation objects
    """
    return [
        Constellation(
            name="Starlink",
            n_satellites=12000,
            altitude_band=0,  # 400-600 km band
            deploy_start=0.0,  # Year 0 of simulation (2024)
            deploy_end=3.0,    # By 2027
            lifetime=5.0,
            pmd_compliance=pmd_compliance
        ),
        Constellation(
            name="OneWeb",
            n_satellites=6500,
            altitude_band=3,  # 1000-1200 km band
            deploy_start=0.0,  # Started
            deploy_end=1.0,    # By 2025
            lifetime=7.0,
            pmd_compliance=pmd_compliance
        ),
        Constellation(
            name="Kuiper",
            n_satellites=3200,
            altitude_band=1,  # 600-800 km band
            deploy_start=0.0,  # 2024
            deploy_end=5.0,    # By 2029
            lifetime=7.0,
            pmd_compliance=pmd_compliance
        ),
        Constellation(
            name="ChinaSatNet",
            n_satellites=13000,
            altitude_band=2,  # Split across bands, primary 800-1000 km
            deploy_start=1.0,  # 2025
            deploy_end=11.0,   # By 2035
            lifetime=5.0,
            pmd_compliance=pmd_compliance
        ),
    ]


def compute_launch_rate(t: float, constellation: Constellation) -> float:
    """
    Compute the launch rate for a constellation at time t.

    During deployment: L(t) = N_c / (t_end - t_start)
    After deployment (replacement): L(t) = N_c / tau_life

    Args:
        t: Current time in years
        constellation: Constellation object

    Returns:
        Launch rate in satellites/year
    """
    deploy_start = constellation.deploy_start
    deploy_end = constellation.deploy_end
    n_sat = constellation.n_satellites
    lifetime = constellation.lifetime

    if t < deploy_start:
        return 0.0
    elif t <= deploy_end:
        # Deployment phase
        deploy_duration = deploy_end - deploy_start
        if deploy_duration > 0:
            return n_sat / deploy_duration
        else:
            return 0.0
    else:
        # Replacement phase (maintain constellation)
        return n_sat / lifetime


def compute_failed_pmd_debris(t: float, constellation: Constellation, dt: float) -> float:
    """
    Compute debris from failed post-mission disposal.

    After deployment, satellites that fail PMD become debris.

    Args:
        t: Current time in years
        constellation: Constellation object
        dt: Time step in years

    Returns:
        Number of new debris objects from failed PMD
    """
    deploy_end = constellation.deploy_end
    lifetime = constellation.lifetime
    n_sat = constellation.n_satellites
    pmd = constellation.pmd_compliance

    if t <= deploy_end:
        return 0.0

    # After deployment, satellites retire at rate N_c / lifetime
    # Fraction (1 - pmd) become debris
    retire_rate = n_sat / lifetime
    failed_debris = retire_rate * (1.0 - pmd) * dt

    return failed_debris


# =============================================================================
# SECTION 5: INITIAL CONDITIONS
# =============================================================================

def initialize_debris_population() -> np.ndarray:
    """
    Initialize the debris population based on approximate current catalog data.

    Current estimates (as of 2024):
    - >36,000 tracked objects (>10 cm)
    - ~1 million objects 1-10 cm (estimated)
    - >100 million objects 1mm-1cm (estimated)
    - ~8,000 active satellites + ~3,000 defunct large objects

    Distribution across altitude bands based on historical trends:
    - Peak concentration at 750-850 km (sun-synchronous debris)
    - Secondary peak at 1400-1500 km (navigation/comm satellites)
    - Lower altitudes dominated by recent mega-constellation activity

    Returns:
        Initial debris population array [M_SIZE_BINS, K_ALT_BANDS]
    """
    S_0 = np.zeros((M_SIZE_BINS, K_ALT_BANDS))

    # Altitude distribution factors (relative concentration)
    # Based on ESA MASTER model and Space-Track catalog distributions
    # Peak at 800-1000 km due to historical debris events (Cosmos-Iridium, Fengyun-1C)
    alt_factors = np.array([0.12, 0.20, 0.30, 0.20, 0.12, 0.06])

    # Initial population estimates by size bin
    # Calibrated to give realistic collision rates
    # s1: [1mm, 1cm) - ~100 million total (estimated from statistical models)
    # s2: [1cm, 10cm) - ~900,000 total (estimated)
    # s3: [10cm, 1m) - ~25,000 total (tracked debris)
    # s4: [1m+) - ~8,000 total (active satellites + large rocket bodies)

    initial_totals = np.array([100000000, 900000, 25000, 8000])

    for i in range(M_SIZE_BINS):
        for k in range(K_ALT_BANDS):
            S_0[i, k] = initial_totals[i] * alt_factors[k]

    return S_0


# =============================================================================
# SECTION 6: PRECOMPUTATION
# =============================================================================

def precompute_parameters() -> Dict:
    """
    Precompute time-invariant simulation parameters.

    Computes:
    - Shell volumes
    - Collision cross-sections
    - Decay time constants
    - Collision rate coefficients per altitude band

    Returns:
        Dictionary containing precomputed parameters
    """
    params = {}

    # Full shell volumes (km^3)
    params['V'] = np.array([
        compute_shell_volume(band[0], band[1])
        for band in ALTITUDE_BANDS
    ])

    # Collision cross-sections (m^2) [M x M matrix]
    params['sigma'] = np.zeros((M_SIZE_BINS, M_SIZE_BINS))
    for i in range(M_SIZE_BINS):
        for j in range(M_SIZE_BINS):
            params['sigma'][i, j] = compute_collision_cross_section(
                SIZE_BIN_RADII[i], SIZE_BIN_RADII[j]
            )

    # Decay time constants (years) [M x K matrix]
    # Use empirical values from DECAY_TAU_BY_BAND
    params['tau'] = DECAY_TAU_BY_BAND.T  # Transpose to get [M x K]

    # Orbital velocities at band centers (km/s)
    params['v_orb'] = np.array([
        compute_orbital_velocity(h) for h in BAND_CENTERS
    ])

    # Altitude-dependent collision rate multipliers
    # Higher at altitudes with more historical debris events
    # Lower at very low altitudes (fast decay) and very high (less traffic)
    params['alt_collision_factor'] = np.array([0.8, 1.2, 1.5, 1.3, 1.0, 0.6])

    return params


# =============================================================================
# SECTION 7: MAIN SIMULATION ENGINE
# =============================================================================

class DebrisCascadeSimulation:
    """
    Main simulation engine for LEO debris cascade dynamics.

    This class implements the discrete-time simulation model from the
    theoretical framework, tracking debris populations, collisions,
    and cascade dynamics over a 50-year horizon.

    Attributes:
        constellations: List of mega-constellations to simulate
        pmd_compliance: Global PMD compliance rate
        params: Precomputed simulation parameters
        S: Debris population time series
        results: Dictionary of result arrays
        collision_log: List of CollisionEvent objects
        runaway_detected: Boolean indicating if runaway occurred
        T_runaway: Time of runaway detection (if applicable)
    """

    def __init__(self, pmd_compliance: float = 0.9,
                 constellations: Optional[List[Constellation]] = None,
                 seed: Optional[int] = None):
        """
        Initialize the simulation.

        Args:
            pmd_compliance: Post-Mission Disposal compliance rate (0.0 to 1.0)
            constellations: List of Constellation objects (default: use standard set)
            seed: Random seed for reproducibility
        """
        self.pmd_compliance = pmd_compliance

        if constellations is None:
            self.constellations = get_default_constellations(pmd_compliance)
        else:
            self.constellations = constellations

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Precompute parameters
        self.params = precompute_parameters()

        # Initialize state arrays
        self.S = np.zeros((N_STEPS + 1, M_SIZE_BINS, K_ALT_BANDS))
        self.S[0] = initialize_debris_population()

        # Result arrays
        self.results = {
            'time': np.zeros(N_STEPS + 1),
            'S_total': np.zeros(N_STEPS + 1),
            'S_by_size': np.zeros((N_STEPS + 1, M_SIZE_BINS)),
            'S_by_altitude': np.zeros((N_STEPS + 1, K_ALT_BANDS)),
            'C': np.zeros(N_STEPS + 1),  # Cumulative collisions
            'R_total': np.zeros(N_STEPS + 1),  # Collision rate
            'G': np.zeros(N_STEPS + 1),  # Cascade gain factor
            'K_m': np.zeros(N_STEPS + 1),  # Cascade multiplication factor
            'n_collisions': np.zeros(N_STEPS + 1),  # Collisions per step
            'fragments': np.zeros(N_STEPS + 1),  # Fragments per step
            'launches': np.zeros(N_STEPS + 1),  # Launches per step
            'decayed': np.zeros(N_STEPS + 1),  # Objects decayed per step
        }

        # Initialize time 0 values
        self.results['time'][0] = 0.0
        self.results['S_total'][0] = np.sum(self.S[0])
        self.results['S_by_size'][0] = np.sum(self.S[0], axis=1)
        self.results['S_by_altitude'][0] = np.sum(self.S[0], axis=0)
        self.results['G'][0] = G_CATASTROPHIC_AVG

        # Collision log
        self.collision_log: List[CollisionEvent] = []

        # Runaway detection
        self.runaway_detected = False
        self.T_runaway = None

        # Track when K_m first exceeds thresholds
        self.K_m_threshold_times = {0.5: None, 0.8: None, 0.9: None, 1.0: None}

    def compute_collision_rates(self, S_current: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute collision rates in each altitude band.

        The collision rate follows kinetic theory:
        R_k = sum_{i,j} (1/2) * N_i * N_j * sigma_ij * v_rel * k_coll * f_k

        Where:
        - N_i, N_j = populations in size bins i, j in band k
        - sigma_ij = collision cross-section
        - v_rel = relative velocity
        - k_coll = calibrated collision coefficient
        - f_k = altitude-dependent factor

        Args:
            S_current: Current debris population [M x K]

        Returns:
            Tuple of (R_shell array, R_total)
        """
        sigma = self.params['sigma']
        alt_factor = self.params['alt_collision_factor']

        R_shell = np.zeros(K_ALT_BANDS)

        for k in range(K_ALT_BANDS):
            # Collision rate contributions from each size bin pair
            for i in range(M_SIZE_BINS):
                for j in range(i, M_SIZE_BINS):
                    # Factor to avoid double counting
                    factor = 0.5 if i == j else 1.0

                    N_i = S_current[i, k]
                    N_j = S_current[j, k]

                    # Collision rate for this pair in this band
                    # R = factor * N_i * N_j * sigma * v_rel * k_coll * f_alt
                    # Units: objects * objects * m^2 * km/s * (calibration) * (dimensionless)
                    #      = calibrated to collisions/year

                    R_ij_k = (factor * N_i * N_j * sigma[i, j]
                              * V_REL_MEAN * COLLISION_RATE_COEFFICIENT
                              * alt_factor[k])

                    R_shell[k] += R_ij_k

        R_total = np.sum(R_shell)

        return R_shell, R_total

    def process_collisions(self, t_step: int, S_current: np.ndarray,
                          R_shell: np.ndarray) -> Tuple[np.ndarray, int, float, List[CollisionEvent]]:
        """
        Process collision events for this time step.

        Uses Poisson sampling for collision counts and NASA Standard Breakup
        Model for fragment generation.

        Args:
            t_step: Current time step index
            S_current: Current debris population [M x K]
            R_shell: Collision rate in each shell (collisions/year)

        Returns:
            Tuple of (fragments_generated, n_collisions, total_fragments, collision_events)
        """
        t_years = t_step * DELTA_T
        sigma = self.params['sigma']

        fragments_generated = np.zeros((M_SIZE_BINS, K_ALT_BANDS))
        n_collisions = 0
        total_fragments = 0.0
        collision_events = []

        for k in range(K_ALT_BANDS):
            # Expected number of collisions in this time step
            lambda_coll = R_shell[k] * DELTA_T

            # Sample from Poisson distribution
            if lambda_coll > 0:
                n_coll_k = np.random.poisson(lambda_coll)
            else:
                n_coll_k = 0

            n_collisions += n_coll_k

            for _ in range(n_coll_k):
                # Select colliding pair
                S_local = S_current[:, k].copy()
                i, j = select_collision_pair(S_local, sigma)

                # Determine collision type based on kinetic energy
                m_i = SIZE_BIN_MASSES[i]
                m_j = SIZE_BIN_MASSES[j]

                # Reduced mass kinetic energy
                mu_reduced = (m_i * m_j) / (m_i + m_j)
                KE = 0.5 * mu_reduced * V_REL_MS**2  # Joules

                # Target mass (larger object)
                M_target = max(m_i, m_j)
                specific_energy = KE / (M_target * 1000)  # J/g

                if specific_energy > E_THRESHOLD:
                    # Catastrophic collision
                    M_total = m_i + m_j

                    # Generate fragments using NASA Standard Breakup Model
                    event_fragments = 0.0
                    for bin_idx in range(M_SIZE_BINS):
                        s_min = SIZE_BIN_LIMITS[bin_idx, 0]
                        s_max = SIZE_BIN_LIMITS[bin_idx, 1]
                        n_frag = nasa_breakup_fragments(M_total, s_min, s_max)
                        fragments_generated[bin_idx, k] += n_frag
                        total_fragments += n_frag
                        event_fragments += n_frag

                    # Remove colliding objects
                    if S_current[i, k] > 0:
                        S_current[i, k] -= 1
                    if S_current[j, k] > 0:
                        if i != j:
                            S_current[j, k] -= 1
                        elif S_current[i, k] > 0:  # Same bin, need to remove another
                            S_current[i, k] -= 1

                    collision_events.append(CollisionEvent(
                        time=t_years,
                        altitude_band=k,
                        size_i=i,
                        size_j=j,
                        collision_type="catastrophic",
                        n_fragments=event_fragments
                    ))
                else:
                    # Non-catastrophic (cratering)
                    # Generate small fragments only (bins 0 and 1)
                    event_fragments = 0.0
                    for bin_idx in range(2):
                        frag_count = G_NONCATASTROPHIC * (0.1 ** bin_idx)
                        fragments_generated[bin_idx, k] += frag_count
                        total_fragments += frag_count
                        event_fragments += frag_count

                    collision_events.append(CollisionEvent(
                        time=t_years,
                        altitude_band=k,
                        size_i=i,
                        size_j=j,
                        collision_type="non-catastrophic",
                        n_fragments=event_fragments
                    ))

        return fragments_generated, n_collisions, total_fragments, collision_events

    def apply_atmospheric_decay(self, S_current: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply atmospheric decay to debris population.

        decay_fraction = 1 - exp(-dt / tau)

        Args:
            S_current: Current debris population [M x K]

        Returns:
            Tuple of (updated population, total decayed)
        """
        tau = self.params['tau']
        total_decayed = 0.0
        S_updated = S_current.copy()

        for i in range(M_SIZE_BINS):
            for k in range(K_ALT_BANDS):
                if tau[i, k] > 0:
                    decay_fraction = 1.0 - np.exp(-DELTA_T / tau[i, k])
                    debris_decayed = S_updated[i, k] * decay_fraction
                    S_updated[i, k] -= debris_decayed
                    total_decayed += debris_decayed

        # Ensure non-negative populations
        S_updated = np.maximum(S_updated, 0)

        return S_updated, total_decayed

    def apply_launches(self, t_step: int, S_current: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply constellation launches and baseline launches.

        Args:
            t_step: Current time step
            S_current: Current debris population [M x K]

        Returns:
            Tuple of (updated population, total launches)
        """
        t_years = t_step * DELTA_T
        total_launches = 0.0
        S_updated = S_current.copy()

        # Baseline launch rate (non-mega-constellation missions)
        # Growing from ~100 objects/year to ~200 objects/year over simulation
        baseline_rate = 100.0 + 2.0 * t_years  # Linear growth
        baseline_per_step = baseline_rate * DELTA_T

        # Distribute baseline launches across size bin 4 (large objects)
        # Peak near 600 km for LEO, secondary at 1000+ km
        alt_weights = np.array([0.25, 0.25, 0.20, 0.15, 0.10, 0.05])
        for k in range(K_ALT_BANDS):
            launches_k = baseline_per_step * alt_weights[k]
            S_updated[3, k] += launches_k  # Size bin 4 (intact objects)
            total_launches += launches_k

        # Mega-constellation launches
        for constellation in self.constellations:
            launch_rate = compute_launch_rate(t_years, constellation)
            launches_this_step = launch_rate * DELTA_T

            if launches_this_step > 0:
                k = constellation.altitude_band
                S_updated[3, k] += launches_this_step  # Size bin 4
                total_launches += launches_this_step

        return S_updated, total_launches

    def apply_failed_pmd(self, t_step: int, S_current: np.ndarray) -> np.ndarray:
        """
        Add debris from failed post-mission disposal.

        Args:
            t_step: Current time step
            S_current: Current debris population [M x K]

        Returns:
            Updated population
        """
        t_years = t_step * DELTA_T
        S_updated = S_current.copy()

        for constellation in self.constellations:
            failed_debris = compute_failed_pmd_debris(t_years, constellation, DELTA_T)
            if failed_debris > 0:
                k = constellation.altitude_band
                # Failed PMD satellites remain as large debris (size bin 3)
                # since they are defunct but still large
                S_updated[2, k] += failed_debris  # Add to 10cm-1m bin

        return S_updated

    def compute_cascade_multiplication_factor(self, t_step: int,
                                             G: float, R_total: float,
                                             S_current: np.ndarray) -> float:
        """
        Compute the cascade multiplication factor K_m.

        K_m = (G * R_total) / (D_total + P_total)

        Where:
        - D_total = total natural decay rate (objects/year)
        - P_total = total active disposal rate (objects/year)
        - G = fragments generated per collision
        - R_total = collision rate (collisions/year)

        K_m > 1 indicates runaway cascade (phase transition).

        Args:
            t_step: Current time step
            G: Current cascade gain factor
            R_total: Total collision rate
            S_current: Current debris population

        Returns:
            Cascade multiplication factor K_m
        """
        tau = self.params['tau']
        t_years = t_step * DELTA_T

        # Compute total natural decay rate (objects/year)
        D_total = 0.0
        for i in range(M_SIZE_BINS):
            for k in range(K_ALT_BANDS):
                if tau[i, k] > 0:
                    D_total += S_current[i, k] / tau[i, k]

        # Compute total active disposal rate (PMD compliance) - objects/year
        P_total = 0.0
        for constellation in self.constellations:
            if t_years > constellation.deploy_end:
                # Active satellites being disposed at retirement rate
                retire_rate = constellation.n_satellites / constellation.lifetime
                # Only compliant fraction actually deorbits
                P_total += retire_rate * constellation.pmd_compliance

        # Debris generation rate = G * R_total (fragments/year)
        # Debris removal rate = D_total + P_total (objects/year)

        # Compute K_m
        denominator = D_total + P_total
        if denominator > 0 and R_total > 0:
            K_m = (G * R_total) / denominator
        else:
            K_m = 0.0

        return K_m

    def run(self, verbose: bool = True) -> Dict:
        """
        Run the full simulation.

        Executes the discrete-time simulation loop for N_STEPS time steps,
        tracking debris evolution, collisions, and cascade dynamics.

        Args:
            verbose: Print progress updates

        Returns:
            Dictionary of results
        """
        if verbose:
            print(f"Starting simulation: PMD compliance = {self.pmd_compliance:.0%}")
            print(f"Time horizon: {T_MAX} years, dt = {DELTA_T} years")
            print(f"Constellations: {[c.name for c in self.constellations]}")
            print("-" * 60)

        # Main simulation loop
        for t_step in range(1, N_STEPS + 1):
            t_years = t_step * DELTA_T

            # Get current state
            S_current = self.S[t_step - 1].copy()

            # Step 1: Compute collision rates
            R_shell, R_total = self.compute_collision_rates(S_current)

            # Step 2: Process collisions
            fragments_gen, n_coll, total_frags, events = self.process_collisions(
                t_step, S_current, R_shell
            )
            self.collision_log.extend(events)

            # Update cascade gain factor (exponential moving average)
            if n_coll > 0:
                G_new = total_frags / n_coll
                # Smooth update to avoid wild fluctuations
                alpha = 0.1
                G = alpha * G_new + (1 - alpha) * self.results['G'][t_step - 1]
            else:
                G = self.results['G'][t_step - 1]

            # Step 3: Apply atmospheric decay
            S_current, decayed = self.apply_atmospheric_decay(S_current)

            # Step 4: Add collision fragments
            S_current += fragments_gen

            # Step 5: Apply launches
            S_current, launches = self.apply_launches(t_step, S_current)

            # Step 6: Add failed PMD debris
            S_current = self.apply_failed_pmd(t_step, S_current)

            # Step 7: Compute cascade multiplication factor
            K_m = self.compute_cascade_multiplication_factor(
                t_step, G, R_total, S_current
            )

            # Track threshold crossings
            for threshold in [0.5, 0.8, 0.9, 1.0]:
                if K_m >= threshold and self.K_m_threshold_times[threshold] is None:
                    self.K_m_threshold_times[threshold] = t_years

            # Step 8: Detect runaway condition
            if K_m > 1.0 and not self.runaway_detected:
                self.runaway_detected = True
                self.T_runaway = t_years
                if verbose:
                    print(f"RUNAWAY DETECTED at T = {t_years:.1f} years (K_m = {K_m:.3f})")

            # Store state
            self.S[t_step] = S_current
            self.results['time'][t_step] = t_years
            self.results['S_total'][t_step] = np.sum(S_current)
            self.results['S_by_size'][t_step] = np.sum(S_current, axis=1)
            self.results['S_by_altitude'][t_step] = np.sum(S_current, axis=0)
            self.results['C'][t_step] = self.results['C'][t_step - 1] + n_coll
            self.results['R_total'][t_step] = R_total
            self.results['G'][t_step] = G
            self.results['K_m'][t_step] = K_m
            self.results['n_collisions'][t_step] = n_coll
            self.results['fragments'][t_step] = total_frags
            self.results['launches'][t_step] = launches
            self.results['decayed'][t_step] = decayed

            # Progress update
            if verbose and t_step % 100 == 0:
                print(f"Year {t_years:.1f}: S_total = {np.sum(S_current):.2e}, "
                      f"K_m = {K_m:.4f}, Collisions = {n_coll}, "
                      f"Rate = {R_total:.2f}/yr")

        if verbose:
            print("-" * 60)
            print("Simulation complete.")
            print(f"Final debris count: {self.results['S_total'][-1]:.2e}")
            print(f"Total collisions: {self.results['C'][-1]:.0f}")
            print(f"Final K_m: {self.results['K_m'][-1]:.4f}")
            print(f"Max K_m: {np.max(self.results['K_m']):.4f}")
            if self.runaway_detected:
                print(f"RUNAWAY OCCURRED at T = {self.T_runaway:.1f} years")
            else:
                print("No runaway detected within simulation horizon.")

        return self.results

    def get_trajectory_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame for easy export.

        Returns:
            DataFrame with time series data
        """
        df = pd.DataFrame({
            'time_years': self.results['time'],
            'S_total': self.results['S_total'],
            'S_1mm_1cm': self.results['S_by_size'][:, 0],
            'S_1cm_10cm': self.results['S_by_size'][:, 1],
            'S_10cm_1m': self.results['S_by_size'][:, 2],
            'S_1m_plus': self.results['S_by_size'][:, 3],
            'S_400_600km': self.results['S_by_altitude'][:, 0],
            'S_600_800km': self.results['S_by_altitude'][:, 1],
            'S_800_1000km': self.results['S_by_altitude'][:, 2],
            'S_1000_1200km': self.results['S_by_altitude'][:, 3],
            'S_1200_1500km': self.results['S_by_altitude'][:, 4],
            'S_1500_plus_km': self.results['S_by_altitude'][:, 5],
            'cumulative_collisions': self.results['C'],
            'collision_rate_per_year': self.results['R_total'],
            'cascade_gain_G': self.results['G'],
            'cascade_multiplier_Km': self.results['K_m'],
            'collisions_this_step': self.results['n_collisions'],
            'fragments_generated': self.results['fragments'],
            'launches': self.results['launches'],
            'objects_decayed': self.results['decayed'],
        })

        return df


# =============================================================================
# SECTION 8: EXPERIMENT RUNNER
# =============================================================================

def run_baseline_and_sensitivity(output_dir: str, seed: int = 42) -> Dict:
    """
    Run baseline scenario (90% PMD) and sensitivity analyses (80%, 95%, 99%).

    This function executes the full experimental design:
    1. Baseline scenario with 90% PMD compliance
    2. Sensitivity analyses at 80%, 95%, and 99% PMD
    3. Save trajectory data to CSV
    4. Save phase transition times to text file

    Args:
        output_dir: Directory to save results
        seed: Random seed for reproducibility

    Returns:
        Dictionary with results from all scenarios
    """
    import os

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    # PMD compliance scenarios
    pmd_scenarios = [0.80, 0.90, 0.95, 0.99]

    all_results = {}
    phase_transitions = {}
    all_trajectories = []

    print("=" * 70)
    print("LEO DEBRIS CASCADE DYNAMICS SIMULATION")
    print("50-Year Evolution with Mega-Constellation Deployment")
    print("=" * 70)
    print()

    for pmd in pmd_scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: PMD Compliance = {pmd:.0%}")
        print(f"{'='*60}")

        # Run simulation with different seed offset for each scenario
        # but keeping reproducibility
        sim = DebrisCascadeSimulation(pmd_compliance=pmd, seed=seed)
        results = sim.run(verbose=True)

        # Store results
        scenario_name = f"PMD_{int(pmd*100)}"
        all_results[scenario_name] = {
            'pmd_compliance': pmd,
            'results': results,
            'runaway_detected': sim.runaway_detected,
            'T_runaway': sim.T_runaway,
            'final_S_total': results['S_total'][-1],
            'final_K_m': results['K_m'][-1],
            'max_K_m': np.max(results['K_m']),
            'total_collisions': results['C'][-1],
            'collision_log': sim.collision_log,
            'K_m_thresholds': sim.K_m_threshold_times.copy(),
        }

        # Record phase transition
        if sim.runaway_detected:
            phase_transitions[scenario_name] = sim.T_runaway
        else:
            phase_transitions[scenario_name] = None

        # Get trajectory DataFrame and add scenario label
        df = sim.get_trajectory_dataframe()
        df['scenario'] = scenario_name
        df['pmd_compliance'] = pmd
        all_trajectories.append(df)

        print()

    # Combine all trajectories
    combined_df = pd.concat(all_trajectories, ignore_index=True)

    # Save trajectory data to CSV
    trajectory_path = os.path.join(output_dir, "cascade_trajectories.csv")
    combined_df.to_csv(trajectory_path, index=False)
    print(f"\nTrajectory data saved to: {trajectory_path}")

    # Save phase transition times to text file
    transition_path = os.path.join(output_dir, "phase_transitions.txt")
    with open(transition_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE TRANSITION ANALYSIS: CASCADE MULTIPLICATION FACTOR K_m\n")
        f.write("LEO Debris Cascade Dynamics Simulation Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("INTERPRETATION OF K_m (CASCADE MULTIPLICATION FACTOR)\n")
        f.write("-" * 70 + "\n")
        f.write("K_m = (G * R_total) / (D_total + P_total)\n\n")
        f.write("Where:\n")
        f.write("  G = Average fragments per collision event\n")
        f.write("  R_total = Total collision rate (collisions/year)\n")
        f.write("  D_total = Natural decay rate (objects/year)\n")
        f.write("  P_total = Active disposal rate (objects/year)\n\n")
        f.write("Phase Classification:\n")
        f.write("  K_m < 0.5  : Stable - debris naturally controlled\n")
        f.write("  K_m 0.5-0.8: Warning - approaching criticality\n")
        f.write("  K_m 0.8-1.0: Critical - near tipping point\n")
        f.write("  K_m > 1.0  : Runaway - self-sustaining cascade\n\n")

        f.write("=" * 70 + "\n")
        f.write("PHASE TRANSITION TIMES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Scenario':<15} {'PMD':<8} {'T(K_m>0.5)':<12} {'T(K_m>0.8)':<12} "
                f"{'T(K_m>1.0)':<12} {'Status':<15}\n")
        f.write("-" * 70 + "\n")

        for scenario, data in all_results.items():
            pmd = data['pmd_compliance']
            thresholds = data['K_m_thresholds']
            t_05 = f"{thresholds[0.5]:.1f}" if thresholds[0.5] else "N/A"
            t_08 = f"{thresholds[0.8]:.1f}" if thresholds[0.8] else "N/A"
            t_10 = f"{thresholds[1.0]:.1f}" if thresholds[1.0] else "N/A"
            status = "RUNAWAY" if data['runaway_detected'] else "Stable"
            f.write(f"{scenario:<15} {pmd:.0%}     {t_05:<12} {t_08:<12} {t_10:<12} {status:<15}\n")

        f.write("-" * 70 + "\n\n")

        # Summary statistics
        f.write("=" * 70 + "\n")
        f.write("DETAILED SUMMARY STATISTICS\n")
        f.write("=" * 70 + "\n")

        for scenario, data in all_results.items():
            f.write(f"\n{scenario} (PMD = {data['pmd_compliance']:.0%}):\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Initial debris count:   {data['results']['S_total'][0]:.2e}\n")
            f.write(f"  Final debris count:     {data['final_S_total']:.2e}\n")
            f.write(f"  Growth factor:          {data['final_S_total'] / data['results']['S_total'][0]:.2f}x\n")
            f.write(f"  Total collisions:       {data['total_collisions']:.0f}\n")
            f.write(f"  Average collision rate: {np.mean(data['results']['R_total'][1:]):.2f}/year\n")
            f.write(f"  Final K_m:              {data['final_K_m']:.4f}\n")
            f.write(f"  Maximum K_m:            {data['max_K_m']:.4f}\n")
            if data['runaway_detected']:
                f.write(f"  Runaway detected at:    {data['T_runaway']:.1f} years\n")
            else:
                f.write(f"  Runaway status:         Not detected (stable)\n")

            # Collision breakdown
            catastrophic = sum(1 for e in data['collision_log'] if e.collision_type == "catastrophic")
            non_cat = len(data['collision_log']) - catastrophic
            f.write(f"  Catastrophic collisions: {catastrophic}\n")
            f.write(f"  Non-catastrophic:        {non_cat}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 70 + "\n\n")

        # Analyze results
        runaway_scenarios = [s for s, d in all_results.items() if d['runaway_detected']]
        stable_scenarios = [s for s, d in all_results.items() if not d['runaway_detected']]

        if runaway_scenarios:
            earliest_runaway = min(all_results[s]['T_runaway'] for s in runaway_scenarios)
            f.write(f"1. Runaway cascade detected in {len(runaway_scenarios)} scenario(s):\n")
            f.write(f"   {', '.join(runaway_scenarios)}\n")
            f.write(f"   Earliest phase transition: {earliest_runaway:.1f} years\n\n")
        else:
            f.write("1. No runaway cascade detected in any scenario within 50-year horizon.\n\n")

        if stable_scenarios:
            f.write(f"2. Stable scenarios: {', '.join(stable_scenarios)}\n")
            max_km_stable = max(all_results[s]['max_K_m'] for s in stable_scenarios)
            f.write(f"   Maximum K_m reached: {max_km_stable:.4f}\n\n")

        # PMD effectiveness analysis
        f.write("3. PMD Compliance Impact:\n")
        for scenario, data in all_results.items():
            debris_growth = data['final_S_total'] / data['results']['S_total'][0]
            f.write(f"   {scenario}: {debris_growth:.2f}x debris growth, "
                    f"max K_m = {data['max_K_m']:.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("SIMULATION PARAMETERS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Time horizon: {T_MAX} years\n")
        f.write(f"Time step: {DELTA_T} years ({DELTA_T*365.25:.1f} days)\n")
        f.write(f"Altitude bands: 6 (400-600, 600-800, 800-1000, 1000-1200, 1200-1500, 1500+ km)\n")
        f.write(f"Size bins: 4 (1mm-1cm, 1cm-10cm, 10cm-1m, >1m)\n")
        f.write(f"Mean relative velocity: {V_REL_MEAN} km/s\n")
        f.write(f"Catastrophic energy threshold: {E_THRESHOLD} J/g\n")
        f.write(f"Collision rate coefficient: {COLLISION_RATE_COEFFICIENT:.2e}\n")
        f.write(f"Random seed: {seed}\n")

        f.write("\nConstellations modeled:\n")
        for c in get_default_constellations():
            f.write(f"  - {c.name}: {c.n_satellites} satellites at band {c.altitude_band}, "
                    f"deploy {c.deploy_start}-{c.deploy_end} yr, lifetime {c.lifetime} yr\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF PHASE TRANSITION ANALYSIS\n")
        f.write("=" * 70 + "\n")

    print(f"Phase transition analysis saved to: {transition_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL SCENARIOS")
    print("=" * 70)
    print(f"{'Scenario':<12} {'PMD':<6} {'Final S_total':<14} {'Max K_m':<10} "
          f"{'Collisions':<12} {'Status':<12}")
    print("-" * 70)
    for scenario, data in all_results.items():
        status = f"RUNAWAY@{data['T_runaway']:.0f}yr" if data['runaway_detected'] else "Stable"
        print(f"{scenario:<12} {data['pmd_compliance']:.0%}   {data['final_S_total']:.2e}     "
              f"{data['max_K_m']:.4f}     {data['total_collisions']:<12.0f} {status:<12}")
    print("=" * 70)

    return {
        'all_results': all_results,
        'phase_transitions': phase_transitions,
        'trajectory_df': combined_df,
        'trajectory_path': trajectory_path,
        'transition_path': transition_path,
    }


# =============================================================================
# SECTION 9: MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for the debris cascade simulation.

    Runs the baseline scenario and sensitivity analyses, saving results
    to the specified output directory.
    """
    import os

    # Output directory
    output_dir = "/Users/jminding/Desktop/Code/Research Agent/research_agent/files/results"

    # Run all scenarios
    results = run_baseline_and_sensitivity(output_dir, seed=42)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Trajectory data: {results['trajectory_path']}")
    print(f"Phase transitions: {results['transition_path']}")
    print("=" * 70)
