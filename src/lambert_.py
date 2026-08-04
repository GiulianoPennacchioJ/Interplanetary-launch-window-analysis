"""
lambert_gooding.py
==================
Robust Lambert Problem Solver using Battin's Universal Variable Formulation.

Based on the algorithm from Vallado, "Fundamentals of Astrodynamics and 
Applications", 4th ed., Chapter 7.

Solves for the orbital transfer between two position vectors given a time of flight.
Handles both short-way (Type-I) and long-way (Type-II) transfers with robust
exception handling and numerical stability.

References:
-----------
Vallado, D.A. "Fundamentals of Astrodynamics and Applications", 4th ed.
    Microcosm Press, 2013. Chapter 7: Lambert's Problem.

Battin, R.H. "An Introduction to the Mathematics and Methods of Astrodynamics",
    AIAA Education Series, 1999.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TransferType(Enum):
    """Classification of Lambert transfer arcs."""
    SHORT_WAY = 1   # Type-I: transfer angle < 180°
    LONG_WAY = 2    # Type-II: transfer angle >= 180°


@dataclass
class LambertSolution:
    """Container for Lambert problem solution."""
    v1: np.ndarray          # Departure velocity vector [km/s]
    v2: np.ndarray          # Arrival velocity vector [km/s]
    transfer_type: TransferType
    tof: float              # Time of flight [s]
    # Orbital elements of the transfer orbit
    sma: float              # Semi-major axis [km]
    ecc: float              # Eccentricity
    inclination: float      # Inclination [rad]
    raan: float             # Right ascension of ascending node [rad]
    arg_periapsis: float    # Argument of periapsis [rad]
    true_anomaly1: float    # True anomaly at departure [rad]
    true_anomaly2: float    # True anomaly at arrival [rad]


class LambertError(Exception):
    """Custom exception for Lambert solver failures."""
    pass


class LambertGooding:
    """
    Lambert Problem Solver using Battin's Universal Variable Formulation.

    Implements the robust iterative algorithm based on universal variables
    with Newton-Raphson iteration for rapid convergence.
    """

    MU_SUN = 1.32712440018e11  # Gravitational parameter of the Sun [km^3/s^2]

    def __init__(self, mu: float = MU_SUN):
        """
        Initialize the solver.

        Parameters
        ----------
        mu : float
            Gravitational parameter [km^3/s^2]. Default is Sun.
        """
        self.mu = mu

    def solve(self, 
              r1: np.ndarray, 
              r2: np.ndarray, 
              tof: float,
              transfer_type: TransferType = TransferType.SHORT_WAY,
              max_iterations: int = 50,
              tolerance: float = 1e-12) -> LambertSolution:
        """
        Solve the Lambert problem using Battin's universal variable formulation.

        Parameters
        ----------
        r1 : np.ndarray
            Departure position vector [km]
        r2 : np.ndarray
            Arrival position vector [km]
        tof : float
            Time of flight [s]
        transfer_type : TransferType
            SHORT_WAY or LONG_WAY transfer
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance

        Returns
        -------
        LambertSolution
            Complete solution with velocity vectors and orbital elements

        Raises
        ------
        LambertError
            If the solver fails to converge or parameters are invalid
        """
        r1 = np.asarray(r1, dtype=float)
        r2 = np.asarray(r2, dtype=float)

        # Validate inputs
        if tof <= 0:
            raise LambertError("Time of flight must be positive")
        r1_norm = np.linalg.norm(r1)
        r2_norm = np.linalg.norm(r2)
        if r1_norm == 0 or r2_norm == 0:
            raise LambertError("Position vectors cannot be zero")

        # Compute transfer angle
        r1r2 = np.dot(r1, r2)
        r1xr2 = np.cross(r1, r2)

        cos_dnu = r1r2 / (r1_norm * r2_norm)
        cos_dnu = np.clip(cos_dnu, -1.0, 1.0)

        # Determine transfer angle
        if transfer_type == TransferType.SHORT_WAY:
            dnu = np.arccos(cos_dnu)
        else:
            dnu = 2.0 * np.pi - np.arccos(cos_dnu)

        # Check for degenerate case (180° transfer with long way)
        sin_dnu = np.linalg.norm(r1xr2) / (r1_norm * r2_norm)
        if sin_dnu < 1e-12 and transfer_type == TransferType.LONG_WAY:
            raise LambertError("180° transfer not possible for long-way trajectory")

        # Parameter A (related to chord)
        A = np.sqrt(r1_norm * r2_norm * (1.0 + np.cos(dnu)))
        if transfer_type == TransferType.LONG_WAY:
            A = -A

        if abs(A) < 1e-12:
            raise LambertError("Transfer angle is 180° (degenerate case)")

        # Solve for universal variable z using Newton-Raphson
        z = self._solve_for_z(r1_norm, r2_norm, A, tof, max_iterations, tolerance)

        # Compute f and g functions
        f, g, g_dot = self._compute_fg(r1_norm, r2_norm, A, z)

        # Compute velocity vectors
        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g

        # Compute orbital elements
        orbital_elements = self._compute_orbital_elements(r1, v1, r1_norm, r2_norm, dnu)

        return LambertSolution(
            v1=v1, v2=v2, transfer_type=transfer_type, tof=tof, **orbital_elements
        )

    def _solve_for_z(self, r1: float, r2: float, A: float, tof: float,
                     max_iter: int, tol: float) -> float:
        """
        Solve for the universal variable z using Newton-Raphson iteration.

        The universal variable z is related to the orbit type:
        z > 0: elliptic
        z = 0: parabolic  
        z < 0: hyperbolic

        Parameters
        ----------
        r1, r2 : float
            Magnitudes of position vectors [km]
        A : float
            Transfer parameter [km]
        tof : float
            Time of flight [s]
        max_iter : int
        tol : float

        Returns
        -------
        float
            Universal variable z
        """
        # Initial guess
        # For Hohmann-like transfers, z is small positive (elliptic)
        # For hyperbolic transfers, z is negative
        z = 0.0  # Start with parabolic guess

        for iteration in range(max_iter):
            # Compute Stumpff functions
            C, S = self._stumpff(z)

            # Compute y
            y = r1 + r2 + A * (z * S - 1.0) / np.sqrt(C) if abs(C) > 1e-15 else r1 + r2

            if y < 0:
                # y must be positive - adjust z
                z = z * 0.9
                continue

            # Time of flight equation
            sqrt_y = np.sqrt(y)
            sqrt_C = np.sqrt(C) if C > 0 else np.sqrt(abs(C))

            if abs(sqrt_C) < 1e-15:
                # Parabolic case
                t_current = (sqrt_y**3) / (3.0 * self.mu)
            else:
                t_current = (sqrt_y**3 * S + A * sqrt_y) / sqrt_C / np.sqrt(self.mu)

            # Residual
            f = t_current - tof

            if abs(f) < tol:
                return z

            # Derivative dt/dz using finite differences
            h = 1e-8
            t_plus = self._compute_tof(r1, r2, A, z + h)
            t_minus = self._compute_tof(r1, r2, A, z - h)

            df = (t_plus - t_minus) / (2.0 * h)

            if abs(df) < 1e-15:
                # Fallback to small step
                z = z + 0.1
                continue

            # Newton-Raphson update
            dz = -f / df
            z_new = z + dz

            # Damping for stability
            if abs(dz) > 1.0:
                dz = np.sign(dz) * 1.0
                z_new = z + dz

            # Keep z in reasonable range
            z_new = np.clip(z_new, -100.0, 100.0)

            if abs(z_new - z) < tol:
                return z_new

            z = z_new

        raise LambertError(f"Failed to converge after {max_iter} iterations. "
                          f"Final residual: {abs(f):.2e}, z = {z:.6f}")

    def _compute_tof(self, r1: float, r2: float, A: float, z: float) -> float:
        """Compute time of flight for a given z."""
        C, S = self._stumpff(z)

        if abs(C) < 1e-15:
            y = r1 + r2
            return (y**1.5) / (3.0 * self.mu)

        sqrt_C = np.sqrt(C)
        y = r1 + r2 + A * (z * S - 1.0) / sqrt_C

        if y < 0:
            return float('inf')

        sqrt_y = np.sqrt(y)
        return (sqrt_y**3 * S + A * sqrt_y) / sqrt_C / np.sqrt(self.mu)

    def _compute_fg(self, r1: float, r2: float, A: float, z: float) -> Tuple[float, float, float]:
        """
        Compute f, g, and g_dot Lagrange coefficients.

        Parameters
        ----------
        r1, r2 : float
            Position magnitudes [km]
        A : float
            Transfer parameter [km]
        z : float
            Universal variable

        Returns
        -------
        tuple (f, g, g_dot)
        """
        C, S = self._stumpff(z)

        if abs(C) < 1e-15:
            y = r1 + r2
        else:
            y = r1 + r2 + A * (z * S - 1.0) / np.sqrt(C)

        if y < 0:
            raise LambertError(f"Invalid y = {y} for z = {z}")

        sqrt_y = np.sqrt(y)
        sqrt_mu = np.sqrt(self.mu)

        # f and g functions
        f = 1.0 - y / r1
        g = A * sqrt_y / sqrt_mu
        g_dot = 1.0 - y / r2

        return f, g, g_dot

    def _stumpff(self, z: float) -> Tuple[float, float]:
        """
        Compute Stumpff functions c2(z) and c3(z).

        Universal functions used in the universal variable formulation.

        Parameters
        ----------
        z : float
            Universal variable

        Returns
        -------
        tuple (c2, c3)
        """
        if z > 1e-6:
            # Elliptic
            sqrt_z = np.sqrt(z)
            c2 = (1.0 - np.cos(sqrt_z)) / z
            c3 = (sqrt_z - np.sin(sqrt_z)) / (z * sqrt_z)
        elif z < -1e-6:
            # Hyperbolic
            sqrt_neg_z = np.sqrt(-z)
            c2 = (1.0 - np.cosh(sqrt_neg_z)) / z
            c3 = (np.sinh(sqrt_neg_z) - sqrt_neg_z) / (-z * sqrt_neg_z)
        else:
            # Series expansion near zero
            c2 = 0.5 - z / 24.0 + z**2 / 720.0 - z**3 / 40320.0
            c3 = 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0 - z**3 / 362880.0

        return c2, c3

    def _compute_orbital_elements(self, r1: np.ndarray, v1: np.ndarray,
                                   r1_norm: float, r2_norm: float,
                                   dnu: float) -> dict:
        """
        Compute classical orbital elements from state vectors.
        """
        # Specific angular momentum
        h_vec = np.cross(r1, v1)
        h = np.linalg.norm(h_vec)

        # Eccentricity vector
        v1_norm = np.linalg.norm(v1)
        e_vec = np.cross(v1, h_vec) / self.mu - r1 / r1_norm
        ecc = np.linalg.norm(e_vec)

        # Semi-major axis from vis-viva
        energy = v1_norm**2 / 2.0 - self.mu / r1_norm
        if abs(energy) < 1e-15:
            sma = np.inf
        else:
            sma = -self.mu / (2.0 * energy)

        # Inclination
        inclination = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

        # RAAN
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)
        if n < 1e-15:
            raan = 0.0
        else:
            raan = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
            if n_vec[1] < 0:
                raan = 2.0 * np.pi - raan

        # Argument of periapsis
        if n < 1e-15 or ecc < 1e-15:
            arg_periapsis = 0.0
        else:
            arg_periapsis = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * ecc), -1.0, 1.0))
            if e_vec[2] < 0:
                arg_periapsis = 2.0 * np.pi - arg_periapsis

        # True anomaly at departure
        if ecc < 1e-15:
            true_anomaly1 = 0.0
        else:
            true_anomaly1 = np.arccos(np.clip(np.dot(e_vec, r1) / (ecc * r1_norm), -1.0, 1.0))
            if np.dot(r1, v1) < 0:
                true_anomaly1 = 2.0 * np.pi - true_anomaly1

        true_anomaly2 = (true_anomaly1 + dnu) % (2.0 * np.pi)

        return {
            'sma': sma,
            'ecc': ecc,
            'inclination': inclination,
            'raan': raan,
            'arg_periapsis': arg_periapsis,
            'true_anomaly1': true_anomaly1,
            'true_anomaly2': true_anomaly2
        }


def solve_lambert(r1: np.ndarray, r2: np.ndarray, tof: float,
                  mu: float = 1.32712440018e11,
                  transfer_type: TransferType = TransferType.SHORT_WAY) -> LambertSolution:
    """
    Convenience function for solving the Lambert problem.

    Parameters
    ----------
    r1, r2 : np.ndarray
        Position vectors [km]
    tof : float
        Time of flight [s]
    mu : float
        Gravitational parameter [km^3/s^2]
    transfer_type : TransferType

    Returns
    -------
    LambertSolution
    """
    solver = LambertGooding(mu=mu)
    return solver.solve(r1, r2, tof, transfer_type=transfer_type)