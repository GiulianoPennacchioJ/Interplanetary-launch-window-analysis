"""
lambert_gooding.py
==================
Robust implementation of Gooding's algorithm (1990) for the Lambert Problem.

Solves for the orbital transfer between two position vectors given a time of flight.
Handles both short-way (Type-I) and long-way (Type-II) transfers.

This implementation uses the universal variable formulation with the 
Lancaster-Blanchard parameterization, following Gooding's original paper.

References:
-----------
Gooding, R.H. "A procedure for the solution of Lambert's orbital boundary-value problem."
Celestial Mechanics and Dynamical Astronomy, 48(2), 145-165, 1990.

Vallado, D.A. "Fundamentals of Astrodynamics and Applications", 4th ed.
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
    Gooding's Lambert Problem Solver.

    Implements the robust iterative algorithm based on the Lancaster-Blanchard
    universal variable formulation with Halley's method for rapid convergence.
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
        Solve the Lambert problem using Gooding's algorithm.

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

        # Geometric parameters
        c = np.sqrt(r1_norm**2 + r2_norm**2 - 2.0 * r1_norm * r2_norm * cos_dnu)
        s = (r1_norm + r2_norm + c) / 2.0

        # Minimum energy semi-major axis
        a_min = s / 2.0

        # Non-dimensional time parameter
        T = np.sqrt(8.0 * self.mu / s**3) * tof

        # q parameter (related to geometry)
        q = np.sqrt(r1_norm * r2_norm) * np.cos(dnu / 2.0) / s

        # Solve for the universal variable x
        x = self._solve_for_x(T, q, transfer_type, max_iterations, tolerance)

        # Compute velocity vectors
        v1, v2 = self._compute_velocities(r1, r2, r1_norm, r2_norm, c, s, x, dnu, transfer_type)

        # Compute orbital elements
        orbital_elements = self._compute_orbital_elements(r1, v1, r1_norm, r2_norm, dnu)

        return LambertSolution(
            v1=v1, v2=v2, transfer_type=transfer_type, tof=tof, **orbital_elements
        )

    def _solve_for_x(self, T: float, q: float, transfer_type: TransferType,
                     max_iter: int, tol: float) -> float:
        """
        Solve for the universal variable x using Gooding's iterative method.

        The variable x is related to the semi-major axis by:
        a = s / (2 * (1 - x^2))

        For elliptic orbits: |x| < 1
        For parabolic: x = 1
        For hyperbolic: x > 1
        """
        # Initial guess based on Gooding's recommendations
        if transfer_type == TransferType.SHORT_WAY:
            # Short-way: initial guess
            if T < np.pi:
                x0 = 0.5
            else:
                x0 = 0.5
            # Clamp to valid range for short-way
            x0 = np.clip(x0, -0.99, 0.99)
        else:
            # Long-way: initial guess
            x0 = 1.5
            x0 = np.clip(x0, 1.01, 10.0)

        # Iterative solution using Halley's method
        x = x0
        for iteration in range(max_iter):
            # Compute time function and derivatives
            t_val, dt_dx, d2t_dx2 = self._compute_time_function(x, q, transfer_type)

            # Residual
            f = t_val - T

            if abs(f) < tol:
                return x

            # Halley's method update
            if abs(dt_dx) < 1e-15:
                raise LambertError("Derivative too small, cannot converge")

            # Halley: x_new = x - f / (f' - f*f''/(2*f'))
            denom = dt_dx - f * d2t_dx2 / (2.0 * dt_dx)
            if abs(denom) < 1e-15:
                dx = -f / dt_dx  # Fall back to Newton
            else:
                dx = -f / denom

            x_new = x + dx

            # Keep x in valid range
            if transfer_type == TransferType.SHORT_WAY:
                x_new = np.clip(x_new, -0.999, 0.999)
            else:
                x_new = np.clip(x_new, 1.001, 20.0)

            if abs(x_new - x) < tol:
                return x_new

            x = x_new

        raise LambertError(f"Failed to converge after {max_iter} iterations. "
                          f"Final residual: {abs(f):.2e}")

    def _compute_time_function(self, x: float, q: float, 
                                transfer_type: TransferType) -> Tuple[float, float, float]:
        """
        Compute the non-dimensional time function t(x) and its derivatives.

        Uses the Lancaster-Blanchard formulation.
        """
        # Compute y
        y_sq = 1.0 - q**2 * (1.0 - x**2)
        if y_sq < 0:
            y_sq = 1e-15
        y = np.sqrt(y_sq)

        # Compute eta
        eta = y - q * x

        # Compute zeta
        zeta = y - x * q

        # Time function based on orbit type
        if transfer_type == TransferType.SHORT_WAY:
            if x < 1.0:
                # Elliptic short-way
                arg = x * q + eta
                arg = np.clip(arg, -1.0, 1.0)

                if x >= 0:
                    t = np.arccos(arg) - x * y + q * zeta
                else:
                    t = -np.arccos(arg) - x * y + q * zeta
            else:
                # Hyperbolic short-way
                arg = x * q + eta
                arg = max(arg, 1.0 + 1e-15)
                t = -np.arccosh(arg) + x * y - q * zeta
        else:
            if x < 1.0:
                # Elliptic long-way
                arg = x * q + eta
                arg = np.clip(arg, -1.0, 1.0)
                t = 2.0 * np.pi - np.arccos(arg) + x * y - q * zeta
            else:
                # Hyperbolic long-way
                arg = x * q + eta
                arg = max(arg, 1.0 + 1e-15)
                t = 2.0 * np.pi + np.arccosh(arg) - x * y + q * zeta

        # Compute derivatives using finite differences
        h = 1e-8
        t_plus = self._compute_time_raw(x + h, q, transfer_type)
        t_minus = self._compute_time_raw(x - h, q, transfer_type)

        dt_dx = (t_plus - t_minus) / (2.0 * h)
        d2t_dx2 = (t_plus - 2.0 * t + t_minus) / (h**2)

        return t, dt_dx, d2t_dx2

    def _compute_time_raw(self, x: float, q: float, transfer_type: TransferType) -> float:
        """Raw time computation for derivative estimation."""
        y_sq = max(1.0 - q**2 * (1.0 - x**2), 1e-15)
        y = np.sqrt(y_sq)
        eta = y - q * x
        zeta = y - x * q

        if transfer_type == TransferType.SHORT_WAY:
            if x < 1.0:
                arg = np.clip(x * q + eta, -1.0, 1.0)
                if x >= 0:
                    return np.arccos(arg) - x * y + q * zeta
                else:
                    return -np.arccos(arg) - x * y + q * zeta
            else:
                arg = max(x * q + eta, 1.0 + 1e-15)
                return -np.arccosh(arg) + x * y - q * zeta
        else:
            if x < 1.0:
                arg = np.clip(x * q + eta, -1.0, 1.0)
                return 2.0 * np.pi - np.arccos(arg) + x * y - q * zeta
            else:
                arg = max(x * q + eta, 1.0 + 1e-15)
                return 2.0 * np.pi + np.arccosh(arg) - x * y + q * zeta

    def _compute_velocities(self, r1: np.ndarray, r2: np.ndarray,
                             r1_norm: float, r2_norm: float,
                             c: float, s: float, x: float,
                             dnu: float,
                             transfer_type: TransferType) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute departure and arrival velocity vectors using f and g functions.
        """
        # Semi-major axis
        a = s / (2.0 * (1.0 - x**2))

        # Parameter A (related to geometry)
        A = np.sqrt(r1_norm * r2_norm * (1.0 + np.cos(dnu)))
        if transfer_type == TransferType.LONG_WAY:
            A = -A

        # Universal variable z = x^2
        z = x**2

        # Stumpff functions
        C, S = self._stumpff(z)

        # y parameter
        if abs(C) < 1e-15:
            y = r1_norm + r2_norm
        else:
            y = r1_norm + r2_norm + A * (z * S - 1.0) / np.sqrt(C)

        if y < 1e-15:
            raise LambertError("Invalid y parameter")

        # f and g Lagrange coefficients
        f = 1.0 - y / r1_norm
        g = A * np.sqrt(y / self.mu)
        g_dot = 1.0 - y / r2_norm

        # Velocity vectors
        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g

        return v1, v2

    def _stumpff(self, z: float) -> Tuple[float, float]:
        """
        Compute Stumpff functions c2(z) and c3(z).

        Universal functions for the universal variable formulation.
        """
        if z > 1e-6:
            sqrt_z = np.sqrt(z)
            c2 = (1.0 - np.cos(sqrt_z)) / z
            c3 = (sqrt_z - np.sin(sqrt_z)) / (z * sqrt_z)
        elif z < -1e-6:
            sqrt_neg_z = np.sqrt(-z)
            c2 = (1.0 - np.cosh(sqrt_neg_z)) / z
            c3 = (np.sinh(sqrt_neg_z) - sqrt_neg_z) / (-z * sqrt_neg_z)
        else:
            # Series expansion
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