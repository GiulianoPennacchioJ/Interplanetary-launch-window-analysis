"""
ephemeris.py
============
Planetary ephemeris management module.

Provides state vectors (position and velocity) and classical orbital elements (COE)
for solar system planets. Designed with a pluggable architecture to support:
- Analytical models (Keplerian/J2000 osculating elements)
- External sources (JPL SPICE via spiceypy)

The analytical model uses mean orbital elements valid for J2000 epoch with
secular rates for approximate positions.

References:
-----------
NASA/JPL: "Approximate Positions of the Planets" (based on Keplerian elements)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Protocol
from dataclasses import dataclass
from datetime import datetime


# Physical constants
AU_KM = 149597870.7  # 1 AU in kilometers
DAY_S = 86400.0      # 1 day in seconds
J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)  # J2000.0 reference epoch


@dataclass
class PlanetState:
    """Planetary state vector container."""
    position: np.ndarray   # [km] in heliocentric ecliptic J2000
    velocity: np.ndarray   # [km/s] in heliocentric ecliptic J2000
    epoch_jd: float        # Julian Date


@dataclass 
class OrbitalElements:
    """Classical orbital elements (COE) container."""
    a: float       # Semi-major axis [AU]
    e: float       # Eccentricity
    i: float       # Inclination [deg]
    L: float       # Mean longitude [deg]
    varpi: float   # Longitude of perihelion [deg]
    Omega: float   # Longitude of ascending node [deg]
    # Rates (per Julian century)
    da: float = 0.0
    de: float = 0.0
    di: float = 0.0
    dL: float = 0.0
    dvarpi: float = 0.0
    dOmega: float = 0.0


class EphemerisSource(Protocol):
    """Protocol for ephemeris data sources."""

    def get_state(self, planet_name: str, epoch_jd: float) -> PlanetState:
        """Get planetary state at given Julian Date."""
        ...

    def get_elements(self, planet_name: str, epoch_jd: float) -> OrbitalElements:
        """Get orbital elements at given Julian Date."""
        ...


class AnalyticalEphemeris:
    """
    Analytical planetary ephemeris using Keplerian elements.

    Based on mean orbital elements from NASA/JPL with secular rates.
    Valid approximately for years 1800-2050.
    """

    # Mean orbital elements at J2000.0 and their secular rates
    # Data from NASA/JPL "Approximate Positions of the Planets"
    # Format: a[AU], e, i[deg], L[deg], varpi[deg], Omega[deg]
    # Rates: da[AU/cy], de[1/cy], di[deg/cy], dL[deg/cy], dvarpi[deg/cy], dOmega[deg/cy]
    PLANET_DATA: Dict[str, Tuple] = {
        # Planet: (a, e, i, L, varpi, Omega, da, de, di, dL, dvarpi, dOmega)
        "Mercury": (0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593,
                    0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081),
        "Venus": (0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255,
                  0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418),
        "Earth": (1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0,
                  0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0),
        "Mars": (1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891,
                 0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343),
        "Jupiter": (5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909,
                    -0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106),
        "Saturn": (9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887813, 113.66242448,
                   -0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.28867794, -0.28867794),
        "Uranus": (19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503,
                   -0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589),
        "Neptune": (30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574,
                    0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664),
        "Pluto": (39.48211675, 0.24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684,
                  -0.00031596, 0.00005170, 0.00004818, 145.20780515, -0.04062942, -0.01183482),
    }

    # Gravitational parameter of the Sun [km^3/s^2]
    MU_SUN = 1.32712440018e11

    def __init__(self):
        """Initialize the analytical ephemeris model."""
        pass

    def _compute_julian_centuries(self, epoch_jd: float) -> float:
        """
        Compute Julian centuries from J2000.0.

        Parameters
        ----------
        epoch_jd : float
            Julian Date

        Returns
        -------
        float
            Julian centuries from J2000.0 (T)
        """
        J2000_JD = 2451545.0
        return (epoch_jd - J2000_JD) / 36525.0

    def get_elements(self, planet_name: str, epoch_jd: float) -> OrbitalElements:
        """
        Get orbital elements for a planet at a given epoch.

        Parameters
        ----------
        planet_name : str
            Name of the planet
        epoch_jd : float
            Julian Date

        Returns
        -------
        OrbitalElements
            Classical orbital elements at the specified epoch
        """
        if planet_name not in self.PLANET_DATA:
            raise ValueError(f"Unknown planet: {planet_name}. Available: {list(self.PLANET_DATA.keys())}")

        data = self.PLANET_DATA[planet_name]
        a0, e0, i0, L0, varpi0, Omega0 = data[:6]
        da, de, di, dL, dvarpi, dOmega = data[6:12]

        T = self._compute_julian_centuries(epoch_jd)

        # Compute elements at epoch
        a = a0 + da * T
        e = e0 + de * T
        i = i0 + di * T
        L = L0 + dL * T
        varpi = varpi0 + dvarpi * T
        Omega = Omega0 + dOmega * T

        # Normalize angles to [0, 360)
        L = L % 360.0
        varpi = varpi % 360.0
        Omega = Omega % 360.0

        return OrbitalElements(
            a=a, e=e, i=i, L=L, varpi=varpi, Omega=Omega,
            da=da, de=de, di=di, dL=dL, dvarpi=dvarpi, dOmega=dOmega
        )

    def _solve_kepler(self, M: float, e: float, tol: float = 1e-12, 
                      max_iter: int = 50) -> float:
        """
        Solve Kepler's equation: M = E - e*sin(E) for eccentric anomaly E.

        Uses Newton-Raphson iteration with a robust initial guess.

        Parameters
        ----------
        M : float
            Mean anomaly [rad]
        e : float
            Eccentricity
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations

        Returns
        -------
        float
            Eccentric anomaly E [rad]
        """
        # Normalize M to [0, 2*pi)
        M = M % (2.0 * np.pi)

        # Initial guess using Mikkola's method (very robust)
        if e < 0.8:
            E = M
        else:
            # Mikkola's initial guess for high eccentricity
            alpha = (1.0 - e) / (4.0 * e + 0.5)
            beta = M / (8.0 * e + 1.0)
            z = np.cbrt(beta + np.sqrt(beta**2 + alpha**3))
            s = z - alpha / z
            E = M + e * (3.0 * s - 4.0 * s**3)

        # Newton-Raphson iteration
        for _ in range(max_iter):
            f = E - e * np.sin(E) - M
            fp = 1.0 - e * np.cos(E)

            if abs(fp) < 1e-15:
                break

            dE = -f / fp
            E = E + dE

            if abs(dE) < tol:
                return E

        return E

    def get_state(self, planet_name: str, epoch_jd: float) -> PlanetState:
        """
        Get heliocentric state vector for a planet at a given epoch.

        Computes position and velocity in the heliocentric ecliptic J2000 frame.

        Parameters
        ----------
        planet_name : str
            Name of the planet
        epoch_jd : float
            Julian Date

        Returns
        -------
        PlanetState
            Position [km] and velocity [km/s] vectors
        """
        elements = self.get_elements(planet_name, epoch_jd)

        # Convert to radians
        i_rad = np.radians(elements.i)
        L_rad = np.radians(elements.L)
        varpi_rad = np.radians(elements.varpi)
        Omega_rad = np.radians(elements.Omega)

        # Argument of perihelion
        omega_rad = varpi_rad - Omega_rad

        # Mean anomaly
        M_rad = L_rad - varpi_rad
        M_rad = M_rad % (2.0 * np.pi)

        # Solve Kepler's equation
        E = self._solve_kepler(M_rad, elements.e)

        # True anomaly
        nu = 2.0 * np.arctan2(
            np.sqrt(1.0 + elements.e) * np.sin(E / 2.0),
            np.sqrt(1.0 - elements.e) * np.cos(E / 2.0)
        )

        # Distance from Sun
        r = elements.a * (1.0 - elements.e * np.cos(E))  # [AU]
        r_km = r * AU_KM  # [km]

        # Position in orbital plane
        x_prime = r * np.cos(nu)  # [AU]
        y_prime = r * np.sin(nu)  # [AU]

        # Velocity in orbital plane [AU/day]
        # From vis-viva: v = sqrt(mu * (2/r - 1/a))
        # In orbital plane coordinates
        n = np.sqrt(self.MU_SUN / (elements.a * AU_KM)**3)  # [rad/s]

        # Velocity components in orbital plane
        # vx' = -sqrt(mu*a)/r * sin(E)
        # vy' =  sqrt(mu*a*(1-e^2))/r * cos(E)
        h = np.sqrt(self.MU_SUN * elements.a * AU_KM * (1.0 - elements.e**2))  # specific angular momentum

        vx_prime = -self.MU_SUN / h * np.sin(nu)  # [km/s] - this needs correction
        vy_prime = self.MU_SUN / h * (elements.e + np.cos(nu))  # [km/s]

        # Correct velocity computation using orbital elements
        # v_r = (mu/h) * e * sin(nu)
        # v_t = (mu/h) * (1 + e * cos(nu))
        v_r = self.MU_SUN / h * elements.e * np.sin(nu)
        v_t = self.MU_SUN / h * (1.0 + elements.e * np.cos(nu))

        # Velocity in orbital plane
        vx_prime = v_r * np.cos(nu) - v_t * np.sin(nu)
        vy_prime = v_r * np.sin(nu) + v_t * np.cos(nu)

        # Rotation matrix from orbital plane to ecliptic J2000
        # R = Rz(-Omega) * Rx(-i) * Rz(-omega)
        cos_O = np.cos(Omega_rad)
        sin_O = np.sin(Omega_rad)
        cos_i = np.cos(i_rad)
        sin_i = np.sin(i_rad)
        cos_w = np.cos(omega_rad)
        sin_w = np.sin(omega_rad)

        # Position transformation
        x = (cos_w * cos_O - sin_w * sin_O * cos_i) * x_prime +             (-sin_w * cos_O - cos_w * sin_O * cos_i) * y_prime
        y = (cos_w * sin_O + sin_w * cos_O * cos_i) * x_prime +             (-sin_w * sin_O + cos_w * cos_O * cos_i) * y_prime
        z = (sin_w * sin_i) * x_prime +             (cos_w * sin_i) * y_prime

        # Velocity transformation (same rotation)
        vx = (cos_w * cos_O - sin_w * sin_O * cos_i) * vx_prime +              (-sin_w * cos_O - cos_w * sin_O * cos_i) * vy_prime
        vy = (cos_w * sin_O + sin_w * cos_O * cos_i) * vx_prime +              (-sin_w * sin_O + cos_w * cos_O * cos_i) * vy_prime
        vz = (sin_w * sin_i) * vx_prime +              (cos_w * sin_i) * vy_prime

        position = np.array([x, y, z]) * AU_KM  # Convert to km
        velocity = np.array([vx, vy, vz])

        return PlanetState(position=position, velocity=velocity, epoch_jd=epoch_jd)

    def get_state_batch(self, planet_name: str, epochs_jd: np.ndarray) -> list:
        """
        Get states for multiple epochs (vectorized where possible).

        Parameters
        ----------
        planet_name : str
        epochs_jd : np.ndarray
            Array of Julian Dates

        Returns
        -------
        list[PlanetState]
        """
        return [self.get_state(planet_name, jd) for jd in epochs_jd]


class JPLSpiceEphemeris:
    """
    JPL SPICE-based ephemeris (placeholder for future implementation).

    Requires spiceypy package and SPICE kernels.
    """

    def __init__(self, kernel_path: Optional[str] = None):
        """
        Initialize SPICE ephemeris.

        Parameters
        ----------
        kernel_path : str, optional
            Path to SPICE kernel files
        """
        self.kernel_path = kernel_path
        self._spice_available = False
        self._spice = None

        try:
            import spiceypy as spice
            self._spice = spice
            self._spice_available = True
            if kernel_path:
                spice.furnsh(kernel_path)
        except ImportError:
            # spiceypy not installed - silently fall back
            self._fallback = AnalyticalEphemeris()

    def get_state(self, planet_name: str, epoch_jd: float) -> PlanetState:
        """Get state using SPICE or fallback to analytical."""
        if not self._spice_available:
            return self._fallback.get_state(planet_name, epoch_jd)

        # SPICE implementation would go here
        # et = self._spice.str2et(f"JD {epoch_jd}")
        # state, _ = self._spice.spkezr(planet_name.upper(), et, 'ECLIPJ2000', 'NONE', 'SUN')
        # ...

        return self._fallback.get_state(planet_name, epoch_jd)

    def get_elements(self, planet_name: str, epoch_jd: float) -> OrbitalElements:
        """Get elements using SPICE or fallback."""
        if not self._spice_available:
            return self._fallback.get_elements(planet_name, epoch_jd)
        return self._fallback.get_elements(planet_name, epoch_jd)


class EphemerisManager:
    """
    Manager class that provides a unified interface for ephemeris data.

    Can switch between analytical and SPICE sources transparently.
    """

    def __init__(self, source: Optional[EphemerisSource] = None):
        """
        Initialize with an ephemeris source.

        Parameters
        ----------
        source : EphemerisSource, optional
            Data source. Defaults to AnalyticalEphemeris.
        """
        self.source = source or AnalyticalEphemeris()

    def get_state(self, planet_name: str, epoch_jd: float) -> PlanetState:
        """Get planetary state."""
        return self.source.get_state(planet_name, epoch_jd)

    def get_elements(self, planet_name: str, epoch_jd: float) -> OrbitalElements:
        """Get orbital elements."""
        return self.source.get_elements(planet_name, epoch_jd)

    def jd_from_datetime(self, dt: datetime) -> float:
        """
        Convert Python datetime to Julian Date.

        Parameters
        ----------
        dt : datetime
            Python datetime object

        Returns
        -------
        float
            Julian Date
        """
        # Algorithm from Meeus, "Astronomical Algorithms"
        year = dt.year
        month = dt.month
        day = dt.day + dt.hour / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0

        if month <= 2:
            year -= 1
            month += 12

        A = int(year / 100)
        B = 2 - A + int(A / 4)

        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

        return jd

    def datetime_from_jd(self, jd: float) -> datetime:
        """
        Convert Julian Date to Python datetime.

        Parameters
        ----------
        jd : float
            Julian Date

        Returns
        -------
        datetime
            Python datetime object (UTC)
        """
        jd = jd + 0.5
        Z = int(jd)
        F = jd - Z

        if Z < 2299161:
            A = Z
        else:
            alpha = int((Z - 1867216.25) / 36524.25)
            A = Z + 1 + alpha - int(alpha / 4)

        B = A + 1524
        C = int((B - 122.1) / 365.25)
        D = int(365.25 * C)
        E = int((B - D) / 30.6001)

        day = B - D - int(30.6001 * E) + F

        if E < 14:
            month = E - 1
        else:
            month = E - 13

        if month > 2:
            year = C - 4716
        else:
            year = C - 4715

        day_int = int(day)
        frac = day - day_int
        hour = int(frac * 24)
        frac = frac * 24 - hour
        minute = int(frac * 60)
        frac = frac * 60 - minute
        second = int(frac * 60)
        microsecond = int((frac * 60 - second) * 1e6)

        return datetime(year, month, day_int, hour, minute, second, microsecond)


def get_planet_state(planet_name: str, epoch_jd: float) -> PlanetState:
    """Convenience function for getting planetary state."""
    manager = EphemerisManager()
    return manager.get_state(planet_name, epoch_jd)