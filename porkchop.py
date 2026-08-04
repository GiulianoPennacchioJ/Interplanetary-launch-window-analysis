"""
porkchop.py
===========
Porkchop plot engine for interplanetary launch window analysis.

Generates 2D grids of departure vs arrival dates and computes key mission metrics:
- C3 (departure energy) [km^2/s^2]
- V_inf (arrival excess velocity) [km/s]
- Time of Flight (TOF) [days]
- Total Delta-V [km/s]

Supports both Type-I (short-way) and Type-II (long-way) transfers with
vectorized computation for efficiency.

References:
-----------
Vallado, D.A. "Fundamentals of Astrodynamics and Applications", 4th ed.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from lambert_ import LambertGooding, TransferType, LambertSolution, LambertError
from ephemeris import EphemerisManager, PlanetState


# Physical constants
MU_SUN = 1.32712440018e11  # [km^3/s^2]
DAY_S = 86400.0            # [s/day]
AU_KM = 149597870.7        # [km/AU]

# Planet gravitational parameters [km^3/s^2] for escape/capture
MU_PLANETS = {
    "Mercury": 2.2032e4,
    "Venus": 3.24859e5,
    "Earth": 3.986004418e5,
    "Mars": 4.282837e4,
    "Jupiter": 1.26686534e8,
    "Saturn": 3.7931187e7,
    "Uranus": 5.793939e6,
    "Neptune": 6.836529e6,
    "Pluto": 8.71e2,
}

# Planet mean radii [km] for parking orbit assumptions
R_PLANETS = {
    "Mercury": 2439.7,
    "Venus": 6051.8,
    "Earth": 6371.0,
    "Mars": 3389.5,
    "Jupiter": 69911.0,
    "Saturn": 58232.0,
    "Uranus": 25362.0,
    "Neptune": 24622.0,
    "Pluto": 1188.3,
}


@dataclass
class PorkchopGrid:
    """Container for porkchop plot grid data."""
    departure_dates: np.ndarray    # Julian Dates [JD]
    arrival_dates: np.ndarray      # Julian Dates [JD]
    tof_grid: np.ndarray           # Time of Flight [days]
    c3_grid: np.ndarray             # C3 departure energy [km^2/s^2]
    vinf_arr_grid: np.ndarray       # Arrival V_inf [km/s]
    dv_total_grid: np.ndarray       # Total Delta-V [km/s]
    transfer_type: TransferType

    @property
    def min_c3(self) -> Tuple[float, float, float]:
        """Return minimum C3 value and its coordinates (dep_jd, arr_jd)."""
        idx = np.unravel_index(np.nanargmin(self.c3_grid), self.c3_grid.shape)
        return self.c3_grid[idx], self.departure_dates[idx[1]], self.arrival_dates[idx[0]]

    @property
    def min_dv(self) -> Tuple[float, float, float]:
        """Return minimum Delta-V value and its coordinates."""
        idx = np.unravel_index(np.nanargmin(self.dv_total_grid), self.dv_total_grid.shape)
        return self.dv_total_grid[idx], self.departure_dates[idx[1]], self.arrival_dates[idx[0]]


class PorkchopEngine:
    """
    Engine for computing porkchop plot data grids.

    Computes the full 2D grid of departure vs arrival dates with
    mission metrics for interplanetary trajectory analysis.
    """

    def __init__(self, 
                 departure_planet: str,
                 arrival_planet: str,
                 ephemeris: Optional[EphemerisManager] = None):
        """
        Initialize the porkchop engine.

        Parameters
        ----------
        departure_planet : str
            Name of departure planet
        arrival_planet : str
            Name of arrival planet
        ephemeris : EphemerisManager, optional
            Ephemeris data source. Defaults to analytical model.
        """
        self.departure_planet = departure_planet
        self.arrival_planet = arrival_planet
        self.ephemeris = ephemeris or EphemerisManager()
        self.lambert = LambertGooding(mu=MU_SUN)

        # Get planet gravitational parameters
        self.mu_dep = MU_PLANETS.get(departure_planet, MU_PLANETS["Earth"])
        self.mu_arr = MU_PLANETS.get(arrival_planet, MU_PLANETS["Earth"])
        self.r_dep = R_PLANETS.get(departure_planet, R_PLANETS["Earth"])
        self.r_arr = R_PLANETS.get(arrival_planet, R_PLANETS["Earth"])

    def compute_grid(self,
                     dep_start_jd: float,
                     dep_end_jd: float,
                     dep_step_days: float,
                     arr_start_jd: float,
                     arr_end_jd: float,
                     arr_step_days: float,
                     transfer_type: TransferType = TransferType.SHORT_WAY,
                     parking_orbit_alt_dep: float = 200.0,   # [km]
                     parking_orbit_alt_arr: float = 200.0,   # [km]
                     parallel: bool = False,
                     n_workers: int = 4) -> PorkchopGrid:
        """
        Compute the full porkchop grid.

        Parameters
        ----------
        dep_start_jd, dep_end_jd : float
            Departure date range [Julian Date]
        dep_step_days : float
            Departure grid step size [days]
        arr_start_jd, arr_end_jd : float
            Arrival date range [Julian Date]
        arr_step_days : float
            Arrival grid step size [days]
        transfer_type : TransferType
            SHORT_WAY or LONG_WAY transfer
        parking_orbit_alt_dep : float
            Departure parking orbit altitude [km]
        parking_orbit_alt_arr : float
            Arrival parking orbit altitude [km]
        parallel : bool
            Use parallel processing
        n_workers : int
            Number of parallel workers

        Returns
        -------
        PorkchopGrid
            Complete grid with all computed metrics
        """
        # Generate date grids
        dep_dates = np.arange(dep_start_jd, dep_end_jd + dep_step_days, dep_step_days)
        arr_dates = np.arange(arr_start_jd, arr_end_jd + arr_step_days, arr_step_days)

        n_dep = len(dep_dates)
        n_arr = len(arr_dates)

        # Initialize output grids with NaN
        tof_grid = np.full((n_arr, n_dep), np.nan)
        c3_grid = np.full((n_arr, n_dep), np.nan)
        vinf_arr_grid = np.full((n_arr, n_dep), np.nan)
        dv_total_grid = np.full((n_arr, n_dep), np.nan)

        # Pre-compute planet states for all dates
        dep_states = [self.ephemeris.get_state(self.departure_planet, jd) 
                      for jd in dep_dates]
        arr_states = [self.ephemeris.get_state(self.arrival_planet, jd) 
                      for jd in arr_dates]

        if parallel:
            # Parallel computation
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for i, dep_state in enumerate(dep_states):
                    for j, arr_state in enumerate(arr_states):
                        tof = (arr_state.epoch_jd - dep_state.epoch_jd) * DAY_S
                        if tof > 0:
                            future = executor.submit(
                                self._compute_cell,
                                dep_state, arr_state, tof, transfer_type,
                                parking_orbit_alt_dep, parking_orbit_alt_arr
                            )
                            futures[future] = (j, i)  # (dep_idx, arr_idx)

                for future in as_completed(futures):
                    j, i = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            tof_grid[i, j] = result['tof_days']
                            c3_grid[i, j] = result['c3']
                            vinf_arr_grid[i, j] = result['vinf_arr']
                            dv_total_grid[i, j] = result['dv_total']
                    except Exception:
                        pass
        else:
            # Sequential computation with progress tracking
            total_cells = n_dep * n_arr
            computed = 0

            for i, dep_state in enumerate(dep_states):
                for j, arr_state in enumerate(arr_states):
                    tof = (arr_state.epoch_jd - dep_state.epoch_jd) * DAY_S

                    if tof <= 0:
                        continue

                    result = self._compute_cell(
                        dep_state, arr_state, tof, transfer_type,
                        parking_orbit_alt_dep, parking_orbit_alt_arr
                    )

                    if result is not None:
                        tof_grid[j, i] = result['tof_days']
                        c3_grid[j, i] = result['c3']
                        vinf_arr_grid[j, i] = result['vinf_arr']
                        dv_total_grid[j, i] = result['dv_total']

                    computed += 1

        return PorkchopGrid(
            departure_dates=dep_dates,
            arrival_dates=arr_dates,
            tof_grid=tof_grid,
            c3_grid=c3_grid,
            vinf_arr_grid=vinf_arr_grid,
            dv_total_grid=dv_total_grid,
            transfer_type=transfer_type
        )

    def _compute_cell(self,
                      dep_state: PlanetState,
                      arr_state: PlanetState,
                      tof: float,
                      transfer_type: TransferType,
                      park_alt_dep: float,
                      park_alt_arr: float) -> Optional[Dict]:
        """
        Compute metrics for a single grid cell.

        Parameters
        ----------
        dep_state, arr_state : PlanetState
            Planet states at departure and arrival
        tof : float
            Time of flight [s]
        transfer_type : TransferType
        park_alt_dep, park_alt_arr : float
            Parking orbit altitudes [km]

        Returns
        -------
        dict or None
            Dictionary with computed metrics, or None if Lambert fails
        """
        try:
            # Solve Lambert problem
            solution = self.lambert.solve(
                dep_state.position,
                arr_state.position,
                tof,
                transfer_type=transfer_type
            )

            # Compute C3 (departure energy)
            v_dep_planet = dep_state.velocity
            v_inf_dep = solution.v1 - v_dep_planet
            vinf_dep_mag = np.linalg.norm(v_inf_dep)
            c3 = vinf_dep_mag**2  # [km^2/s^2]

            # Compute arrival V_inf
            v_arr_planet = arr_state.velocity
            v_inf_arr = solution.v2 - v_arr_planet
            vinf_arr_mag = np.linalg.norm(v_inf_arr)

            # Compute Delta-V for departure (escape from parking orbit)
            r_park_dep = self.r_dep + park_alt_dep
            v_park_dep = np.sqrt(self.mu_dep / r_park_dep)
            v_esc_dep = np.sqrt(vinf_dep_mag**2 + 2 * self.mu_dep / r_park_dep)
            dv_dep = abs(v_esc_dep - v_park_dep)

            # Compute Delta-V for arrival (capture into parking orbit)
            r_park_arr = self.r_arr + park_alt_arr
            v_park_arr = np.sqrt(self.mu_arr / r_park_arr)
            v_esc_arr = np.sqrt(vinf_arr_mag**2 + 2 * self.mu_arr / r_park_arr)
            dv_arr = abs(v_esc_arr - v_park_arr)

            dv_total = dv_dep + dv_arr

            return {
                'tof_days': tof / DAY_S,
                'c3': c3,
                'vinf_dep': vinf_dep_mag,
                'vinf_arr': vinf_arr_mag,
                'dv_dep': dv_dep,
                'dv_arr': dv_arr,
                'dv_total': dv_total,
                'solution': solution
            }

        except LambertError:
            return None
        except Exception:
            return None

    def compute_grid_vectorized(self,
                                dep_start_jd: float,
                                dep_end_jd: float,
                                dep_step_days: float,
                                arr_start_jd: float,
                                arr_end_jd: float,
                                arr_step_days: float,
                                transfer_type: TransferType = TransferType.SHORT_WAY,
                                parking_orbit_alt_dep: float = 200.0,
                                parking_orbit_alt_arr: float = 200.0) -> PorkchopGrid:
        """
        Vectorized grid computation (experimental, may have memory issues for large grids).

        Uses numpy broadcasting for speed but requires significant memory.
        """
        dep_dates = np.arange(dep_start_jd, dep_end_jd + dep_step_days, dep_step_days)
        arr_dates = np.arange(arr_start_jd, arr_end_jd + arr_step_days, arr_step_days)

        n_dep = len(dep_dates)
        n_arr = len(arr_dates)

        # Pre-compute states
        dep_states = [self.ephemeris.get_state(self.departure_planet, jd) 
                      for jd in dep_dates]
        arr_states = [self.ephemeris.get_state(self.arrival_planet, jd) 
                      for jd in arr_dates]

        # Extract positions and velocities
        dep_pos = np.array([s.position for s in dep_states])  # (n_dep, 3)
        dep_vel = np.array([s.velocity for s in dep_states])  # (n_dep, 3)
        arr_pos = np.array([s.position for s in arr_states])  # (n_arr, 3)
        arr_vel = np.array([s.velocity for s in arr_states])  # (n_arr, 3)

        # Create meshgrids for broadcasting
        # dep_pos_expanded: (n_arr, n_dep, 3)
        # arr_pos_expanded: (n_arr, n_dep, 3)
        dep_pos_m = np.tile(dep_pos[np.newaxis, :, :], (n_arr, 1, 1))
        arr_pos_m = np.tile(arr_pos[:, np.newaxis, :], (1, n_dep, 1))
        dep_vel_m = np.tile(dep_vel[np.newaxis, :, :], (n_arr, 1, 1))
        arr_vel_m = np.tile(arr_vel[:, np.newaxis, :], (1, n_dep, 1))

        # TOF grid [s]
        dep_jd_m = np.tile(dep_dates[np.newaxis, :], (n_arr, 1))
        arr_jd_m = np.tile(arr_dates[:, np.newaxis], (1, n_dep))
        tof_grid = (arr_jd_m - dep_jd_m) * DAY_S

        # Initialize output grids
        c3_grid = np.full((n_arr, n_dep), np.nan)
        vinf_arr_grid = np.full((n_arr, n_dep), np.nan)
        dv_total_grid = np.full((n_arr, n_dep), np.nan)

        # Process in batches to avoid memory issues
        batch_size = min(50, n_dep)

        for batch_start in range(0, n_dep, batch_size):
            batch_end = min(batch_start + batch_size, n_dep)

            for j in range(batch_start, batch_end):
                for i in range(n_arr):
                    if tof_grid[i, j] <= 0:
                        continue

                    try:
                        solution = self.lambert.solve(
                            dep_pos_m[i, j],
                            arr_pos_m[i, j],
                            tof_grid[i, j],
                            transfer_type=transfer_type
                        )

                        # C3
                        v_inf_dep = solution.v1 - dep_vel_m[i, j]
                        c3_grid[i, j] = np.linalg.norm(v_inf_dep)**2

                        # V_inf arrival
                        v_inf_arr = solution.v2 - arr_vel_m[i, j]
                        vinf_arr_grid[i, j] = np.linalg.norm(v_inf_arr)

                        # Delta-V
                        vinf_dep_mag = np.sqrt(c3_grid[i, j])
                        vinf_arr_mag = vinf_arr_grid[i, j]

                        r_park_dep = self.r_dep + parking_orbit_alt_dep
                        r_park_arr = self.r_arr + parking_orbit_alt_arr

                        v_park_dep = np.sqrt(self.mu_dep / r_park_dep)
                        v_park_arr = np.sqrt(self.mu_arr / r_park_arr)

                        v_esc_dep = np.sqrt(vinf_dep_mag**2 + 2 * self.mu_dep / r_park_dep)
                        v_esc_arr = np.sqrt(vinf_arr_mag**2 + 2 * self.mu_arr / r_park_arr)

                        dv_total_grid[i, j] = abs(v_esc_dep - v_park_dep) + abs(v_esc_arr - v_park_arr)

                    except (LambertError, ValueError):
                        continue

        return PorkchopGrid(
            departure_dates=dep_dates,
            arrival_dates=arr_dates,
            tof_grid=tof_grid / DAY_S,
            c3_grid=c3_grid,
            vinf_arr_grid=vinf_arr_grid,
            dv_total_grid=dv_total_grid,
            transfer_type=transfer_type
        )

    def find_optimal_window(self,
                          grid: PorkchopGrid,
                          criterion: str = 'c3',
                          max_c3: Optional[float] = None,
                          max_dv: Optional[float] = None,
                          min_tof: Optional[float] = None,
                          max_tof: Optional[float] = None) -> Dict:
        """
        Find optimal launch window from computed grid.

        Parameters
        ----------
        grid : PorkchopGrid
            Computed porkchop grid
        criterion : str
            Optimization criterion: 'c3', 'dv', 'tof'
        max_c3 : float, optional
            Maximum allowable C3 [km^2/s^2]
        max_dv : float, optional
            Maximum allowable Delta-V [km/s]
        min_tof, max_tof : float, optional
            TOF constraints [days]

        Returns
        -------
        dict
            Optimal solution details
        """
        # Create mask for valid cells
        mask = np.ones(grid.c3_grid.shape, dtype=bool)

        if max_c3 is not None:
            mask &= grid.c3_grid <= max_c3
        if max_dv is not None:
            mask &= grid.dv_total_grid <= max_dv
        if min_tof is not None:
            mask &= grid.tof_grid >= min_tof
        if max_tof is not None:
            mask &= grid.tof_grid <= max_tof

        # Ensure we have valid cells
        valid_c3 = np.where(mask, grid.c3_grid, np.inf)
        valid_dv = np.where(mask, grid.dv_total_grid, np.inf)
        valid_tof = np.where(mask, grid.tof_grid, np.inf)

        if criterion == 'c3':
            idx = np.unravel_index(np.nanargmin(valid_c3), valid_c3.shape)
        elif criterion == 'dv':
            idx = np.unravel_index(np.nanargmin(valid_dv), valid_dv.shape)
        elif criterion == 'tof':
            idx = np.unravel_index(np.nanargmin(valid_tof), valid_tof.shape)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return {
            'departure_jd': grid.departure_dates[idx[1]],
            'arrival_jd': grid.arrival_dates[idx[0]],
            'tof_days': grid.tof_grid[idx],
            'c3': grid.c3_grid[idx],
            'vinf_arr': grid.vinf_arr_grid[idx],
            'dv_total': grid.dv_total_grid[idx],
            'transfer_type': grid.transfer_type
        }