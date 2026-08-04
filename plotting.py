"""
plotting.py
===========
Comprehensive visualization suite for interplanetary trajectory analysis.

Provides publication-quality plots for:
- Classic porkchop plots (C3, V_inf, TOF iso-contours)
- Type-I / Type-II split plots
- 2D/3D heliocentric trajectory visualization

All plots use matplotlib with carefully tuned aesthetics for technical publications.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import warnings

from porkchop import PorkchopGrid, PorkchopEngine
from ephemeris import EphemerisManager
from lambert_ import LambertGooding, TransferType


# Default plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palettes
C3_CMAP = 'RdYlGn_r'      # Red = high energy, Green = low energy
VINF_CMAP = 'plasma'      # Plasma for arrival velocity
TOF_CMAP = 'viridis'      # Viridis for time of flight
DV_CMAP = 'coolwarm'      # Coolwarm for delta-V


class PorkchopPlotter:
    """
    Professional porkchop plot generator with publication-quality aesthetics.
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Initialize plotter.

        Parameters
        ----------
        figsize : tuple
            Default figure size (width, height) in inches
        """
        self.figsize = figsize
        self.ephemeris = EphemerisManager()

    def _jd_to_datetime(self, jd: float) -> datetime:
        """Convert Julian Date to datetime for plotting."""
        return self.ephemeris.datetime_from_jd(jd)

    def _format_dates(self, ax, dep_dates, arr_dates):
        """Format axis with datetime labels."""
        dep_dt = [self._jd_to_datetime(jd) for jd in dep_dates]
        arr_dt = [self._jd_to_datetime(jd) for jd in arr_dates]

        # Set ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

        return dep_dt, arr_dt

    def plot_classic(self,
                     grid: PorkchopGrid,
                     departure_planet: str,
                     arrival_planet: str,
                     c3_levels: Optional[List[float]] = None,
                     vinf_levels: Optional[List[float]] = None,
                     tof_levels: Optional[List[float]] = None,
                     show_optimal: bool = True,
                     save_path: Optional[str] = None,
                     dpi: int = 150) -> plt.Figure:
        """
        Generate a classic porkchop plot with C3, V_inf, and TOF contours.

        Parameters
        ----------
        grid : PorkchopGrid
            Computed porkchop grid data
        departure_planet, arrival_planet : str
            Planet names for title
        c3_levels : list, optional
            Custom C3 contour levels [km^2/s^2]
        vinf_levels : list, optional
            Custom V_inf contour levels [km/s]
        tof_levels : list, optional
            Custom TOF contour levels [days]
        show_optimal : bool
            Mark the minimum C3 point
        save_path : str, optional
            Path to save the figure
        dpi : int
            Figure resolution

        Returns
        -------
        plt.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        dep_dates = grid.departure_dates
        arr_dates = grid.arrival_dates

        # Convert to matplotlib date numbers
        dep_mpl = mdates.date2num([self._jd_to_datetime(jd) for jd in dep_dates])
        arr_mpl = mdates.date2num([self._jd_to_datetime(jd) for jd in arr_dates])

        dep_grid, arr_grid = np.meshgrid(dep_mpl, arr_mpl)

        # Default contour levels
        if c3_levels is None:
            c3_min = np.nanmin(grid.c3_grid)
            c3_max = np.nanmax(grid.c3_grid)
            c3_levels = np.linspace(c3_min, min(c3_max, c3_min * 5), 15)

        if vinf_levels is None:
            vinf_min = np.nanmin(grid.vinf_arr_grid)
            vinf_max = np.nanmax(grid.vinf_arr_grid)
            vinf_levels = np.linspace(vinf_min, min(vinf_max, vinf_min * 4), 10)

        if tof_levels is None:
            tof_min = np.nanmin(grid.tof_grid)
            tof_max = np.nanmax(grid.tof_grid)
            tof_levels = np.linspace(tof_min, tof_max, 8)

        # C3 contours (filled)
        c3_contourf = ax.contourf(dep_grid, arr_grid, grid.c3_grid, 
                                   levels=c3_levels, cmap=C3_CMAP, alpha=0.7)
        c3_contour = ax.contour(dep_grid, arr_grid, grid.c3_grid,
                                  levels=c3_levels, colors='black', 
                                  linewidths=0.5, alpha=0.5)
        ax.clabel(c3_contour, inline=True, fontsize=7, fmt='C3=%.0f')

        # V_inf contours (dashed lines)
        vinf_contour = ax.contour(dep_grid, arr_grid, grid.vinf_arr_grid,
                                    levels=vinf_levels, colors='blue',
                                    linewidths=1.5, linestyles='--', alpha=0.8)
        ax.clabel(vinf_contour, inline=True, fontsize=8, fmt='V\u221e=%.1f',
                   colors='blue')

        # TOF contours (dotted lines)
        tof_contour = ax.contour(dep_grid, arr_grid, grid.tof_grid,
                                   levels=tof_levels, colors='red',
                                   linewidths=1.0, linestyles=':', alpha=0.8)
        ax.clabel(tof_contour, inline=True, fontsize=8, fmt='TOF=%.0f d',
                   colors='red')

        # Mark optimal point
        if show_optimal:
            min_c3, min_dep_jd, min_arr_jd = grid.min_c3
            min_dep_mpl = mdates.date2num(self._jd_to_datetime(min_dep_jd))
            min_arr_mpl = mdates.date2num(self._jd_to_datetime(min_arr_jd))

            ax.plot(min_dep_mpl, min_arr_mpl, 'k*', markersize=20, 
                    markeredgecolor='white', markeredgewidth=1.5,
                    label=f'Min C3 = {min_c3:.1f} km\u00b2/s\u00b2')

            # Add annotation
            ax.annotate(f'Min C3\n{min_c3:.1f} km\u00b2/s\u00b2',
                        xy=(min_dep_mpl, min_arr_mpl),
                        xytext=(min_dep_mpl + 20, min_arr_mpl + 20),
                        fontsize=9, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                  edgecolor='black', alpha=0.9))

        # Formatting
        ax.set_xlabel('Departure Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Arrival Date', fontsize=12, fontweight='bold')

        transfer_name = "Short-Way (Type-I)" if grid.transfer_type == TransferType.SHORT_WAY else "Long-Way (Type-II)"
        ax.set_title(f'Porkchop Plot: {departure_planet} to {arrival_planet}\n'
                     f'{transfer_name} Transfer',
                     fontsize=14, fontweight='bold', pad=15)

        # Colorbar for C3
        cbar = fig.colorbar(c3_contourf, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('C3 Departure Energy [km\u00b2/s\u00b2]', fontsize=11)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=0.5, label='C3 contours'),
            Line2D([0], [0], color='blue', linewidth=1.5, linestyle='--', label='V\u221e arrival'),
            Line2D([0], [0], color='red', linewidth=1.0, linestyle=':', label='TOF'),
        ]
        if show_optimal:
            legend_elements.append(Line2D([0], [0], marker='*', color='w',
                                          markerfacecolor='black', markersize=15,
                                          label=f'Min C3 = {min_c3:.1f}'))
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                  framealpha=0.9)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print(f"Figure saved to {save_path}")

        return fig

    def plot_split_types(self,
                         grid_short: PorkchopGrid,
                         grid_long: PorkchopGrid,
                         departure_planet: str,
                         arrival_planet: str,
                         save_path: Optional[str] = None,
                         dpi: int = 150) -> plt.Figure:
        """
        Create side-by-side porkchop plots for Type-I and Type-II transfers.

        Parameters
        ----------
        grid_short, grid_long : PorkchopGrid
            Grids for short-way and long-way transfers
        departure_planet, arrival_planet : str
            Planet names
        save_path : str, optional
        dpi : int

        Returns
        -------
        plt.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))

        grids = [grid_short, grid_long]
        titles = ['Type-I: Short-Way Transfer', 'Type-II: Long-Way Transfer']

        for idx, (ax, grid, title) in enumerate(zip(axes, grids, titles)):
            dep_dates = grid.departure_dates
            arr_dates = grid.arrival_dates

            dep_mpl = mdates.date2num([self._jd_to_datetime(jd) for jd in dep_dates])
            arr_mpl = mdates.date2num([self._jd_to_datetime(jd) for jd in arr_dates])
            dep_grid, arr_grid = np.meshgrid(dep_mpl, arr_mpl)

            # C3 filled contours
            c3_min = np.nanmin(grid.c3_grid)
            c3_max = np.nanmax(grid.c3_grid)
            levels = np.linspace(c3_min, min(c3_max, c3_min * 4), 20)

            cf = ax.contourf(dep_grid, arr_grid, grid.c3_grid, 
                              levels=levels, cmap=C3_CMAP, alpha=0.8)

            # C3 line contours
            cs = ax.contour(dep_grid, arr_grid, grid.c3_grid,
                            levels=levels[::2], colors='black',
                            linewidths=0.5, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')

            # Mark minimum
            min_c3, min_dep, min_arr = grid.min_c3
            min_dep_mpl = mdates.date2num(self._jd_to_datetime(min_dep))
            min_arr_mpl = mdates.date2num(self._jd_to_datetime(min_arr))
            ax.plot(min_dep_mpl, min_arr_mpl, 'k*', markersize=18,
                    markeredgecolor='white', markeredgewidth=1.5)
            ax.annotate(f'{min_c3:.1f}', 
                        xy=(min_dep_mpl, min_arr_mpl),
                        xytext=(min_dep_mpl + 15, min_arr_mpl + 15),
                        fontsize=9, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black'))

            # V_inf contours
            vinf_levels = np.linspace(np.nanmin(grid.vinf_arr_grid),
                                       np.nanmax(grid.vinf_arr_grid), 8)
            ax.contour(dep_grid, arr_grid, grid.vinf_arr_grid,
                       levels=vinf_levels, colors='blue', linewidths=1.2,
                       linestyles='--', alpha=0.7)

            # Formatting
            ax.set_xlabel('Departure Date', fontsize=11, fontweight='bold')
            ax.set_ylabel('Arrival Date', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # Colorbar
            cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
            cbar.set_label('C3 [km\u00b2/s\u00b2]', fontsize=10)

        fig.suptitle(f'Porkchop Plot Comparison: {departure_planet} to {arrival_planet}',
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')

        return fig

    def plot_dv_contour(self,
                        grid: PorkchopGrid,
                        departure_planet: str,
                        arrival_planet: str,
                        save_path: Optional[str] = None,
                        dpi: int = 150) -> plt.Figure:
        """
        Plot Delta-V total as filled contours.

        Parameters
        ----------
        grid : PorkchopGrid
        departure_planet, arrival_planet : str
        save_path : str, optional
        dpi : int

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        dep_dates = grid.departure_dates
        arr_dates = grid.arrival_dates

        dep_mpl = mdates.date2num([self._jd_to_datetime(jd) for jd in dep_dates])
        arr_mpl = mdates.date2num([self._jd_to_datetime(jd) for jd in arr_dates])
        dep_grid, arr_grid = np.meshgrid(dep_mpl, arr_mpl)

        # Delta-V contours
        dv_min = np.nanmin(grid.dv_total_grid)
        dv_max = np.nanmax(grid.dv_total_grid)
        levels = np.linspace(dv_min, min(dv_max, dv_min * 3), 20)

        cf = ax.contourf(dep_grid, arr_grid, grid.dv_total_grid,
                          levels=levels, cmap=DV_CMAP, alpha=0.85)
        cs = ax.contour(dep_grid, arr_grid, grid.dv_total_grid,
                        levels=levels[::2], colors='black',
                        linewidths=0.5, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

        # Mark minimum
        min_dv, min_dep, min_arr = grid.min_dv
        min_dep_mpl = mdates.date2num(self._jd_to_datetime(min_dep))
        min_arr_mpl = mdates.date2num(self._jd_to_datetime(min_arr))
        ax.plot(min_dep_mpl, min_arr_mpl, 'k*', markersize=20,
                markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'Min ΔV = {min_dv:.2f} km/s',
                    xy=(min_dep_mpl, min_arr_mpl),
                    xytext=(min_dep_mpl + 20, min_arr_mpl + 20),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                              edgecolor='black', alpha=0.9))

        ax.set_xlabel('Departure Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Arrival Date', fontsize=12, fontweight='bold')
        ax.set_title(f'Total ΔV Map: {departure_planet} to {arrival_planet}',
                     fontsize=14, fontweight='bold')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
        cbar.set_label('Total ΔV [km/s]', fontsize=11)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')

        return fig


class TrajectoryPlotter:
    """
    2D and 3D trajectory visualization in heliocentric frame.
    """

    def __init__(self):
        self.ephemeris = EphemerisManager()
        self.lambert = LambertGooding()

    def _propagate_orbit(self, r0: np.ndarray, v0: np.ndarray, 
                         mu: float, t_span: np.ndarray) -> np.ndarray:
        """
        Propagate a Keplerian orbit numerically.

        Parameters
        ----------
        r0, v0 : np.ndarray
            Initial position and velocity [km, km/s]
        mu : float
            Gravitational parameter [km^3/s^2]
        t_span : np.ndarray
            Time points [s]

        Returns
        -------
        np.ndarray
            Array of position vectors (n, 3)
        """
        from scipy.integrate import solve_ivp

        def equations(t, state):
            r = state[:3]
            v = state[3:]
            r_norm = np.linalg.norm(r)
            acc = -mu / r_norm**3 * r
            return np.concatenate([v, acc])

        y0 = np.concatenate([r0, v0])
        sol = solve_ivp(equations, [t_span[0], t_span[-1]], y0, 
                        t_eval=t_span, method='DOP853', rtol=1e-10, atol=1e-12)

        return sol.y[:3, :].T

    def plot_2d_trajectory(self,
                         departure_planet: str,
                         arrival_planet: str,
                         dep_jd: float,
                         arr_jd: float,
                         transfer_type: TransferType = TransferType.SHORT_WAY,
                         n_points: int = 500,
                         show_orbits: bool = True,
                         save_path: Optional[str] = None,
                         dpi: int = 150) -> plt.Figure:
        """
        Plot 2D heliocentric trajectory in the ecliptic plane.

        Parameters
        ----------
        departure_planet, arrival_planet : str
        dep_jd, arr_jd : float
            Departure and arrival Julian Dates
        transfer_type : TransferType
        n_points : int
            Number of points for trajectory propagation
        show_orbits : bool
            Show full planet orbits
        save_path : str, optional
        dpi : int

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        # Get planet states
        dep_state = self.ephemeris.get_state(departure_planet, dep_jd)
        arr_state = self.ephemeris.get_state(arrival_planet, arr_jd)

        # Solve Lambert problem
        tof = (arr_jd - dep_jd) * 86400.0
        solution = self.lambert.solve(dep_state.position, arr_state.position, 
                                       tof, transfer_type=transfer_type)

        # Propagate transfer orbit
        t_span = np.linspace(0, tof, n_points)
        transfer_r = self._propagate_orbit(dep_state.position, solution.v1,
                                            1.32712440018e11, t_span)

        # Plot Sun
        ax.plot(0, 0, 'yo', markersize=20, markeredgecolor='orange',
                markeredgewidth=2, label='Sun', zorder=5)

        # Plot planet orbits (approximate circles/ellipses)
        if show_orbits:
            for planet, color, style in [(departure_planet, 'blue', '-'),
                                          (arrival_planet, 'red', '-')]:
                # Generate orbit by propagating for one period
                elements = self.ephemeris.get_elements(planet, dep_jd)
                a_km = elements.a * 149597870.7
                period = 2 * np.pi * np.sqrt(a_km**3 / 1.32712440018e11)

                state0 = self.ephemeris.get_state(planet, dep_jd)
                t_orbit = np.linspace(0, period, 500)
                orbit_r = self._propagate_orbit(state0.position, state0.velocity,
                                                 1.32712440018e11, t_orbit)
                ax.plot(orbit_r[:, 0], orbit_r[:, 1], color=color, 
                        linestyle=style, alpha=0.3, linewidth=1)

        # Plot transfer trajectory
        ax.plot(transfer_r[:, 0], transfer_r[:, 1], 'g-', linewidth=2.5,
                label='Transfer Trajectory', zorder=3)

        # Mark departure and arrival points
        ax.plot(dep_state.position[0], dep_state.position[1], 'bo',
                markersize=12, markeredgecolor='white', markeredgewidth=2,
                label=f'{departure_planet} (departure)', zorder=4)
        ax.plot(arr_state.position[0], arr_state.position[1], 'ro',
                markersize=12, markeredgecolor='white', markeredgewidth=2,
                label=f'{arrival_planet} (arrival)', zorder=4)

        # Draw velocity vectors (scaled)
        scale = 5e6  # Scale factor for visualization
        ax.arrow(dep_state.position[0], dep_state.position[1],
                 solution.v1[0] * scale, solution.v1[1] * scale,
                 head_width=2e7, head_length=3e7, fc='green', ec='green',
                 alpha=0.7, linewidth=1.5, zorder=4)
        ax.arrow(arr_state.position[0], arr_state.position[1],
                 solution.v2[0] * scale, solution.v2[1] * scale,
                 head_width=2e7, head_length=3e7, fc='purple', ec='purple',
                 alpha=0.7, linewidth=1.5, zorder=4)

        # Formatting
        ax.set_aspect('equal')
        ax.set_xlabel('X [km]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y [km]', fontsize=12, fontweight='bold')

        dep_dt = self.ephemeris.datetime_from_jd(dep_jd)
        arr_dt = self.ephemeris.datetime_from_jd(arr_jd)

        ax.set_title(f'Heliocentric Transfer: {departure_planet} to {arrival_planet}\n'
                     f'Departure: {dep_dt.strftime("%Y-%m-%d")} | '
                     f'Arrival: {arr_dt.strftime("%Y-%m-%d")} | '
                     f'TOF: {tof/86400:.0f} days',
                     fontsize=13, fontweight='bold')

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Set axis limits
        max_r = max(np.linalg.norm(dep_state.position),
                    np.linalg.norm(arr_state.position)) * 1.3
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')

        return fig

    def plot_3d_trajectory(self,
                           departure_planet: str,
                           arrival_planet: str,
                           dep_jd: float,
                           arr_jd: float,
                           transfer_type: TransferType = TransferType.SHORT_WAY,
                           n_points: int = 500,
                           save_path: Optional[str] = None,
                           dpi: int = 150) -> plt.Figure:
        """
        Plot 3D heliocentric trajectory.

        Parameters
        ----------
        departure_planet, arrival_planet : str
        dep_jd, arr_jd : float
        transfer_type : TransferType
        n_points : int
        save_path : str, optional
        dpi : int

        Returns
        -------
        plt.Figure
        """
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Get states and solve Lambert
        dep_state = self.ephemeris.get_state(departure_planet, dep_jd)
        arr_state = self.ephemeris.get_state(arrival_planet, arr_jd)
        tof = (arr_jd - dep_jd) * 86400.0
        solution = self.lambert.solve(dep_state.position, arr_state.position,
                                       tof, transfer_type=transfer_type)

        # Propagate transfer orbit
        t_span = np.linspace(0, tof, n_points)
        transfer_r = self._propagate_orbit(dep_state.position, solution.v1,
                                            1.32712440018e11, t_span)

        # Plot Sun
        ax.scatter([0], [0], [0], c='yellow', s=200, marker='o',
                   edgecolors='orange', linewidths=2, label='Sun')

        # Plot transfer trajectory
        ax.plot(transfer_r[:, 0], transfer_r[:, 1], transfer_r[:, 2],
                'g-', linewidth=2.5, label='Transfer', alpha=0.8)

        # Plot departure and arrival points
        ax.scatter([dep_state.position[0]], [dep_state.position[1]], 
                   [dep_state.position[2]], c='blue', s=150, marker='o',
                   edgecolors='white', linewidths=2, label=departure_planet)
        ax.scatter([arr_state.position[0]], [arr_state.position[1]],
                   [arr_state.position[2]], c='red', s=150, marker='o',
                   edgecolors='white', linewidths=2, label=arrival_planet)

        # Plot ecliptic plane reference
        max_r = max(np.linalg.norm(dep_state.position),
                    np.linalg.norm(arr_state.position)) * 1.2
        xx, yy = np.meshgrid(np.linspace(-max_r, max_r, 10),
                             np.linspace(-max_r, max_r, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.05, color='gray')

        # Formatting
        ax.set_xlabel('X [km]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y [km]', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z [km]', fontsize=11, fontweight='bold')

        dep_dt = self.ephemeris.datetime_from_jd(dep_jd)
        ax.set_title(f'3D Heliocentric Transfer: {departure_planet} to {arrival_planet}\n'
                     f'Departure: {dep_dt.strftime("%Y-%m-%d")} | TOF: {tof/86400:.0f} days',
                     fontsize=13, fontweight='bold')

        ax.legend(loc='upper left', fontsize=10)

        # Equal aspect ratio
        max_range = max_r
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range * 0.3, max_range * 0.3])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')

        return fig