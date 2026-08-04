"""
main.py
=======
Example script for Interplanetary Launch Window Analysis.

Demonstrates the complete workflow:
1. Compute porkchop grids for Earth-Mars 2026-2028 window
2. Generate classic porkchop plots
3. Compare Type-I vs Type-II transfers
4. Visualize optimal trajectory in 2D/3D

Usage:
    python main.py

Requirements:
    numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from ephemeris import EphemerisManager
from lambert_gooding import TransferType
from porkchop import PorkchopEngine
from plotting import PorkchopPlotter, TrajectoryPlotter


def main():
    """Run the complete interplanetary launch window analysis."""

    print("=" * 70)
    print("INTERPLANETARY LAUNCH WINDOW ANALYSIS")
    print("Earth to Mars Transfer - 2026/2028 Window")
    print("=" * 70)

    # Initialize components
    ephemeris = EphemerisManager()

    # Define date ranges for the 2026-2028 Mars launch window
    # Mars opposition in 2027: ~Feb 2027
    # Launch window typically opens ~3 months before opposition

    dep_start = datetime(2026, 6, 1)
    dep_end = datetime(2027, 6, 1)
    arr_start = datetime(2026, 12, 1)
    arr_end = datetime(2028, 1, 1)

    # Convert to Julian Dates
    dep_start_jd = ephemeris.jd_from_datetime(dep_start)
    dep_end_jd = ephemeris.jd_from_datetime(dep_end)
    arr_start_jd = ephemeris.jd_from_datetime(arr_start)
    arr_end_jd = ephemeris.jd_from_datetime(arr_end)

    # Grid resolution (coarser for faster computation)
    dep_step = 5.0   # days
    arr_step = 5.0   # days

    print(f"\nDeparture window: {dep_start.strftime('%Y-%m-%d')} to {dep_end.strftime('%Y-%m-%d')}")
    print(f"Arrival window:   {arr_start.strftime('%Y-%m-%d')} to {arr_end.strftime('%Y-%m-%d')}")
    print(f"Grid resolution:  {dep_step} days x {arr_step} days")

    # Initialize engine
    engine = PorkchopEngine(departure_planet="Earth", arrival_planet="Mars")

    # ============================================================
    # 1. Compute Type-I (Short-Way) Porkchop Grid
    # ============================================================
    print("\n[1/4] Computing Type-I (Short-Way) porkchop grid...")

    grid_short = engine.compute_grid(
        dep_start_jd=dep_start_jd,
        dep_end_jd=dep_end_jd,
        dep_step_days=dep_step,
        arr_start_jd=arr_start_jd,
        arr_end_jd=arr_end_jd,
        arr_step_days=arr_step,
        transfer_type=TransferType.SHORT_WAY,
        parking_orbit_alt_dep=200.0,
        parking_orbit_alt_arr=200.0
    )

    min_c3, min_dep, min_arr = grid_short.min_c3
    min_dv, _, _ = grid_short.min_dv
    print(f"  Type-I Minimum C3:  {min_c3:.2f} km²/s²")
    print(f"  Type-I Minimum ΔV:  {min_dv:.2f} km/s")

    # ============================================================
    # 2. Compute Type-II (Long-Way) Porkchop Grid
    # ============================================================
    print("\n[2/4] Computing Type-II (Long-Way) porkchop grid...")

    grid_long = engine.compute_grid(
        dep_start_jd=dep_start_jd,
        dep_end_jd=dep_end_jd,
        dep_step_days=dep_step,
        arr_start_jd=arr_start_jd,
        arr_end_jd=arr_end_jd,
        arr_step_days=arr_step,
        transfer_type=TransferType.LONG_WAY,
        parking_orbit_alt_dep=200.0,
        parking_orbit_alt_arr=200.0
    )

    min_c3_long, min_dep_long, min_arr_long = grid_long.min_c3
    min_dv_long, _, _ = grid_long.min_dv
    print(f"  Type-II Minimum C3: {min_c3_long:.2f} km²/s²")
    print(f"  Type-II Minimum ΔV: {min_dv_long:.2f} km/s")

    # ============================================================
    # 3. Generate Plots
    # ============================================================
    print("\n[3/4] Generating visualizations...")

    plotter = PorkchopPlotter()
    traj_plotter = TrajectoryPlotter()

    # Classic porkchop plot (Type-I)
    print("  - Classic porkchop plot (Type-I)...")
    fig1 = plotter.plot_classic(
        grid=grid_short,
        departure_planet="Earth",
        arrival_planet="Mars",
        save_path="porkchop_classic_type1.png"
    )

    # Type-I / Type-II comparison
    print("  - Type-I vs Type-II comparison...")
    fig2 = plotter.plot_split_types(
        grid_short=grid_short,
        grid_long=grid_long,
        departure_planet="Earth",
        arrival_planet="Mars",
        save_path="porkchop_split_types.png"
    )

    # Delta-V map
    print("  - Delta-V contour map...")
    fig3 = plotter.plot_dv_contour(
        grid=grid_short,
        departure_planet="Earth",
        arrival_planet="Mars",
        save_path="porkchop_dv_map.png"
    )

    # 2D Trajectory for optimal Type-I transfer
    print("  - 2D heliocentric trajectory (optimal Type-I)...")
    min_c3, opt_dep_jd, opt_arr_jd = grid_short.min_c3
    fig4 = traj_plotter.plot_2d_trajectory(
        departure_planet="Earth",
        arrival_planet="Mars",
        dep_jd=opt_dep_jd,
        arr_jd=opt_arr_jd,
        transfer_type=TransferType.SHORT_WAY,
        save_path="trajectory_2d_optimal.png"
    )

    # 3D Trajectory
    print("  - 3D heliocentric trajectory (optimal Type-I)...")
    fig5 = traj_plotter.plot_3d_trajectory(
        departure_planet="Earth",
        arrival_planet="Mars",
        dep_jd=opt_dep_jd,
        arr_jd=opt_arr_jd,
        transfer_type=TransferType.SHORT_WAY,
        save_path="trajectory_3d_optimal.png"
    )

    # ============================================================
    # 4. Summary Report
    # ============================================================
    print("\n[4/4] Analysis Summary")
    print("-" * 50)

    opt_dep_dt = ephemeris.datetime_from_jd(opt_dep_jd)
    opt_arr_dt = ephemeris.datetime_from_jd(opt_arr_jd)
    tof_days = (opt_arr_jd - opt_dep_jd)

    print(f"\nOPTIMAL TYPE-I TRANSFER:")
    print(f"  Departure Date:     {opt_dep_dt.strftime('%Y-%m-%d')}")
    print(f"  Arrival Date:       {opt_arr_dt.strftime('%Y-%m-%d')}")
    print(f"  Time of Flight:     {tof_days:.1f} days")
    print(f"  C3 (departure):     {min_c3:.2f} km²/s²")
    print(f"  V∞ (arrival):       {grid_short.vinf_arr_grid[np.unravel_index(np.nanargmin(grid_short.c3_grid), grid_short.c3_grid.shape)]:.2f} km/s")
    print(f"  Total ΔV:           {min_dv:.2f} km/s")

    print(f"\nOPTIMAL TYPE-II TRANSFER:")
    opt_dep_dt_long = ephemeris.datetime_from_jd(min_dep_long)
    opt_arr_dt_long = ephemeris.datetime_from_jd(min_arr_long)
    tof_days_long = (min_arr_long - min_dep_long)

    print(f"  Departure Date:     {opt_dep_dt_long.strftime('%Y-%m-%d')}")
    print(f"  Arrival Date:       {opt_arr_dt_long.strftime('%Y-%m-%d')}")
    print(f"  Time of Flight:     {tof_days_long:.1f} days")
    print(f"  C3 (departure):     {min_c3_long:.2f} km²/s²")
    print(f"  Total ΔV:           {min_dv_long:.2f} km/s")

    print("\n" + "=" * 70)
    print("Analysis complete. Figures saved to current directory.")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()