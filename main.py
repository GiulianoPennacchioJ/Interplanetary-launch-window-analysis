import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Mars
from poliastro.iod import lambert
from poliastro.constants import GM_sun
from poliastro.ephem import Ephem

from porkchop import PorkchopCalculator
from plotting import plot_porkchop

def poliastro_lambert_solver(dep_mjd, arr_mjd):
    """
    Real planetary ephemeris lookup and Lambert problem solver using Poliastro.
    
    Parameters:
    - dep_mjd: Departure date in Modified Julian Date (MJD).
    - arr_mjd: Arrival date in Modified Julian Date (MJD).
    
    Returns:
    - v_inf_dep_vec: 3D numpy array of departure hyperbolic excess velocity (km/s).
    - v_inf_arr_vec: 3D numpy array of arrival hyperbolic excess velocity (km/s).
    """
    # Convert Modified Julian Dates to Astropy Time objects (TDB scale for dynamical accuracy)
    t_dep = Time(dep_mjd, format="mjd", scale="tdb")
    t_arr = Time(arr_mjd, format="mjd", scale="tdb")
    
    # Retrieve Sun gravitational parameter in km^3 / s^2
    k = GM_sun.to(u.km**3 / u.s**2).value
    
    # Get position (r) and velocity (v) vectors using Ephem.from_body (compatible with modern poliastro versions)
    r1_q, v1_q = Ephem.from_body(Earth, t_dep).rv(t_dep)
    r2_q, v2_q = Ephem.from_body(Mars, t_arr).rv(t_arr)
    
    # Extract numerical values in km and km/s
    r1 = r1_q.to(u.km).value
    v1 = v1_q.to(u.km / u.s).value
    r2 = r2_q.to(u.km).value
    v2 = v2_q.to(u.km / u.s).value
    
    # Calculate time of flight in seconds
    tof_sec = (arr_mjd - dep_mjd) * 86400.0
    
    # Solve Lambert's problem (returns initial and final velocity vectors of the transfer orbit)
    v_init, v_final = lambert(k, r1, r2, tof_sec)
    
    # Compute hyperbolic excess velocity vectors
    # Departure: Transfer initial velocity minus Earth's heliocentric velocity
    v_inf_dep_vec = v_init - v1
    
    # Arrival: Mars' heliocentric velocity minus Transfer final velocity
    v_inf_arr_vec = v2 - v_final
    
    return v_inf_dep_vec, v_inf_arr_vec

def main():
    # 1. Define the search space grid for an Earth-Mars transfer (Dates in MJD)
    # Example window for a 2026/2027 Earth-Mars opportunity
    dep_dates = np.linspace(61200.0, 61350.0, 50)  # 50 departure steps
    arr_dates = np.linspace(61400.0, 61800.0, 60)  # 60 arrival steps

    print(f"Initializing PorkchopCalculator for Earth-Mars Transfer...")
    print(f"Departure Grid Size: {len(dep_dates)} steps (MJD {dep_dates[0]:.1f} to {dep_dates[-1]:.1f})")
    print(f"Arrival Grid Size:   {len(arr_dates)} steps (MJD {arr_dates[0]:.1f} to {arr_dates[-1]:.1f})")

    # 2. Instantiate the calculator using the real Poliastro Lambert solver
    calculator = PorkchopCalculator(
        dep_dates=dep_dates, 
        arr_dates=arr_dates, 
        lambert_solver=poliastro_lambert_solver
    )

    # 3. Execute the grid computations
    print("Computing ephemerides and solving Lambert's problem across the grid...")
    c3_grid, vinf_arr_grid, tof_grid = calculator.compute()

    # 4. Render the output porkchop plot
    print("Generating high-fidelity porkchop plot visualization...")
    fig, ax = plot_porkchop(
        dep_dates=dep_dates, 
        arr_dates=arr_dates, 
        c3_grid=c3_grid, 
        tof_grid=tof_grid, 
        max_c3=45.0, 
        title="Earth-Mars Interplanetary Transfer Porkchop Plot"
    )

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()