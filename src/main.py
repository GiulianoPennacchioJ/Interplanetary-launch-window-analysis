import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Mars
from poliastro.iod import lambert
from poliastro.constants import GM_sun
from poliastro.ephem import Ephem

from porkchop import PorkchopCalculator
from plotting import plot_porkchop, plot_vinf_arrival, plot_trajectory_3d

def poliastro_lambert_solver(dep_mjd, arr_mjd):
    t_dep = Time(dep_mjd, format="mjd", scale="tdb")
    t_arr = Time(arr_mjd, format="mjd", scale="tdb")
    
    k = GM_sun.to(u.km**3 / u.s**2)
    
    r1_q, v1_q = Ephem.from_body(Earth, t_dep).rv(t_dep)
    r2_q, v2_q = Ephem.from_body(Mars, t_arr).rv(t_arr)
    
    r1 = r1_q.to(u.km)
    v1 = v1_q.to(u.km / u.s)
    r2 = r2_q.to(u.km)
    v2 = v2_q.to(u.km / u.s)
    
    tof_sec = ((arr_mjd - dep_mjd) * 86400.0) * u.s
    v_init, v_final = lambert(k, r1, r2, tof_sec)
    
    v_inf_dep_vec = (v_init - v1).to_value(u.km / u.s)
    v_inf_arr_vec = (v2 - v_final).to_value(u.km / u.s)
    
    return v_inf_dep_vec, v_inf_arr_vec

def main():
    # 1. Griglia di ricerca spaziale (MJD)
    dep_dates = np.linspace(61200.0, 61350.0, 50)
    arr_dates = np.linspace(61400.0, 61800.0, 60)

    print("Initializing PorkchopCalculator for Earth-Mars Transfer...")
    calculator = PorkchopCalculator(
        dep_dates=dep_dates, 
        arr_dates=arr_dates, 
        lambert_solver=poliastro_lambert_solver
    )

    # 2. Calcolo della griglia
    print("Computing ephemerides and solving Lambert's problem across the grid...")
    c3_grid, vinf_arr_grid, tof_grid = calculator.compute()

    # 3. Generazione dei grafici di analisi
    print("Generating mission analysis plots...")
    
    # A. Porkchop Plot C3 + TOF contours
    fig1, ax1, (opt_dep, opt_arr) = plot_porkchop(
        dep_dates=dep_dates, 
        arr_dates=arr_dates, 
        c3_grid=c3_grid, 
        tof_grid=tof_grid, 
        max_c3=45.0, 
        title="Earth-Mars Transfer: C3 & TOF Analysis"
    )
    
    # B. Grafico della Velocità d'Arrivo
    fig2, ax2 = plot_vinf_arrival(
        dep_dates=dep_dates, 
        arr_dates=arr_dates, 
        vinf_arr_grid=vinf_arr_grid, 
        max_vinf=12.0, 
        title="Earth-Mars Transfer: Arrival Excess Velocity ($v_\infty$)"
    )
    
    # C. Visualizzazione 3D della Traiettoria Ottimale
    print(f"Plotting optimal 3D transfer trajectory for Departure MJD {opt_dep:.1f} and Arrival MJD {opt_arr:.1f}...")
    fig3 = plot_trajectory_3d(opt_dep, opt_arr)

    # Mostra tutte le finestre grafiche generate
    plt.show()

if __name__ == "__main__":
    main()