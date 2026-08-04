import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Sun, Earth, Mars
from poliastro.twobody import Orbit
from poliastro.ephem import Ephem
from poliastro.iod import lambert
from poliastro.constants import GM_sun

def plot_porkchop(dep_dates, arr_dates, c3_grid, tof_grid, max_c3=45.0, title="Porkchop Plot"):
    X, Y = np.meshgrid(dep_dates, arr_dates)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    c3_plot = np.ma.masked_invalid(c3_grid)
    c3_plot = np.ma.masked_where(c3_plot > max_c3, c3_plot)
    
    levels_c3 = np.linspace(c3_plot.min(), max_c3, 25)
    cs = ax.contourf(X, Y, c3_plot, levels=levels_c3, cmap="turbo", extend="max")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"$C_3$ ($\mathrm{km}^2/\mathrm{s}^2$)")
    
    tof_days = tof_grid
    levels_tof = np.arange(50, 600, 29)
    cc = ax.contour(X, Y, tof_days, levels=levels_tof, colors="black", linewidths=0.8, alpha=0.7)
    ax.clabel(cc, fmt="%d d", fontsize=9, inline=True)
    
    min_idx = np.unravel_index(np.nanargmin(c3_grid), c3_grid.shape)
    opt_dep = dep_dates[min_idx[1]]
    opt_arr = arr_dates[min_idx[0]]
    opt_c3 = c3_grid[min_idx]
    
    ax.plot(opt_dep, opt_arr, "r*", markersize=12, label=f"Min C3: {opt_c3:.2f} km^2/s^2")
    
    ax.set_title(title)
    ax.set_xlabel("Departure Date (MJD)")
    ax.set_ylabel("Arrival Date (MJD)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    
    return fig, ax, (opt_dep, opt_arr)

def plot_vinf_arrival(dep_dates, arr_dates, vinf_arr_grid, max_vinf=15.0, title="Arrival Velocity Plot"):
    X, Y = np.meshgrid(dep_dates, arr_dates)
    fig, ax = plt.subplots(figsize=(10, 8))
    vinf_plot = np.ma.masked_invalid(vinf_arr_grid)
    vinf_plot = np.ma.masked_where(vinf_plot > max_vinf, vinf_plot)
    
    levels = np.linspace(vinf_plot.min(), max_vinf, 20)
    cs = ax.contourf(X, Y, vinf_plot, levels=levels, cmap="viridis", extend="max")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"Arrival v_inf (km/s)")
    
    ax.set_title(title)
    ax.set_xlabel("Departure Date (MJD)")
    ax.set_ylabel("Arrival Date (MJD)")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    return fig, ax

def plot_trajectory_3d(dep_mjd, arr_mjd):
    """
    Genera la visualizzazione 3D nativa in Matplotlib della traiettoria 
    eliocentrica di trasferimento tra la Terra e Marte.
    """
    t_dep = Time(dep_mjd, format="mjd", scale="tdb")
    t_arr = Time(arr_mjd, format="mjd", scale="tdb")
    
    k = GM_sun.to(u.km**3 / u.s**2)
    
    earth_ephem = Ephem.from_body(Earth, t_dep)
    mars_ephem = Ephem.from_body(Mars, t_arr)
    
    r1_q, v1_q = earth_ephem.rv(t_dep)
    r2_q, v2_q = mars_ephem.rv(t_arr)
    
    tof_sec = ((arr_mjd - dep_mjd) * 86400.0) * u.s
    v_init, v_final = lambert(k, r1_q, r2_q, tof_sec)
    
    ss_transfer = Orbit.from_vectors(Sun, r1_q, v_init, epoch=t_dep)
    
    # Campiona i punti dell'orbita di trasferimento in unità astronomiche (AU)
    n_points = 200
    times = t_dep + np.linspace(0, tof_sec.value, n_points) * u.s
    transfer_coords = np.array([ss_transfer.propagate(t).r.to(u.au).value for t in times])
    
    r1_au = r1_q.to(u.au).value
    r2_au = r2_q.to(u.au).value
    
    # Configurazione della figura 3D con Matplotlib puro
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    
    # Sole nell'origine
    ax.scatter([0], [0], [0], color="yellow", s=300, label="Sun", depthpath=None)
    
    # Orbita di trasferimento
    ax.plot(transfer_coords[:, 0], transfer_coords[:, 1], transfer_coords[:, 2], 
            color="orange", linestyle="--", linewidth=2, label="Transfer Orbit")
    
    # Posizioni di Terra e Marte
    ax.scatter([r1_au[0]], [r1_au[1]], [r1_au[2]], color="blue", s=80, label="Earth (Departure)")
    ax.scatter([r2_au[0]], [r2_au[1]], [r2_au[2]], color="red", s=80, label="Mars (Arrival)")
    
    ax.set_title(f"3D Interplanetary Trajectory (TOF: {(arr_mjd - dep_mjd):.1f} days)")
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_zlabel("Z (AU)")
    ax.legend()
    
    return fig