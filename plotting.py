"""
Plotting Suite for Porkchop Analysis.
Visualizzazione ottimizzata senza artefatti di bordo e con contorni ben definiti.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_porkchop(porkchop_data, title="Earth-Mars 2026 Launch Window"):
    dep_jds = porkchop_data["dep_jds"]
    arr_jds = porkchop_data["arr_jds"]
    
    X, Y = np.meshgrid(dep_jds - dep_jds[0], arr_jds - arr_jds[0])
    c3_matrix = porkchop_data["C3_dep"]
    vinf_matrix = porkchop_data.get("Vinf_arr", None)
    tof_matrix = porkchop_data["TOF"]
    
    if np.all(np.isnan(c3_matrix)):
        print("ERRORE: Griglia C3 vuota.")
        return
        
    fig, ax = plt.subplots(figsize=(12, 8.5))
    
    # 1. Contorni C3 di partenza
    min_c3 = np.nanmin(c3_matrix)
    max_c3 = min(min_c3 + 30.0, np.nanmax(c3_matrix))
    c3_levels = np.linspace(min_c3, max_c3, 22)
    
    cs_c3 = ax.contourf(X, Y, c3_matrix, levels=c3_levels, cmap="viridis_r", alpha=0.9, extend="max")
    cbar = fig.colorbar(cs_c3, ax=ax, pad=0.02)
    cbar.set_label(r"$C_3$ Departure ($\mathrm{km}^2/\mathrm{s}^2$)", fontsize=11, fontweight="bold")
    
    # 2. Isolinee Time of Flight (TOF)
    cs_tof = ax.contour(X, Y, tof_matrix, levels=np.arange(120, 480, 30), colors="white", alpha=0.6, linestyles="--", linewidths=1.0)
    ax.clabel(cs_tof, fmt="%1.0f d", inline=True, fontsize=8, colors="white")
    
    # 3. Isolinee V_infinity di arrivo
    if vinf_matrix is not None and not np.all(np.isnan(vinf_matrix)):
        vinf_min = np.nanmin(vinf_matrix)
        vinf_max = min(vinf_min + 3.5, np.nanmax(vinf_matrix))
        vinf_levels = np.arange(np.floor(vinf_min), np.ceil(vinf_max) + 0.5, 0.5)
        
        cs_vinf = ax.contour(X, Y, vinf_matrix, levels=vinf_levels, colors="#FF6B35", alpha=0.85, linestyles="-.", linewidths=1.2)
        ax.clabel(cs_vinf, fmt="%1.1f km/s", inline=True, fontsize=8, colors="#FF6B35")
        
        ax.plot([], [], color="#FF6B35", linestyle="-.", label=r"Arrival $V_{\infty}$ (km/s)")
    
    ax.plot([], [], color="white", linestyle="--", label="Time of Flight (days)")
    
    # 4. Marker Minimo C3
    min_idx = np.unravel_index(np.nanargmin(c3_matrix), c3_matrix.shape)
    ax.scatter(
        [X[min_idx]], [Y[min_idx]], 
        color="gold", marker="*", s=300, edgecolors="black", 
        zorder=5, label=f"Min $C_3$ ({min_c3:.2f} $\mathrm{{km}}^2/\mathrm{{s}}^2$)"
    )
    
    ax.set_xlabel(f"Days after Departure Epoch (JD {dep_jds[0]:.1f})", fontsize=11)
    ax.set_ylabel(f"Days after Arrival Epoch (JD {arr_jds[0]:.1f})", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.85, facecolor="#222222", labelcolor="white")
    ax.grid(True, alpha=0.15, linestyle=":")
    
    plt.tight_layout()
    plt.show()

def plot_type_split(data_t1, data_t2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, data, ptype in zip([ax1, ax2], [data_t1, data_t2], ["Type-I (Short Way)", "Type-II (Long Way)"]):
        c3 = np.copy(data["C3_dep"])
        c3[c3 > 50.0] = np.nan
        
        if np.all(np.isnan(c3)):
            ax.set_title(f"{ptype} - Geometria Fuori Finestra", fontsize=12)
            ax.grid(True, alpha=0.2)
            continue
            
        X, Y = np.meshgrid(data["dep_jds"] - data["dep_jds"][0], data["arr_jds"] - data["arr_jds"][0])
        min_c3 = np.nanmin(c3)
        max_c3 = min(min_c3 + 30.0, np.nanmax(c3))
        
        levels = np.linspace(min_c3, max_c3, 20)
        cs = ax.contourf(X, Y, c3, levels=levels, cmap="viridis_r", alpha=0.9, extend="max")
        fig.colorbar(cs, ax=ax, label=r"$C_3$ Departure ($\mathrm{km}^2/\mathrm{s}^2$)")
        
        ax.contour(X, Y, data["TOF"], levels=8, colors="white", alpha=0.4, linestyles="--")
        
        if "Vinf_arr" in data and not np.all(np.isnan(data["Vinf_arr"])):
            ax.contour(X, Y, data["Vinf_arr"], levels=6, colors="#FF6B35", alpha=0.7, linestyles="-.")
            
        min_idx = np.unravel_index(np.nanargmin(c3), c3.shape)
        ax.scatter([X[min_idx]], [Y[min_idx]], color="gold", marker="*", s=200, edgecolors="black", zorder=5)
        
        ax.set_title(ptype, fontsize=12, fontweight="bold")
        ax.set_xlabel("Departure Relative Days")
        ax.set_ylabel("Arrival Relative Days")
        ax.grid(True, alpha=0.2)
        
    plt.tight_layout()
    plt.show()