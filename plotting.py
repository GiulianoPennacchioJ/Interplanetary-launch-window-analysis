import matplotlib.pyplot as plt
import numpy as np

def plot_porkchop(dep_dates, arr_dates, c3_grid, tof_grid, max_c3=45.0, title="Interplanetary Porkchop Plot"):
    """
    Generates a professional porkchop plot containing filled C3 contours 
    and overlaid Time of Flight (TOF) contour lines.
    
    Parameters:
    - dep_dates: Array of departure dates.
    - arr_dates: Array of arrival dates.
    - c3_grid: Matrix of C3 values with shape (n_arr, n_dep).
    - tof_grid: Matrix of TOF values with shape (n_arr, n_dep).
    - max_c3: Upper bound threshold for C3 to filter out degenerate Type-II branches.
    - title: Plot title string.
    """
    # Mask C3 values exceeding the threshold to preserve color contrast for low-energy paths
    c3_masked = np.ma.masked_greater(c3_grid, max_c3)

    # Create a cartesian meshgrid aligned with the (n_arr, n_dep) grid shape ('xy' indexing)
    X, Y = np.meshgrid(dep_dates, arr_dates, indexing='xy')

    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. Filled contour plot for C3 (Departure Energy)
    c3_levels = np.linspace(np.nanmin(c3_grid), max_c3, 30)
    cs_c3 = ax.contourf(X, Y, c3_masked, levels=c3_levels, cmap='turbo', extend='max')
    cbar = fig.colorbar(cs_c3, ax=ax)
    cbar.set_label(r'$C_3$ ($\text{km}^2/\text{s}^2$)')

    # 2. Contour lines for Time of Flight (TOF)
    tof_min = np.nanmin(tof_grid)
    tof_max = np.nanmax(tof_grid)
    tof_levels = np.linspace(tof_min, tof_max, 20)
    
    cs_tof = ax.contour(X, Y, tof_grid, levels=tof_levels, colors='black', linewidths=0.7, alpha=0.8)
    ax.clabel(cs_tof, inline=True, fontsize=9, fmt='%1.0f d')

    # 3. Identify and plot the global optimum (minimum C3 point)
    min_idx = np.nanargmin(c3_grid)
    j_opt, i_opt = np.unravel_index(min_idx, c3_grid.shape)
    
    ax.plot(dep_dates[i_opt], arr_dates[j_opt], 'r*', markersize=12, 
            label=f'Min $C_3$: {c3_grid[j_opt, i_opt]:.2f} $\\text{{km}}^2/\\text{{s}}^2$')

    # Axis styling and formatting
    ax.set_xlabel('Departure Date (MJD)')
    ax.set_ylabel('Arrival Date (MJD)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig, ax