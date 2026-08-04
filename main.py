"""
Main Execution Script per Interplanetary Launch Window Analysis (Earth-Mars 2026).
Finestre temporali calibrate per catturare l'intero dominio Type-I e Type-II.
"""

import numpy as np
import warnings
from ephemeris import julian_date
from plotting import plot_porkchop, plot_type_split
from porkchop import PorkchopEngine

def main():
    print("==================================================")
    print(" Interplanetary Launch Window Analysis (Earth-Mars)")
    print("==================================================")
    
    # Finestra di Partenza (Ottobre 2026 - Gennaio 2027)
    dep_start = julian_date(2026, 9, 15)
    dep_end = julian_date(2027, 1, 15)
    
    # Finestra di Arrivo AMPLIATA (Febbraio 2027 - Aprile 2028)
    # FONDAMENTALE: Febbraio 2027 serve per catturare il lobo di Type-I!
    arr_start = julian_date(2027, 2, 1)
    arr_end = julian_date(2028, 4, 1)
    
    # Risoluzione griglia incrementata per isolinee perfettamente lisce
    dep_jds = np.linspace(dep_start, dep_end, 120)
    arr_jds = np.linspace(arr_start, arr_end, 120)
    
    print("Generating Porkchop grid metrics...")
    engine = PorkchopEngine(departure_body='earth', arrival_body='mars')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        porkchop_t1 = engine.generate_grid(dep_jds, arr_jds, lw=0)
        porkchop_t2 = engine.generate_grid(dep_jds, arr_jds, lw=1)
    
    # Unione logica delle due geometrie di trasferimento
    c3_t1 = np.nan_to_num(porkchop_t1["C3_dep"], nan=np.inf)
    c3_t2 = np.nan_to_num(porkchop_t2["C3_dep"], nan=np.inf)
    
    mask_t1 = c3_t1 < c3_t2
    c3_combined = np.fmin(c3_t1, c3_t2)
    c3_combined[c3_combined == np.inf] = np.nan
    
    # Mascheramento per eliminare artefatti ad altissima energia (> 50 km²/s²)
    c3_combined[c3_combined > 50.0] = np.nan
    
    tof_combined = np.where(mask_t1, porkchop_t1["TOF"], porkchop_t2["TOF"])
    vinf_arr_combined = np.where(mask_t1, porkchop_t1["Vinf_arr"], porkchop_t2["Vinf_arr"])
    
    porkchop_combined = {
        'dep_jds': dep_jds,
        'arr_jds': arr_jds,
        'C3_dep': c3_combined,
        'Vinf_arr': vinf_arr_combined,
        'TOF': tof_combined
    }
    
    if np.all(np.isnan(c3_combined)):
        print("\nNessuna traiettoria trovata per le date fornite.")
        return
        
    min_idx = np.unravel_index(np.nanargmin(c3_combined), c3_combined.shape)
    opt_dep_jd = dep_jds[min_idx[1]]
    opt_arr_jd = arr_jds[min_idx[0]]
    opt_c3 = c3_combined[min_idx]
    opt_vinf = vinf_arr_combined[min_idx]
    opt_tof = tof_combined[min_idx]
    
    best_type = "Type-I" if mask_t1[min_idx] else "Type-II"
    
    print(f"\n--- Optimal Trajectory Parameters ({best_type}) ---")
    print(f"Departure JD      : {opt_dep_jd:.2f}")
    print(f"Arrival JD        : {opt_arr_jd:.2f}")
    print(f"Time of Flight    : {opt_tof:.1f} days")
    print(f"Minimum C3 Dep    : {opt_c3:.2f} km²/s²")
    print(f"Arrival V_infinity: {opt_vinf:.2f} km/s")
    
    print("\nDisplaying Porkchop plots...")
    plot_porkchop(porkchop_combined, title="Earth-Mars 2026 Launch Window (Combined)")
    plot_type_split(porkchop_t1, porkchop_t2)

if __name__ == "__main__":
    main()