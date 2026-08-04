import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.constants import GM_sun

def run_validation():
    print("=" * 65)
    print("       INTERPLANETARY MISSION SCIENTIFIC VALIDATION (EARTH-MARS)")
    print("=" * 65)
    
    # 1. Optimal dates extracted from your previous analysis (MJD)
    dep_mjd = 61343.9
    arr_mjd = 61637.3
    
    t_dep = Time(dep_mjd, format="mjd", scale="tdb")
    t_arr = Time(arr_mjd, format="mjd", scale="tdb")
    
    print(f"\n[1] Analyzed Time Window (Your Model):")
    print(f"    - Departure: {t_dep.iso[:10]} (MJD {dep_mjd})")
    print(f"    - Arrival:   {t_arr.iso[:10]} (MJD {arr_mjd})")
    print(f"    - TOF:       {(arr_mjd - dep_mjd):.1f} days")

    # 2. Ideal Hohmann Model (Curtis)
    r_e = 1.0     # AU (Mean Earth orbit radius)
    r_m = 1.524   # AU (Mean Mars orbit radius)
    a_h = (r_e + r_m) / 2.0  # Semi-major axis
    
    au_km = 149597870.7
    mu_s_val = GM_sun.to(u.km**3 / u.s**2).value
    
    # Hohmann TOF in seconds converted to days
    tof_hohmann_days = (np.pi * np.sqrt((a_h * au_km)**3 / mu_s_val)) / 86400.0
    
    # Theoretical Hohmann C3 energy
    v_e = np.sqrt(mu_s_val / (r_e * au_km))
    v_t1 = np.sqrt(mu_s_val * (2.0 / (r_e * au_km) - 1.0 / (a_h * au_km)))
    c3_hohmann = (v_t1 - v_e)**2

    print(f"\n[2] Ideal Hohmann Model (Curtis):")
    print(f"    - Theoretical TOF (circular/coplanar): {tof_hohmann_days:.1f} days")
    print(f"    - Theoretical C3:                      {c3_hohmann:.2f} km^2/s^2")

    # 3. Real Benchmark Mission: NASA Mars 2020 (Perseverance)
    print(f"\n[3] Real Mission Benchmark (NASA Mars 2020):")
    print(f"    - Launch: July 30, 2020 | Arrival: February 18, 2021")
    print(f"    - Real TOF:       203 days (Accelerated trajectory)")
    print(f"    - Real C3:        ~14.49 km^2/s^2 (Atlas V + Heavy payload)")

    # 4. Final Comparative Summary Table
    print("\n" + "=" * 65)
    print(f"{'MISSION PARAMETER':<24} | {'YOUR MODEL':<12} | {'HOHMANN':<10} | {'MARS 2020':<10}")
    print("=" * 65)
    print(f"{'Launch Energy C3':<24} | {9.82:<12.2f} | {c3_hohmann:<10.2f} | {14.49:<10.2f}")
    print(f"{'Time of Flight (TOF)':<24} | {293.4:<12.1f} | {tof_hohmann_days:<10.1f} | {203.0:<10.1f}")
    print("=" * 65)

if __name__ == "__main__":
    run_validation()