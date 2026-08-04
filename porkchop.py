"""
Porkchop Engine.
Generates valid transfer matrices utilizing vectorial V_infinity limits.
"""

import numpy as np
from ephemeris import get_planet_state
from lambert import solve_lambert

MU_SUN = 1.32712440018e11  # km^3/s^2

class PorkchopEngine:
    def __init__(self, departure_body='earth', arrival_body='mars'):
        self.dep_body = departure_body
        self.arr_body = arrival_body

    def generate_grid(self, dep_jds, arr_jds, lw=0):
        num_dep = len(dep_jds)
        num_arr = len(arr_jds)

        c3_dep = np.full((num_arr, num_dep), np.nan)
        vinf_arr = np.full((num_arr, num_dep), np.nan)
        tof_grid = np.full((num_arr, num_dep), np.nan)

        for j, arr_jd in enumerate(arr_jds):
            r_arr, v_arr = get_planet_state(self.arr_body, arr_jd)

            for i, dep_jd in enumerate(dep_jds):
                tof_days = arr_jd - dep_jd
                if tof_days <= 30:
                    continue

                tof_sec = tof_days * 86400.0
                r_dep, v_dep = get_planet_state(self.dep_body, dep_jd)

                v1_trans, v2_trans = solve_lambert(
                    r_dep, r_arr, tof_sec, MU_SUN, lw=lw
                )

                if np.isnan(v1_trans[0]):
                    continue

                v_inf_dep_vec = v1_trans - v_dep
                c3 = np.dot(v_inf_dep_vec, v_inf_dep_vec)

                # Limite realistico per contorni ben definiti
                if c3 < 150.0:
                    c3_dep[j, i] = c3
                    v_inf_arr_vec = v2_trans - v_arr
                    vinf_arr[j, i] = np.linalg.norm(v_inf_arr_vec)
                    tof_grid[j, i] = tof_days

        return {
            'dep_jds': dep_jds,
            'arr_jds': arr_jds,
            'C3_dep': c3_dep,
            'Vinf_arr': vinf_arr,
            'TOF': tof_grid,
        }