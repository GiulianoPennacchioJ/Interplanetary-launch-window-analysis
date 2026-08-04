"""
Lambert's Problem Solver.
Implemented using Universal Variables (Vallado Algorithm 58) for absolute 
grid stability and strict prograde trajectory filtering.
"""

import numpy as np

def solve_lambert(r1, r2, tof, mu, lw=0, revs=0):
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    if r1_mag < 1e-6 or r2_mag < 1e-6 or tof <= 0:
        return np.full(3, np.nan), np.full(3, np.nan)
        
    cross_r = np.cross(r1, r2)
    dot_r = np.dot(r1, r2)
    
    cos_dnu = np.clip(dot_r / (r1_mag * r2_mag), -1.0, 1.0)
    dnu = np.arccos(cos_dnu)
    
    # Controllo del piano orbitale: forza trasferimenti progridi.
    # Se il momento angolare z è negativo, la via più breve è retrograda.
    if cross_r[2] < 0:
        dnu = 2.0 * np.pi - dnu
        
    # Filtraggio rigoroso Type-I (Short Way) e Type-II (Long Way)
    if lw == 0 and dnu >= np.pi:
        return np.full(3, np.nan), np.full(3, np.nan)
    if lw == 1 and dnu < np.pi:
        return np.full(3, np.nan), np.full(3, np.nan)
        
    A = np.sin(dnu) * np.sqrt(r1_mag * r2_mag / (1.0 - np.cos(dnu)))
    
    # Singolarità geometrica dei 180 gradi
    if abs(A) < 1e-8:
        return np.full(3, np.nan), np.full(3, np.nan)
        
    # Metodo di Bisezione per la variabile universale z
    z = 0.0
    z_up = 4.0 * np.pi**2
    z_low = -4.0 * np.pi**2
    
    for _ in range(100):
        if z > 1e-6:
            sqz = np.sqrt(z)
            C = (1.0 - np.cos(sqz)) / z
            S = (sqz - np.sin(sqz)) / (sqz**3)
        elif z < -1e-6:
            sqz = np.sqrt(-z)
            C = (1.0 - np.cosh(sqz)) / z
            S = (np.sinh(sqz) - sqz) / (sqz**3)
        else:
            C = 1.0 / 2.0
            S = 1.0 / 6.0
            
        y = r1_mag + r2_mag - A * (1.0 - z * S) / np.sqrt(C)
        
        # Previene orbite non fisiche
        if y < 0:
            z_low = z
            z = (z + z_up) / 2.0
            continue
            
        x = np.sqrt(y / C)
        t_calc = (x**3 * S + A * np.sqrt(y)) / np.sqrt(mu)
        
        dt = t_calc - tof
        if abs(dt) < 1e-6:
            break
            
        if dt > 0:
            z_up = z
        else:
            z_low = z
            
        z = (z + z_up) / 2.0
        
    f = 1.0 - y / r1_mag
    g_dot = 1.0 - y / r2_mag
    g = A * np.sqrt(y / mu)
    
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    
    return v1, v2