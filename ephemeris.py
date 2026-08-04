"""
Analytic Ephemeris Engine.
Computes precise Heliocentric J2000 state vectors.
"""

import numpy as np

MU_SUN = 1.32712440018e11  # km^3/s^2
AU_TO_KM = 149597870.7

PLANETS = {
    'earth': {
        'a': 1.00000011,
        'e': 0.01671022,
        'i': 0.00005,
        'Omega': -11.26064,
        'w_bar': 102.94719,
        'L0': 100.46435,
        'n': 0.98560912,
    },
    'mars': {
        'a': 1.52366231,
        'e': 0.09341233,
        'i': 1.85061,
        'Omega': 49.57854,
        'w_bar': 336.04084,
        'L0': 355.45332,
        'n': 0.52403930,
    },
}

def julian_date(year, month, day, hour=0, minute=0, second=0):
    """Calcolo robusto e privo di offset della Data Giuliana."""
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    jd += (hour + minute / 60.0 + second / 3600.0) / 24.0
    return jd

def get_planet_state(body_name, jd):
    p = PLANETS[body_name.lower()]
    d = jd - 2451545.0  # Giorni da J2000
    
    a = p['a'] * AU_TO_KM
    e = p['e']
    i = np.radians(p['i'])
    Omega = np.radians(p['Omega'])
    w = np.radians(p['w_bar'] - p['Omega'])
    
    L = np.radians((p['L0'] + p['n'] * d) % 360.0)
    M = (L - np.radians(p['w_bar'])) % (2.0 * np.pi)
    
    E = M
    for _ in range(15):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0)
    )
    
    r_mag = a * (1.0 - e * np.cos(E))
    h = np.sqrt(MU_SUN * a * (1.0 - e**2))
    
    r_pqw = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
    v_pqw = np.array([-np.sin(nu), e + np.cos(nu), 0.0]) * (MU_SUN / h)
    
    # Rotazione da piano perifocale a Eliocentrico J2000
    R3_O = np.array([
        [np.cos(Omega), -np.sin(Omega), 0],
        [np.sin(Omega), np.cos(Omega), 0],
        [0, 0, 1]
    ])
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    R3_w = np.array([
        [np.cos(w), -np.sin(w), 0],
        [np.sin(w), np.cos(w), 0],
        [0, 0, 1]
    ])
    
    Q = R3_O @ R1_i @ R3_w
    return Q @ r_pqw, Q @ v_pqw