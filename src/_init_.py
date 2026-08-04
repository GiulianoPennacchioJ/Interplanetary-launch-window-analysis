"""
Interplanetary Launch Window Analysis Toolkit
=============================================

A modular Python toolkit for interplanetary trajectory design and launch window
analysis, featuring:

- Gooding's Lambert problem solver (1990)
- Analytical planetary ephemeris (J2000 Keplerian elements)
- Porkchop plot generation engine
- Publication-quality visualization suite

Modules:
--------
lambert_gooding : Lambert problem solver using Gooding's algorithm
ephemeris       : Planetary state vectors and orbital elements
porkchop        : Launch window grid computation engine
plotting        : Visualization suite for trajectories and porkchop plots

Example:
--------
>>> from porkchop import PorkchopEngine
>>> from lambert_gooding import TransferType
>>> engine = PorkchopEngine("Earth", "Mars")
>>> grid = engine.compute_grid(dep_start_jd, dep_end_jd, 5.0,
...                            arr_start_jd, arr_end_jd, 5.0,
...                            transfer_type=TransferType.SHORT_WAY)
"""

__version__ = "1.0.0"
__author__ = "Interplanetary Launch Window Analysis Project"

from .lambert_ import LambertGooding, TransferType, LambertSolution, solve_lambert
from .ephemeris import EphemerisManager, PlanetState, OrbitalElements
from .porkchop import PorkchopEngine, PorkchopGrid
from .plotting import PorkchopPlotter, TrajectoryPlotter

__all__ = [
    "LambertGooding",
    "TransferType", 
    "LambertSolution",
    "solve_lambert",
    "EphemerisManager",
    "PlanetState",
    "OrbitalElements",
    "PorkchopEngine",
    "PorkchopGrid",
    "PorkchopPlotter",
    "TrajectoryPlotter",
]