import numpy as np

class PorkchopCalculator:
    """
    Calculates interplanetary transfer grids (C3, TOF, arrival Vinf) 
    for porkchop plots with rigorous matrix indexing:
    Rows correspond to arrival dates (j), and columns to departure dates (i).
    """
    
    def __init__(self, dep_dates, arr_dates, lambert_solver):
        """
        Initializes the porkchop grid calculator.
        
        Parameters:
        - dep_dates: Array-like of departure dates (e.g., MJD or JD).
        - arr_dates: Array-like of arrival dates (e.g., MJD or JD).
        - lambert_solver: Callable function that takes (dep_date, arr_date) 
                          and returns (v_inf_dep_vec, v_inf_arr_vec).
        """
        self.dep_dates = np.asarray(dep_dates)
        self.arr_dates = np.asarray(arr_dates)
        self.lambert_solver = lambert_solver
        
        self.n_dep = len(self.dep_dates)
        self.n_arr = len(self.arr_dates)
        
        # Initialize grids with NaNs. Shape is strictly (n_arr, n_dep)
        self.c3_grid = np.full((self.n_arr, self.n_dep), np.nan)
        self.vinf_arr_grid = np.full((self.n_arr, self.n_dep), np.nan)
        self.tof_grid = np.full((self.n_arr, self.n_dep), np.nan)

    def compute(self):
        """
        Iterates through all departure and arrival date pairs to compute 
        the transfer parameters via Lambert's problem.
        
        Returns:
        - c3_grid: Characteristic energy at departure (km^2/s^2)
        - vinf_arr_grid: Hyperbolic excess velocity magnitude at arrival (km/s)
        - tof_grid: Time of flight (days)
        """
        for i, dep in enumerate(self.dep_dates):
            for j, arr in enumerate(self.arr_dates):
                tof_days = arr - dep
                
                # Discard backward-in-time or unrealistically short transfers
                if tof_days <= 10.0:
                    continue

                try:
                    # Solve Lambert's problem for the current leg
                    v_inf_dep_vec, v_inf_arr_vec = self.lambert_solver(dep, arr)
                    
                    # C3 is the squared magnitude of the departure hyperbolic excess velocity vector
                    self.c3_grid[j, i] = np.dot(v_inf_dep_vec, v_inf_dep_vec)
                    
                    # Magnitude of the arrival hyperbolic excess velocity vector
                    self.vinf_arr_grid[j, i] = np.linalg.norm(v_inf_arr_vec)
                    
                    # Store time of flight
                    self.tof_grid[j, i] = tof_days
                    
                except (RuntimeError, ValueError):
                    # Safely skip Lambert solver singularities or convergence failures
                    pass

        return self.c3_grid, self.vinf_arr_grid, self.tof_grid