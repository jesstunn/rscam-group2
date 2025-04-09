# ========================================================
# Parameters Class
# ========================================================

class Parameters:
    """
    Holds all user-defined parameters, validates these and
    computes non-dimensionalised values.
    """
    def __init__(self):

        # -----------------------------------------------
        # Domain / geometry parameters
        # -----------------------------------------------
        self.L_mm = 10.0
        self.H_mm = 1.0

        self.nx = 1000
        self.ny = 100

        self.sulci_n = 1
        self.sulci_h_mm = 2.0
        self.sulci_width_mm = 1.0
        
        # Resolution for mesh generation
        self.resolution = 50

        # -----------------------------------------------
        # Fluid flow parameters
        # -----------------------------------------------
        self.U_ref = 0.012      # Max. fluid flow velocity in mm/s
        self.viscosity = 1.0    # Dimensionless 
        
        # -----------------------------------------------
        # Solute parameters
        # -----------------------------------------------       
        self.D_mms2 = 0.0003   # Diffusion coefficient 3×10-10 m2/s converted to mm2/s
        self.mu = 10            # for Robin boundary (dc/dn = -mu * c)
        
        # -----------------------------------------------
        # Validate parameters
        # -----------------------------------------------
        self.validate()

        # -----------------------------------------------
        # Non-Dimensionalise
        # -----------------------------------------------
        self.nondim()

    # -----------------------------------------------
    # Function for validating inputs
    # -----------------------------------------------

    def validate(self):
        """
        Validate input parameters for the simulation
        (raises ValueError if invalid).
        """
        # Domain checks
        if self.L_mm <= 0:
            raise ValueError("Domain length L_mm must be strictly positive.")
        if self.H_mm <= 0:
            raise ValueError("Domain height H_mm must be strictly positive.")

        # Sulci checks
        if self.sulci_n < 0:
            raise ValueError("Number of sulci must be strictly positive.")
        if self.sulci_h_mm < 0:
            raise ValueError("Sulcus height cannot be negative.")
        if self.sulci_n > 0 and self.sulci_h_mm <= 0:
            raise ValueError("If there are sulci defined, then sulcus height must be greater than 0.")
        if self.sulci_width_mm < 0:
            raise ValueError("Sulci width cannot be negative.")
        if self.sulci_n > 0 and self.sulci_width_mm <= 0:
            raise ValueError("If there are sulci defined, then sulcus width must be greater than 0.")
        if self.sulci_n > 0 and self.sulci_width_mm * self.sulci_n >= self.L_mm:
            raise ValueError("Total sulcus width must be less than domain length.")

        # PDE parameter checks
        if self.U_ref < 0:
            raise ValueError("Reference velocity cannot be negative.")
        if self.D_mms2 < 0:
            raise ValueError("Diffusion coefficient cannot be negative.")
        if self.mu < 0:
            raise ValueError("Uptake parameter cannot be negative for the boundary condition.")
        
    # -----------------------------------------------
    # Function for non-dimensionalisastion
    # -----------------------------------------------

    def nondim(self):
        """
        Converts domain and PDE parameters to dimensionless form
        and stores them back as attributes.
        """
        # Using domain height as the length scale
        self.L_ref = self.H_mm   

        # Geometry in dimensionless units
        self.L = self.L_mm / self.L_ref
        self.H = self.H_mm / self.L_ref
        self.sulci_h = self.sulci_h_mm / self.L_ref
        self.sulci_width = self.sulci_width_mm / self.L_ref

        # Peclet number and dimensionless diffusion coefficient
        self.Pe = (self.U_ref * self.L_ref) / self.D_mms2
        self.D = 1.0 / self.Pe

    def __str__(self):
        """Statement of parameters for logging and debugging."""
        param_str = "Simulation Parameters:\n"
        param_str += f"  Domain: L={self.L_mm}mm x H={self.H_mm}mm\n"
        param_str += f"  Mesh: {self.nx}x{self.ny} points, resolution={self.resolution}\n"
        param_str += f"  Sulci: n={self.sulci_n}, height={self.sulci_h_mm}mm, width={self.sulci_width_mm}mm\n"
        param_str += f"  Flow: U_ref={self.U_ref}mm/s, viscosity={self.viscosity}\n"
        param_str += f"  Solute: D={self.D_mms2}mm²/s, mu={self.mu}\n"
        param_str += f"  Non-dimensional: Pe={self.Pe}, D={self.D}\n"
        return param_str
    
    def load_from_dict(self, params_dict):
        """
        Load parameters from a dictionary.
        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        # Validate and non-dimensionalise
        self.validate()
        self.nondim()
        
        return self
    
    def to_dict(self):
        """
        Convert parameters to a dictionary for saving/loading.
        """
        # Only include original parameters, not derived ones
        param_dict = {
            'L_mm': self.L_mm,
            'H_mm': self.H_mm,
            'nx': self.nx,
            'ny': self.ny,
            'sulci_n': self.sulci_n,
            'sulci_h_mm': self.sulci_h_mm,
            'sulci_width_mm': self.sulci_width_mm,
            'resolution': self.resolution,
            'U_ref': self.U_ref,
            'viscosity': self.viscosity,
            'D_mms2': self.D_mms2,
            'mu': self.mu
        }
        
        return param_dict
