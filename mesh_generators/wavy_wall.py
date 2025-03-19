# ========================================================
# Packages
# ========================================================

from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

# ========================================================
# Domain Generator Classes
# ========================================================

class WavyWallMeshGenerator:

    def __init__(self, 
                 L_mm, H_mm, 
                 nx=200, ny=200, 
                 sulcus_depth_mm=1.0, 
                 num_sulci=2):
        """
        Creates a 2D mesh with a wavy lower boundary which extends below the base rectangle
        representing the SAS space.

        Parameters:
        - L_mm (float): Domain length (mm).
        - H_mm (float): Base domain height (mm), from sulci peaks to top of rectangular area.
        - nx, ny (int): Mesh density.
        - sulcus_depth_mm (float): Depth of each sulci (mm).
        - num_sulci (int): Number of caveties.
        """

        # -----------------------------------------------------------
        # Input Validation
        # -----------------------------------------------------------

        if L_mm <= 0 or H_mm <=0:
             raise ValueError("Error: L_mm and H_mm must be strictly positive.")
        
        if not isinstance(nx, int) or nx <= 0:
            raise ValueError("Error: nx must be a strictly positive integer.")

        if not isinstance(ny, int) or ny <= 0:
            raise ValueError("Error: ny must be a strictly positive integer.")

        if sulcus_depth_mm <= 0:
            raise ValueError("Error: sulcus_depth_mm must be strictly positive.")
        
        if not isinstance(num_sulci, int) or num_sulci <= 0:
            raise ValueError("Error: num_sulci must be a positive integer.")

        # -----------------------------------------------------------
        # Non-Dimensionalise and Store Variables
        # -----------------------------------------------------------

        # Store dimensional measurements
        self.L_mm = L_mm  
        self.H_mm = H_mm 
        self.sulcus_depth_mm = sulcus_depth_mm

        # Define a characteristic length
        self.L_ref = L_mm  

        # Length of domain, L_mm/L_mm
        self.L = 1.0  

        # Base height (peaks to top of rectangular region)
        self.H_base = H_mm / L_mm  

        # Trough depth
        self.sulcus_depth = sulcus_depth_mm / L_mm  

        self.nx = nx
        self.ny = ny
        self.num_sulci = num_sulci

        # Compute sulcus width
        self.sulcus_width_mm = L_mm / num_sulci  # Width of each cavity

    # -----------------------------------------------------------
    # Generate wavy wall
    # -----------------------------------------------------------

    def _generate_wavy_boundary(self):
        """Generate boundary points with peaks at y=0, troughs at y=-sulcus_depth"""

        # Generate evenly spaced x-values.
        # The range 0 to L is in non-dim form, thus ranges from 0 to 1.
        x_values = np.linspace(0, self.L, self.nx+1)

        # Calculate the y-value for each x-value based on cosine function
        y = (0.5 * self.sulcus_depth) * (np.cos(self.num_sulci * (2 * np.pi) * x_values) - 1)

        # Return coordinate pairs
        return list(zip(x_values, y))
    
    # -----------------------------------------------------------
    # Generate mesh
    # -----------------------------------------------------------

    def generate_mesh(self, resolution=50):
        """Generate mesh with total height = base_height + sulcus_depth"""

        # Define boundaries of the domain
        lower = self._generate_wavy_boundary() 
        upper = [(x, self.H_base) for x in np.linspace(0, self.L, self.nx+1)]
        left = [(0, y) for y in np.linspace(-self.sulcus_depth, self.H_base, self.ny+1)[1:-1]]
        right = [(self.L, y) for y in np.linspace(-self.sulcus_depth, self.H_base, self.ny+1)[1:-1]]

        # Build polygon
        points = lower + right + upper[::-1] + left[::-1]
        return generate_mesh(Polygon([Point(x,y) for x,y in points]), resolution)
    
    # -----------------------------------------------------------
    # Return parameters
    # -----------------------------------------------------------

    def get_parameters(self):
        """Returns computed sulcus width along with other key parameters."""
        return {
            "Domain Length (mm)": self.L_mm,
            "Domain Height (mm)": self.H_mm,
            "Sulcus Depth (mm)": self.sulcus_depth_mm,
            "Number of Sulci": self.num_sulci,
            "Sulcus Width (mm)": self.sulcus_width_mm,  # Computed value
        }
    

