# ========================================================
# Packages
# ========================================================

from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

# ========================================================
# Custom Sulci - Domain Generator Class
# ========================================================

class SulciMeshGenerator:
    
    def __init__(self, 
                 resolution,
                 L_mm, H_mm,  
                 nx, ny, 
                 sulci_n, sulci_h_mm, sulci_width_mm, sulci_spacing="Uniform"
                 ):
        """
        Creates a 2D mesh for Subarachnoid Space (SAS) simulations, taking real world
        measurements and dimensions as input.

        Parameters
        - resolution (int): Resolution for mesh generation.

        - L_mm (float): Length of the domain in mm.
        - H_mm (float): Height of the domain in mm.

        - nx (int): Number of points in the x-direction.
        - ny (int): Number of points in the y-direction.

        Choose number, depth, width, spacing method of sulcus:
        - sulci_n (int): Number of sulci (caveties).
        - sulci_h_mm (float): Depth of the sulci in mm.
        - sulci_width_mm (float): Width of a single sulcus in mm.
        - sulci_spacing (str): 'Uniform' for even spacing, 'Noisy' for randomised positions.
        """
        
        # -----------------------------------------------------------
        # Input Validation
        # -----------------------------------------------------------

        # Check resolution
        if not isinstance(resolution, int) or resolution <= 0:
            raise ValueError("Error: Resolution must be a strictly positive integer.")

        # Check domain dimensions
        if L_mm <= 0 or H_mm <= 0:
            raise ValueError("Error: Domain dimensions (L_mm, H_mm) must be strictly positive.")

        # Check number of points in x and y directions
        if not isinstance(nx, int) or not isinstance(ny, int) or nx <= 0 or ny <= 0:
            raise ValueError("Error: The number of points (nx, ny) must be strictly positive integers.")

        # Check sulci parameters
        if sulci_n < 0:
            raise ValueError("Error: The number of sulci (sulci_n) must be non-negative.")

        if sulci_h_mm < 0:
            raise ValueError("Error: Sulcus height (sulci_h_mm) must be non-negative.")
        
        if sulci_h_mm <=0 and sulci_n > 0:
            raise ValueError("Error: Sulcus height (sulci_h_mm) must be strictly positive.")

        if sulci_width_mm <= 0:
            raise ValueError("Error: Sulcus width (sulci_width_mm) must be strictly positive.")

        # Ensure total sulcus width does not exceed the domain length
        if sulci_n > 0 and (sulci_width_mm * sulci_n >= L_mm):
            raise ValueError(f"Error: The total sulcus width ({sulci_width_mm * sulci_n} mm) must be smaller than the domain length ({L_mm} mm). Reduce sulci_n or sulci_width_mm.")

        # -----------------------------------------------------------
        # Non-Dimensionalisation & Store Variables
        # -----------------------------------------------------------

        # Defining characteristic length (using domain length as reference)
        self.L_ref = L_mm  

        # Store real-world dimensions
        self.L_mm = L_mm
        self.H_mm = H_mm
        self.sulci_h_mm = sulci_h_mm
        self.sulci_width_mm = sulci_width_mm

        # Convert to non-dimensional units
        self.L = L_mm / self.L_ref
        self.H = H_mm / self.L_ref
        self.sulci_h = sulci_h_mm / self.L_ref
        self.sulci_width = sulci_width_mm / self.L_ref

        # Store other parameters
        self.nx = nx
        self.ny = ny
        self.sulci_n = sulci_n
        self.sulci_spacing = sulci_spacing
        self.resolution = resolution

    # -----------------------------------------------------------
    # Generate sulci
    # -----------------------------------------------------------

    def _generate_sulcus_positions(self):
        """Generates centre x-positions for the sulci based on specified spacing method."""

        # If no sulci specified return nothing
        if self.sulci_n == 0:
            return []  

        # Generate uniformly spaced sulci centers
        sulcus_centers = np.linspace(self.L / (self.sulci_n + 1), self.L - self.L / (self.sulci_n + 1), self.sulci_n)

        # Introduce noise if specified
        if self.sulci_spacing == 'Noisy':
            spacing_noise = 0.05 * self.L  # Small perturbations
            sulcus_centers += np.random.uniform(-spacing_noise, spacing_noise, self.sulci_n)

        return sulcus_centers

    def _generate_lower_boundary(self):
        """Creates the lower boundary with sulci."""
        points_lower = []
        sulcus_centers = self._generate_sulcus_positions()

        for i in range(self.nx + 1):

            x = i * self.L / self.nx  # x-coordinate in non-dimensional space
            y = 0  # Default flat bottom

            for x0 in sulcus_centers:
                if x0 - self.sulci_width / 2 <= x <= x0 + self.sulci_width / 2:
                    y += -self.sulci_h * np.sin(np.pi * (x - (x0 - self.sulci_width / 2)) / self.sulci_width)  # Sinusoidal sulcus

            points_lower.append((x, y))

        return points_lower
   
    # -----------------------------------------------------------
    # Generate full mesh
    # -----------------------------------------------------------

    def generate_mesh(self):
        """
        Generates the finite element mesh for the SAS domain.
        """
        # Define boundaries
        points_lower = self._generate_lower_boundary()
        points_upper = [(i / self.nx * self.L, self.H) for i in range(self.nx + 1)]  
        points_left = [(0, j / self.ny * self.H) for j in range(1, self.ny)]  
        points_right = [(self.L, j / self.ny * self.H) for j in range(1, self.ny)] 

        # Combine all points in counterclockwise order
        points = points_lower + points_right + points_upper[::-1] + points_left[::-1]

        # Define the polygon
        polygon = Polygon([Point(x, y) for x, y in points])

        # Generate the mesh
        return generate_mesh(polygon, self.resolution)

# ========================================================
# Wavy Wall - Domain Generator Class
# ========================================================

class WavyWallMeshGenerator:

    def __init__(self, 
                 resolution,
                 L_mm, H_mm, 
                 nx, ny, 
                 sulcus_depth_mm, 
                 num_sulci):
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
        self.resolution = resolution

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

    def generate_mesh(self):
        """Generate mesh with total height = base_height + sulcus_depth"""

        # Define boundaries of the domain
        lower = self._generate_wavy_boundary() 
        upper = [(x, self.H_base) for x in np.linspace(0, self.L, self.nx+1)]
        left = [(0, y) for y in np.linspace(-self.sulcus_depth, self.H_base, self.ny+1)[1:-1]]
        right = [(self.L, y) for y in np.linspace(-self.sulcus_depth, self.H_base, self.ny+1)[1:-1]]

        # Build polygon
        points = lower + right + upper[::-1] + left[::-1]
        return generate_mesh(Polygon([Point(x,y) for x,y in points]), self.resolution)




