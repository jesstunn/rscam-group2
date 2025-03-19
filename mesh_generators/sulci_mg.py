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
                 L_mm, H_mm,  
                 nx, ny, 
                 sulci_n, sulci_h_mm, sulci_width_mm, sulci_spacing="Uniform"):
        """
        Creates a 2D mesh for Subarachnoid Space (SAS) simulations, taking real world
        measurements and dimensions as input.

        Parameters
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
        # Non-Dimensionalisation
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

        # Other parameters
        self.nx = nx
        self.ny = ny
        self.sulci_n = sulci_n
        self.sulci_spacing = sulci_spacing

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

    def generate_mesh(self, resolution=50):
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
        return generate_mesh(polygon, resolution)


