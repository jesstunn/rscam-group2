# ========================================================
# Packages
# ========================================================

from dolfin import *
from mshr import *
# import mesh_generators as mg
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
        """Generate mesh"""

        # Store lower boundary points as an attribute
        self.points_lower = self._generate_lower_boundary()

        # Define boundaries
        points_upper = [(i / self.nx * self.L, self.H) for i in range(self.nx + 1)]  
        points_left = [(0, j / self.ny * self.H) for j in range(1, self.ny)]  
        points_right = [(self.L, j / self.ny * self.H) for j in range(1, self.ny)] 

        # Combine all points in counterclockwise order
        points = self.points_lower + points_right + points_upper[::-1] + points_left[::-1]

        # Define the polygon
        polygon = Polygon([Point(x, y) for x, y in points])

        # Generate the mesh
        return generate_mesh(polygon, self.resolution)
    
    def define_boundaries(self):
        """Defines boundary functions for use in FEniCS."""
        
        def left(x, on_boundary): 
            return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

        def right(x, on_boundary): 
            return on_boundary and near(x[0], self.L, DOLFIN_EPS)

        min_y = min(p[1] for p in self.points_lower) if self.points_lower else 0.0

        def bottom(x, on_boundary):  
            return on_boundary and min_y < x[1] < 0 + DOLFIN_EPS

        def top(x, on_boundary): 
            return on_boundary and near(x[1], self.H, DOLFIN_EPS)

        return left, right, bottom, top
    
    
# ========================================================
# Defining Domain & Creating Mesh
# ========================================================

mesh_generator = SulciMeshGenerator(
    resolution=50, 
    L_mm=10, 
    H_mm=1, 
    nx=1000, 
    ny=100, 
    sulci_n=1, 
    sulci_h_mm=1, 
    sulci_width_mm=2, 
    sulci_spacing="Uniform"
)

# Generate mesh and get boundaries
mesh = mesh_generator.generate_mesh() 
left, right, bottom, top = mesh_generator.define_boundaries()

# ========================================================
# Defining Function Spaces
# ========================================================

V = VectorFunctionSpace(mesh, "P", 2)  
Q = FunctionSpace(mesh, "P", 1)  
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
C = FunctionSpace(mesh, "CG", 1)

# ========================================================
# Defining Boundary Conditions
# ========================================================

inflow = Expression(("4.0*x[1]*(1.0-x[1])", "0.0"), degree=2)  # Poiseuille
noslip = Constant((0.0, 0.0))  
outflow = Constant(0.0)  

# Boundary condition
bc_inlet = DirichletBC(W.sub(0), inflow, left)  
bc_noslip_bottom = DirichletBC(W.sub(0), noslip, bottom)  
bc_noslip_top = DirichletBC(W.sub(0), noslip, top)  
bc_outlet = DirichletBC(W.sub(1), outflow, right)  

bcs = [bc_inlet, bc_noslip_bottom, bc_noslip_top, bc_outlet]

# ========================================================
# Solving the Stokes Equations (Fluid Flow)
# ========================================================

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0.0, 0.0))
n = Constant((0.0,1.0))
S = Constant((0.0,0.0))
a = inner(grad(u)+grad(u).T, grad(v))*dx + div(v)*p*dx + q*div(u)*dx
L = inner(f, v)*dx + inner(S,v)*ds

# Form for use in constructing preconditioner matrix
b = inner(grad(u), grad(v))*dx + p*q*dx

# Assemble system
A, bb = assemble_system(a, L, bcs)

# Assemble preconditioner system
P, btmp = assemble_system(b, L, bcs)

solver = KrylovSolver("minres", "amg")
solver.set_operators(A, P)

# Solve
U = Function(W)
solver.solve(U.vector(), bb)

# Get sub-functions
u, p = U.split()
ux, uy = u.split()

# ========================================================
# Solving the Advection-Diffusion Equation (Solute Transport)
# ========================================================

bc_left = DirichletBC(C, Constant(1.0), left)  # c = 1 at x=0 (left)
bc_right = DirichletBC(C, Constant(0.0), right)  # c = 0 at x=L (right)
bcs_c = [bc_left, bc_right]  # Combine boundary conditions

# Advection-diffusion Problem
c_sol = TrialFunction(C)
phi = TestFunction(C)
D = Constant(0.05)

a_c = (D * inner(grad(c_sol), grad(phi)) + inner(dot(u, grad(c_sol)), phi)) * dx
L_c = Constant(0) * phi * dx

c_sol = Function(C)
solve(a_c == L_c, c_sol, bcs_c)  # Apply both Dirichlet BCs

# ========================================================
# Saving the Results
# ========================================================

File("solution_P.pvd") << u
File("solute_concentration_P.pvd") << c_sol
