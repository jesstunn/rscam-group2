# ========================================================
# Packages
# ========================================================

from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("Plots", exist_ok=True)

# ========================================================
# Class for storing Parameters
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
        self.sulci_spacing = 'Uniform'
        
        # Resolution for mesh generation
        self.resolution = 50

        # -----------------------------------------------
        # Fluid flow parameters
        # -----------------------------------------------
        self.U_ref = 2.0           # Max. fluid flow velocity in mm/s
        self.viscosity = 1.0       # Dimensionless 
        
        # -----------------------------------------------
        # Solute parameters
        # -----------------------------------------------       
        self.D_mms2 = 0.5   # Diffusion coefficient in mm^2 / s
        self.mu = 1.0       # for Robin boundary (dc/dn = -mu * c)
        
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
            raise ValueError("Number of sulci must be >= 0.")
        if self.sulci_h_mm < 0:
            raise ValueError("Sulcus height cannot be negative.")
        if self.sulci_n > 0 and self.sulci_h_mm <= 0:
            raise ValueError("If sulci_n>0, then sulci_h_mm must be > 0.")
        if self.sulci_width_mm <= 0:
            raise ValueError("Sulci width must be strictly positive.")
        if self.sulci_n > 0 and self.sulci_width_mm * self.sulci_n >= self.L_mm:
            raise ValueError("Total sulcus width must be less than domain length.")

        # PDE parameter checks
        if self.U_ref < 0:
            raise ValueError("Reference velocity U_ref cannot be negative.")
        if self.D_mms2 < 0:
            raise ValueError("Diffusion coefficient D_mms2 cannot be negative.")
        if self.mu < 0:
            raise ValueError("mu cannot be negative for the boundary condition.")
        
    # -----------------------------------------------
    # Function for non-dimensionalisastion
    # -----------------------------------------------

    def nondim(self):
        """
        Converts geometry and PDE parameters to dimensionless form
        and stores them back as attributes.
        """
        # Using domain height as the length scale
        self.L_ref = self.H_mm   

        # Geometry in dimensionless units
        self.L = self.L_mm / self.L_ref
        self.H = self.H_mm / self.L_ref
        self.sulci_h = self.sulci_h_mm / self.L_ref
        self.sulci_width = self.sulci_width_mm / self.L_ref

        # Péclet number and dimensionless diffusion coefficient
        self.Pe = (self.U_ref * self.L_ref) / self.D_mms2
        self.D = 1.0 / self.Pe   

# ========================================================
# Mesh Generator Class
# ========================================================

def mesh_generator(resolution,
                   L, H,  
                   nx, ny, 
                   sulci_n, sulci_h, sulci_width, 
                   sulci_spacing="Uniform"):

    # -----------------------------------------------------------
    # Generate sulci
    # -----------------------------------------------------------
    # Generates centre x-positions for the sulci based on specified spacing method.

    # If no sulci specified return nothing
    if sulci_n == 0:
        sulcus_centers = []  

    # Generate uniformly spaced sulci centers
    sulcus_centers = np.linspace(L / (sulci_n + 1), L - L / (sulci_n + 1), sulci_n)

    # Introduce noise if specified
    if sulci_spacing == 'Noisy':
        spacing_noise = 0.05 * L  # Small perturbations
        sulcus_centers += np.random.uniform(-spacing_noise, spacing_noise, sulci_n)

    # -----------------------------------------------------------
    # Lower boundary with sulci
    # -----------------------------------------------------------
    # Create the lower boundary with sulci.

    # Store lower boundary points as an attribute
    points_lower = []

    for i in range(nx + 1):

        x = i * L / nx  # x-coord in non-dimensional space
        y = 0           # Default flat bottom

        for x0 in sulcus_centers:
            if x0 - sulci_width / 2 <= x <= x0 + sulci_width / 2:
                y += -sulci_h * np.sin(np.pi * (x - (x0 - sulci_width / 2)) / sulci_width)  # Sinusoidal sulcus

        points_lower.append((x, y))

    # -----------------------------------------------------------
    # Generate full mesh
    # -----------------------------------------------------------

    # Define boundaries
    points_upper = [(i / nx *L, H) for i in range(nx + 1)]  
    points_left = [(0, j / ny * H) for j in range(1, ny)]  
    points_right = [(L, j / ny * H) for j in range(1, ny)] 

    # Combine all points in counterclockwise order
    points = points_lower + points_right + points_upper[::-1] + points_left[::-1]

    # Define the polygon and generate mesh
    polygon = Polygon([Point(x, y) for x, y in points])
    mesh = generate_mesh(polygon, resolution)

    # -----------------------------------------------------------
    # Define boundaries
    # -----------------------------------------------------------
 
    def left(x, on_boundary): 
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

    def right(x, on_boundary): 
        return on_boundary and near(x[0], L, DOLFIN_EPS)

    # Find lowest sulci point across whole lower boundary
    min_y = min(p[1] for p in points_lower) if points_lower else 0.0

    def bottom(x, on_boundary):  
        return on_boundary and min_y - 1e-4 < x[1] < 0 + DOLFIN_EPS

    def top(x, on_boundary): 
        return on_boundary and near(x[1], H, DOLFIN_EPS)

    return mesh, left, right, bottom, top
    
# ========================================================
# Stokes Solver Function
# ========================================================

def stokes_solver(mesh, W, H, left, right, bottom, top):

    # --------------------------------------------------------
    # Defining Boundary Conditions
    # --------------------------------------------------------

    # Poiseuille flow
    inflow = Expression(
        ("4.0*x[1]*(H - x[1])", "0.0"), 
        H=H,  
        degree=2
    )

    noslip = Constant((0.0, 0.0))  
    outflow = Constant(0.0)  

    bc_inlet = DirichletBC(W.sub(0), inflow, left)  
    bc_noslip_bottom = DirichletBC(W.sub(0), noslip, bottom)  
    bc_noslip_top = DirichletBC(W.sub(0), noslip, top)  
    bc_outlet = DirichletBC(W.sub(1), outflow, right)  

    bcs = [bc_inlet, bc_noslip_bottom, bc_noslip_top, bc_outlet]

    # --------------------------------------------------------
    # Solve system
    # --------------------------------------------------------

    f = Constant((0.0, 0.0))   # body force (volume force) 
    n = Constant((0.0, 1.0))   # a 'normal' vector, though not used in PDE
    S = Constant((0.0, 0.0))   # boundary traction

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    a = inner(grad(u)+grad(u).T, grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    L = inner(f, v)*dx + inner(S,v)*ds

    # Form for use in constructing preconditioner matrix
    b = inner(grad(u), grad(v))*dx + p*q*dx

    # Assemble system
    A, bb = assemble_system(a, L, bcs)

    # Assemble preconditioner system
    P, _ = assemble_system(b, L, bcs)

    solver = KrylovSolver("minres", "amg")
    solver.set_operators(A, P)

    # Solve
    U = Function(W)
    solver.solve(U.vector(), bb)

    # Get sub-functions
    u, p = U.split()
    return u, p

# ========================================================
# Adv-Diff Solver Function
# ========================================================

def advdiff_solver(mesh, left, right, bottom,
                   u, C,
                   D, mu):

    # ----------------------------------------------------
    # Boundary Conditions
    # ----------------------------------------------------

    # Dirichlet BC on left and right
    bc_left = DirichletBC(C, Constant(1.0), left)
    bc_right = DirichletBC(C, Constant(0.0), right)
    bcs_c = [bc_left, bc_right]

    # ----------------------------------------------------
    # Solving System
    # ----------------------------------------------------

    # Mark bottom boundary and imaginary boundary
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return bottom(x, on_boundary) 
            # your user-defined function checking x[1] ~ 0      

    # Mark boundary
    bottom_id = 1
    BottomBoundary().mark(boundaries, bottom_id)

    # Create a measure
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Robin condition parameters for bottom:
    alpha = D*mu  # If D is a Constant(1.0/Pe_value)

    # Build the variational form
    c_sol = TrialFunction(C)
    phi = TestFunction(C)

    a_c = (D * inner(grad(c_sol), grad(phi)) 
        + inner(dot(u, grad(c_sol)), phi)) * dx

    # Add the Robin term: alpha*c_sol*phi on bottom boundary
    a_c += alpha * c_sol * phi * ds(bottom_id)

    L_c = Constant(0.0)*phi*dx
    # No boundary term => 0 on the bottom side

    # Solve
    c_sol = Function(C)
    solve(a_c == L_c, c_sol, bcs_c)

    return c_sol

# ========================================================
# Post Processing Functions
# ========================================================

# ========================================================
# Validation Functions
# ========================================================

# ========================================================
# Running Simulation
# ========================================================

# Define parameters
epsilons = [0.5, 1.0, 3.0, 5.0]

for eps in epsilons:
    print(f"\nRunning simulation for ε = {eps}")

    params = Parameters()
    params.sulci_h_mm = eps * params.H_mm  # ε = sulci_h/H
    params.nondim()
    # ----------------------------------
    # 1) Generate mesh
    # ----------------------------------
    mesh, left, right, bottom, top = mesh_generator(params.resolution,
                                                    params.L, params.H,  
                                                    params.nx, params.ny, 
                                                    params.sulci_n, params.sulci_h, params.sulci_width, 
                                                    params.sulci_spacing)

    # ----------------------------------
    # 2) Build function spaces
    # ----------------------------------
    V = VectorFunctionSpace(mesh, "P", 2)  
    Q = FunctionSpace(mesh, "P", 1)  
    W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
    C = FunctionSpace(mesh, "CG", 1)

    # ----------------------------------
    # 3) Solve Stokes
    # ----------------------------------
    u, p = stokes_solver(mesh, W, params.H, left, right, bottom, top)

    # ----------------------------------
    # 4) Solve Advection-Diffusion
    # ----------------------------------
    c = advdiff_solver(mesh, left, right, bottom,
                    u, C,
                    D=Constant(params.D),
                    mu=Constant(params.mu))

# ----------------------------------
# 5) Postprocess
# ----------------------------------
    File(f"Plots/velocity_eps{eps}.pvd") << u
    File(f"Plots/pressure_eps{eps}.pvd") << p
    File(f"Plots/concentration_eps{eps}.pvd") << c

# ----------------------------------
# 6) Validation
# ----------------------------------
