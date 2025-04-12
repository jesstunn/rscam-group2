##########################################################
# Advection-Diffusion Solver Module
##########################################################

# ========================================================
# Imports
# ========================================================

from dolfin import *
import numpy as np
import os

# ========================================================
# Advection-Diffusion Solver
# ========================================================

def advdiff_solver(mesh, left, right, bottom,
                   u, C,
                   D, mu):
    """
    Solve the advection-diffusion equation for concentration field.
    
    Parameters:
    mesh : dolfin.Mesh
        The computational mesh
    left, right, bottom : function
        Functions to identify boundaries
    u : dolfin.Function
        Velocity field from Stokes solution
    C : dolfin.FunctionSpace
        Function space for concentration
    D : dolfin.Constant
        Diffusion coefficient (1/Pe)
    mu : dolfin.Constant
        Robin boundary condition coefficient (uptake parameter)
    
    Returns:
    c_sol : dolfin.Function
        Concentration field solution
    """
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

    # Mark bottom boundary 
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return bottom(x, on_boundary)     

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
# Post-Process Functions
# ========================================================

def calculate_total_mass(c, mesh):
    """
    Calculate the total mass of solute in the domain by integrating
    the concentration field over the entire domain.
    
    Parameters:
    c : dolfin.Function
        Concentration field
    mesh : dolfin.Mesh
        Mesh on which c is defined
        
    Returns:
    float
        Total mass of solute in the domain
    """
    # Define measure for integration over the domain
    # "dx" = this is a volume (domain) integral
    dx = Measure("dx", domain=mesh)
    
    # Compute the double integral of c over the domain
    total_mass = assemble(c*dx)
    
    return total_mass

def calculate_average_mass(c, mesh):
    """
    Calculate the average massof solute in the domain by integrating 
    the concentration field over the entire domain and dividing by 
    the domain volume.
    
    Parameters:
    c : dolfin.Function
        Concentration field
    mesh : dolfin.Mesh
        Mesh on which c is defined
        
    Returns:
    float
        Average mass (concentration) of solute in the domain
    """
    # Define measure for integration over the domain
    dx = Measure("dx", domain=mesh)
    
    # Compute the integral of c over the domain
    total_mass = assemble(c*dx)
    
    # Compute the domain volume
    domain_volume = assemble(Constant(1.0)*dx)
    
    # Compute the average mass (concentration)
    average_mass = total_mass / domain_volume
    
    return average_mass
  
def compute_uptake_flux(c, mesh, bottom, D, mu):
    """
    Compute the total uptake flux through the bottom boundary.
    
    Parameters:
    -----------
    c : dolfin.Function
        Concentration field
    mesh : dolfin.Mesh
        Computational mesh
    bottom : function
        Function to identify bottom boundary
    D : dolfin.Constant
        Diffusion coefficient
    mu : dolfin.Constant
        Robin boundary condition coefficient
    
    Returns:
    --------
    flux : float
        Total uptake flux through bottom boundary
    """
    # Mark bottom boundary
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    
    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return bottom(x, on_boundary)
    
    BottomBoundary().mark(boundaries, 1)
    
    # Create measure
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    # Normal vector (pointing outward from domain)
    n = FacetNormal(mesh)
    
    # Compute flux: integral of (D*grad(c)·n + mu*c) over bottom boundary
    # Note: For Robin condition, flux = D*grad(c) dot n = -mu*c
    flux = assemble(D * dot(grad(c), n) * ds(1))
    
    return flux

