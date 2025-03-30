# ========================================================
# Stokes Solver Module
# ========================================================

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os

def stokes_solver(mesh, W, H, left, right, bottom, top):
    """
    Solve the Stokes equations for incompressible flow.
    
    Parameters:
    mesh : dolfin.Mesh
        The computational mesh
    W : dolfin.FunctionSpace
        Mixed function space for velocity and pressure
    H : float
        Domain height for inlet profile
    left, right, bottom, top : function
        Functions to identify boundaries
    
    Returns:
    u : dolfin.Function
        Velocity field solution
    p : dolfin.Function
        Pressure field solution
    """
    # --------------------------------------------------------
    # Define Boundary Conditions
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

    # Configure solver
    solver = KrylovSolver("minres", "amg")
    solver.set_operators(A, P)

    # Solve
    U = Function(W)
    solver.solve(U.vector(), bb)

    # Get sub-functions
    u, p = U.split()
    return u, p

def compute_flow_rate(u, mesh, cross_section_x=None):
    """
    Compute the flow rate (flux) across a vertical cross-section.
    
    Parameters:
    u : dolfin.Function
        Velocity field
    mesh : dolfin.Mesh
        The computational mesh
    cross_section_x : float, optional
        x-coordinate of the cross-section (defaults to domain midpoint)
    
    Returns:
    flow_rate : float
        Flow rate across the cross-section
    """
    if cross_section_x is None:
        # Default to domain midpoint
        cross_section_x = mesh.coordinates()[:, 0].max() / 2
    
    # Define a plane (line in 2D) at the specified x-coordinate
    class CrossSection(SubDomain):
        def __init__(self, x):
            self.x = x
            super().__init__()
            
        def inside(self, x, on_boundary):
            return near(x[0], self.x, 1e-14)
    
    # Mark the cross-section
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    CrossSection(cross_section_x).mark(boundaries, 1)
    
    # Define normal vector (pointing in positive x direction)
    n = Constant((1.0, 0.0))
    
    # Create a measure
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    # Compute flow rate by integrating the normal component of velocity
    flow_rate = assemble(dot(u, n)*ds(1))
    
    return flow_rate

def compute_max_velocity(u):
    """
    Compute the maximum velocity magnitude in the domain.
    
    Parameters:
    u : dolfin.Function
        Velocity field
    
    Returns:
    max_vel : float
        Maximum velocity magnitude
    """
    # Compute velocity magnitude
    vel_mag = sqrt(dot(u, u))
    
    # Find maximum
    max_vel = vel_mag.vector().max()
    
    return max_vel

def save_flow_fields(u, p, directory="results"):
    """
    Save velocity and pressure fields to files.
    
    Parameters:
    u : dolfin.Function
        Velocity field
    p : dolfin.Function
        Pressure field
    directory : str, optional
        Directory to save the files (default: "results")
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save fields
    File(os.path.join(directory, "velocity.pvd")) << u
    File(os.path.join(directory, "pressure.pvd")) << p
    
    print(f"Flow fields saved to {directory}")

def visualise_velocity(u, mesh, filename="visuals/velocity.png"):
    """
    Create and save a visualisation of the velocity field.
    
    Parameters:
    u : dolfin.Function
        Velocity field
    mesh : dolfin.Mesh
        Computational mesh
    filename : str, optional
        Path to save the visualisation (default: "visuals/velocity.png")
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot velocity magnitude
    vel_mag = sqrt(dot(u, u))
    c = plot(vel_mag, title="Velocity Magnitude")
    plt.colorbar(c)
    
    # Save figure
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Velocity visualisation saved to {filename}")