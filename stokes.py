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

def compute_multiple_flow_rates(u, mesh, num_sections=3):
    """
    Compute flow rates at multiple cross-sections along the channel.
    
    Parameters:
    u : dolfin.Function
        Velocity field
    mesh : dolfin.Mesh
        The computational mesh
    num_sections : int, optional
        Number of cross-sections to evaluate (default: 3)
    
    Returns:
    tuple
        (x_positions, flow_rates): Lists of x-positions and corresponding flow rates
    """
    # Get domain bounds
    x_min = mesh.coordinates()[:, 0].min()
    x_max = mesh.coordinates()[:, 0].max()
    L = x_max - x_min
    
    # Define cross-section positions (20%, 50%, 80% of domain length)
    if num_sections == 3:
        x_positions = [x_min + 0.2*L, x_min + 0.5*L, x_min + 0.8*L]
    else:
        # If not exactly 3 sections, space them evenly with margins
        margin = 0.1 * L
        x_positions = np.linspace(x_min + margin, x_max - margin, num_sections)
    
    # Calculate flow rates
    flow_rates = []
    
    print("\nComputing flow rates at multiple cross-sections:")
    for i, x_pos in enumerate(x_positions):
        # Define a cross-section boundary similar to other boundaries
        def cross_section(x, on_boundary):
            # This checks for points near the vertical line at x_pos
            # We don't need on_boundary here since we want interior points too
            return near(x[0], x_pos, mesh.hmin()/2)
        
        # Mark the facets on this cross-section
        facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
        
        # Mark all facets, not just boundary facets
        for facet in facets(mesh):
            mp = facet.midpoint()  # Get facet midpoint
            if cross_section(mp, False):  # Check if midpoint is on cross-section
                facet_markers[facet] = 1
        
        # Count marked facets
        num_marked = sum(1 for f in facets(mesh) if facet_markers[f] == 1)
        
        # Create a measure for integration
        ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)
        
        # Use interior facet measure for internal facets
        dS = Measure("dS", domain=mesh, subdomain_data=facet_markers)
        
        # Normal vector (pointing in positive x direction)
        n = Constant((1.0, 0.0))
        
        # Compute flow rate by integrating over marked facets
        # For boundary facets: dot(u, n)*ds(1)
        # For interior facets: dot(avg(u), n('+'))*dS(1)
        boundary_flow = assemble(dot(u, n)*ds(1))
        interior_flow = assemble(dot(avg(u), n('+'))*dS(1))
        flow_rate = boundary_flow + interior_flow
        
        flow_rates.append(flow_rate)
        print(f"Cross-section {i+1} at x = {x_pos:.3f}: Marked {num_marked} facets, Flow = {flow_rate:.6f}")
    
    if flow_rates:
        print(f"Average flow rate: {sum(flow_rates)/len(flow_rates):.6f}")
        print(f"Max flow rate: {max(flow_rates):.6f}")
    else:
        print("No valid flow rates calculated")
    
    return x_positions, flow_rates


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

    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })
    
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
