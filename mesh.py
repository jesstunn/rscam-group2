# ========================================================
# Packages
# ========================================================

from dolfin import *
from mshr import *
import numpy as np

# ========================================================
# Mesh Generator Class
# ========================================================

def mesh_generator(resolution,
                   L, H,  
                   nx, ny, 
                   sulci_n, sulci_h, sulci_width):
    """
    Generate a mesh with optional sulci on the bottom boundary.
    
    Parameters:
    resolution : int
        Resolution for mesh generation, must be positive
    L, H : float
        Non-dimensional length and height of the domain, must be positive
    nx, ny : int
        Number of points in x and y directions for boundary definition, must be positive
    sulci_n : int
        Number of sulci (0 for flat bottom), must be non-negative
    sulci_h : float
        Non-dimensional height of sulci, must be non-negative
    sulci_width : float
        Non-dimensional width of sulci, must be positive if sulci_n > 0
    
    Returns:
    dict
        Dictionary containing:
        - mesh: The generated FEniCS mesh
        - boundary_markers: Dictionary with functions to identify boundaries (left, right, bottom, top)
        - mesh_info: Dictionary with additional mesh information
    """
    # -----------------------------------------------------------
    # Validate Inputs
    # -----------------------------------------------------------
    if resolution <= 0:
        raise ValueError("Resolution must be positive.")
    if L <= 0 or H <= 0:
        raise ValueError("Domain dimensions L and H must be positive.")
    if nx <= 0 or ny <= 0:
        raise ValueError("Number of points nx and ny must be positive.")
    if sulci_n < 0:
        raise ValueError("Number of sulci must be non-negative.")
    if sulci_n > 0 and sulci_h <= 0:
        raise ValueError("If sulci are present, their height must be positive.")
    if sulci_n > 0 and sulci_width <= 0:
        raise ValueError("If sulci are present, their width must be positive.")
    if sulci_n > 0 and sulci_width * sulci_n >= L:
        raise ValueError("Total width of sulci must be less than domain length.")
        
    # -----------------------------------------------------------
    # Generate sulci
    # -----------------------------------------------------------
    # Generates the centre positions for the sulci on the x dimension.

    # If no sulci specified return empty list
    if sulci_n == 0:
        sulcus_centers = []  
    else:
        # Generate uniformly spaced sulci centers
        sulcus_centers = np.linspace(L / (sulci_n + 1), L - L / (sulci_n + 1), sulci_n)

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
    points_upper = [(i / nx * L, H) for i in range(nx + 1)]  
    points_left = [(0, j / ny * H) for j in range(1, ny)]  
    points_right = [(L, j / ny * H) for j in range(1, ny)] 

    # Combine all points in counterclockwise order
    points = points_lower + points_right + points_upper[::-1] + points_left[::-1]

    try:
        # Define the polygon and generate mesh
        polygon = Polygon([Point(x, y) for x, y in points])
        mesh = generate_mesh(polygon, resolution)
        
        # Check if mesh generation was successful
        if mesh.num_cells() == 0:
            raise RuntimeError("Mesh generation failed: No cells were created.")
            
    except Exception as e:
        raise RuntimeError(f"Mesh generation failed: {str(e)}")

    # -----------------------------------------------------------
    # Define boundaries
    # -----------------------------------------------------------
    
    # Find extreme y-values across whole lower boundary
    min_y = min(p[1] for p in points_lower) if points_lower else 0.0
    max_y_lower = max(p[1] for p in points_lower) if points_lower else 0.0

    def left(x, on_boundary): 
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

    def right(x, on_boundary): 
        return on_boundary and near(x[0], L, DOLFIN_EPS)

    def bottom(x, on_boundary):  
        # Improved bottom boundary detection
        return on_boundary and min_y - 1e-4 <= x[1] <= max_y_lower + DOLFIN_EPS

    def top(x, on_boundary): 
        return on_boundary and near(x[1], H, DOLFIN_EPS)

    # Store boundary markers in a dictionary
    boundary_markers = {
        "left": left,
        "right": right, 
        "bottom": bottom,
        "top": top
    }

    # -----------------------------------------------------------
    # Calculate additional mesh information
    # -----------------------------------------------------------

    mesh_info = {
        "num_vertices": mesh.num_vertices(),
        "num_cells": mesh.num_cells(),
        "hmin": mesh.hmin(),  # Minimum cell diameter
        "hmax": mesh.hmax(),  # Maximum cell diameter
        "domain_area": assemble(Constant(1.0)*dx(domain=mesh)),
        "sulci_depth": abs(min_y) if min_y < 0 else 0.0,
        "sulci_centers": sulcus_centers.tolist() if sulci_n > 0 else [],
        "boundary_points": {
            "lower": points_lower,
            "upper": points_upper,
            "left": points_left,
            "right": points_right
        }
    }

    # -----------------------------------------------------------
    # Return mesh, boundary markers and mesh info
    # -----------------------------------------------------------

    return {
        "mesh": mesh,
        "boundary_markers": boundary_markers,
        "mesh_info": mesh_info
    }

# ========================================================
# Function to view the mesh as a plot
# ========================================================

def visualise_mesh(mesh_data, filename="mesh.png"):
    """
    Save a visualization of the mesh to a file in the 'visuals' folder.
    
    Parameters:
    mesh_data : dict or dolfin.Mesh
        Either the mesh object directly or the dictionary returned by mesh_generator
    filename : str, optional
        Name of the file of the mesh image, default is "mesh.png"
    """
    import matplotlib.pyplot as plt
    import os
    
    # Create the visuals directory if it doesn't exist
    visuals_dir = "visuals"
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Complete file path
    filepath = os.path.join(visuals_dir, filename)
    
    # Extract mesh from input
    if isinstance(mesh_data, dict) and "mesh" in mesh_data:
        mesh = mesh_data["mesh"]
        mesh_info = mesh_data.get("mesh_info", {})
    else:
        mesh = mesh_data
        mesh_info = {}
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot the mesh
    plot(mesh, title="Mesh")
    
    # Add mesh statistics annotation if available
    if mesh_info:
        info_text = (
            f"Vertices: {mesh_info.get('num_vertices', mesh.num_vertices())}\n"
            f"Cells: {mesh_info.get('num_cells', mesh.num_cells())}\n"
            f"Min cell size: {mesh_info.get('hmin', mesh.hmin()):.4f}\n"
            f"Sulci depth: {mesh_info.get('sulci_depth', 0):.4f}"
        )
        plt.annotate(info_text, xy=(0.02, 0.02), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Save the file
    try:
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Mesh visualization saved to {filepath}")
    except Exception as e:
        print(f"Warning: Failed to save mesh plot to {filepath}: {str(e)}")