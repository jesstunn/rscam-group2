##########################################################
# Plotting Module
##########################################################

# Contains all visualisation functions for simulation results.

# ========================================================
# Imports
# ========================================================

import matplotlib.pyplot as plt
import numpy as np
import os
from dolfin import *
import json

# ========================================================
# Formatting
# ========================================================

# Set default plotting style
def set_plotting_style():
    """Set default styling for all plots"""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })

# If want to change this for a specific graph, then just copy 
# from "plt.rcParams.update" onwards, place in to the
# plotting function and amend where required.
    
# ========================================================
# Mesh Visualisation
# ========================================================

def visualise_mesh(mesh_data, filename="Results/mesh.png"):
    """
    Save a visualisation of the mesh to a file.
    
    Parameters:
    mesh_data : dict or dolfin.Mesh
        Either the mesh object directly or the dictionary returned by mesh_generator
    filename : str, optional
        Name of the file of the mesh image
    """
    set_plotting_style()
        
    # Extract mesh from input
    if isinstance(mesh_data, dict) and "mesh" in mesh_data:
        mesh = mesh_data["mesh"]
        mesh_info = mesh_data.get("mesh_info", {})
    else:
        mesh = mesh_data
        mesh_info = {}
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
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
        # Add mesh statistics outside the plot
        plt.subplots_adjust(right=0.7)  # Shrink plot to make space on right side
        plt.figtext(0.72, 0.5, info_text, fontsize=10, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    # Save the file
    try:
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Mesh visualisation saved to {filename}")
    except Exception as e:
        print(f"Error: Failed to save mesh plot to {filename}: {str(e)}")

# ========================================================
# Flow Field Visualisation
# ========================================================

def visualise_velocity(u, mesh, filename="Results/velocity.png"):
    """
    Create and save a visualisation of the velocity field.
    
    Parameters:
    u : dolfin.Function
        Velocity field
    mesh : dolfin.Mesh
        Computational mesh
    filename : str, optional
        Path to save the visualisation
    """
    set_plotting_style()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot velocity magnitude
    vel_mag = sqrt(dot(u, u))
    c = plot(vel_mag, title="Velocity Field")
    plt.colorbar(c)
    
    # Save figure
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Velocity visualisation saved to {filename}")

# ========================================================
# Concentration Field Visualisation
# ========================================================

def visualise_concentration(c, mesh, filename="Results/concentration.png"):
    """
    Create and save a visualisation of the concentration field.
    
    Parameters:
    c : dolfin.Function
        Concentration field
    mesh : dolfin.Mesh
        Computational mesh
    filename : str, optional
        Path to save the visualisation
    """
    set_plotting_style()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot concentration
    plot_c = plot(c, title="Concentration Field")
    plt.colorbar(plot_c)
    
    # Save figure
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Concentration visualisation saved to {filename}")

# ========================================================
# Mass Analysis Plotting
# ========================================================

def plot_mass_results(results, x_param, y_param=None, log_scale=True, output_dir="Results/mass_analysis"):
    """
    Plot the results of a parameter sweep.
    
    Parameters:
    results : dict
        Results from parameter_sweep_mass
    x_param : str
        Parameter name to plot on x-axis
    y_param : str, optional
        If provided, create a heatmap with x_param and y_param
    log_scale : bool, optional
        Whether to use log scale for axes
    output_dir : str, optional
        Directory to save the plots
    """
    set_plotting_style()
    
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    params = results['parameters']
    mass = results['total_mass']
    
    if y_param is None:
        # 1D plot (line plot)
        # Group by x_param
        x_values = sorted(set(p[x_param] for p in params))
        y_values = []
        
        for x in x_values:
            # Find all masses for this x value
            matching_indices = [i for i, p in enumerate(params) if p[x_param] == x]
            avg_mass = np.mean([mass[i] for i in matching_indices])
            y_values.append(avg_mass)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        if log_scale and all(x > 0 for x in x_values):
            plt.semilogx(x_values, y_values, 'o-')
        else:
            plt.plot(x_values, y_values, 'o-')
        
        plt.xlabel(f'{x_param}')
        plt.ylabel('Average mass')
        plt.grid(True)
        plt.title(f'Average mass vs {x_param}')
        
        # Save plot
        plt.savefig(f'{output_dir}/mass_vs_{x_param}.png', dpi=300)
        plt.close()
        
    else:
        # 2D plot (heatmap)
        # Get unique values for both parameters
        x_values = sorted(set(p[x_param] for p in params))
        y_values = sorted(set(p[y_param] for p in params))
        
        # Create a grid for the heatmap
        heat_data = np.zeros((len(y_values), len(x_values)))
        
        # Fill the grid
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                # Find matching parameter combination
                matches = [idx for idx, p in enumerate(params) 
                          if p[x_param] == x_val and p[y_param] == y_val]
                
                if matches:
                    heat_data[i, j] = mass[matches[0]]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(heat_data, origin='lower', aspect='auto', cmap='viridis')
        
        # Set x and y ticks
        plt.xticks(range(len(x_values)), x_values)
        plt.yticks(range(len(y_values)), y_values)
        
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.colorbar(label='Average mass')
        plt.title(f'Average mass as a function of {x_param} and {y_param}')
        
        # Save plot
        plt.savefig(f'{output_dir}/mass_heatmap_{x_param}_{y_param}.png', dpi=300)
        plt.close()

# ========================================================
# No Sulci Analysis Plotting
# ========================================================

def plot_no_sulci_comparison(no_sulci_results, with_sulci_results, output_dir="Results/no_sulci_analysis/comparison"):
    """
    Create comparative visualisations for the no sulci analysis.
    """
    set_plotting_style()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Compare concentration fields
        plt.figure(figsize=(12, 5))
        
        # Plot no sulci concentration - first subplot
        ax1 = plt.subplot(1, 2, 1)
        try:
            # Try to get the velocity magnitude for a more reliable plot
            c1 = no_sulci_results['c']
            c1_plot = plot(c1)
            plt.colorbar(c1_plot)
            plt.title("Concentration (No Sulci)")
        except Exception as e:
            print(f"Error plotting no-sulci concentration: {e}")
            plt.title("Concentration (No Sulci) - Plot Failed")
        
        # Plot with sulci concentration - second subplot
        ax2 = plt.subplot(1, 2, 2)
        try:
            # Try to get the velocity magnitude for a more reliable plot
            c2 = with_sulci_results['c']
            c2_plot = plot(c2)
            plt.colorbar(c2_plot)
            plt.title("Concentration (With Sulci)")
        except Exception as e:
            print(f"Error plotting with-sulci concentration: {e}")
            plt.title("Concentration (With Sulci) - Plot Failed")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "concentration_comparison.png"), dpi=300)
        plt.close()
        
        # Compare velocity fields
        plt.figure(figsize=(12, 5))
        
        # Plot no sulci velocity - first subplot
        ax1 = plt.subplot(1, 2, 1)
        try:
            # Try to get the velocity magnitude for a more reliable plot
            u1 = no_sulci_results['u']
            u1_mag = sqrt(dot(u1, u1))  # This might be more reliable for plotting
            u1_plot = plot(u1_mag)
            plt.colorbar(u1_plot)
            plt.title("Velocity Magnitude (No Sulci)")
        except Exception as e:
            print(f"Error plotting no-sulci velocity: {e}")
            plt.title("Velocity (No Sulci) - Plot Failed")
        
        # Plot with sulci velocity - second subplot
        ax2 = plt.subplot(1, 2, 2)
        try:
            # Try to get the velocity magnitude for a more reliable plot
            u2 = with_sulci_results['u']
            u2_mag = sqrt(dot(u2, u2))  # This might be more reliable for plotting
            u2_plot = plot(u2_mag)
            plt.colorbar(u2_plot)
            plt.title("Velocity Magnitude (With Sulci)")
        except Exception as e:
            print(f"Error plotting with-sulci velocity: {e}")
            plt.title("Velocity (With Sulci) - Plot Failed")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "velocity_comparison.png"), dpi=300)
        plt.close()
        
    except Exception as e:
        print("Error: Could not create comparison visualisations: {}".format(e))
        import traceback
        traceback.print_exc()

# ========================================================
# Sulci Geometry Analysis Plotting
# ========================================================

def plot_sulci_geometry_comparison(results, output_dir, pe_values, mu_values, fixed_pe, fixed_mu):
    """
    Create bar charts comparing average mass across different sulci geometries,
    with both Pe and mu variations.
    
    Parameters:
    results : dict
        Nested dictionary with results for each geometry and parameter variation
    output_dir : str
        Directory to save output plots
    pe_values : list
        List of Pe values used in the simulations
    mu_values : list
        List of mu values used in the simulations
    fixed_pe : float
        The fixed Pe value used when varying mu
    fixed_mu : float
        The fixed mu value used when varying Pe
    """
    set_plotting_style()
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract geometry names
    geometries = list(results["varying_pe"].keys())

    # Format geometry names for display with h and l in italics
    display_names = []
    for geom_name in geometries:
        if "small_height" in geom_name:
            height_str = "Small $h$"
        else:
            height_str = "Large $h$"
            
        if "small_width" in geom_name:
            width_str = "small $l$"
        else:
            width_str = "large $l$"
        
        display_names.append(f"{height_str}, {width_str}")
    
    # --------------------------------------------------------
    # Create grouped bar chart for varying Pe (fixed mu)
    plt.figure(figsize=(14, 8))

    # Set bar width and positions
    num_groups = len(geometries)
    num_bars = len(pe_values)
    bar_width = 0.8 / num_bars
    
    # Define colors for different Pe values (blue scheme)
    pe_colors = ['#9CC3E6', '#2E75B6', '#203864']  # Light to dark blue
    
    # Plot bars for each Pe value
    for i, pe in enumerate(pe_values):
        # Extract mass values for this Pe across all geometries
        mass_values = [results["varying_pe"][geom][pe]["total_mass"] for geom in geometries]
        
        # Calculate x positions for this group of bars
        x_positions = [j + (i - num_bars/2 + 0.5) * bar_width for j in range(num_groups)]
        
        # Create bars
        bars = plt.bar(x_positions, mass_values, bar_width, 
                       label=f'Pe = {pe}', color=pe_colors[i % len(pe_colors)])
        
        # Add value labels on top of bars
        for bar, value in zip(bars, mass_values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01*max(mass_values), 
                    f"{value:.3f}", ha='center', va='bottom', rotation=0)
    
    # Add labels and legend
    plt.xlabel("Sulci Geometry", fontsize=20)
    plt.ylabel("Average mass", fontsize=20)
    plt.title(f"Effect of Sulci Geometry on Average Mass for Different Pe Values (Fixed μ={fixed_mu})")
    plt.xticks(range(num_groups), display_names)
    plt.legend(title="Péclet Number")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(comparison_dir, "mass_comparison_varying_pe.png"), dpi=300)
    plt.close()
    
    # --------------------------------------------------------
    # Create grouped bar chart for varying mu (fixed Pe)
    plt.figure(figsize=(14, 8))
    
    # Set bar width and positions
    num_groups = len(geometries)
    num_bars = len(mu_values)
    bar_width = 0.8 / num_bars
    
    # Define colors for different mu values (green scheme)
    mu_colors = ['#A8D08D', '#548235', '#385723']  # Light to dark green
    
    # Plot bars for each mu value
    for i, mu in enumerate(mu_values):
        # Extract mass values for this mu across all geometries
        mass_values = [results["varying_mu"][geom][mu]["total_mass"] for geom in geometries]
        
        # Calculate x positions for this group of bars
        x_positions = [j + (i - num_bars/2 + 0.5) * bar_width for j in range(num_groups)]
        
        # Create bars
        bars = plt.bar(x_positions, mass_values, bar_width, 
                       label=f'μ = {mu}', color=mu_colors[i % len(mu_colors)])
        
        # Add value labels on top of bars
        for bar, value in zip(bars, mass_values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01*max(mass_values), 
                    f"{value:.3f}", ha='center', va='bottom', rotation=0)
    
    # Add labels and legend
    plt.xlabel("Sulci Geometry", fontsize=20)
    plt.ylabel("Average mass", fontsize=20)
    plt.title(f"Effect of Sulci Geometry on Average Mass for Different μ Values (Fixed Pe={fixed_pe})")
    plt.xticks(range(num_groups), display_names)
    plt.legend(title="Uptake (μ)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(comparison_dir, "mass_comparison_varying_mu.png"), dpi=300)
    plt.close()
    
    # --------------------------------------------------------
    # Create multiline labels for better readability
    multiline_labels = [
        "Small Height,\nSmall Width",
        "Small Height,\nLarge Width",
        "Large Height,\nSmall Width",
        "Large Height,\nLarge Width"
    ]

    # Panel figure for varying Pe
    fig1, axes1 = plt.subplots(1, len(pe_values), figsize=(18, 6), sharey=True)

    for i, pe in enumerate(pe_values):
        ax = axes1[i]
        
        # Extract mass values for this Pe across all geometries
        mass_values = [results["varying_pe"][geom][pe]["total_mass"] for geom in geometries]
        
        # Create bars with multiline labels
        bars = ax.bar(multiline_labels, mass_values, color=pe_colors[i % len(pe_colors)])
        
        # Add value labels
        for bar, value in zip(bars, mass_values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.01*max(mass_values), 
                f"{value:.3f}", ha='center', va='bottom')
        
        # Set title and grid
        ax.set_title(f'Pe = {pe}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Only add y-label to the first subplot
        if i == 0:
            ax.set_ylabel("Average mass")
        
        # Adjust x-tick label size if needed
        ax.tick_params(axis='x', labelsize=11)

    # Set overall title
    fig1.suptitle(f"Effect of Sulci Geometry on average mass for Different Pe Values (Fixed μ={fixed_mu})")

    # Adjust layout and add x-label
    plt.xlabel("Sulci Geometry")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  

    # Save the plot
    plt.savefig(os.path.join(comparison_dir, "mass_comparison_panels_pe.png"), dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Panel figure for varying mu
    fig2, axes2 = plt.subplots(1, len(mu_values), figsize=(18, 6), sharey=True)

    for i, mu in enumerate(mu_values):
        ax = axes2[i]
        
        # Extract mass values for this mu across all geometries
        mass_values = [results["varying_mu"][geom][mu]["total_mass"] for geom in geometries]
        
        # Create bars with multiline labels
        bars = ax.bar(multiline_labels, mass_values, color=mu_colors[i % len(mu_colors)])
        
        # Add value labels
        for bar, value in zip(bars, mass_values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.01*max(mass_values), 
                f"{value:.3f}", ha='center', va='bottom')
        
        # Set title and grid
        ax.set_title(f'μ = {mu}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Only add y-label to the first subplot
        if i == 0:
            ax.set_ylabel("Average mass")
        
        # Adjust x-tick label size if needed
        ax.tick_params(axis='x', labelsize=11)

    # Set overall title
    fig2.suptitle(f"Effect of Sulci Geometry on average mass for Different μ Values (Fixed Pe={fixed_pe})")

    # Adjust layout and add x-label
    plt.xlabel("Sulci Geometry")
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 

    # Save the plot
    plt.savefig(os.path.join(comparison_dir, "mass_comparison_panels_mu.png"), dpi=300)
    plt.close()
