# ========================================================
# Sulci Analysis Module
# ========================================================

# Import general modules
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from datetime import datetime
from copy import deepcopy

# Import project modules
from parameters import Parameters
from mesh import mesh_generator, visualise_mesh
from stokes import stokes_solver, compute_multiple_flow_rates, visualise_velocity, save_flow_fields
from adv_diff import advdiff_solver, calculate_total_mass, visualise_concentration, save_concentration_field

def run_single_sulci_simulation(params, output_dir):
    """
    Run a single simulation with given sulci parameters.
    This is similar to run_simulation but defined directly in this module
    to avoid circular imports.
    """
    # Start timing
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visuals directory if it doesn't exist
    visuals_dir = os.path.join(output_dir, "plots")
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Print parameters
    print("\nSimulation Parameters:")
    print(params)
    
    # Generate mesh
    print("\n1. Generating mesh...")
    mesh_result = mesh_generator(
        params.resolution,
        params.L, params.H,  
        params.nx, params.ny, 
        params.sulci_n, params.sulci_h, params.sulci_width
    )
    
    # Extract mesh and boundary markers
    if isinstance(mesh_result, dict):
        mesh = mesh_result["mesh"]
        left = mesh_result["boundary_markers"]["left"]
        right = mesh_result["boundary_markers"]["right"]
        bottom = mesh_result["boundary_markers"]["bottom"]
        top = mesh_result["boundary_markers"]["top"]
        mesh_info = mesh_result["mesh_info"]
    else:
        mesh, left, right, bottom, top = mesh_result
        mesh_info = {"num_vertices": mesh.num_vertices(), "num_cells": mesh.num_cells()}
    
    # Save mesh visualisation
    try:
        visualise_mesh(mesh_result, os.path.join(visuals_dir, "mesh.png"))
    except Exception as e:
        print(f"Error: Could not save mesh visualisation: {e}")
    
    # Build function spaces
    print("2. Building function spaces...")
    V = VectorFunctionSpace(mesh, "P", 2)  
    Q = FunctionSpace(mesh, "P", 1)  
    W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
    C = FunctionSpace(mesh, "CG", 1)
    
    # Solve Stokes
    print("3. Solving Stokes equations...")
    u, p = stokes_solver(mesh, W, params.H, left, right, bottom, top)
    
    # Save flow visualisation
    try:
        visualise_velocity(u, mesh, os.path.join(visuals_dir, "velocity.png"))
        save_flow_fields(u, p, directory=output_dir)
    except Exception as e:
        print(f"Error: Could not save flow fields: {e}")
    
    # Solve Advection-Diffusion
    print("4. Solving advection-diffusion equation...")
    c = advdiff_solver(
        mesh, left, right, bottom,
        u, C,
        D=Constant(params.D),
        mu=Constant(params.mu)
    )
    
    # Save concentration visualisation
    try:
        visualise_concentration(c, mesh, os.path.join(visuals_dir, "concentration.png"))
        save_concentration_field(c, directory=output_dir)
    except Exception as e:
        print(f"Error: Could not save concentration field: {e}")
    
    # Post-process
    print("5. Post-processing results...")
    
    # Calculate mass
    total_mass = calculate_total_mass(c, mesh)
    
    # Calculate flow rate
    try:
        x_positions, flow_rates = compute_multiple_flow_rates(u, mesh, num_sections=5)
        flow_rate = max(flow_rates) if flow_rates else 0.0
    except Exception as e:
        print(f"Error: Could not compute flow rate: {e}")
        flow_rate = None
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print summary
    print("\nSimulation Results:")
    print(f"  Total mass: {total_mass:.6f}")
    if flow_rate is not None:
        print(f"  Flow rate: {flow_rate:.6f}")
    print(f"  Péclet number: {params.Pe:.2f}")
    print(f"  Elapsed time: {elapsed_time:.2f} seconds")
    
    # Return results
    return {
        "params": params,
        "mesh": mesh,
        "u": u,
        "p": p,
        "c": c,
        "total_mass": total_mass,
        "flow_rate": flow_rate,
        "elapsed_time": elapsed_time
    }

def run_sulci_analysis(output_dir="sulci_results"):
    """
    Runs a study on the effect of sulci geometry on flow and concentration.

    Examines 4 cases:
    1. Small height, small width
    2. Small height, large width
    3. Large height, small width
    4. Large height, large width
    
    Across multiple Pe and mu values for comparison.

    Parameters:
    output_dir : str
        Directory to save results
    
    Returns:
    dict
        Dictionary with results for each case
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define sulci configurations
    small_height = 0.5  # Small sulcus height in mm
    large_height = 2.0  # Large sulcus height in mm
    small_width = 0.5   # Small sulcus width in mm
    large_width = 2.0   # Large sulcus width in mm
    
    # Define the 4 geometry cases
    geometries = [
        {"name": "small_height_small_width", "height": small_height, "width": small_width},
        {"name": "small_height_large_width", "height": small_height, "width": large_width},
        {"name": "large_height_small_width", "height": large_height, "width": small_width},
        {"name": "large_height_large_width", "height": large_height, "width": large_width}
    ]
    
    # Define different Pe and mu values to test
    pe_values = [1, 10, 100]      # Low, medium, high Pe
    mu_values = [0.1, 1.0, 10.0]  # Low, medium, high uptake
    
    # Store results for all combinations
    all_results = {"varying_pe": {}, "varying_mu": {}}
    
    # --------------------------------------------------------

    # Run simulations for each geometry with varying Pe (fixed mu)
    fixed_mu = 1.0  # Fixed mu value for the varying Pe part
    
    for geometry in geometries:

        all_results["varying_pe"][geometry["name"]] = {}
        
        for pe in pe_values:

            print("\n" + "="*50)
            print(f"Running case: {geometry['name']} with Pe={pe}, mu={fixed_mu}")
            print("="*50)
            
            # Create parameters with default values
            params = Parameters()
            
            # Set geometry parameters for this case
            params.sulci_n = 1  
            params.sulci_h_mm = geometry["height"]
            params.sulci_width_mm = geometry["width"]
            
            # Set Pe value (adjusting U_ref to achieve desired Pe)
            params.U_ref = pe * params.D_mms2 / params.H_mm
            params.mu = fixed_mu
            
            # Validate and recalculate derived parameters
            params.validate()
            params.nondim()
            
            # Create case-specific output directory
            case_dir = os.path.join(output_dir, f"{geometry['name']}_Pe{pe}_mu{fixed_mu}")
            
            # Run simulation with these parameters
            case_results = run_single_sulci_simulation(params, case_dir)
            
            # Store key results
            all_results["varying_pe"][geometry["name"]][pe] = {
                "params": {
                    "sulci_height": params.sulci_h_mm,
                    "sulci_width": params.sulci_width_mm,
                    "Pe": params.Pe,
                    "mu": params.mu
                },
                "total_mass": float(case_results["total_mass"]),
                "flow_rate": float(case_results["flow_rate"]) if case_results["flow_rate"] is not None else None
            }
    
    # --------------------------------------------------------

    # Run simulations for each geometry with varying mu (fixed Pe)
    fixed_pe = 10  # Fixed Pe value for the varying mu part
    
    for geometry in geometries:
        all_results["varying_mu"][geometry["name"]] = {}
        
        for mu in mu_values:
            print("\n" + "="*50)
            print(f"Running case: {geometry['name']} with Pe={fixed_pe}, mu={mu}")
            print("="*50)
            
            # Create parameters with default values
            params = Parameters()
            
            # Set geometry parameters for this case
            params.sulci_n = 1  # Fixed number of sulci
            params.sulci_h_mm = geometry["height"]
            params.sulci_width_mm = geometry["width"]
            
            # Set Pe and mu values
            params.U_ref = fixed_pe * params.D_mms2 / params.H_mm
            params.mu = mu
            
            # Validate and recalculate derived parameters
            params.validate()
            params.nondim()
            
            # Create case-specific output directory
            case_dir = os.path.join(output_dir, f"{geometry['name']}_Pe{fixed_pe}_mu{mu}")
            
            # Run simulation with these parameters
            case_results = run_single_sulci_simulation(params, case_dir)
            
            # Store key results
            all_results["varying_mu"][geometry["name"]][mu] = {
                "params": {
                    "sulci_height": params.sulci_h_mm,
                    "sulci_width": params.sulci_width_mm,
                    "Pe": params.Pe,
                    "mu": params.mu
                },
                "total_mass": float(case_results["total_mass"]),
                "flow_rate": float(case_results["flow_rate"]) if case_results["flow_rate"] is not None else None
            }
    
    # Generate enhanced comparison charts for both parameter variations
    compare_results_dual(all_results, output_dir, pe_values, mu_values, fixed_pe, fixed_mu)
    
    return all_results


def compare_results_dual(results, output_dir, pe_values, mu_values, fixed_pe, fixed_mu):
    """
    Create bar charts comparing total mass across different sulci geometries,
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
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract geometry names
    geometries = list(results["varying_pe"].keys())
    
    # Format geometry names for display
    display_names = [name.replace("_", " ").title() for name in geometries]
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 13,
        'axes.labelsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
    })
    
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
                    f"{value:.3f}", ha='center', va='bottom', fontsize=11, rotation=0)
    
    # Add labels and legend
    plt.xlabel("Sulci Geometry")
    plt.ylabel("Total Mass")
    plt.title(f"Effect of Sulci Geometry on Total Mass for Different Pe Values (Fixed μ={fixed_mu})")
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
                    f"{value:.3f}", ha='center', va='bottom', fontsize=11, rotation=0)
    
    # Add labels and legend
    plt.xlabel("Sulci Geometry")
    plt.ylabel("Total Mass")
    plt.title(f"Effect of Sulci Geometry on Total Mass for Different μ Values (Fixed Pe={fixed_pe})")
    plt.xticks(range(num_groups), display_names)
    plt.legend(title="Uptake Parameter (μ)")
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
        
        # Create bars with numbered labels
        bars = ax.bar(numbered_labels, mass_values, color=pe_colors[i % len(pe_colors)])
        
        # Add value labels
        for bar, value in zip(bars, mass_values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.01*max(mass_values), 
                f"{value:.3f}", ha='center', va='bottom', fontsize=11)
        
        # Set title and grid
        ax.set_title(f'Pe = {pe}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Only add y-label to the first subplot
        if i == 0:
            ax.set_ylabel("Total Mass")

    # Add a text box with label explanations
    fig1.text(0.15, 0.03, 
            "\n".join([f"{num} = {desc}" for num, desc in multiline_labels .items()]),
            fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

    # Set overall title
    fig1.suptitle(f"Effect of Sulci Geometry on Total Mass for Different Pe Values (Fixed μ={fixed_mu})", fontsize=15)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Make room for the suptitle and legend

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
        
        # Create bars with numbered labels
        bars = ax.bar(numbered_labels, mass_values, color=mu_colors[i % len(mu_colors)])
        
        # Add value labels
        for bar, value in zip(bars, mass_values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.01*max(mass_values), 
                f"{value:.3f}", ha='center', va='bottom', fontsize=11)
        
        # Set title and grid
        ax.set_title(f'μ = {mu}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Only add y-label to the first subplot
        if i == 0:
            ax.set_ylabel("Total Mass")

    # Add a text box with label explanations
    fig2.text(0.15, 0.03, 
            "\n".join([f"{num} = {desc}" for num, desc in multiline_labels .items()]),
            fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

    # Set overall title
    fig2.suptitle(f"Effect of Sulci Geometry on Total Mass for Different μ Values (Fixed Pe={fixed_pe})", fontsize=15)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Make room for the suptitle and legend

    # Save the plot
    plt.savefig(os.path.join(comparison_dir, "mass_comparison_panels_mu.png"), dpi=300)
    plt.close()
    
    # --------------------------------------------------------
    
    # Save comparison data
    comparison_data = {
        "geometries": geometries,
        "pe_values": pe_values,
        "mu_values": mu_values,
        "fixed_pe": fixed_pe,
        "fixed_mu": fixed_mu,
        "results": results
    }
    
    with open(os.path.join(comparison_dir, "comparison_data_dual.json"), "w") as f:
        json.dump(comparison_data, f, indent=4)