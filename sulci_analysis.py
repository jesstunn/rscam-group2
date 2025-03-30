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
from stokes import stokes_solver, compute_flow_rate, visualise_velocity, save_flow_fields
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
        flow_rate = compute_flow_rate(u, mesh)
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
    Run a study on the effect of sulci geometry on flow and concentration.
    Examines 4 cases:
    1. Small height, small width
    2. Small height, large width
    3. Large height, small width
    4. Large height, large width
    
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
    
    # Define the 4 cases
    cases = [
        {"name": "small_height_small_width", "height": small_height, "width": small_width},
        {"name": "small_height_large_width", "height": small_height, "width": large_width},
        {"name": "large_height_small_width", "height": large_height, "width": small_width},
        {"name": "large_height_large_width", "height": large_height, "width": large_width}
    ]
    
    # Store results
    results = {}
    
    # Run simulations for each case
    for case in cases:
        print("\n" + "="*50)
        print(f"Running case: {case['name']}")
        print("="*50)
        
        # Create parameters with default values
        params = Parameters()
        
        # Set sulci parameters for this case
        params.sulci_n = 1  # Fixed number of sulci
        params.sulci_h_mm = case["height"]
        params.sulci_width_mm = case["width"]
        
        # Validate and recalculate derived parameters
        params.validate()
        params.nondim()
        
        # Create case-specific output directory
        case_dir = os.path.join(output_dir, case["name"])
        
        # Run simulation with these parameters
        case_results = run_single_sulci_simulation(params, case_dir)
        
        # Store key results
        results[case["name"]] = {
            "params": {
                "sulci_height": params.sulci_h_mm,
                "sulci_width": params.sulci_width_mm,
                "Pe": params.Pe,
                "mu": params.mu
            },
            "total_mass": float(case_results["total_mass"]),
            "flow_rate": float(case_results["flow_rate"]) if case_results["flow_rate"] is not None else None
        }
    
    # Comparative analysis
    compare_results(results, output_dir)
    
    return results

def compare_results(results, output_dir):
    """
    Compare and visualise results from different sulci configurations.
    
    Parameters:
    -----------
    results : dict
        Dictionary with results for each case
    output_dir : str
        Directory to save comparison plots
    """
    # Extract data for plotting
    case_names = list(results.keys())
    mass_values = [results[case]["total_mass"] for case in case_names]
    flow_values = [results[case]["flow_rate"] for case in case_names]
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Format case names for display
    display_names = [name.replace("_", " ").title() for name in case_names]
    
    # Plot total mass comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(display_names, mass_values)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, mass_values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.01*max(mass_values), 
                f"{value:.4f}", ha='center', va='bottom')
    
    plt.ylabel("Total Mass")
    plt.title("Effect of Sulci Geometry on Total Mass")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "mass_comparison.png"), dpi=300)
    plt.close()
    
    # Plot flow rate comparison
    if all(flow is not None for flow in flow_values):
        plt.figure(figsize=(12, 6))
        bars = plt.bar(display_names, flow_values)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, flow_values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01*max(flow_values), 
                    f"{value:.4f}", ha='center', va='bottom')
        
        plt.ylabel("Flow Rate")
        plt.title("Effect of Sulci Geometry on Flow Rate")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "flow_comparison.png"), dpi=300)
        plt.close()
    
    # Save comparison data
    comparison_data = {
        "cases": case_names,
        "total_mass": mass_values,
        "flow_rate": flow_values,
        "normalized_mass": [m/max(mass_values) for m in mass_values],
        "normalized_flow": [f/max(flow_values) if f is not None else None for f in flow_values]
    }
    
    with open(os.path.join(comparison_dir, "comparison_data.json"), "w") as f:
        json.dump(comparison_data, f, indent=4)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze effect of sulci geometry")
    parser.add_argument("--output-dir", type=str, default="results/sulci_analysis", 
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run the analysis
    results = run_sulci_analysis(args.output_dir)
    
    print("\nSulci geometry analysis completed!")