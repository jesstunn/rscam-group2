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
from run_simulation import run_simulation

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
        case_results = run_simulation(params, case_dir)
        
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