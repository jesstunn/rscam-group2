##########################################################
# No Sulci Analysis Module
##########################################################

# ========================================================
# Imports
# ========================================================

# Import general modules
from dolfin import *
import numpy as np
import os
import json
from datetime import datetime

# Import project modules
from parameters import Parameters
from plotting import plot_no_sulci_comparison

# ========================================================
# No Sulci Analysis Functions
# ========================================================

def run_no_sulci_analysis(output_dir="Results/no_sulci_analysis"):
    """
    Run an analysis comparing a simulation with no sulci to the default case with sulci.
    
    Parameters:
    output_dir : str
        Directory where results will be saved
    
    Returns:
    results : dict
        Dictionary containing simulation results for both cases
    """
    print("\nRunning no sulci analysis...")
    
    # Import here to avoid circular imports
    from run_simulation import run_simulation
    
    # Create output directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    no_sulci_dir = os.path.join(output_dir, "no_sulci")
    with_sulci_dir = os.path.join(output_dir, "with_sulci")
    comparison_dir = os.path.join(output_dir, "comparison")
    
    os.makedirs(no_sulci_dir, exist_ok=True)
    os.makedirs(with_sulci_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Run simulation with no sulci
    print("\nRunning simulation with no sulci (flat surface)...")
    no_sulci_params = Parameters()
    no_sulci_params.sulci_n = 0  # Set number of sulci to zero
    no_sulci_results = run_simulation(no_sulci_params, no_sulci_dir)
    
    # Run simulation with default sulci parameters
    print("\nRunning simulation with default sulci parameters...")
    with_sulci_params = Parameters()  # Use default parameters
    with_sulci_results = run_simulation(with_sulci_params, with_sulci_dir)
    
    # Compare results
    print("\nComparing results:")
    print(f"No sulci mass: {no_sulci_results['total_mass']:.6f}")
    print(f"With sulci mass: {with_sulci_results['total_mass']:.6f}")
    print(f"Mass difference: {with_sulci_results['total_mass'] - no_sulci_results['total_mass']:.6f}")
    
    # if no_sulci_results['flow_rate'] is not None and with_sulci_results['flow_rate'] is not None:
    #     print(f"No sulci flow rate: {no_sulci_results['flow_rate']:.6f}")
    #     print(f"With sulci flow rate: {with_sulci_results['flow_rate']:.6f}")
    #     print(f"Flow rate difference: {with_sulci_results['flow_rate'] - no_sulci_results['flow_rate']:.6f}")
    
    # Generate comparative plots using the plotting module
    plot_no_sulci_comparison(no_sulci_results, with_sulci_results, comparison_dir)
    
    # Save comparison summary to file
    comparison_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "no_sulci": {
            "parameters": {
                "sulci_n": no_sulci_params.sulci_n,
                "Pe": no_sulci_params.Pe,
                "mu": no_sulci_params.mu
            },
            "results": {
                "total_mass": float(no_sulci_results["total_mass"])
                # "flow_rate": float(no_sulci_results["flow_rate"]) if no_sulci_results["flow_rate"] is not None else None
            }
        },
        "with_sulci": {
            "parameters": {
                "sulci_n": with_sulci_params.sulci_n,
                "sulci_h": with_sulci_params.sulci_h,
                "sulci_width": with_sulci_params.sulci_width,
                "Pe": with_sulci_params.Pe,
                "mu": with_sulci_params.mu
            },
            "results": {
                "total_mass": float(with_sulci_results["total_mass"])
                # "flow_rate": float(with_sulci_results["flow_rate"]) if with_sulci_results["flow_rate"] is not None else None
            }
        },
        "comparison": {
            "mass_difference": float(with_sulci_results["total_mass"] - no_sulci_results["total_mass"])
            # "flow_rate_difference": float(with_sulci_results["flow_rate"] - no_sulci_results["flow_rate"]) 
            #    if (with_sulci_results["flow_rate"] is not None and no_sulci_results["flow_rate"] is not None) else None
        }
    }
    
    with open(os.path.join(comparison_dir, "comparison_summary.json"), "w") as f:
        json.dump(comparison_summary, f, indent=4)
    
    print(f"\nNo sulci analysis completed. Results saved in: {output_dir}")
    
    return {
        "no_sulci": no_sulci_results,
        "with_sulci": with_sulci_results,
        "comparison": comparison_summary["comparison"]
    }