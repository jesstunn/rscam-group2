##########################################################
# Varying Sulci Geometry Analysis Module
##########################################################

# ========================================================
# Imports
# ========================================================

# Import general modules
import os
import json
import time
from datetime import datetime

# Import project modules
from parameters import Parameters
from plotting import plot_sulci_geometry_comparison

# ========================================================
# Varying Sulci Geometry Functions
# ========================================================

def run_single_sulci_simulation(params, output_dir):
    """
    Run a single simulation with given sulci parameters.
    
    Parameters:
    params : Parameters
        Parameter object with configured parameters
    output_dir : str
        Directory to save results
        
    Returns:
    dict
        Dictionary with simulation results
    """
    # Import here to avoid circular imports
    from run_simulation import run_simulation
    
    # Run simulation and return results
    return run_simulation(params, output_dir)

def run_sulci_geometry_analysis(output_dir="Results/sulci_geometry_analysis"):
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
                "total_mass": float(case_results["total_mass"])
                # "flow_rate": float(case_results["flow_rate"]) if case_results["flow_rate"] is not None else None
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
                "total_mass": float(case_results["total_mass"])
                # "flow_rate": float(case_results["flow_rate"]) if case_results["flow_rate"] is not None else None
            }
    
    # Generate enhanced comparison charts for both parameter variations
    plot_sulci_geometry_comparison(all_results, output_dir, pe_values, mu_values, fixed_pe, fixed_mu)
    
    # Save comparison data to JSON
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    comparison_data = {
        "geometries": [geom["name"] for geom in geometries],
        "pe_values": pe_values,
        "mu_values": mu_values,
        "fixed_pe": fixed_pe,
        "fixed_mu": fixed_mu,
        "results": all_results
    }
    
    with open(os.path.join(comparison_dir, "comparison_data.json"), "w") as f:
        json.dump(comparison_data, f, indent=4)
    
    print(f"Sulci geometry analysis completed. Results saved to {output_dir}")
    
    return all_results