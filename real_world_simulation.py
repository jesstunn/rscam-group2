# ======================================================
# Run SAS Simulations with Real-World Parameters
# ======================================================

import os
import numpy as np
from parameters import Parameters
from run_simulation import run_simulation, run_mass_parameter_study, run_sulci_parameter_study

def run_real_sas_simulation():
    """
    Run simulation and all analysis with real world parameters.
    """
    
    # Create base output directory
    base_dir = "real_world_results"  # Changed to "real_world_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # ------------------------------------------------
    # Run simulation with base parameters
    # ------------------------------------------------
    # Create parameters with default values
    params = Parameters()
    
    # Set real world parameters
    params.U_ref = 50.0 # 5 cm/s in mm/s
    params.D_mms2 = 0.0003  # 3×10-10 m2/s converted to mm2/s
    
    # Non-dimensionalise
    params.validate()
    params.nondim()
    
    # Run simulation
    output_dir = os.path.join(base_dir, "base_simulation")
    print(f"\nRunning simulation for biologically relevant parameters")
    results = run_simulation(params, output_dir)
    
    # Report key results
    print(f"  Average mass: {results['total_mass']:.6f}")
    print(f"  Peclet number: {params.Pe:.2f}")
    print(f"  Note: High Pe number ({params.Pe:.0f}) indicates flow-dominated transport")
    
    # ------------------------------------------------
    # Run the mass parameter study
    # ------------------------------------------------
    print("\n" + "="*50)
    print("Running mass parameter study")
    print("="*50)
    
    mass_study_dir = os.path.join(base_dir, "mass_study")
    mass_results = run_mass_parameter_study(mass_study_dir)
    
    # ------------------------------------------------
    # Run the sulci parameter study
    # ------------------------------------------------
    print("\n" + "="*50)
    print("Running sulci geometry parameter study")
    print("="*50)
    
    sulci_study_dir = os.path.join(base_dir, "sulci_study")
    sulci_results = run_sulci_parameter_study(sulci_study_dir)
    
    print("\n" + "="*50)
    print("All simulations completed successfully!")
    print("="*50)
    print(f"Results saved in: {base_dir}")

if __name__ == "__main__":
    run_real_sas_simulation()
