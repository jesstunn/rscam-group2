# ========================================================
# Run Simulation Module
# ========================================================


# Import general modules
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from datetime import datetime

# Import our modules
from parameters import Parameters
from mesh import mesh_generator, visualise_mesh
from stokes import stokes_solver, compute_multiple_flow_rates, visualise_velocity, save_flow_fields
from adv_diff import advdiff_solver, calculate_total_mass, calculate_average_mass, visualise_concentration, save_concentration_field
from mass_analysis import run_mass_analysis
from sulci_analysis import run_sulci_analysis 

# ------------------------------------------

# Set global plot settings
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
})

# --------------------------------------------------------

def run_simulation(params=None, output_dir="simulation_results"):
    """
    Run a complete simulation with the given parameters.
    
    Parameters:
    params : Parameters, optional
        Parameter object, uses default if None
    output_dir : str
        Directory where results will be saved
    
    Returns:
    results : dict
        Dictionary containing simulation results
    """
    # Start timing
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visuals directory if it doesn't exist
    visuals_dir = os.path.join(output_dir, "plots")
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Use default parameters if none provided
    if params is None:
        params = Parameters()
    
    # Print parameters
    print("\nSimulation Parameters:")
    print(params)
    
    # ----------------------------------
    # 1) Generate mesh
    # ----------------------------------
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
        # For backwards compatibility with older mesh_generator
        mesh, left, right, bottom, top = mesh_result
        mesh_info = {"num_vertices": mesh.num_vertices(), "num_cells": mesh.num_cells()}
    
    # Save mesh visualisation
    try:
        visualise_mesh(mesh_result, os.path.join(visuals_dir, "mesh.png"))
    except Exception as e:
        print("Error: Could not save mesh visualisation: {}".format(e))
    
    # ----------------------------------
    # 2) Build function spaces
    # ----------------------------------
    print("2. Building function spaces...")
    V = VectorFunctionSpace(mesh, "P", 2)  
    Q = FunctionSpace(mesh, "P", 1)  
    W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
    C = FunctionSpace(mesh, "CG", 1)
    
    # ----------------------------------
    # 3) Solve Stokes
    # ----------------------------------
    print("3. Solving Stokes equations...")
    u, p = stokes_solver(mesh, W, params.H, left, right, bottom, top)
    
    # Save flow visualisation
    try:
        visualise_velocity(u, mesh, os.path.join(visuals_dir, "velocity.png"))
        save_flow_fields(u, p, directory=output_dir)
    except Exception as e:
        print("Error: Could not save flow fields: {}".format(e))
    
    # ----------------------------------
    # 4) Solve Advection-Diffusion
    # ----------------------------------
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
        print("Error: Could not save concentration field: {}".format(e))
    
    # ----------------------------------
    # 5) Postprocess
    # ----------------------------------
    print("5. Post-processing results...")
    
    # Calculate mass
    total_mass = calculate_average_mass(c, mesh)
    
    # Calculate flow rate
    try:
        x_positions, flow_rates = compute_multiple_flow_rates(u, mesh, num_sections=5)
        flow_rate = max(flow_rates) if flow_rates else 0.0
    except Exception as e:
        print("Error: Could not compute flow rate: {}".format(e))
        flow_rate = None
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print summary
    print("\nSimulation Results:")
    print("  Total mass: {:.6f}".format(total_mass))
    if flow_rate is not None:
        print("  Flow rate: {:.6f}".format(flow_rate))
    print("  Peclet number: {:.2f}".format(params.Pe))
    print("  Elapsed time: {:.2f} seconds".format(elapsed_time))
    
    # Save summary to file
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "L": params.L,
            "H": params.H,
            "resolution": params.resolution,
            "sulci_n": params.sulci_n,
            "sulci_h": params.sulci_h,
            "sulci_width": params.sulci_width,
            "U_ref": params.U_ref,
            "D": params.D,
            "mu": params.mu,
            "Pe": params.Pe,
        },
        "mesh": {
            "vertices": mesh.num_vertices(),
            "cells": mesh.num_cells()
        },
        "results": {
            "total_mass": float(total_mass),
            "flow_rate": float(flow_rate) if flow_rate is not None else None,
            "elapsed_time": elapsed_time
        }
    }
    
    # Save summary to file
    import json
    with open(os.path.join(output_dir, "simulation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
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

def run_mass_parameter_study(output_dir="mass_study_results"):
    """Run a mass analysis parameter study"""
    print("\nRunning mass parameter study...")
    
    # This will run a series of simulations to analyse how mass depends on Pe and mu
    results = run_mass_analysis()
    
    print("Mass parameter study completed!")
    return results

def run_sulci_parameter_study(output_dir="sulci_study_results"):
    """Run a study on the effect of sulci geometry with varying Pe and mu"""
    print("\nRunning sulci geometry parameter study with varying Pe and mu...")
    
    # This will run a series of simulations to analyse how shape of the sulci
    # affects the solution under different Pe and mu conditions
    results = run_sulci_analysis(output_dir)
    
    print("Sulci geometry parameter study completed!")
    print("Created visualisations showing the effect of geometry for:")
    print("  - Different Peclet numbers (Pe = 1, 10, 100) with fixed μ = 1.0")
    print("  - Different uptake parameters (μ = 0.1, 1.0, 10.0) with fixed Pe = 10")
    print(f"Results saved in: {output_dir}/comparison/")
    
    return results

# This part makes the script executable from the command line:
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run advection-diffusion simulation with sulci")
    
    # Study type arguments (mutually exclusive)
    study_group = parser.add_mutually_exclusive_group()
    study_group.add_argument("--mass_study", action="store_true", help="Run parameter study for mass analysis")
    study_group.add_argument("--sulci_study", action="store_true", help="Run study of sulci geometry effects")
    
    # Other arguments
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--sulci", type=int, default=None, help="Number of sulci")
    parser.add_argument("--pe", type=float, default=None, help="Peclet number")
    parser.add_argument("--mu", type=float, default=None, help="Uptake parameter (mu)")
    
    args = parser.parse_args()
    
    if args.mass_study:
        # Run mass parameter study
        output_dir = args.output_dir if args.output_dir else "mass_study_results"
        run_mass_parameter_study(output_dir)
    elif args.sulci_study:
        # Run sulci geometry study
        output_dir = args.output_dir if args.output_dir else "sulci_study_results"
        run_sulci_parameter_study(output_dir)  
    else:
        # Run single simulation
        params = Parameters()
        
        # Override parameters if provided via command line
        if args.sulci is not None:
            params.sulci_n = args.sulci
        
        if args.pe is not None:
            # Update U_ref to achieve the desired Pe
            params.U_ref = args.pe * params.D_mms2 / params.H_mm
            params.nondim()  # Recalculate derived parameters
        
        if args.mu is not None:
            params.mu = args.mu
            params.nondim()  # Recalculate derived parameters
        
        # Run simulation
        output_dir = args.output_dir if args.output_dir else "simulation_results"
        results = run_simulation(params, output_dir)
