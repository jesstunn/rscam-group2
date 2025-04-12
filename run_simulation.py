##########################################################
# Run Simulation Module
##########################################################

# ========================================================
# Imports
# ========================================================

# Import general modules
from dolfin import *
import numpy as np
import os
import time
import sys
from datetime import datetime
import json

# Import project modules
from parameters import Parameters
from mesh import mesh_generator
from stokes import stokes_solver
# from stokes import compute_multiple_flow_rates
 
from adv_diff import advdiff_solver, calculate_average_mass
from plotting import (
    visualise_mesh, 
    visualise_velocity, 
    visualise_concentration
)

# ========================================================
# Run Simulation Functions
# ========================================================

def save_flow_fields(u, p, directory="Results/simulation"):
    """
    Save velocity and pressure fields to files.
    
    Parameters:
    u : dolfin.Function
        Velocity field
    p : dolfin.Function
        Pressure field
    directory : str, optional
        Directory to save the files
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save fields
    File(os.path.join(directory, "velocity.pvd")) << u
    File(os.path.join(directory, "pressure.pvd")) << p
    
    print(f"Flow fields saved to {directory}")


def save_concentration_field(c, directory="Results/simulation"):
    """
    Save concentration field to a file.
    
    Parameters:
    c : dolfin.Function
        Concentration field
    directory : str, optional
        Directory to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save field
    File(os.path.join(directory, "concentration.pvd")) << c
    
    print(f"Concentration field saved to {directory}")


def run_simulation(params=None, output_dir="Results/simulation"):
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
    # Going to comment out all flow rate as we aren't reporting this,
    # but the code is useful I think.
    # try:
    #     x_positions, flow_rates = compute_multiple_flow_rates(u, mesh, num_sections=5)
    #     flow_rate = max(flow_rates) if flow_rates else 0.0
    # except Exception as e:
    #     print("Error: Could not compute flow rate: {}".format(e))
    #     flow_rate = None
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print summary
    print("\nSimulation Results:")
    print("  Average mass: {:.6f}".format(total_mass))
    # if flow_rate is not None:
    #     print("  Flow rate: {:.6f}".format(flow_rate))
    print("  Peclet number: {:.2f}".format(params.Pe))
    print("  Elapsed time: {:.2f} seconds".format(elapsed_time))
    
    # Save summary to file
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "L": float(params.L),
            "H": float(params.H),
            "resolution": params.resolution,
            "sulci_n": params.sulci_n,
            "sulci_h": float(params.sulci_h),
            "sulci_width": float(params.sulci_width),
            "U_ref": float(params.U_ref),
            "D": float(params.D),
            "mu": float(params.mu),
            "Pe": float(params.Pe),
        },
        "mesh": {
            "vertices": mesh.num_vertices(),
            "cells": mesh.num_cells()
        },
        "results": {
            "total_mass": float(total_mass),
            # "flow_rate": float(flow_rate) if flow_rate is not None else None,
            "elapsed_time": elapsed_time
        }
    }
    
    # Save summary to file
    with open(os.path.join(output_dir, "simulation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    return {
        "params": params,
        "mesh": mesh,
        "u": u,
        "p": p,
        "c": c,
        "total_mass": total_mass,
        # "flow_rate": flow_rate,
        "elapsed_time": elapsed_time
    }


def run_all_analyses(base_output_dir="Results"):
    """
    Run all available analyses: single simulation, mass analysis, 
    no sulci analysis, and sulci geometry analysis.
    
    Parameters:
    base_output_dir : str
        Base directory for all results
    
    Returns:
    dict
        Dictionary containing results from all analyses
    """
    print("\n" + "="*60)
    print("RUNNING ALL ANALYSES")
    print("="*60)
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 1. Run a single simulation with default parameters
    print("\n" + "="*50)
    print("1. Running Single Simulation with Default Parameters")
    print("="*50)
    sim_output_dir = os.path.join(base_output_dir, "simulation")
    sim_results = run_simulation(None, sim_output_dir)
    
    # 2. Run mass analysis
    print("\n" + "="*50)
    print("2. Running Mass Analysis Parameter Study")
    print("="*50)
    from mass_analysis import run_mass_analysis
    mass_output_dir = os.path.join(base_output_dir, "mass_analysis")
    mass_results = run_mass_analysis(mass_output_dir)
    
    # 3. Run no sulci analysis
    print("\n" + "="*50)
    print("3. Running No-Sulci Comparison Analysis")
    print("="*50)
    from no_sulci_analysis import run_no_sulci_analysis
    no_sulci_output_dir = os.path.join(base_output_dir, "no_sulci_analysis")
    no_sulci_results = run_no_sulci_analysis(no_sulci_output_dir)
    
    # 4. Run sulci geometry analysis
    print("\n" + "="*50)
    print("4. Running Sulci Geometry Analysis")
    print("="*50)
    from sulci_geometry_analysis import run_sulci_geometry_analysis
    sulci_geom_output_dir = os.path.join(base_output_dir, "sulci_geometry_analysis")
    sulci_geom_results = run_sulci_geometry_analysis(sulci_geom_output_dir)
    
    # Create a summary file
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analyses_run": [
            "single_simulation",
            "mass_analysis",
            "no_sulci_analysis",
            "sulci_geometry_analysis"
        ],
        "output_directories": {
            "single_simulation": sim_output_dir,
            "mass_analysis": mass_output_dir,
            "no_sulci_analysis": no_sulci_output_dir,
            "sulci_geometry_analysis": sulci_geom_output_dir
        }
    }
    
    with open(os.path.join(base_output_dir, "all_analyses_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETED SUCCESSFULLY")
    print("Results saved to:", base_output_dir)
    print("="*60)
    
    return {
        "simulation": sim_results,
        "mass_analysis": mass_results,
        "no_sulci_analysis": no_sulci_results,
        "sulci_geometry_analysis": sulci_geom_results
    }

# ========================================================
# Executable Terminal Script
# ========================================================

# This part makes script executable in the command line of the terminal

if __name__ == "__main__":

    # Imports
    import argparse
    import os
    import sys
        
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run advection-diffusion simulation with sulci")

    # Analysis type arguments (mutually exclusive)
    study_group = parser.add_mutually_exclusive_group()
    study_group.add_argument("--mass_study", action="store_true", help="Run parameter sweep for mass analysis")
    study_group.add_argument("--no_sulci_study", action="store_true", help="Run comparison study of domains with/without sulci")
    study_group.add_argument("--sulci_geometry_study", action="store_true", help="Run study of sulci geometry effects")
    study_group.add_argument("--all", action="store_true", help="Run all analyses")

    # Plot regeneration arguments (mutually exclusive with study_group)
    replot_group = parser.add_mutually_exclusive_group()
    replot_group.add_argument("--replot_simulation", action="store_true", 
                            help="Regenerate plots for single simulation (typically faster to just rerun the simulation)")
    replot_group.add_argument("--replot_mass", action="store_true", 
                            help="Regenerate plots for mass analysis")
    replot_group.add_argument("--replot_no_sulci", action="store_true", 
                            help="Regenerate plots for no-sulci analysis")
    replot_group.add_argument("--replot_sulci_geometry", action="store_true", 
                            help="Regenerate plots for sulci geometry analysis")
    # Other arguments
    parser.add_argument("--json-file", type=str, default=None, help="Path to the JSON file for plot regeneration")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--sulci", type=int, default=None, help="Number of sulci")
    parser.add_argument("--pe", type=float, default=None, help="Peclet number")
    parser.add_argument("--mu", type=float, default=None, help="Uptake parameter (mu)")

    args = parser.parse_args()

    # Default output dir
    base_output_dir = "Results"

    # Handle reploting requests
    if args.replot_simulation:
        import json
        from plotting import visualise_concentration, visualise_velocity, visualise_mesh
        
        # Default JSON file path if not specified
        json_file = args.json_file if args.json_file else os.path.join(base_output_dir, "simulation/simulation_summary.json")
        output_dir = args.output_dir if args.output_dir else os.path.join(base_output_dir, "simulation")
        
        print("Note: For single simulations, it's typically faster to rerun the simulation.")
        print("Regenerating plots requires the original solution fields which may not be available.")
        print("Only plots that can be generated from summary data will be created.")
        
        # Since the actual visualisation needs the solution fields (u, p, c),
        # which aren't saved in the JSON, this would have limited functionality
        # unless we save those fields in a format that can be loaded later.
        
        try:
            with open(json_file, 'r') as f:
                summary = json.load(f)
            
            print(f"Summary data loaded from {json_file}")
            print(f"To fully regenerate plots, please rerun the simulation with:")
            print(f"python run_simulation.py --output-dir={output_dir}")
            
        except Exception as e:
            print(f"Error reading simulation summary: {e}")

    if args.replot_mass:
        import json
        from plotting import plot_mass_results
        
        # Default JSON file path if not specified
        json_file = args.json_file if args.json_file else os.path.join(base_output_dir, "mass_analysis/mass_analysis_summary.json")
        output_dir = args.output_dir if args.output_dir else os.path.join(base_output_dir, "mass_analysis")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"Regenerating mass analysis plots from {json_file}...")
            
            # Generate plots for Pe sweep
            if 'pe_results' in data:
                pe_results = {
                    'parameters': data['pe_results']['parameters'],
                    'total_mass': data['pe_results']['total_mass']
                }
                plot_mass_results(pe_results, 'Pe', output_dir=output_dir)
            
            # Generate plots for mu sweep
            if 'mu_results' in data:
                mu_results = {
                    'parameters': data['mu_results']['parameters'],
                    'total_mass': data['mu_results']['total_mass']
                }
                plot_mass_results(mu_results, 'mu', output_dir=output_dir)
            
            # Generate joint plots
            if 'joint_results' in data:
                joint_results = {
                    'parameters': data['joint_results']['parameters'],
                    'total_mass': data['joint_results']['total_mass']
                }
                plot_mass_results(joint_results, 'Pe', 'mu', log_scale=True, output_dir=output_dir)
            
            print(f"Successfully regenerated mass analysis plots in {output_dir}")
        except Exception as e:
            print(f"Error regenerating mass analysis plots: {e}")
            import traceback
            traceback.print_exc()

    elif args.replot_no_sulci:
        import json
        from plotting import plot_no_sulci_comparison
        
        # Default JSON file path if not specified
        json_file = args.json_file if args.json_file else os.path.join(base_output_dir, "no_sulci_analysis/comparison/comparison_summary.json")
        output_dir = args.output_dir if args.output_dir else os.path.join(base_output_dir, "no_sulci_analysis/comparison")
        
        print("Note: Regenerating no-sulci comparison plots requires the original solution fields.")
        print("This functionality is limited to basic comparisons that can be done from summary data.")
        
        # This would require the actual solution fields to be fully implemented
        # For now, just provide a placeholder/warning

    elif args.replot_sulci_geometry:
        import json
        from plotting import plot_sulci_geometry_comparison
        
        # Default JSON file path if not specified
        json_file = args.json_file if args.json_file else os.path.join(base_output_dir, "sulci_geometry_analysis/comparison/comparison_data.json")
        
        try:
            # Load data from JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract necessary parameters
            geometries = data["geometries"]
            pe_values = data["pe_values"]
            mu_values = data["mu_values"]
            fixed_pe = data["fixed_pe"]
            fixed_mu = data["fixed_mu"]
            results = data["results"]
            
            # Convert string keys to numeric in the results dictionary
            fixed_results = {"varying_pe": {}, "varying_mu": {}}
            
            # Process varying_pe data
            for geom in geometries:
                fixed_results["varying_pe"][geom] = {}
                for pe_str, values in results["varying_pe"][geom].items():
                    # Convert the string key to float or int as needed
                    pe_key = int(float(pe_str)) if float(pe_str).is_integer() else float(pe_str)
                    fixed_results["varying_pe"][geom][pe_key] = values
            
            # Process varying_mu data
            for geom in geometries:
                fixed_results["varying_mu"][geom] = {}
                for mu_str, values in results["varying_mu"][geom].items():
                    # Convert the string key to float or int as needed
                    mu_key = int(float(mu_str)) if float(mu_str).is_integer() else float(mu_str)
                    fixed_results["varying_mu"][geom][mu_key] = values
            
            # Determine output directory
            output_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(json_file))
            
            print(f"Regenerating sulci geometry comparison plots from {json_file}...")
            # Call the plotting function with the loaded data
            plot_sulci_geometry_comparison(fixed_results, output_dir, pe_values, mu_values, fixed_pe, fixed_mu)
            print(f"Successfully regenerated plots in {output_dir}/comparison/")
            
        except Exception as e:
            print(f"Error while regenerating plots: {e}")
            import traceback
            traceback.print_exc()

    # Original analysis runs
    elif args.all:
        # Run all analyses
        run_all_analyses(base_output_dir)
    elif args.mass_study:
        # Run mass parameter study
        from mass_analysis import run_mass_analysis
        output_dir = args.output_dir if args.output_dir else os.path.join(base_output_dir, "mass_analysis")
        run_mass_analysis(output_dir)
    elif args.no_sulci_study:
        # Run no-sulci comparison analysis
        from no_sulci_analysis import run_no_sulci_analysis
        output_dir = args.output_dir if args.output_dir else os.path.join(base_output_dir, "no_sulci_analysis")
        run_no_sulci_analysis(output_dir)
    elif args.sulci_geometry_study:
        # Run sulci geometry study
        from sulci_geometry_analysis import run_sulci_geometry_analysis
        output_dir = args.output_dir if args.output_dir else os.path.join(base_output_dir, "sulci_geometry_analysis")
        run_sulci_geometry_analysis(output_dir)
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
        output_dir = args.output_dir if args.output_dir else os.path.join(base_output_dir, "simulation")
        results = run_simulation(params, output_dir)