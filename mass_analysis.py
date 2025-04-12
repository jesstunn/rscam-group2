##########################################################
# Mass Analysis Module
##########################################################

# ========================================================
# Imports
# ========================================================

# Import general modules
from dolfin import *
import numpy as np
import itertools
import os
from copy import deepcopy

# Import project modules
from parameters import Parameters
from mesh import mesh_generator
from stokes import stokes_solver
from adv_diff import advdiff_solver, calculate_average_mass
from plotting import plot_mass_results

# ========================================================
# Mass Parameter Sweep Functions
# ========================================================

def parameter_sweep_mass(params_range, fixed_params=None, save_fields=True, output_dir="Results/mass_analysis"):
    """
    Perform a parameter sweep to compute average mass for different parameter values.
    
    Parameters:
    params_range : dict
        Dictionary with parameter names as keys and lists of values to sweep over.
        E.g., {'Pe': [1, 10, 100], 'mu': [0, 0.1, 1, 10]}
    fixed_params : dict, optional
        Dictionary with fixed parameter values for parameters not in params_range.
    save_fields : bool, optional
        Whether to save velocity, pressure, and concentration fields for ParaView
    output_dir : str, optional
        Directory to save results
        
    Returns:
    dict
        Dictionary with parameter combinations and corresponding average mass values
    """
    if fixed_params is None:
        fixed_params = {}
    
    # Store results
    results = {'parameters': [], 'total_mass': []}
    
    # Create parameter grid
    param_names = list(params_range.keys())
    param_values = list(params_range.values())
    
    # Create directories for ParaView files if needed
    if save_fields:
        paraview_dir = os.path.join(output_dir, 'paraview')
        os.makedirs(paraview_dir, exist_ok=True)
    
    # For all parameter combinations
    for values in itertools.product(*param_values):

        # Create parameter set for this run
        current_params = Parameters()
        
        # Set fixed parameters
        for param, value in fixed_params.items():
            setattr(current_params, param, value)
            
        # Set swept parameters
        for param, value in zip(param_names, values):
            if param == 'Pe':
                # Pe affects D in the nondim process
                current_params.U_ref = value * current_params.D_mms2 / current_params.L_ref
            else:
                setattr(current_params, param, value)
        
        # Re-validate and non-dimensionalise
        current_params.validate()
        current_params.nondim()
        
        # Generate mesh
        mesh_data = mesh_generator(
            current_params.resolution,
            current_params.L, current_params.H,  
            current_params.nx, current_params.ny, 
            current_params.sulci_n, current_params.sulci_h, current_params.sulci_width
        )
        
        # Extract mesh and boundary markers from result dictionary
        if isinstance(mesh_data, dict):
            mesh = mesh_data["mesh"]
            left = mesh_data["boundary_markers"]["left"]
            right = mesh_data["boundary_markers"]["right"]
            bottom = mesh_data["boundary_markers"]["bottom"]
            top = mesh_data["boundary_markers"]["top"]
        else:
            # Fallback for older mesh_generator that returns a tuple
            mesh, left, right, bottom, top = mesh_data
        
        # Build function spaces
        V = VectorFunctionSpace(mesh, "P", 2)  
        Q = FunctionSpace(mesh, "P", 1)  
        W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
        C = FunctionSpace(mesh, "CG", 1)
        
        # Solve Stokes
        u, p = stokes_solver(mesh, W, current_params.H, left, right, bottom, top)
        
        # Solve Advection-Diffusion
        c = advdiff_solver(
            mesh, left, right, bottom,
            u, C,
            D=Constant(current_params.D),
            mu=Constant(current_params.mu)
        )
        
        # Calculate average mass
        total_mass = calculate_average_mass(c, mesh)
        
        # Store parameters and results
        param_dict = {param: value for param, value in zip(param_names, values)}
        results['parameters'].append(param_dict)
        results['total_mass'].append(total_mass)
        
        # Save fields for ParaView if requested
        if save_fields:
            # Create parameter-specific directory name
            param_str = '_'.join([f"{k}_{v}" for k, v in param_dict.items()])
            param_dir = os.path.join(paraview_dir, param_str)
            os.makedirs(param_dir, exist_ok=True)
            
            # Save fields
            File(os.path.join(param_dir, "velocity.pvd")) << u
            File(os.path.join(param_dir, "pressure.pvd")) << p
            File(os.path.join(param_dir, "concentration.pvd")) << c
        
        # Print progress
        print(f"Parameters: {param_dict}, Average mass: {total_mass}")
    
    return results

def run_mass_analysis(output_dir="Results/mass_analysis"):
    """
    Run a comprehensive analysis of how average mass depends on Pe and mu.
    
    Parameters:
    output_dir : str
        Directory to save results
        
    Returns:
    dict
        Dictionary containing results for all parameter sweeps
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Single parameter sweeps
    
    # a. Effect of Pe with mu=0 (no uptake)
    print("Running Pe sweep with mu=0...")
    pe_results = parameter_sweep_mass(
        params_range={'Pe': [0.1, 1, 5, 10, 20, 50, 100, 200, 500]},
        fixed_params={'mu': 0.0},
        output_dir=output_dir
    )
    plot_mass_results(pe_results, 'Pe', output_dir=output_dir)
    
    # b. Effect of mu with Pe=0 (no flow)
    print("Running mu sweep with Pe=0...")
    mu_results = parameter_sweep_mass(
        params_range={'mu': [0, 0.1, 0.5, 1, 2, 5, 10, 20]},
        fixed_params={'Pe': 0},
        output_dir=output_dir
    )
    plot_mass_results(mu_results, 'mu', output_dir=output_dir)
    
    # 2. Two-parameter sweep
    print("Running Pe-mu parameter sweep...")
    joint_results = parameter_sweep_mass(
        params_range={
            'Pe': [0.1, 1, 10, 100, 1000],
            'mu': [0, 0.1, 1, 10, 100]
        },
        output_dir=output_dir
    )
    plot_mass_results(joint_results, 'Pe', 'mu', log_scale=True, output_dir=output_dir)
    
    # 3. Special case: mu >> Pe
    print("Analysing case where mu >> Pe...")
    high_mu_results = parameter_sweep_mass(
        params_range={
            'Pe': [0.1, 1, 10],
            'mu': [100, 1000]
        },
        output_dir=output_dir
    )
    plot_mass_results(high_mu_results, 'Pe', 'mu', log_scale=True, output_dir=output_dir)
    
    # 4. Special case: mu << Pe
    print("Analysing case where mu << Pe...")
    high_pe_results = parameter_sweep_mass(
        params_range={
            'Pe': [100, 1000],
            'mu': [0.01, 0.1, 1]
        },
        output_dir=output_dir
    )
    plot_mass_results(high_pe_results, 'Pe', 'mu', log_scale=True, output_dir=output_dir)
    
    # Save a summary of all results to a json file
    import json
    summary = {
        'pe_results': {
            'parameters': pe_results['parameters'],
            'total_mass': [float(m) for m in pe_results['total_mass']]
        },
        'mu_results': {
            'parameters': mu_results['parameters'],
            'total_mass': [float(m) for m in mu_results['total_mass']]
        },
        'joint_results': {
            'parameters': joint_results['parameters'],
            'total_mass': [float(m) for m in joint_results['total_mass']]
        },
        'high_mu_results': {
            'parameters': high_mu_results['parameters'],
            'total_mass': [float(m) for m in high_mu_results['total_mass']]
        },
        'high_pe_results': {
            'parameters': high_pe_results['parameters'],
            'total_mass': [float(m) for m in high_pe_results['total_mass']]
        }
    }
    
    with open(os.path.join(output_dir, 'mass_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Mass analysis completed. Results saved to {output_dir}")
    
    return {
        'pe_results': pe_results,
        'mu_results': mu_results,
        'joint_results': joint_results,
        'high_mu_results': high_mu_results,
        'high_pe_results': high_pe_results
    }