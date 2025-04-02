# ========================================================
# Mass Analysis Module
# ========================================================

# Import general moduels
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D  

# Import our modules
from parameters import Parameters
from mesh import mesh_generator
from stokes import stokes_solver
from adv_diff import advdiff_solver, calculate_average_mass
   
def parameter_sweep_mass(params_range, fixed_params=None, save_fields=True):
    """
    Perform a parameter sweep to compute average mass for different parameter values.
    
    Parameters:
    -----------
    params_range : dict
        Dictionary with parameter names as keys and lists of values to sweep over.
        E.g., {'Pe': [1, 10, 100], 'mu': [0, 0.1, 1, 10]}
    fixed_params : dict, optional
        Dictionary with fixed parameter values for parameters not in params_range.
    save_fields : bool, optional
        Whether to save velocity, pressure, and concentration fields for ParaView
        
    Returns:
    --------
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
        paraview_dir = 'mass_results/paraview'
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

def plot_mass_results(results, x_param, y_param=None, log_scale=True):
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
    """
    # Set font sizes for the plots
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })
    
    # Create plots directory if it doesn't exist
    os.makedirs('mass_results', exist_ok=True)
    
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
        plt.savefig(f'mass_results/mass_vs_{x_param}.png', dpi=300)
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
        plt.savefig(f'mass_results/mass_heatmap_{x_param}_{y_param}.png', dpi=300)
        plt.close()

def run_mass_analysis():
    """
    Run a comprehensive analysis of how average mass depends on Pe and mu.
    """
    # 1. Single parameter sweeps
    
    # a. Effect of Pe with mu=0 (no uptake)
    print("Running Pe sweep with mu=0...")
    pe_results = parameter_sweep_mass(
        params_range={'Pe': [0.1, 1, 5, 10, 20, 50, 100, 200, 500]},
        fixed_params={'mu': 0.0}
    )
    plot_mass_results(pe_results, 'Pe')
    
    # b. Effect of mu with Pe=0 (no flow)
    print("Running mu sweep with Pe=0...")
    mu_results = parameter_sweep_mass(
        params_range={'mu': [0, 0.1, 0.5, 1, 2, 5, 10, 20]},
        fixed_params={'Pe': 0}
    )
    plot_mass_results(mu_results, 'mu')
    
    # 2. Two-parameter sweep
    print("Running Pe-mu parameter sweep...")
    joint_results = parameter_sweep_mass(
        params_range={
            'Pe': [0.1, 1, 10, 100, 1000],
            'mu': [0, 0.1, 1, 10, 100]
        }
    )
    plot_mass_results(joint_results, 'Pe', 'mu', log_scale=True)
    
    # 3. Special case: mu >> Pe
    print("Analysing case where mu >> Pe...")
    high_mu_results = parameter_sweep_mass(
        params_range={
            'Pe': [0.1, 1, 10],
            'mu': [100, 1000]
        }
    )
    plot_mass_results(high_mu_results, 'Pe', 'mu', log_scale=True)
    
    # 4. Special case: mu << Pe
    print("Analysing case where mu << Pe...")
    high_pe_results = parameter_sweep_mass(
        params_range={
            'Pe': [100, 1000],
            'mu': [0.01, 0.1, 1]
        }
    )
    plot_mass_results(high_pe_results, 'Pe', 'mu', log_scale=True)
    
    return {
        'pe_results': pe_results,
        'mu_results': mu_results,
        'joint_results': joint_results,
        'high_mu_results': high_mu_results,
        'high_pe_results': high_pe_results
    }
