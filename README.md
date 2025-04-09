# Advection-Diffusion Simulation with Sulci

This project simulates advection-diffusion in a channel with optional sulci on the bottom boundary. The code is designed to study how flow and solute transport are affected by various parameters including Peclet number, uptake rate, and sulci geometry.

## Project Structure

The repository is organised into the following structure:

- Main simulation files:
  - `parameters.py`: Defines simulation base parameters
  - `mesh.py`: Handles mesh generation and visualisation
  - `stokes.py`: Solves the Stokes equations for fluid flow
  - `adv_diff.py`: Solves the advection-diffusion equation for concentration
  - `mass_analysis.py`: Analyses how mass depends on Peclet number and uptake
  - `sulci_analysis.py`: Studies the effect of differing sulci geometry
  - `run_simulation.py`: Main script to run simulations and parameter studies

- `development/`: Contains preliminary work and test files used during the development process
  - Draft versions of simulation components
  - Test problems with simplified geometries
  - Experimental implementations
  - Alternative approaches that were explored

## Prerequisites

- Python 3.6 or higher
- FEniCS (with DOLFIN)
- NumPy
- Matplotlib

## Basic Usage

### Running a Single Simulation

To run a single simulation with default parameters:

```bash
python3 run_simulation.py
```

This will save results in the `simulation_results` directory.

### Customising Parameters

You can specify parameters directly from the command line:

```bash
python3 run_simulation.py --sulci 2 --pe 10 --mu 5
```

This runs a simulation with:
- 2 sulci
- Peclet number of 10
- Uptake parameter of 5

### Specifying Output Directory

Change the output directory using `--output-dir`:

```bash
python3 run_simulation.py --output-dir custom_results
```

## Parameter Studies

### Mass Analysis Study

To run a comprehensive analysis of how average mass depends on Peclet number and uptake parameter:

```bash
python3 run_simulation.py --mass_study
```

This will:
1. Run simulations for various Pe and mu combinations
2. Generate plots showing how mass varies with each parameter
3. Save results in `mass_study_results` directory

### Sulci Geometry Study

To analyse how sulci geometry (height and width) affects flow and concentration:

```bash
python3 run_simulation.py --sulci_study
```

This will:
1. Run simulations for four geometry cases:
   - Small height, small width
   - Small height, large width
   - Large height, small width
   - Large height, large width
2. Generate comparison plots
3. Save results in `sulci_study_results` directory

## Visualisation

The simulation results can be visualised in several ways:

1. **PNG Images**: Automatically generated in the directory of results
   - Mesh visualisation
   - Velocity field
   - Concentration field

2. **ParaView Files**: Saved in the results directory
   - `velocity.pvd`: Velocity field for ParaView
   - `pressure.pvd`: Pressure field for ParaView
   - `concentration.pvd`: Concentration field for ParaView

## Output Data

Each simulation saves:
- Solution fields (velocity, pressure, concentration)
- Visual plots
- A `simulation_summary.json` file with parameters and key results

Parameter studies generate additional plots and comparison data.

## Examples

```bash
# Basic simulation with default parameters
python3 run_simulation.py

# Simulation with custom parameters
python3 run_simulation.py --sulci 3 --pe 20 --mu 2

# Mass parameter study with custom output directory
python3 run_simulation.py --mass_study --output-dir mass_analysis_june

# Sulci geometry study with custom output directory
python3 run_simulation.py --sulci_study --output-dir sulci_study_june
```

## Extending the Code

To modify or extend this code:

1. **Add New Parameters**: Edit the `Parameters` class in `parameters.py`
2. **Change Boundary Conditions**: Modify the appropriate solver function
3. **Add New Analysis Types**: Create a new analysis module following the pattern of `mass_analysis.py` or `sulci_analysis.py`
