# CSF Flow in the SAS - Simulation Framework

> **Warning**: This project should work with GitHub Codespaces using the provided devcontainer.json configuration. However, compatibility issues may occur with the latest VS Code versions. If you encounter any problems, please try using an earlier version of VS Code or run the code locally with the required dependencies installed.

This repository contains a computational simulation framework for studying Cerebrospinal fluid (CSF) flow and solute transporation phenomena in a domain that represents the Subarachnoid Space (SAS), with a focus on studying the effects of sulci (indentations on the lower boundary) on these processes.

## Code Structure

The codebase is organised into several modules with clear responsibilities:

### Core Simulation Modules

These modules contain the fundamental numerical methods and physics models:

- **parameters.py**: Defines the `Parameters` class that stores all simulation parameters, handles validation, and performs non-dimensionalisation.
- **mesh.py**: Contains the mesh generation functionality, including creation of domains with sulci.
- **stokes.py**: Implements the Stokes equations solver for incompressible fluid flow.
- **adv_diff.py**: Implements the advection-diffusion equation solver for concentration fields.

### Analysis Modules

These modules implement specific analysis types:

- **mass_analysis.py**: Performs parameter sweeps to analyse how average mass depends on Péclet number (Pe) and uptake parameter (μ).
- **no_sulci_analysis.py**: Compares simulations with and without sulci to quantify their effects.
- **sulci_geometry_analysis.py**: Studies how different sulci geometries (height and width variations) affect transport under different flow and uptake conditions.

### Visualisation and Runner Modules

- **plotting.py**: Contains all visualisation functions, separated for reusability.
- **run_simulation.py**: The main entry point that ties everything together and provides command-line functionality.

### Results Organisation

All simulation results are stored in a structured way:

```
Results/
  ├── simulation/               # Default single simulation results
  ├── mass_analysis/            # Results from mass parameter sweeps
  ├── no_sulci_analysis/        # Comparison of cases with/without sulci
  └── sulci_geometry_analysis/  # Studies on different sulci geometries
```

Each analysis directory contains:
- Input parameter records
- Solution fields in ParaView format
- Visualisation plots
- Summary JSON files with key metrics

## How to Run Simulations

The framework can be used in various ways depending on your needs.

### Basic Simulation

To run a single simulation with default parameters:

```bash
python run_simulation.py
```

### Custom Parameters

For a  single simulation, you can customise parameters from the command line:

```bash
# Run with custom parameters
python run_simulation.py --sulci 2 --pe 10 --mu 5
```

### Specific Analysis Types

To run specific types of analysis:

```bash
# Run mass parameter study
python run_simulation.py --mass_study

# Run no-sulci comparison
python run_simulation.py --no_sulci_study

# Run sulci geometry study
python run_simulation.py --sulci_geometry_study
```
### Running All Analyses

To run all analyses sequentially (single simulation, mass analysis, no-sulci analysis, and sulci geometry analysis):

```bash
python run_simulation.py --all
```

> **Warning**: Running all analyses may take a considerable amount of time (depending on your hardware), as it involves multiple parameter sweeps and numerous individual simulations. Consider running specific analyses separately if you're only interested in particular results.

### Rerunning Visualisations

You can regenerate plots without rerunning simulations, which is useful when you want to:
- Change plot styles or formatting
- Create new visualisations from existing data
- Export figures with different resolutions or formats

Each analysis module saves its data in JSON format that can be reused. For example:

```bash
# Regenerate sulci geometry comparison plots using the default location
python sulci_geometry_analysis.py --rerun

# Or specify a custom JSON file path
python sulci_geometry_analysis.py --rerun --json-file /path/to/your/comparison_data.json
```

The `--json-file` parameter specifies the path to the JSON file containing previously calculated results. Each analysis saves these files automatically in their respective results directories (e.g., `Results/sulci_geometry_analysis/comparison/comparison_data.json`).

> **Tip**: If you don't specify a JSON file, the script will look for data in the default location.

## Key Parameters

- **Pe (Péclet number)**: Ratio of advective to diffusive transport rates
- **μ (Mu)**: Uptake parameter for Robin boundary condition
- **sulci_n**: Number of sulci on the bottom boundary
- **sulci_h_mm**: Height of sulci in mm
- **sulci_width_mm**: Width of sulci in mm

## Requirements

- FEniCS (with DOLFIN and mshr modules)
- NumPy
- Matplotlib
- Python 3.6+

## Example Usage

Here's a complete example of running a custom simulation followed by analysis:

```python
from parameters import Parameters
from run_simulation import run_simulation, run_all_analyses

# Create custom parameters
params = Parameters()
params.sulci_n = 3            # Three sulci
params.sulci_h_mm = 1.5       # 1.5mm high
params.sulci_width_mm = 0.8   # 0.8mm wide
params.U_ref = 0.02           # Fluid velocity reference (mm/s)
params.mu = 5                 # Uptake parameter

# Run single simulation
results = run_simulation(params, "Results/my_custom_simulation")

# Run all analyses with default settings
all_results = run_all_analyses()

# Access results
print(f"Total mass: {results['total_mass']}")
print(f"Flow rate: {results['flow_rate']}")
```

## Extending the Framework

To add new analysis types:

1. Create a new module based on existing analysis patterns
2. Add plotting functions to the plotting.py module
3. Add a runner function in run_simulation.py
