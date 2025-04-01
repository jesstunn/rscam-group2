# ========================================================
# Module for Replotting Existing Results
# ========================================================

import os
import json
import matplotlib.pyplot as plt
import sys

# Import the compare_results_dual function from sulci_analysis
from sulci_analysis import compare_results_dual

def replot_sulci_results(output_dir="sulci_results"):
    """
    Replot the comparison charts for existing sulci analysis results
    without re-running the simulations.
    
    Parameters:
    output_dir : str
        Directory where the original results are stored
    """
    # Path to the comparison data file
    data_file = os.path.join(output_dir, "comparison", "comparison_data_dual.json")
    
    # Check if the file exists
    if not os.path.exists(data_file):
        print(f"Error: Could not find results data at {data_file}")
        print("Please specify the correct output directory using --output-dir.")
        return False
    
    try:
        # Load the existing results
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Extract necessary parameters for replotting
        results = data['results']
        pe_values = data['pe_values']
        mu_values = data['mu_values']
        fixed_pe = data['fixed_pe']
        fixed_mu = data['fixed_mu']
        
        print(f"Found existing results with:")
        print(f"  Pe values: {pe_values}")
        print(f"  Mu values: {mu_values}")
        print(f"  Fixed Pe: {fixed_pe}")
        print(f"  Fixed Mu: {fixed_mu}")
        
        # Run only the plotting function
        print(f"Regenerating plots in {output_dir}/comparison/...")
        compare_results_dual(results, output_dir, pe_values, mu_values, fixed_pe, fixed_mu)
        
        print("Plots successfully regenerated!")
        return True
        
    except Exception as e:
        print(f"Error while replotting results: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Replot existing sulci analysis results")
    parser.add_argument("--output-dir", type=str, default="sulci_results", 
                      help="Directory containing original simulation results")
    
    args = parser.parse_args()
    
    replot_sulci_results(args.output_dir)

# Run this in terminal to replot:
# python replot_results.py --output-dir=sulci_study_results