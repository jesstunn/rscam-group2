# rerun_sulci_comparison.py

import os
import json
import matplotlib.pyplot as plt
from sulci_analysis import compare_results_dual

def rerun_sulci_comparison(json_file="comparison_data_dual.json", output_dir=None):
    """
    Rerun the compare_results_dual function using data from a previously saved JSON file.
    
    Parameters:
    json_file : str
        Path to the JSON file containing the results data
    output_dir : str, optional
        Directory to save the plots (defaults to directory containing the JSON file)
    """
    # Check if the JSON file exists
    if not os.path.exists(json_file):
        print(f"Error: Could not find JSON file at {json_file}")
        return False
    
    try:
        # Load the saved data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract the necessary parameters
        geometries = data["geometries"]
        pe_values = data["pe_values"]
        mu_values = data["mu_values"]
        fixed_pe = data["fixed_pe"]
        fixed_mu = data["fixed_mu"]
        results = data["results"]
        
        print(f"Loaded data with:")
        print(f"  Geometries: {geometries}")
        print(f"  Pe values: {pe_values}")
        print(f"  Mu values: {mu_values}")
        print(f"  Fixed Pe: {fixed_pe}")
        print(f"  Fixed Mu: {fixed_mu}")
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(json_file))
            if output_dir == "":
                output_dir = "."
        
        # Call the existing function with the loaded data
        print(f"Rerunning compare_results_dual function...")
        compare_results_dual(results, output_dir, pe_values, mu_values, fixed_pe, fixed_mu)
        
        print(f"Successfully regenerated plots in {output_dir}/comparison/")
        return True
        
    except Exception as e:
        print(f"Error while rerunning comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rerun compare_results_dual with saved JSON data")
    parser.add_argument("--json-file", type=str, default="comparison_data_dual.json", 
                      help="Path to the JSON file containing results")
    parser.add_argument("--output-dir", type=str, default=None, 
                      help="Directory to save the new plots (defaults to JSON file's directory)")
    
    args = parser.parse_args()
    
    rerun_sulci_comparison(args.json_file, args.output_dir)


# Use this in terminal to run:
# python rerun_sulci_comparison.py --json-file=sulci_study_results/comparison/comparison_data_dual.json --output-dir=new_plots