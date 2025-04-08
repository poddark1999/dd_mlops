import os
import sys
import pandas as pd

# Attempt to import PyTorch for GPU detection
try:
    import torch
except ImportError:
    torch = None

def is_gpu_available():
    """
    Checks if any GPU is available to PyTorch, including CUDA, ROCm, or MPS.
    Returns a tuple: (bool: whether a GPU is found, str: GPU description or message).
    """
    if torch is None:
        return False, "PyTorch not installed; cannot detect GPU."
    
    # If CUDA/ROCm GPUs are recognized
    if torch.cuda.device_count() > 0:
        return True, f"GPU found: {torch.cuda.get_device_name(0)}"
    
    # If Apple Silicon GPU via MPS is available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True, "GPU found: Apple MPS (Metal)"
    
    return False, "No GPU detected."


if __name__ == "__main__":
    # --- GPU detection ---
    gpu_found, gpu_info = is_gpu_available()
    if not gpu_found:
        print(f"{gpu_info} This script cannot be run without a GPU. Exiting.")
        sys.exit(1)
    else:
        print(f"{gpu_info}\n")

    print("#########################################################")
    print("Welcome to the Dynamic Discounting model evaluation app!")
    print("#########################################################\n")
    print("\nPreReqs")
    print("_________________________________________________________\n")
    print("This app can help you run and evaluate the different forecasting models you have built.")
    print("1. Models folder:\n   - Make sure you have the model in whatever format stored as a folder here.")
    print("2. evalutation_scripts folder:\n   - For each model, make sure its relevant model evaluation script is stored as a .py file here.")
    print("\nMake sure both the model folder and its evaluation function work together.")
    print("_________________________________________________________\n")
    
    # Prepare model list for display
    path = "evaluation_scripts"
    dir_list = os.listdir(path)
    compare_script_file_name = "compare_evals.py"
    
    # Identify evaluation scripts
    models = dir_list.copy()
    
    # Dynamically build metadata to link model scripts with their output CSV files
    models_metadata = {}
    for model_script in models:
        if model_script.endswith('.py') and model_script != compare_script_file_name:
            model_name = model_script.replace('.py', '')
            result_path = f"model_evaluation_results/{model_name}/forecast_evaluation_details.csv"
            models_metadata[model_script] = result_path
    
    # Remove the comparison script from the model list (it's not a model script)
    if compare_script_file_name in models:
        models.remove(compare_script_file_name)

    # Display available scripts
    print(f"\nFound the following evaluation scripts in '{path}':")
    formatted_models = [model.replace('_', ' ').replace('.py', '') for model in models]
    print("- " + ", ".join(formatted_models))
    print(f"{len(models)} Model scripts found\n")
    
    # Build a dictionary to hold the scripts for the menu
    scripts_dict = {}
    invalid_models = []

    def csv_valid(csv_path):
        """Checks if the CSV file exists, is not empty, and is loadable."""
        return (
            os.path.exists(csv_path)
            and os.path.getsize(csv_path) > 0
            and not pd.read_csv(csv_path).empty
        )

    # Populate scripts_dict and identify invalid CSVs
    for i, model_script in enumerate(models):
        scripts_dict[i + 1] = model_script

        if model_script in models_metadata:
            result_path = models_metadata[model_script]
            if not csv_valid(result_path):
                invalid_models.append(model_script)

    # If all CSVs are valid, allow comparison
    comparaison_available = False
    if not invalid_models:  # Means everything is valid
        comparaison_available = True
        scripts_dict[len(scripts_dict) + 1] = compare_script_file_name

    # Main loop for user interaction
    while True:
        print("Please choose an action by typing the corresponding number:")
        print("0: Exit the app")
        
        for index, script in scripts_dict.items():
            # Clean up the script name for printing
            pretty_name = script.replace('_', ' ').replace('.py', '')
            if script == compare_script_file_name:
                print(f"\n{index}: Compare models using {pretty_name}\n")
            else:
                print(f"{index}: Run {pretty_name}")
        
        if not comparaison_available and invalid_models:
            print("\n⚠️ Cannot compare models yet because some evaluations are missing:")
            for model in invalid_models:
                print(f"   - {model.replace('_', ' ').replace('.py', '')} needs to be run first.")
        
        # Get user choice
        try:
            inp = int(input("Choose your action: "))
        except ValueError:
            print("\n🛑 Input must be an integer. Please try again.\n")
            continue
        
        if inp == 0:  # Exit
            print("Exiting.")
            break
        elif inp in scripts_dict:
            selected_script = scripts_dict[inp]
            script_path = os.path.join(path, selected_script)
            
            print(f"Running: {selected_script.replace('_', ' ').replace('.py', '')}")
            try:
                with open(script_path, 'r') as f:
                    script_content = f.read()
                    exec(script_content)
            except Exception as e:
                print(f"Error running {selected_script}: {e}")
            
            print("\nExecution completed.")
            again = input("Do you want to run another script? (y/n): ").strip().lower()
            if again != 'y':
                print("Exiting the app.")
                break
        else:
            print("Invalid option. Please try again.")
            continue
