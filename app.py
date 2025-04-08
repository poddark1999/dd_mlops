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
    Check if a GPU is available through any supported PyTorch backend (CUDA, ROCm, MPS).
    Returns (bool, str): (gpu_detected, message).
    """
    if torch is not None:
        # Check for CUDA or ROCm devices
        if torch.cuda.device_count() > 0:
            return True, f"GPU detected: {torch.cuda.get_device_name(0)}"
        # Check for Apple Silicon GPU via MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "GPU detected: Apple MPS (Metal)"
    return False, "No GPU detected."

if __name__ == "__main__":
    # Detect GPU
    gpu_found, gpu_msg = is_gpu_available()
    if not gpu_found:
        print(f"{gpu_msg}. This script cannot be run without a GPU. Exiting.")
        sys.exit(1)
    else:
        print(gpu_msg)

    print("\nWelcome to DD model evaluation\n")

    # Set up paths and model metadata
    path = "evaluation_scripts"
    compare_script_name = "compare_evals.py"

    model_metadata = {
        "base_model_eval.py": "model_evaluation_results/base_model_eval/forecast_evaluation_details0.csv",
        "neural_model_eval.py": "model_evaluation_results/neural_net_eval/forecast_evaluation_details.csv"
    }

    # Gather the scripts
    scripts_list = os.listdir(path)
    if compare_script_name in scripts_list:
        scripts_list.remove(compare_script_name)

    print("Model evaluation scripts available:")
    print(scripts_list)
    print(f"{len(scripts_list)} Model scripts found\n")

    # Prepare script menu
    scripts_dict = {}
    invalid_models = []

    def csv_valid(p):
        return os.path.exists(p) and os.path.getsize(p) > 0 and not pd.read_csv(p).empty

    # Populate the dictionary with scripts
    for i, script in enumerate(scripts_list):
        scripts_dict[i + 1] = script
        if script in model_metadata:
            if not csv_valid(model_metadata[script]):
                invalid_models.append(script)

    # If there are no invalid CSV files, allow comparison
    if not invalid_models:
        scripts_dict[len(scripts_dict) + 1] = compare_script_name
        comparison_available = True
    else:
        comparison_available = False

    while True:
        print("0: Exit the app")
        for idx, script in scripts_dict.items():
            if script == compare_script_name:
                print(f"{idx}: Compare models using {script}")
            else:
                print(f"{idx}: Run model in {script}")

        if not comparison_available and invalid_models:
            print("\n⚠️ Comparison is unavailable because some model evaluations are missing:")
            for m in invalid_models:
                print(f"   - {m} needs to be run first.")

        try:
            choice = int(input("Choose your action: "))
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if choice == 0:
            print("Exiting.")
            break
        elif choice in scripts_dict:
            selected_script = scripts_dict[choice]
            script_path = os.path.join(path, selected_script)

            print(f"Running: {selected_script}")
            try:
                with open(script_path, 'r') as f:
                    exec(f.read())
            except Exception as e:
                print(f"Error running {selected_script}: {e}")
            
            print("\nExecution completed.")
            another = input("Run another script? (y/n): ").strip().lower()
            if another != 'y':
                print("Exiting.")
                break
        else:
            print("Invalid option. Please try again.")
