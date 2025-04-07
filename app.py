import os
import pandas as pd

if __name__ == "__main__":
    
    print("\nWelcome to DD model evaluation\n")

    #show models
    path = "evaluation_scripts"
    dir_list = os.listdir(path)
    
    compare_script_file_name = "compare_evals.py"

    # Retrieving the list of the models
    models = dir_list.copy()

    # Metadata to link model scripts with their output CSV files
    models_metadata = {
        "base_model_eval.py": "model_evaluation_results/base_model_eval/forecast_evaluation_details0.csv",
        "neural_model_eval.py": "model_evaluation_results/neural_net_eval/forecast_evaluation_details.csv"
    }    

    if compare_script_file_name in models : # removing the compare_evals from the models list as it is not a model
        models.remove(compare_script_file_name)

    # Printing the models available 
    print("Model evaluation scripts available:")
    print(models)
    num_models = len(models)
    print(f"{num_models} Model scripts found\n")
    
    # Changed the variable name to scripts_dict : will contain the available scripts to run
    scripts_dict = {}

    # Check if comparaison should be available
    comparaison_available = False
    invalid_models = []

    def csv_valid(path):
        """Check if the csv containing the models' results exists and are valid for use"""
        return os.path.exists(path) and os.path.getsize(path) > 0 and not pd.read_csv(path).empty

    # Add all model scripts to the menu
    for i, model_script in enumerate(models):
        scripts_dict[i + 1] = model_script

        # Check if this model has an associated result file and whether it is valid
        if model_script in models_metadata:
            result_path = models_metadata[model_script]
            if not csv_valid(result_path):
                invalid_models.append(model_script)

    # If all csvs are present and not empty we allow comparaison
    if invalid_models == []:
        comparaison_available = True
        # Add the comparaison script as the last element of the dictionary 
        scripts_dict[len(scripts_dict) + 1] = compare_script_file_name

    # Main loop
    while True:
        # MENU
        # 0 : Exit option
        # Last index : compare the models
        # inbetween  : run the models 

        print("0: Exit the app") # exit option
        # Scripts options
        for index, script in scripts_dict.items():
            if script == compare_script_file_name : 
                print(f"{index}: Compare models using {script}")
            else :
                print(f"{index}: Run model in {script}")
        
        # If the comparaison script was not in the options we explain why :
        if not comparaison_available:
            print("\n⚠️ Comparison is unavailable because some model evaluations are missing:")
            for model in invalid_models:
                print(f"   - {model} needs to be run first.")
            

        # Get user input
        try:
            inp = int(input("Choose your action: "))  # Fixed: added colon and space
        except ValueError:
            print("Input needs to be an integer")  # Fixed: print instead of nested input
            continue
        
        # Fixed this condition check
        if inp == 0: # exit option
            print("Exiting")
            break
        elif inp in scripts_dict.keys():
            selected_script = scripts_dict[inp]
            script_path = os.path.join(path,selected_script)

            print(f"Running model: {scripts_dict[inp]}")
            
            try:
                with open(script_path,'r') as f:
                    script_content = f.read()
                    exec(script_content)
            except Exception as e:
                print(f"Error running {selected_script}: {e}")
            
            # Added : Let the user choose if they want to quit the app or run another script
            print("Running completed\n.")
            again = input("Do you want to run another script? (y/n): ").strip().lower()
            if again != 'y':
                print("Exiting the app.")
                break
        else:
            print("Model not found. Please enter a valid number.")  # Fixed: print message
            continue