import os
import pandas as pd

if __name__ == "__main__":
    print("\n#######################################################")
    print("Welcome to the Dynamic Discounting model evaluation app!")
    print("########################################################\n")
    print("\nPreReqs")
    print("______________________________________________________\n")
    print("This app can help you run and evaluate the different forecasting models you have built")
    print("1. Models folder:                Make sure you have the model in whatever format stored as a folder here")
    print("2. evalutation_scripts folder:   For each model, make sure its relevant model evaluation script in stored as a .py file here")
    print("\nMake sure both model folder and its evaluation function work together")
    print("______________________________________________________\n")
    
    # Prepping model list for display
    #get evaluation script directory
    path = "evaluation_scripts"
    dir_list = os.listdir(path)
    compare_script_file_name = "compare_evals.py"
    # Retrieving the list of the models
    models = dir_list.copy()
    # Dynamically build metadata to link model scripts with their output CSV files
    models_metadata = {}
    for model_script in models:
        if model_script.endswith('.py') and model_script != compare_script_file_name:
            model_name = model_script.replace('.py', '')
            result_path = f"model_evaluation_results/{model_name}/forecast_evaluation_details.csv"
            models_metadata[model_script] = result_path
    if compare_script_file_name in models : # removing the compare_evals from the models list as it is not a model
        models.remove(compare_script_file_name)

    # Displaying models available
    print(f"\nChecked the {path} file and found the following evaluation scripts:")
    print("To add more, add the files to the folders as mentioned above")
    formatted_models = [model.replace('_', ' ').replace('.py', '') for model in models]
    print("- "+", ".join(formatted_models))
    print(f"{len(models)} Model scripts found\n")
    
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
        
        print("Please choose and print from the following actions by typing and entering the corresponding number:\n") # exit option
        print("At any point type 0 to leave the app") # exit option
        # Scripts options
        for index, script in scripts_dict.items():
            if script == compare_script_file_name : 
                print(f"\n{index}: Compare models using {script.replace('_', ' ').replace('.py', '')}\n")
            else :
                print(f"{index}: Run {script.replace('_', ' ').replace('.py', '')}")
        
        # If the comparaison script was not in the options we explain why :
        if not comparaison_available:
            print("\n⚠️ Can not compare models now because some model evaluations are missing:")
            for model in invalid_models:
                print(f"   - {model.replace('_', ' ').replace('.py', '')} needs to be run first.")
            

        # Get user input
        try:
            inp = int(input("Choose your action: ")) 
        except ValueError:
            print("\n🛑Input needs to be an integer🛑\n") 
            continue
        
        # Fixed this condition check
        if inp == 0: # exit option
            print("Exiting")
            break
        elif inp in scripts_dict.keys():
            selected_script = scripts_dict[inp]
            script_path = os.path.join(path,selected_script)

            print(f"Running model: {scripts_dict[inp].replace('_', ' ').replace('.py', '')}")
            
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