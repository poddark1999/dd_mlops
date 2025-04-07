import os
if __name__ == "__main__":
    
    print("\nWelcome to DD model evaluation\n")

    #show models
    path = "evaluation_scripts"
    dir_list = os.listdir(path)
    
    compare_script_file_name = "compare_evals.py"

    # Retrieving the list of the models
    models = dir_list.copy()
    
    if compare_script_file_name in models : # removing the compare_evals from the models list as it is not a model
        models.remove(compare_script_file_name)

    # Printing the models available 
    print("Model evaluation scripts available:")
    print(models)
    num_models = len(models)
    print(f"{num_models} Model scripts found\n")
    
    # Changed the variable name to scripts_dict as we want to separe models from comparaison file
    scripts_dict = {}
    for mod_int in range(len(models)):
        scripts_dict[mod_int+1] = models[mod_int]
    # Add the comparaison script as the last element of the dictionary 
    scripts_dict[len(scripts_dict) + 1] = compare_script_file_name
    

    # Action of the user
    print("Choose your action : ")

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
            

        # Get user input
        try:
            inp = int(input("Choose the script to run: "))  # Fixed: added colon and space
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
        


        




