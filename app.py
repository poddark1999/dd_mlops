import os
if __name__ == "__main__":
    
    print("\nWelcome to DD model evaluation\n")

    #show models
    path = "evaluation_scripts"
    dir_list = os.listdir(path)
    
    print("Model evaluation scripts found:")
    print(dir_list)
    num_models = len(dir_list)
    print(f"{num_models} Model scripts found\n")
    
    models_dict = {}
    for mod_int in range(len(dir_list)):
        models_dict[mod_int+1] = dir_list[mod_int]
    
    
    print("Choose which model to evaluate by pressing the number with the corresponding model")

    while True:
        # display models
        print("\ntype 0 to see models")
        for index, model in models_dict.items():
            print(f"{index}: {model}")

        # Get user input
        try:
            inp = int(input("Choose model to run: "))  # Fixed: added colon and space
        except ValueError:
            print("Input needs to be an integer")  # Fixed: print instead of nested input
            continue
        
        # Fixed this condition check
        if inp == 0:
            # Just show the models again (loop will repeat)
            continue
        elif inp in models_dict.keys():
            selected_script = models_dict[inp]
            script_path = os.path.join(path,selected_script)

            print(f"Running model: {models_dict[inp]}")
            
            try:
                with open(script_path,'r') as f:
                    script_content = f.read()
                    exec(script_content)
            except Exception as e:
                print(f"Error running {selected_script}: {e}")

            break
        else:
            print("Model not found. Please enter a valid number.")  # Fixed: print message
            continue
        


        




