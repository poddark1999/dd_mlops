#AI CODE
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Compare results with neural network model (if available)
def compare_models(linear_results_path, neural_results_path, output_dir='model_comparison'):
    """
    Compare the performance of linear regression and neural network models
    
    Parameters:
    -----------
    linear_results_path : str
        Path to linear model evaluation results CSV
    neural_results_path : str
        Path to neural network evaluation results CSV
    output_dir : str
        Directory to save comparison results
    """
    print("\n===== Comparing Model Performance =====")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load evaluation results
        linear_results = pd.read_csv(linear_results_path)
        neural_results = pd.read_csv(neural_results_path)
        
        print(f"Linear model: {len(linear_results)} predictions")
        print(f"Neural network: {len(neural_results)} predictions")
        
        # Find common keys and dates for fair comparison
        linear_results['KEY_DATE'] = linear_results['KEY'] + '_' + linear_results['DATE'].astype(str)
        neural_results['KEY_DATE'] = neural_results['KEY'] + '_' + neural_results['DATE'].astype(str)
        
        common_key_dates = set(linear_results['KEY_DATE']).intersection(set(neural_results['KEY_DATE']))
        print(f"Common predictions: {len(common_key_dates)}")
        
        if len(common_key_dates) == 0:
            print("No common predictions found for comparison.")
            return
        
        # Filter to common predictions
        linear_common = linear_results[linear_results['KEY_DATE'].isin(common_key_dates)]
        neural_common = neural_results[neural_results['KEY_DATE'].isin(common_key_dates)]
        
        # Ensure same order
        linear_common = linear_common.sort_values('KEY_DATE')
        neural_common = neural_common.sort_values('KEY_DATE')
        
        # Calculate comparative metrics
        comparison = pd.DataFrame({
            'KEY': linear_common['KEY'],
            'DATE': linear_common['DATE'],
            'Actual_Sales': linear_common['Actual_Sales'],
            'Linear_Predicted': linear_common['Predicted_Sales'],
            'Neural_Predicted': neural_common['Predicted_Sales'],
            'Linear_Error': linear_common['Absolute_Error'],
            'Neural_Error': neural_common['Absolute_Error'],
            'Linear_Pct_Error': linear_common['Percent_Error'],
            'Neural_Pct_Error': neural_common['Percent_Error'],
            'DISCOUNT': linear_common['DISCOUNT'],
            'is_eoss': linear_common['is_eoss']
        })
        
        # Add which model performed better
        comparison['Better_Model'] = 'Equal'
        comparison.loc[comparison['Linear_Error'] < comparison['Neural_Error'], 'Better_Model'] = 'Linear'
        comparison.loc[comparison['Neural_Error'] < comparison['Linear_Error'], 'Better_Model'] = 'Neural'
        
        # Overall comparison
        linear_mae = comparison['Linear_Error'].mean()
        neural_mae = comparison['Neural_Error'].mean()
        linear_mape = comparison['Linear_Pct_Error'].mean()
        neural_mape = comparison['Neural_Pct_Error'].mean()
        
        linear_win_rate = (comparison['Better_Model'] == 'Linear').mean() * 100
        neural_win_rate = (comparison['Better_Model'] == 'Neural').mean() * 100
        
        print("\n===== Overall Comparison =====")
        print(f"Linear Model MAE: {linear_mae:.2f}, MAPE: {linear_mape:.2f}%")
        print(f"Neural Network MAE: {neural_mae:.2f}, MAPE: {neural_mape:.2f}%")
        print(f"Linear model wins: {linear_win_rate:.1f}% of predictions")
        print(f"Neural network wins: {neural_win_rate:.1f}% of predictions")
        
        # Create comparison visualizations
        plt.figure(figsize=(16, 12))
        
        # 1. Overall error comparison
        plt.subplot(2, 2, 1)
        models = ['Linear Regression', 'Neural Network']
        maes = [linear_mae, neural_mae]
        mapes = [linear_mape, neural_mape]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, maes, width, label='MAE')
        plt.bar(x + width/2, mapes, width, label='MAPE (%)')
        
        plt.xlabel('Model')
        plt.ylabel('Error')
        plt.title('Error Comparison')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Win rate comparison
        plt.subplot(2, 2, 2)
        win_rates = [linear_win_rate, neural_win_rate, 100 - linear_win_rate - neural_win_rate]
        labels = ['Linear Better', 'Neural Better', 'Equal']
        
        plt.pie(win_rates, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Model Win Rate Comparison')
        
        # 3. Error by discount level
        if 'DISCOUNT' in comparison.columns:
            plt.subplot(2, 2, 3)
            
            # Create discount bins
            comparison['Discount_Bin'] = pd.cut(
                comparison['DISCOUNT'],
                bins=[-0.001, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 1.0],
                labels=['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-40%', '40%+']
            )
            
            # Calculate average error by bin for each model
            bin_errors = comparison.groupby('Discount_Bin').agg({
                'Linear_Error': 'mean',
                'Neural_Error': 'mean'
            }).reset_index()
            
            # Reshape for seaborn
            bin_errors_melted = pd.melt(
                bin_errors, 
                id_vars=['Discount_Bin'],
                value_vars=['Linear_Error', 'Neural_Error'],
                var_name='Model',
                value_name='Error'
            )
            bin_errors_melted['Model'] = bin_errors_melted['Model'].map({
                'Linear_Error': 'Linear',
                'Neural_Error': 'Neural'
            })
            
            # Plot
            sns.barplot(x='Discount_Bin', y='Error', hue='Model', data=bin_errors_melted)
            plt.xlabel('Discount Bin')
            plt.ylabel('Mean Absolute Error')
            plt.title('Error by Discount Level')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 4. Error distribution comparison
        plt.subplot(2, 2, 4)
        
        plt.hist(comparison['Linear_Error'], bins=30, alpha=0.5, label='Linear Model')
        plt.hist(comparison['Neural_Error'], bins=30, alpha=0.5, label='Neural Network')
        
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
        print(f"Comparison visualization saved to '{os.path.join(output_dir, 'model_comparison.png')}'")
        
        # Save comparison data
        comparison.to_csv(os.path.join(output_dir, 'model_comparison_details.csv'), index=False)
        print(f"Detailed comparison saved to '{os.path.join(output_dir, 'model_comparison_details.csv')}'")
        
        # Create summary by discount bin
        if 'Discount_Bin' in comparison.columns:
            discount_comparison = comparison.groupby('Discount_Bin').agg({
                'Actual_Sales': 'sum',
                'Linear_Predicted': 'sum',
                'Neural_Predicted': 'sum',
                'Linear_Error': 'mean',
                'Neural_Error': 'mean',
                'Linear_Pct_Error': 'mean',
                'Neural_Pct_Error': 'mean',
                'Better_Model': lambda x: (x == 'Linear').mean() * 100
            }).reset_index()
            
            discount_comparison.rename(columns={'Better_Model': 'Linear_Win_Rate'}, inplace=True)
            discount_comparison['Neural_Win_Rate'] = 100 - discount_comparison['Linear_Win_Rate']
            
            discount_comparison.to_csv(os.path.join(output_dir, 'discount_bin_comparison.csv'), index=False)
            print(f"Discount bin comparison saved to '{os.path.join(output_dir, 'discount_bin_comparison.csv')}'")
        
        return comparison
    
    except Exception as e:
        print(f"Error comparing models: {e}")
        return None

if __name__ == "__main__":
    print("\n===== Starting Model Comparison =====")
    
    # Dynamically discover available model evaluation scripts
    eval_scripts_dir = "evaluation_scripts"
    model_results_dir = "model_evaluation_results"
    
    # Get all evaluation scripts except the comparison script
    eval_scripts = [f for f in os.listdir(eval_scripts_dir) 
                    if f.endswith('.py') and f != 'compare_evals.py']
    
    # Build model paths dictionary dynamically
    model_paths = {}
    for script in eval_scripts:
        # Extract model name without extension
        model_name = script.replace('.py', '')
        # Construct expected results path
        result_path = os.path.join(model_results_dir, model_name, 'forecast_evaluation_details.csv')
        # Check if results exist
        if os.path.exists(result_path):
            model_paths[model_name] = result_path
    
    if len(model_paths) < 2:
        print("\n⚠️ Not enough model evaluation results available for comparison.")
        print("Please run at least two model evaluations first.")
        exit()
    
    # List available models
    print("\nAvailable models for comparison:")
    for i, model in enumerate(model_paths.keys()):
        print(f"{i+1}. {model.replace('_', ' ')}")
    
    # Get user selections
    try:
        first_model_idx = int(input("\nSelect first model (number): ")) - 1
        second_model_idx = int(input("Select second model (number): ")) - 1
        
        model_names = list(model_paths.keys())
        
        if first_model_idx < 0 or first_model_idx >= len(model_names) or \
           second_model_idx < 0 or second_model_idx >= len(model_names):
            print("Invalid selection. Please try again.")
            exit()
        
        first_model = model_names[first_model_idx]
        second_model = model_names[second_model_idx]
    except ValueError:
        print("Invalid input. Please enter a number.")
        exit()
    
    # Get paths
    first_path = model_paths[first_model]
    second_path = model_paths[second_model]
    
    # Output directory
    output_dir = os.path.join(model_results_dir, 'model_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist and run evaluations if needed
    for model, path in [(first_model, first_path), (second_model, second_path)]:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"\n⚠️ {model.replace('_', ' ')} results not found.")
            run_eval = input(f"Run {model.replace('_', ' ')} evaluation first? (y/n): ").strip().lower()
            
            if run_eval == 'y':
                print(f"\nRunning {model} evaluation...")
                try:
                    script_path = os.path.join(eval_scripts_dir, f"{model}.py")
                    with open(script_path, 'r') as f:
                        script_content = f.read()
                        exec(script_content)
                except Exception as e:
                    print(f"Error running {model} evaluation: {e}")
                    print("Comparison cannot continue.")
                    exit()
            else:
                print("Comparison cannot continue without required data.")
                exit()
    
    print(f"\nComparing models using:")
    print(f"First model ({first_model.replace('_', ' ')}): {first_path}")
    print(f"Second model ({second_model.replace('_', ' ')}): {second_path}")
    print(f"Output directory: {output_dir}\n")
    
    # Run the comparison
    comparison = compare_models(first_path, second_path, output_dir=output_dir)
