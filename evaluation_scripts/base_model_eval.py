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

def add_time_and_discount_features(df):
    """
    Exactly the same transformations used in training for time-based columns,
    discount squares/bins, etc.
    """
    # Ensure DATE is datetime
    df['DATE'] = pd.to_datetime(df['DATE'])

    # Time-based features
    df['month'] = df['DATE'].dt.month
    df['year'] = df['DATE'].dt.year
    df['quarter'] = df['DATE'].dt.quarter
    df['day_of_week'] = df['DATE'].dt.dayofweek
    df['day_of_year'] = df['DATE'].dt.dayofyear
    df['week_of_year'] = df['DATE'].dt.isocalendar().week

    # Cyclical transforms
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year']/52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year']/52)

    # Discount transformations
    if 'DISCOUNT' in df.columns:
        df['DISCOUNT_SQUARED'] = df['DISCOUNT'] ** 2
        df['DISCOUNT_BIN'] = pd.cut(
            df['DISCOUNT'],
            bins=[-0.001, 0.05, 0.15, 0.25, 0.35, 1.0],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        if 'is_eoss' in df.columns:
            df['DISCOUNT_EOSS'] = df['DISCOUNT'] * df['is_eoss']

    return df

def add_lag_features(df, group_col='KEY'):
    """Add lag features with proper type conversion"""
    lag_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_8', 'lag_12', 'lag_26', 'lag_52']
    for col in lag_cols:
        df[col] = np.nan

    # Process each product separately
    for key, group in df.groupby(group_col):
        # Sort by date
        group = group.sort_values('DATE')

        # Create lag features
        for lag_col in lag_cols:
            lag_value = int(lag_col.split('_')[1])
            group[lag_col] = group['QUANTITY_SOLD'].shift(lag_value)

        # Update the dataframe
        idx = df[df[group_col] == key].index
        for col in lag_cols:
            df.loc[idx, col] = group[col].values

    # Replace NaNs with 0
    df[lag_cols] = df[lag_cols].fillna(0)
    
    return df

def add_rolling_and_trend_features(df, group_col='KEY'):
    """Add rolling statistics and trend features"""
    # Add rolling statistics
    rolling_cols = ['rolling_mean_4', 'rolling_mean_8', 'rolling_mean_12']
    for col in rolling_cols:
        df[col] = np.nan

    # Add trend and seasonality indicators
    trend_cols = ['trend', 'trend_diff', 'seasonal_indicator']
    for col in trend_cols:
        df[col] = np.nan

    # Process each product separately
    for key, group in df.groupby(group_col):
        # Sort by date
        group = group.sort_values('DATE')

        # Create rolling statistics
        for col in rolling_cols:
            window = int(col.split('_')[2])
            group[col] = group['QUANTITY_SOLD'].rolling(window=window, min_periods=1).mean()

        # Add trend components
        if len(group) > 1:
            # Linear trend
            group['trend'] = np.arange(len(group)) / len(group)

            # Trend difference (acceleration/deceleration)
            group['trend_diff'] = group['QUANTITY_SOLD'].diff().fillna(0)

            # Simple seasonal indicator (based on month)
            group['seasonal_indicator'] = group['month'].map(
                lambda m: 1 if m in [12, 1, 6, 7] else 0
            )
        else:
            # Default values for single-point groups
            group['trend'] = 0.5
            group['trend_diff'] = 0
            group['seasonal_indicator'] = 0

        # Update the dataframe
        idx = df[df[group_col] == key].index
        for col in rolling_cols + trend_cols:
            df.loc[idx, col] = group[col].values

    # Replace NaNs with 0
    all_dynamic_cols = rolling_cols + trend_cols
    df[all_dynamic_cols] = df[all_dynamic_cols].fillna(0)

    return df

def load_model_package(model_path):
    """
    Load the trained models and metadata from pickle file
    
    Parameters:
    - model_path: Path to the saved model package
    
    Returns:
    - Dictionary containing models, key_metadata, feature_cols, and scaler
    """
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        print(f"Loaded model package with {len(model_package['models'])} models")
        return model_package
    except Exception as e:
        print(f"Error loading model package: {e}")
        return None

def apply_discount_calibration(prediction, key, discount, calibration_strategies, is_eoss=None):
    """Apply calibration to predictions based on discount level and product"""
    # If no calibration strategies, return prediction as is
    if calibration_strategies is None or not calibration_strategies:
        return prediction

    # Determine discount bin
    if discount <= 0.1:
        discount_bin = '0-10%'
    elif discount <= 0.2:
        discount_bin = '10-20%'
    elif discount <= 0.3:
        discount_bin = '20-30%'
    elif discount <= 0.4:
        discount_bin = '30-40%'
    else:
        discount_bin = '40%+'

    # Try most specific calibration first
    if is_eoss is not None and 'product_eoss' in calibration_strategies:
        if (key in calibration_strategies['product_eoss'] and
            (is_eoss, discount_bin) in calibration_strategies['product_eoss'][key]):
            return prediction * calibration_strategies['product_eoss'][key][(is_eoss, discount_bin)]

    # Try product + discount bin calibration
    if 'product_discount' in calibration_strategies and key in calibration_strategies['product_discount']:
        if discount_bin in calibration_strategies['product_discount'][key]:
            return prediction * calibration_strategies['product_discount'][key][discount_bin]

    # Try product-only calibration
    if 'product' in calibration_strategies and key in calibration_strategies['product']:
        return prediction * calibration_strategies['product'][key]

    # Try discount bin calibration
    if 'discount_bin' in calibration_strategies and discount_bin in calibration_strategies['discount_bin']:
        return prediction * calibration_strategies['discount_bin'][discount_bin]

    # Default: no calibration
    return prediction

def evaluate_linear_regressor(
    model_path,
    train_data_path,
    test_data_path,
    output_dir='linear_model_evaluation',
    calibration_path=None
):
    """
    Evaluates the linear regression models on test data.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model package
    train_data_path : str
        Path to training data CSV
    test_data_path : str
        Path to test data CSV
    output_dir : str
        Directory to save evaluation results
    calibration_path : str, optional
        Path to calibration strategies file
    
    Returns:
    --------
    pd.DataFrame
        Forecast comparison results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n===== Loading Model Resources =====")
    
    # Load model package
    model_package = load_model_package(model_path)
    if model_package is None:
        return None
    
    # Extract components from model package
    models = model_package['models']
    key_metadata = model_package.get('key_metadata', {})
    feature_cols = model_package['feature_cols']
    scaler = model_package.get('scaler', None)
    
    print(f"Loaded {len(models)} models and {len(feature_cols)} feature columns")
    
    # Load calibration strategies if provided
    calibration_strategies = {}
    if calibration_path is not None and os.path.exists(calibration_path):
        try:
            with open(calibration_path, 'rb') as f:
                calibration_strategies = pickle.load(f)
            print(f"Loaded calibration strategies")
        except Exception as e:
            print(f"Warning: Error loading calibration strategies: {e}")
    
    # Load training and test data
    print("\n===== Loading Datasets =====")
    try:
        df_train = pd.read_csv(train_data_path)
        df_test = pd.read_csv(test_data_path)
        
        # Ensure DATE is in datetime format
        df_train['DATE'] = pd.to_datetime(df_train['DATE'])
        df_test['DATE'] = pd.to_datetime(df_test['DATE'])
        
        print(f"Train data: {len(df_train)} rows, {df_train['KEY'].nunique()} products")
        print(f"Test data: {len(df_test)} rows, {df_test['KEY'].nunique()} products")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None
    
    # Track results
    results = []
    
    # Get test product keys
    test_keys = df_test['KEY'].unique()
    print(f"\n===== Evaluating {len(test_keys)} Products =====")
    
    # Ensure feature_cols doesn't include DATE or KEY
    feature_cols = [col for col in feature_cols if col not in ['DATE', 'KEY']]
    
    # Process each product in the test set
    for i, key in enumerate(tqdm(test_keys, desc="Evaluating products")):
        # Get test data for this product
        test_data_key = df_test[df_test['KEY'] == key].copy()
        
        # Skip if no test data
        if len(test_data_key) == 0:
            continue
        
        # Skip if no model for this key
        if key not in models:
            print(f"Warning: No model found for key {key}")
            continue
        
        # Get training data for this product
        train_data_key = df_train[df_train['KEY'] == key].copy()
        
        # Skip if not enough training data
        if len(train_data_key) < 5:  # Minimum data required
            print(f"Warning: Insufficient training data for key {key}")
            continue
        
        # Concatenate train and test for feature engineering
        combined = pd.concat([train_data_key, test_data_key], ignore_index=True)
        combined.sort_values('DATE', inplace=True)
        
        try:
            # Apply all feature engineering steps from training
            combined = add_time_and_discount_features(combined)
            combined = add_lag_features(combined)
            combined = add_rolling_and_trend_features(combined)
            
            # Ensure all necessary columns exist
            missing_cols = [col for col in feature_cols if col not in combined.columns]
            if missing_cols:
                print(f"Warning: {len(missing_cols)} missing features for {key}: {missing_cols[:5]}...")
                for col in missing_cols:
                    combined[col] = 0.0
            
            # Clean data - handle NaN and Inf values
            for col in feature_cols:
                combined[col] = combined[col].replace([np.inf, -np.inf], np.nan)
                if combined[col].isna().any():
                    combined[col] = combined[col].fillna(0)
            
            # Apply scaling if scaler is available
            if scaler is not None:
                combined_features = combined[feature_cols].values
                combined_features = scaler.transform(combined_features)
                
                # Replace original values with scaled values
                for j, col in enumerate(feature_cols):
                    combined[col] = combined_features[:, j]
            
            # Get model for this key
            model = models[key]
            
            # Process each test date for this product
            for _, test_row in test_data_key.iterrows():
                test_date = test_row['DATE']
                
                # Get all data up to this test date for features
                test_features = combined[combined['DATE'] == test_date][feature_cols].values
                
                # Skip if data is incomplete
                if test_features.shape[0] == 0 or test_features.shape[1] != len(feature_cols):
                    print(f"Warning: Incomplete features for {key} on {test_date}")
                    continue
                
                # Get actual values from test data
                actual_sales = float(test_row['QUANTITY_SOLD'])
                discount = float(test_row.get('DISCOUNT', 0))
                is_eoss = float(test_row.get('is_eoss', 0)) if 'is_eoss' in test_row else 0
                
                try:
                    # Make prediction
                    raw_pred = float(model.predict(test_features)[0])
                    
                    # Apply calibration if available
                    calibrated_pred = apply_discount_calibration(
                        raw_pred, key, discount, calibration_strategies, is_eoss
                    )
                    
                    # Ensure non-negative prediction
                    calibrated_pred = max(0, calibrated_pred)
                    
                    # Calculate metrics
                    abs_diff = abs(calibrated_pred - actual_sales)
                    pct_error = (abs_diff / (actual_sales + 1e-5)) * 100
                    
                    # Store result
                    results.append({
                        'KEY': key,
                        'DATE': test_date,
                        'Actual_Sales': actual_sales,
                        'Predicted_Sales': calibrated_pred,
                        'Raw_Prediction': raw_pred,
                        'Absolute_Error': abs_diff,
                        'Percent_Error': pct_error,
                        'DISCOUNT': discount,
                        'is_eoss': is_eoss
                    })
                except Exception as e:
                    print(f"Error predicting for {key} on {test_date}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing product {key}: {e}")
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No predictions were generated. Check data compatibility and model weights.")
        return None
    
    print("\n===== Generating Evaluation Results =====")
    
    # Calculate overall metrics
    mean_abs_error = results_df['Absolute_Error'].mean()
    median_abs_error = results_df['Absolute_Error'].median()
    mean_pct_error = results_df['Percent_Error'].mean()
    median_pct_error = results_df['Percent_Error'].median()
    
    total_actual = results_df['Actual_Sales'].sum()
    total_predicted = results_df['Predicted_Sales'].sum()
    overall_error = ((total_predicted - total_actual) / total_actual) * 100
    
    # Display metrics
    print("\n===== Summary Metrics =====")
    print(f"Total products evaluated: {results_df['KEY'].nunique()}")
    print(f"Total predictions: {len(results_df)}")
    print(f"Mean Absolute Error: {mean_abs_error:.2f}")
    print(f"Median Absolute Error: {median_abs_error:.2f}")
    print(f"Mean Percent Error: {mean_pct_error:.2f}%")
    print(f"Median Percent Error: {median_pct_error:.2f}%")
    print(f"Total Actual Sales: {total_actual:.2f}")
    print(f"Total Predicted Sales: {total_predicted:.2f}")
    print(f"Overall Error: {overall_error:.2f}%")
    
    # Save metrics to summary file
    metrics_summary = {
        'Total Products': results_df['KEY'].nunique(),
        'Total Predictions': len(results_df),
        'Mean Absolute Error': mean_abs_error,
        'Median Absolute Error': median_abs_error,
        'Mean Percent Error': mean_pct_error,
        'Median Percent Error': median_pct_error,
        'Total Actual Sales': total_actual,
        'Total Predicted Sales': total_predicted,
        'Overall Error': overall_error
    }
    
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        for key, value in metrics_summary.items():
            f.write(f"{key}: {value}\n")
    
    # Create visualizations
    print("\n===== Creating Visualizations =====")
    
    try:
        plt.figure(figsize=(16, 12))
        
        # 1. Predicted vs Actual scatter plot
        plt.subplot(2, 3, 1)
        plt.scatter(results_df['Actual_Sales'], results_df['Predicted_Sales'], alpha=0.5)
        max_val = max(results_df['Actual_Sales'].max(), results_df['Predicted_Sales'].max()) * 1.1
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title('Predicted vs Actual Sales')
        plt.grid(True, alpha=0.3)
        
        # 2. Error distribution
        plt.subplot(2, 3, 2)
        plt.hist(results_df['Absolute_Error'], bins=50, alpha=0.7)
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # 3. Error by discount level (if available)
        if 'DISCOUNT' in results_df.columns:
            plt.subplot(2, 3, 3)
            
            # Create discount bins
            results_df['Discount_Bin'] = pd.cut(
                results_df['DISCOUNT'],
                bins=[-0.001, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 1.0],
                labels=['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-40%', '40%+']
            )
            
            # Calculate average error by bin
            bin_errors = results_df.groupby('Discount_Bin')['Absolute_Error'].mean().reset_index()
            
            # Plot
            sns.barplot(x='Discount_Bin', y='Absolute_Error', data=bin_errors)
            plt.xlabel('Discount Bin')
            plt.ylabel('Mean Absolute Error')
            plt.title('Error by Discount Level')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 4. Error by EOSS status (if available)
        if 'is_eoss' in results_df.columns and results_df['is_eoss'].nunique() > 1:
            plt.subplot(2, 3, 4)
            eoss_errors = results_df.groupby('is_eoss')['Absolute_Error'].mean().reset_index()
            eoss_errors['EOSS_Status'] = eoss_errors['is_eoss'].map({0: 'Non-EOSS', 1: 'EOSS'})
            
            sns.barplot(x='EOSS_Status', y='Absolute_Error', data=eoss_errors)
            plt.xlabel('EOSS Status')
            plt.ylabel('Mean Absolute Error')
            plt.title('Error by EOSS Status')
            plt.grid(True, alpha=0.3)
        
        # 5. Actual vs Predicted by discount level
        if 'DISCOUNT' in results_df.columns and 'Discount_Bin' in results_df.columns:
            plt.subplot(2, 3, 5)
            
            # Calculate averages by discount bin
            avg_by_bin = results_df.groupby('Discount_Bin').agg({
                'Actual_Sales': 'mean',
                'Predicted_Sales': 'mean'
            }).reset_index()
            
            # Plot as grouped bar chart
            x = np.arange(len(avg_by_bin))
            width = 0.35
            
            plt.bar(x - width/2, avg_by_bin['Actual_Sales'], width, label='Actual')
            plt.bar(x + width/2, avg_by_bin['Predicted_Sales'], width, label='Predicted')
            
            plt.xlabel('Discount Bin')
            plt.ylabel('Average Sales')
            plt.title('Actual vs Predicted Sales by Discount')
            plt.xticks(x, avg_by_bin['Discount_Bin'], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. Performance by top products
        plt.subplot(2, 3, 6)
        
        # Get top 10 products by sales volume
        top_products = results_df.groupby('KEY')['Actual_Sales'].sum().nlargest(10).index
        top_products_df = results_df[results_df['KEY'].isin(top_products)]
        
        # Calculate performance by product
        product_performance = top_products_df.groupby('KEY').agg({
            'Actual_Sales': 'sum',
            'Predicted_Sales': 'sum'
        }).reset_index()
        
        # Sort by actual sales
        product_performance = product_performance.sort_values('Actual_Sales', ascending=False)
        
        # Plot as grouped bar chart
        x = np.arange(len(product_performance))
        width = 0.35
        
        plt.bar(x - width/2, product_performance['Actual_Sales'], width, label='Actual')
        plt.bar(x + width/2, product_performance['Predicted_Sales'], width, label='Predicted')
        
        plt.xlabel('Product')
        plt.ylabel('Total Sales')
        plt.title('Performance for Top Products')
        plt.xticks(x, [p[:10] + '...' for p in product_performance['KEY']], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'forecast_evaluation.png'))
        print(f"Visualization saved to '{os.path.join(output_dir, 'forecast_evaluation.png')}'")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Generate detailed reports
    try:
        # Save full results
        results_df.to_csv(os.path.join(output_dir, 'forecast_evaluation_details.csv'), index=False)
        print(f"Detailed results saved to '{os.path.join(output_dir, 'forecast_evaluation_details.csv')}'")
        
        # Create product-level summary
        product_summary = results_df.groupby('KEY').agg({
            'Actual_Sales': 'sum',
            'Predicted_Sales': 'sum',
            'Absolute_Error': 'mean',
            'DISCOUNT': 'mean'
        }).reset_index()
        
        product_summary['Total_Error'] = product_summary['Predicted_Sales'] - product_summary['Actual_Sales']
        product_summary['Percent_Error'] = (product_summary['Total_Error'] / product_summary['Actual_Sales']) * 100
        
        # Sort by largest absolute error
        product_summary = product_summary.sort_values(by='Percent_Error', ascending=False)
        
        # Save product summary
        product_summary.to_csv(os.path.join(output_dir, 'product_forecast_summary.csv'), index=False)
        print(f"Product summary saved to '{os.path.join(output_dir, 'product_forecast_summary.csv')}'")
        
        # If discount bins exist, create discount bin summary
        if 'Discount_Bin' in results_df.columns:
            discount_summary = results_df.groupby('Discount_Bin').agg({
                'Actual_Sales': ['sum', 'mean', 'count'],
                'Predicted_Sales': ['sum', 'mean'],
                'Absolute_Error': 'mean',
                'Percent_Error': 'mean'
            })
            
            # Flatten column names
            discount_summary.columns = ['_'.join(col).strip() for col in discount_summary.columns.values]
            
            # Calculate overall error for each bin
            discount_summary['Total_Error'] = discount_summary['Predicted_Sales_sum'] - discount_summary['Actual_Sales_sum']
            discount_summary['Error_Percent'] = (discount_summary['Total_Error'] / discount_summary['Actual_Sales_sum']) * 100
            
            # Save discount bin summary
            discount_summary.to_csv(os.path.join(output_dir, 'discount_bin_performance.csv'))
            print(f"Discount bin summary saved to '{os.path.join(output_dir, 'discount_bin_performance.csv')}'")
    except Exception as e:
        print(f"Error generating detailed reports: {e}")
    
    # Display top products with largest errors
    try:
        print("\nTop 10 Products with Largest Percentage Errors:")
        print(product_summary[['KEY', 'Actual_Sales', 'Predicted_Sales', 'Percent_Error', 'DISCOUNT']].head(10))
    except:
        print("Could not display top products with errors")
    
    return results_df

def run_linear_model_evaluation(model_path, train_data_path, test_data_path, output_dir='linear_model_evaluation', calibration_path=None):
    """
    Main function to run the evaluation pipeline for linear regression models
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model pickle file
    train_data_path : str
        Path to training data CSV
    test_data_path : str
        Path to test data CSV
    output_dir : str
        Directory to save evaluation results
    calibration_path : str, optional
        Path to calibration strategies file
    """
    print("\n===============================================")
    print("LINEAR REGRESSION DISCOUNT-SENSITIVE FORECASTING MODEL EVALUATION")
    print("===============================================\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Log evaluation start time
    start_time = datetime.now()
    print(f"Evaluation started at: {start_time}")
    
    # Run evaluation
    results = evaluate_linear_regressor(
        model_path=model_path,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        output_dir=output_dir,
        calibration_path=calibration_path
    )
    
    # Log evaluation end time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nEvaluation completed at: {end_time}")
    print(f"Total duration: {duration}")
    
    if results is not None:
        print(f"\nEvaluation completed successfully. Results saved to '{output_dir}'")
    else:
        print("\nEvaluation failed. Check errors above.")
    
    return results

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

# Example usage
if __name__ == "__main__":
    # Run linear model evaluation
    linear_results = run_linear_model_evaluation(
        model_path='models/base_model/sales_forecasting_models.pkl',
        train_data_path='data_files/df_forecasting_train.csv',
        test_data_path='data_files/df_forecasting_test.csv',
        output_dir='model_evaluation_results/base_model_eval'
    )
    
    # Optional: Compare with neural network model if results are available
    # compare_models(
    #     linear_results_path='linear_model_evaluation/forecast_evaluation_details.csv',
    #     neural_results_path='model_evaluation_results/forecast_evaluation_details.csv',
    #     output_dir='model_comparison'
    # )