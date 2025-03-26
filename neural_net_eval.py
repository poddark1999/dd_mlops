import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D,
                                    Input, GlobalAveragePooling1D, Concatenate, 
                                    BatchNormalization, MultiHeadAttention,
                                    LayerNormalization, Add, Multiply)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#####################################################################
# 1. FEATURE ENGINEERING FUNCTIONS (IMPORTED FROM TRAINING)
#####################################################################

def add_time_features(df):
    """Add time-based features to the dataframe with explicit type conversion"""
    df['DATE'] = pd.to_datetime(df['DATE'])

    # Time-based features
    df['month'] = df['DATE'].dt.month.astype(float)
    df['quarter'] = df['DATE'].dt.quarter.astype(float)
    df['day_of_week'] = df['DATE'].dt.dayofweek.astype(float)
    df['week_of_year'] = df['DATE'].dt.isocalendar().week.astype(float)

    # Sine and cosine transformations for cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12).astype(float)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12).astype(float)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7).astype(float)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7).astype(float)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year']/52).astype(float)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year']/52).astype(float)

    # Add interaction features
    if 'DISCOUNT' in df.columns:
        df['DISCOUNT'] = df['DISCOUNT'].astype(float)
        if 'month' in df.columns:
            df['DISCOUNT_MONTH'] = (df['DISCOUNT'] * df['month_sin']).astype(float)

        if 'holiday_label' in df.columns:
            df['DISCOUNT_HOLIDAY'] = (df['DISCOUNT'] * df['holiday_label']).astype(float)

        if 'is_eoss' in df.columns:
            df['is_eoss'] = df['is_eoss'].astype(float)
            df['DISCOUNT_EOSS'] = (df['DISCOUNT'] * df['is_eoss']).astype(float)

            # Create tier indicators for high/medium/low discount
            df['discount_tier_high'] = (df['DISCOUNT'] >= 0.4).astype(float)
            df['discount_tier_med'] = ((df['DISCOUNT'] >= 0.2) & (df['DISCOUNT'] < 0.4)).astype(float)
            df['discount_tier_low'] = (df['DISCOUNT'] < 0.2).astype(float)

            # Create EOSS × discount tier interactions
            df['EOSS_HIGH_DISCOUNT'] = (df['is_eoss'] * df['discount_tier_high']).astype(float)
            df['EOSS_MED_DISCOUNT'] = (df['is_eoss'] * df['discount_tier_med']).astype(float)
            df['EOSS_LOW_DISCOUNT'] = (df['is_eoss'] * df['discount_tier_low']).astype(float)

    return df

def enhance_discount_features(df):
    """
    Add enhanced discount features with explicit type conversion and robust safety measures
    for handling extreme values
    """
    if 'DISCOUNT' in df.columns:
        # Convert DISCOUNT to float first to ensure all derivations are float
        df['DISCOUNT'] = df['DISCOUNT'].astype(float)
        
        # Clean DISCOUNT values first
        df['DISCOUNT'] = df['DISCOUNT'].replace([np.inf, -np.inf], np.nan)
        df['DISCOUNT'] = df['DISCOUNT'].fillna(0)
        
        # Ensure DISCOUNT is within valid range [0, 1]
        df['DISCOUNT'] = df['DISCOUNT'].clip(0, 1)

        # Add non-linear transformations of discount with safety measures
        df['DISCOUNT_SQUARED'] = (df['DISCOUNT'] ** 2).astype(float)
        df['DISCOUNT_CUBE'] = (df['DISCOUNT'] ** 3).astype(float)
        df['DISCOUNT_SQRT'] = np.sqrt(df['DISCOUNT']).astype(float)  # Already ensured non-negative above

        # Limit extreme values for log and exp transformations
        df['DISCOUNT_LOG'] = np.log1p(df['DISCOUNT'] * 10).astype(float)

        # Cap inputs to exp to avoid overflow - using even stricter limits
        capped_discount = df['DISCOUNT'].clip(0, 0.3)  # Cap at 0.3 to avoid extreme exp values
        df['DISCOUNT_EXP'] = (np.exp(capped_discount) - 1).astype(float)

        # Add discount bins with explicit float conversion and error handling
        try:
            bins = pd.cut(df['DISCOUNT'],
                        bins=[-0.001, 0.05, 0.15, 0.25, 0.35, 1.0],
                        labels=[0, 1, 2, 3, 4])
            df['DISCOUNT_BIN'] = bins.astype(float)
        except Exception as e:
            print(f"Error creating DISCOUNT_BIN: {e}")
            # Create manual bins as fallback
            df['DISCOUNT_BIN'] = 0  # Default bin
            df.loc[(df['DISCOUNT'] >= 0.05) & (df['DISCOUNT'] < 0.15), 'DISCOUNT_BIN'] = 1
            df.loc[(df['DISCOUNT'] >= 0.15) & (df['DISCOUNT'] < 0.25), 'DISCOUNT_BIN'] = 2
            df.loc[(df['DISCOUNT'] >= 0.25) & (df['DISCOUNT'] < 0.35), 'DISCOUNT_BIN'] = 3
            df.loc[df['DISCOUNT'] >= 0.35, 'DISCOUNT_BIN'] = 4

        # Finer-grained binning with similar safety measures
        try:
            fine_bins = pd.cut(df['DISCOUNT'],
                            bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0],
                            labels=range(10))
            df['DISCOUNT_FINE_BIN'] = fine_bins.astype(float)
        except Exception as e:
            print(f"Error creating DISCOUNT_FINE_BIN: {e}")
            # Create manual bins as fallback
            df['DISCOUNT_FINE_BIN'] = 0  # Default bin
            for i, threshold in enumerate([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75]):
                if i < len(range(10))-1:  # All except the last bin
                    mask = (df['DISCOUNT'] >= threshold) & (df['DISCOUNT'] < [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0][i])
                    df.loc[mask, 'DISCOUNT_FINE_BIN'] = i+1

    return df

def add_lag_features(df, group_col='KEY'):
    """Add lag features with explicit type conversion"""
    lag_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_8', 'lag_12', 'lag_26', 'lag_52']
    for col in lag_cols:
        df[col] = np.nan

    # Add discount lag features
    if 'DISCOUNT' in df.columns:
        discount_lag_cols = ['discount_lag_1', 'discount_lag_2', 'discount_lag_4']
        discount_dynamics_cols = ['discount_change', 'discount_acceleration', 'days_since_discount_change']

        for col in discount_lag_cols + discount_dynamics_cols:
            df[col] = np.nan
    else:
        discount_lag_cols = []
        discount_dynamics_cols = []

    # Process each product separately
    for key, group in df.groupby(group_col):
        # Sort by date
        group = group.sort_values('DATE')

        # Ensure QUANTITY_SOLD is float type
        group['QUANTITY_SOLD'] = group['QUANTITY_SOLD'].astype(float)

        # Create lag features
        for lag_col in lag_cols:
            lag_value = int(lag_col.split('_')[1])
            group[lag_col] = group['QUANTITY_SOLD'].shift(lag_value)

        # Create discount lag features
        if 'DISCOUNT' in group.columns:
            # Ensure DISCOUNT is float type
            group['DISCOUNT'] = group['DISCOUNT'].astype(float)

            for lag_col in discount_lag_cols:
                lag_value = int(lag_col.split('_')[2])
                group[lag_col] = group['DISCOUNT'].shift(lag_value)

            # Discount change (first derivative)
            group['discount_change'] = group['DISCOUNT'].diff()

            # Discount acceleration (second derivative)
            group['discount_acceleration'] = group['discount_change'].diff()

            # Days since discount changed significantly
            significant_change = 0.05  # 5% discount change is significant
            change_points = (group['discount_change'].abs() >= significant_change).astype(int)

            # Count days (rows) since last change
            days_since = np.zeros(len(group))
            counter = 0
            for i in range(len(group)):
                if change_points.iloc[i] == 1:
                    counter = 0
                else:
                    counter += 1
                days_since[i] = counter

            group['days_since_discount_change'] = days_since

        # Update the dataframe
        idx = df[df[group_col] == key].index
        all_dynamic_cols = lag_cols + discount_lag_cols + discount_dynamics_cols
        for col in all_dynamic_cols:
            if col in group.columns:  # Safety check
                df.loc[idx, col] = group[col].values

    # Replace NaNs with 0 and convert to float
    all_dynamic_cols = lag_cols + discount_lag_cols + discount_dynamics_cols
    for col in all_dynamic_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)

    # Add discount × previous sales interaction
    if 'DISCOUNT' in df.columns and 'lag_1' in df.columns:
        df['DISCOUNT_PREV_SALES'] = (df['DISCOUNT'] * df['lag_1'] / (df['lag_1'].mean() + 1e-5)).astype(float)

    return df

def add_rolling_and_trend_features(df, group_col='KEY'):
    """Add rolling statistics and trend features with explicit type conversion"""
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

        # Ensure QUANTITY_SOLD is float type
        group['QUANTITY_SOLD'] = group['QUANTITY_SOLD'].astype(float)

        # Create rolling statistics
        for col in rolling_cols:
            window = int(col.split('_')[2])
            group[col] = group['QUANTITY_SOLD'].rolling(window=window, min_periods=1).mean()

        # Add trend components
        if len(group) > 1:
            # Linear trend
            group['trend'] = (np.arange(len(group)) / len(group)).astype(float)

            # Trend difference (acceleration/deceleration)
            group['trend_diff'] = group['QUANTITY_SOLD'].diff().fillna(0).astype(float)

            # Simple seasonal indicator (based on month)
            if 'month' in group.columns:
                group['seasonal_indicator'] = group['month'].map(
                    lambda m: 1.0 if m in [12, 1, 6, 7] else 0.0
                ).astype(float)
            else:
                group['seasonal_indicator'] = 0.0
        else:
            # Default values for single-point groups
            group['trend'] = 0.5
            group['trend_diff'] = 0.0
            group['seasonal_indicator'] = 0.0

        # Update the dataframe
        idx = df[df[group_col] == key].index
        all_dynamic_cols = rolling_cols + trend_cols
        for col in all_dynamic_cols:
            if col in group.columns:  # Safety check
                df.loc[idx, col] = group[col].values

    # Replace NaNs with 0 and convert to float
    all_dynamic_cols = rolling_cols + trend_cols
    for col in all_dynamic_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)

    return df

#####################################################################
# 2. CUSTOM LOSS FUNCTION FOR DISCOUNT SENSITIVITY
#####################################################################

def create_discount_sensitive_loss(underprediction_weight=1.5, overprediction_weight=1.0):
    """
    Creates a loss function that is more sensitive to discount-related errors
    """
    def discount_sensitive_loss(y_true, y_pred):
        # Basic error calculations
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Apply small epsilon
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, float('inf'))

        # Calculate error
        error = y_true - y_pred

        # Apply weights based on error direction
        weights = tf.where(error > 0,
                           tf.ones_like(error) * underprediction_weight,
                           tf.ones_like(error) * overprediction_weight)

        # Squared error with safe clipping
        squared_error = tf.square(tf.clip_by_value(error, -1e6, 1e6))

        # Apply weights
        weighted_squared_error = weights * squared_error

        return tf.reduce_mean(weighted_squared_error)

    return discount_sensitive_loss

#####################################################################
# 3. CUSTOM LAYERS FOR FEATURE EXTRACTION
#####################################################################

class DiscountFeatureExtractor(tf.keras.layers.Layer):
    """
    Custom layer to extract discount-related features by indices,
    replacing the Lambda layer which can cause loading issues.
    """
    def __init__(self, discount_indices, **kwargs):
        super(DiscountFeatureExtractor, self).__init__(**kwargs)
        self.discount_indices = discount_indices
        
    def call(self, inputs):
        # Use tf.gather to extract the discount features
        return tf.gather(inputs, self.discount_indices, axis=2)
        
    def get_config(self):
        config = super(DiscountFeatureExtractor, self).get_config()
        config.update({'discount_indices': self.discount_indices})
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

#####################################################################
# 4. TRANSFORMER BLOCK - REPLACING LSTM
#####################################################################

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    """
    Implements a Transformer Encoder block with multi-head attention,
    residual connections, and feed-forward network.
    """
    # Multi-head attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    
    # Add residual connection and normalize
    x = Add()([inputs, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network
    ffn_output = Dense(ff_dim, activation="relu")(x)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    
    # Second residual connection and normalize
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    return x

#####################################################################
# 5. IMPROVED MODEL ARCHITECTURE - SAME AS TRAINING
#####################################################################

def build_improved_model(input_shape, discount_feature_indices, params):
    """
    Build an improved model with:
    1. Custom layer instead of Lambda for discount feature extraction
    2. Transformer blocks instead of LSTM
    3. Additional residual connections for better gradient flow
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial normalization
    normalized = BatchNormalization()(inputs)
    
    # Extract discount features using custom layer
    discount_features = DiscountFeatureExtractor(
        discount_feature_indices, name="discount_feature_extractor"
    )(normalized)
    
    # Process discount features
    discount_branch = Conv1D(
        filters=params['discount_filters'],
        kernel_size=params['discount_kernel'],
        padding='same',
        activation='relu'
    )(discount_features)
    
    discount_branch = BatchNormalization()(discount_branch)
    
    # Add attention for discount features
    discount_branch = MultiHeadAttention(
        num_heads=params['discount_heads'],
        key_dim=params['discount_filters'] // params['discount_heads'],
        dropout=params['attn_dropout']
    )(discount_branch, discount_branch)
    
    discount_branch = LayerNormalization(epsilon=1e-6)(discount_branch)
    discount_branch = GlobalAveragePooling1D()(discount_branch)
    discount_output = Dense(params['discount_dense'], activation='relu')(discount_branch)
    
    # CNN branch with residual connections
    if params['use_cnn']:
        cnn1 = Conv1D(
            filters=params['cnn_filters'],
            kernel_size=params['cnn_kernel'],
            padding='same',
            activation='relu'
        )(normalized)
        
        cnn2 = Conv1D(
            filters=params['cnn_filters'],
            kernel_size=params['cnn_kernel'] + 2,
            padding='same',
            activation='relu'
        )(normalized)
        
        # Combine CNN outputs
        cnn = Concatenate()([cnn1, cnn2])
        cnn = BatchNormalization()(cnn)
        
        # Add residual connection (transform input to match cnn shape if needed)
        if normalized.shape[-1] != cnn.shape[-1]:
            normalized_projected = Conv1D(
                filters=cnn.shape[-1],
                kernel_size=1,
                padding='same'
            )(normalized)
            cnn = Add()([cnn, normalized_projected])
        else:
            cnn = Add()([cnn, normalized])
            
        cnn = MaxPooling1D(pool_size=2)(cnn)
        
        # Apply additional transformer encoder blocks
        for _ in range(params['cnn_transformer_blocks']):
            cnn = transformer_encoder_block(
                cnn,
                head_size=params['head_size'],
                num_heads=params['num_heads'],
                ff_dim=params['ff_dim'],
                dropout=params['dropout']
            )
            
        cnn_output = GlobalAveragePooling1D()(cnn)
    else:
        cnn_output = None
    
    # Transformer blocks branch (replacing LSTM)
    if params['use_transformer']:
        # Initial projection to ensure compatible dimensions
        transformer_input = Conv1D(
            filters=params['transformer_dim'],
            kernel_size=1,
            padding='same'
        )(normalized)
        
        # Stack transformer blocks
        transformer_output = transformer_input
        for _ in range(params['transformer_blocks']):
            transformer_output = transformer_encoder_block(
                transformer_output,
                head_size=params['head_size'],
                num_heads=params['num_heads'],
                ff_dim=params['ff_dim'],
                dropout=params['dropout']
            )
            
        # Global pooling to get fixed-size output
        transformer_pooled = GlobalAveragePooling1D()(transformer_output)
    else:
        transformer_pooled = None
    
    # Dense branch (always used)
    dense_branch = Dense(params['dense_units1'], activation='relu')(normalized)
    dense_branch = BatchNormalization()(dense_branch)
    dense_branch = Dropout(params['dropout'])(dense_branch)
    
    # Add residual connection if shapes match
    if normalized.shape[-1] == params['dense_units1']:
        residual_projection = Dense(params['dense_units1'])(normalized)
        dense_branch = Add()([dense_branch, residual_projection])
        
    dense_output = GlobalAveragePooling1D()(dense_branch)
    
    # Combine all branches
    to_combine = []
    if cnn_output is not None:
        to_combine.append(cnn_output)
    if transformer_pooled is not None:
        to_combine.append(transformer_pooled)
    to_combine.extend([dense_output, discount_output])
    
    combined = Concatenate()(to_combine)
    
    # Create a gating mechanism where discount information modulates the combined features
    discount_gate = Dense(combined.shape[-1], activation='sigmoid')(discount_output)
    gated_combined = Multiply()([combined, discount_gate])
    
    # Dense layers with residual connections
    x = Dense(params['dense_units2'], activation='relu')(gated_combined)
    x = BatchNormalization()(x)
    x = Dropout(params['dropout'])(x)
    
    # Add residual connection with projection if needed
    if gated_combined.shape[-1] != params['dense_units2']:
        projected_residual = Dense(params['dense_units2'])(gated_combined)
        x = Add()([x, projected_residual])
    else:
        x = Add()([x, gated_combined])
    
    x = Dense(params['dense_units3'], activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Add another residual connection with projection
    projected_for_res = Dense(params['dense_units3'])(combined)
    x = Add()([x, projected_for_res])
    
    # Final dense layer for prediction
    output = Dense(1, activation='relu')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=output)
    
    # Custom loss function
    loss_fn = create_discount_sensitive_loss(
        underprediction_weight=params['underpred_weight'],
        overprediction_weight=params['overpred_weight']
    )
    
    # Compile with custom loss and gradient clipping
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params['learning_rate'],
            clipnorm=params['clipnorm']
        ),
        loss=loss_fn,
        metrics=['mae']
    )
    
    return model

#####################################################################
# 6. EVALUATION FUNCTION
#####################################################################

def evaluate_forecast(
    model_dir,
    train_data_path,
    test_data_path,
    output_dir='evaluation_results',
    sequence_length=52,
    batch_size=64
):
    """
    Evaluates the discount-sensitive forecasting model on test data.
    
    Parameters:
    -----------
    model_dir : str
        Directory containing model files (weights, parameters, etc.)
    train_data_path : str
        Path to training data CSV
    test_data_path : str
        Path to test data CSV
    output_dir : str
        Directory to save evaluation results
    sequence_length : int
        Sequence length used for predictions
    batch_size : int
        Batch size for model prediction
    
    Returns:
    --------
    pd.DataFrame
        Forecast comparison results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n===== Loading Model Resources =====")

    # Load feature columns
    feature_cols_path = os.path.join(model_dir, 'feature_cols.pkl')
    if os.path.exists(feature_cols_path):
        try:
            feature_cols = joblib.load(feature_cols_path)
            print(f"Loaded {len(feature_cols)} feature columns")
        except Exception as e:
            print(f"Error loading feature columns: {e}")
            return None
    else:
        print(f"Feature columns file not found at {feature_cols_path}")
        return None

    # Load discount feature indices
    disc_indices_path = os.path.join(model_dir, 'discount_feature_indices.pkl')
    if os.path.exists(disc_indices_path):
        try:
            discount_feature_indices = joblib.load(disc_indices_path)
            print(f"Loaded {len(discount_feature_indices)} discount feature indices")
        except Exception as e:
            print(f"Error loading discount feature indices: {e}")
            return None
    else:
        print(f"Discount feature indices file not found at {disc_indices_path}")
        return None

    # Load best parameters
    params_path = os.path.join(model_dir, 'best_params.pkl')
    if os.path.exists(params_path):
        try:
            best_params = joblib.load(params_path)
            print("Loaded model parameters")
        except Exception as e:
            print(f"Error loading best parameters: {e}")
            return None
    else:
        print(f"Parameters file not found at {params_path}")
        return None

    # Load feature scalers
    scalers_path = os.path.join(model_dir, 'scalers.pkl')
    if os.path.exists(scalers_path):
        try:
            scalers = joblib.load(scalers_path)
            print("Loaded feature scalers")
        except Exception as e:
            print(f"Error loading scalers: {e}")
            scalers = None
    else:
        print("Warning: Scalers file not found, proceeding without scaling")
        scalers = None

    # Optional: Load calibration strategies
    calibration_path = os.path.join(model_dir, 'discount_calibration_strategies.pkl')
    if os.path.exists(calibration_path):
        try:
            calibration_strategies = joblib.load(calibration_path)
            print("Loaded calibration strategies")
        except Exception as e:
            print(f"Warning: Error loading calibration strategies: {e}")
            calibration_strategies = {}
    else:
        print("No calibration file found, proceeding without calibration")
        calibration_strategies = {}

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
    
    # Define input shape
    input_shape = (sequence_length, len(feature_cols))
    print(f"Model input shape: {input_shape}")
    
    # Build model with same architecture as training
    print("\n===== Building Model Architecture =====")
    model = build_improved_model(
        input_shape=input_shape,
        discount_feature_indices=discount_feature_indices,
        params=best_params
    )
    
    # Load weights - try several options to be robust
    print("\n===== Loading Model Weights =====")
    # First try: direct weights file
    weights_path = os.path.join(model_dir, 'final_model.weights.h5')
    weights_loaded = False
    
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded weights from {weights_path}")
            weights_loaded = True
        except Exception as e:
            print(f"Error loading weights from {weights_path}: {e}")
    
    # Second try: full model file
    if not weights_loaded:
        model_path = os.path.join(model_dir, 'final_model.h5')
        if os.path.exists(model_path):
            try:
                # Define custom objects for loading
                custom_objects = {
                    'DiscountFeatureExtractor': DiscountFeatureExtractor,
                    'discount_sensitive_loss': create_discount_sensitive_loss()
                }
                
                # Load full model
                loaded_model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects=custom_objects,
                    compile=False
                )
                
                # Copy weights to our model
                model.set_weights(loaded_model.get_weights())
                print(f"Successfully loaded model from {model_path}")
                weights_loaded = True
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
    
    # Third try: best model from checkpoints
    if not weights_loaded:
        best_model_path = os.path.join(model_dir, 'best_model.h5')
        if os.path.exists(best_model_path):
            try:
                # Define custom objects for loading
                custom_objects = {
                    'DiscountFeatureExtractor': DiscountFeatureExtractor,
                    'discount_sensitive_loss': create_discount_sensitive_loss()
                }
                
                # Load full model
                loaded_model = tf.keras.models.load_model(
                    best_model_path, 
                    custom_objects=custom_objects,
                    compile=False
                )
                
                # Copy weights to our model
                model.set_weights(loaded_model.get_weights())
                print(f"Successfully loaded model from {best_model_path}")
                weights_loaded = True
            except Exception as e:
                print(f"Error loading model from {best_model_path}: {e}")
    
    if not weights_loaded:
        print("WARNING: Could not load any model weights. Using untrained model.")
        print("Evaluation will not be meaningful without trained weights.")
        return None
    
    # Track results
    results = []
    
    # Get test product keys
    test_keys = df_test['KEY'].unique()
    print(f"\n===== Evaluating {len(test_keys)} Products =====")
    
    # Function to apply discount calibration
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
    
    # Function to scale features consistently
    def apply_safe_scaling(df, feature_cols, scalers):
        """Apply scaling to features with safety measures"""
        if scalers is None:
            return df
            
        scaled_df = df.copy()
        
        # Handle discount features (using MinMaxScaler)
        if 'discount' in scalers:
            discount_scaler = scalers['discount']
            discount_features = [col for col in feature_cols if 'discount' in col.lower() or 'DISCOUNT' in col]
            discount_features = [col for col in discount_features if col in scaled_df.columns]
            
            if discount_features:
                # Clean and prepare data
                for col in discount_features:
                    scaled_df[col] = scaled_df[col].replace([np.inf, -np.inf], np.nan)
                    scaled_df[col] = scaled_df[col].fillna(0)
                    scaled_df[col] = scaled_df[col].astype(float)
                
                try:
                    # Apply scaling
                    scaled_data = discount_scaler.transform(scaled_df[discount_features])
                    for i, col in enumerate(discount_features):
                        scaled_df[col] = scaled_data[:, i]
                except Exception as e:
                    print(f"Warning: Error scaling discount features: {e}")
        
        # Handle other continuous features (using StandardScaler)
        if 'other' in scalers:
            other_scaler = scalers['other']
            categorical_features = scalers.get('categorical', [])
            
            # Identify continuous features (not discount, not categorical)
            discount_features = [col for col in feature_cols if 'discount' in col.lower() or 'DISCOUNT' in col]
            continuous_features = [col for col in feature_cols 
                                  if col not in discount_features 
                                  and col not in categorical_features
                                  and col in scaled_df.columns]
            
            if continuous_features:
                # Clean and prepare data
                for col in continuous_features:
                    scaled_df[col] = scaled_df[col].replace([np.inf, -np.inf], np.nan)
                    scaled_df[col] = scaled_df[col].fillna(0)
                    scaled_df[col] = scaled_df[col].astype(float)
                
                try:
                    # Apply scaling
                    scaled_data = other_scaler.transform(scaled_df[continuous_features])
                    for i, col in enumerate(continuous_features):
                        scaled_df[col] = scaled_data[:, i]
                except Exception as e:
                    print(f"Warning: Error scaling continuous features: {e}")
        
        return scaled_df
    
    # Process each product
    for i, key in enumerate(test_keys):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(test_keys)} products")
        
        # Get test data for this product
        test_data_key = df_test[df_test['KEY'] == key].copy()
        
        # Skip if no test data
        if len(test_data_key) == 0:
            continue
        
        # Get training data for this product
        train_data_key = df_train[df_train['KEY'] == key].copy()
        
        # Skip if not enough training data
        if len(train_data_key) < sequence_length:
            continue
        
        # Concatenate train and test for feature engineering
        combined = pd.concat([train_data_key, test_data_key], ignore_index=True)
        combined.sort_values('DATE', inplace=True)
        
        try:
            # Apply all feature engineering steps from training
            combined = add_time_features(combined)
            combined = enhance_discount_features(combined)
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
                combined[col] = combined[col].astype(float)
            
            # Apply scaling if scalers are available
            combined = apply_safe_scaling(combined, feature_cols, scalers)
            
            # Process each test date for this product
            for _, test_row in test_data_key.iterrows():
                test_date = test_row['DATE']
                
                # Get historical data before test date
                hist_data = combined[combined['DATE'] < test_date].copy()
                
                # Skip if insufficient history
                if len(hist_data) < sequence_length:
                    continue
                
                # Get the most recent sequence
                sequence_data = hist_data.iloc[-sequence_length:][feature_cols].values
                
                # Check sequence shape
                if sequence_data.shape != (sequence_length, len(feature_cols)):
                    print(f"Warning: Shape mismatch for {key} - Expected {(sequence_length, len(feature_cols))}, got {sequence_data.shape}")
                    continue
                
                # Prepare input for model prediction
                model_input = np.expand_dims(sequence_data, axis=0)
                
                # Get actual values from test data
                actual_sales = float(test_row['QUANTITY_SOLD'])
                discount = float(test_row.get('DISCOUNT', 0))
                is_eoss = float(test_row.get('is_eoss', 0)) if 'is_eoss' in test_row else 0
                
                try:
                    # Make prediction
                    raw_pred = float(model.predict(model_input, verbose=0, batch_size=batch_size)[0][0])
                    
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

#####################################################################
# 7. MAIN FUNCTION TO RUN EVALUATION
#####################################################################

def run_evaluation(model_dir, train_data_path, test_data_path, output_dir='evaluation_results'):
    """
    Main function to run the evaluation pipeline
    
    Parameters:
    -----------
    model_dir : str
        Directory containing model files
    train_data_path : str
        Path to training data CSV
    test_data_path : str
        Path to test data CSV
    output_dir : str
        Directory to save evaluation results
    """
    print("\n===============================================")
    print("TRANSFORMER-BASED DISCOUNT-SENSITIVE FORECASTING MODEL EVALUATION")
    print("===============================================\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Log evaluation start time
    start_time = datetime.now()
    print(f"Evaluation started at: {start_time}")
    
    # Run evaluation
    results = evaluate_forecast(
        model_dir=model_dir,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        output_dir=output_dir,
        sequence_length=52,
        batch_size=256  # Increased batch size for faster evaluation
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

# Example usage
if __name__ == "__main__":
    results = run_evaluation(
        model_dir='Neural_net',
        train_data_path='data_files/df_forecasting_test.csv',
        test_data_path='data_files/df_forecasting_test.csv',
        output_dir='model_evaluation_results'
    )