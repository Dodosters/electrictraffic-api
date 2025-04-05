from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import json
import pickle
from datetime import datetime
import os
import warnings
import uvicorn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Time Series Forecasting API",
    description="API for forecasting time series data using an ensemble model",
    version="1.0.0"
)

# Models directory
MODELS_DIR = "models"
DATA_DIR = "data"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Define request and response models
class ForecastAndTrainRequest(BaseModel):
    months_ahead: int = Field(..., description="Number of months to forecast", ge=1, le=24)

class ForecastResponse(BaseModel):
    forecast: List[Dict[str, Any]]

# Function to load and prepare data (adapted from original code)
def load_and_prepare_data(df):
    """Prepare dataframe for time series analysis"""
    
    # Check and fix zero values
    zero_count = (df['volume'] == 0).sum()
    if zero_count > 0:
        df.loc[df['volume'] == 0, 'volume'] = np.nan
        print(f"Found {zero_count} zero values in volume, replaced with NaN")
    
    # Create season (based on month)
    df['season'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else  # Winter
                                           2 if x in [3, 4, 5] else      # Spring
                                           3 if x in [6, 7, 8] else 4)   # Summer, otherwise Fall
    
    # Fill any missing values
    df['volume'] = df['volume'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    # Create quarter
    df['quarter'] = df['month'].apply(lambda x: 1 if x in [1, 2, 3] else 
                                           2 if x in [4, 5, 6] else
                                           3 if x in [7, 8, 9] else 4)
    
    # Sort by time
    df = df.sort_values(by=['year', 'month']).reset_index(drop=True)
    
    # Create lags
    df['volume_lag_1'] = df['volume'].shift(1)
    df['volume_lag_3'] = df['volume'].shift(3)
    
    # Create moving averages
    df['volume_ma_3'] = df['volume'].rolling(window=3, min_periods=1).mean()
    
    # Create exponential smoothing
    df['volume_exp_smoothed'] = df['volume'].ewm(span=3).mean()
    
    # Fill NaN values in lag/MA columns
    for col in df.columns:
        if df[col].isnull().any():
            # Use season means for imputation first
            season_means = df.groupby('season')[col].transform('mean')
            df[col] = df[col].fillna(season_means)
            
            # Then use overall mean for any remaining NAs
            df[col] = df[col].fillna(df[col].mean())
            
            # If all were NaN, fill with zeros
            df[col] = df[col].fillna(0)
    
    # Create date column for visualization
    df['date'] = df.apply(lambda row: f"{int(row['year'])}-{int(row['month']):02d}", axis=1)
    
    return df

# Function to train all component models that make up the ensemble
def train_ensemble_model(df):
    """Train all models needed for the ensemble"""
    # Use all data for training - no test split
    X = df.drop(['volume', 'date', 'month', 'year'], axis=1, errors='ignore')
    y = df['volume']
    
    # Dictionary to store trained models
    models = {}
    
    # Linear Models
    linear_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=0.1),
        'Lasso Regression': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5)
    }
    
    for name, model in linear_models.items():
        model.fit(X, y)
        models[name] = model
    
    # Tree-based Models
    tree_models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42),
        'XGBoost': XGBRegressor(
            n_estimators=20,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42
        )
    }
    
    for name, model in tree_models.items():
        model.fit(X, y)
        models[name] = model
    
    # Time Series Models
    # We'll need the raw data for Holt-Winters, not the feature matrix
    if len(y) >= 4:  # Need at least 4 observations for exponential smoothing
        try:
            # Simple exponential smoothing
            hw_model = ExponentialSmoothing(
                y, 
                trend='add',
                seasonal='add',
                seasonal_periods=min(12, len(y)-1)
            ).fit()
            
            models['Holt-Winters'] = hw_model
        except Exception as e:
            print(f"Error with Holt-Winters model: {e}")
    
    return models, X.columns.tolist()  # Return models and feature names

# Function to make ensemble prediction
def predict_ensemble(models, feature_names, future_df):
    """Make predictions using ensemble of models"""
    predictions = {}
    
    # Ensure future_df has the right features
    X_future = future_df[feature_names].values
    
    # Make predictions with each model (except Holt-Winters)
    for name, model in models.items():
        if name != 'Holt-Winters':
            try:
                predictions[name] = model.predict(X_future)
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                # Skip this model in the ensemble
                continue
    
    # Make prediction with Holt-Winters if available
    if 'Holt-Winters' in models:
        try:
            hw_forecast = models['Holt-Winters'].forecast(len(future_df))
            predictions['Holt-Winters'] = hw_forecast.values
        except Exception as e:
            print(f"Error predicting with Holt-Winters: {e}")
    
    # Create ensemble prediction by averaging all available model predictions
    all_preds = []
    for model_name, preds in predictions.items():
        all_preds.append(preds)
    
    if all_preds:
        ensemble_pred = np.mean(np.array(all_preds), axis=0)
        return ensemble_pred
    else:
        raise Exception("No models were able to make predictions")

# Function to generate future dataframe
def generate_future_dataframe(df, months_ahead):
    """Generate a dataframe for future months to forecast"""
    # Get last date in dataset
    last_year = int(df['year'].iloc[-1])
    last_month = int(df['month'].iloc[-1])
    
    # Create future dates
    future_dates = []
    for i in range(months_ahead):
        future_month = (last_month + i + 1) % 12
        if future_month == 0:
            future_month = 12
        future_year = last_year + (last_month + i + 1 - 1) // 12
        future_dates.append((future_year, future_month))
    
    # Create base dataframe
    future_df = pd.DataFrame({
        'year': [date[0] for date in future_dates],
        'month': [date[1] for date in future_dates]
    })
    
    # Add derived features
    future_df['season'] = future_df['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else  # Winter
               2 if x in [3, 4, 5] else      # Spring
               3 if x in [6, 7, 8] else 4     # Summer, otherwise Fall
    )
    
    future_df['quarter'] = future_df['month'].apply(
        lambda x: 1 if x in [1, 2, 3] else 
               2 if x in [4, 5, 6] else
               3 if x in [7, 8, 9] else 4
    )
    
    # Create date column
    future_df['date'] = future_df.apply(lambda row: f"{int(row['year'])}-{int(row['month']):02d}", axis=1)
    
    # Initialize lagged features with values from the original dataframe
    if len(df) > 0:
        # Get last volume values for lagged features
        last_volume = df['volume'].iloc[-1]
        last_volume_lag_3 = df['volume'].iloc[-3] if len(df) > 3 else df['volume'].mean()
        
        # Create initial values for the first row of future dataframe
        future_df.loc[0, 'volume_lag_1'] = last_volume
        future_df.loc[0, 'volume_lag_3'] = last_volume_lag_3
        future_df.loc[0, 'volume_ma_3'] = df['volume'].iloc[-3:].mean() if len(df) >= 3 else df['volume'].mean()
        future_df.loc[0, 'volume_exp_smoothed'] = df['volume_exp_smoothed'].iloc[-1] if 'volume_exp_smoothed' in df.columns else df['volume'].mean()
    
    return future_df

# Function to recursively forecast multiple months ahead
def forecast_recursive(models, feature_names, df, months_ahead):
    """Make a recursive forecast for multiple months ahead"""
    # Clone the original dataframe to avoid modifying it
    df_copy = df.copy()
    
    # Create future dataframe
    future_df = generate_future_dataframe(df_copy, months_ahead)
    
    # Make predictions recursively
    predictions = []
    
    for i in range(months_ahead):
        # For the first month, use data from original dataframe
        if i == 0:
            # Populate lagged features from original data
            current_df = future_df.iloc[[0]].copy()
            
            # Make prediction
            pred = predict_ensemble(models, feature_names, current_df)
            future_df.loc[0, 'volume'] = pred[0]
            
            # Store prediction
            predictions.append({
                'date': future_df['date'].iloc[0],
                'year': int(future_df['year'].iloc[0]),
                'month': int(future_df['month'].iloc[0]),
                'volume': float(pred[0])
            })
        else:
            # Update lag features based on previous predictions
            if i >= 1:
                future_df.loc[i, 'volume_lag_1'] = future_df.loc[i-1, 'volume']
            
            if i >= 3:
                future_df.loc[i, 'volume_lag_3'] = future_df.loc[i-3, 'volume']
            else:
                # Use values from original dataframe for the early predictions
                idx = -3 + i
                if idx < 0 and abs(idx) <= len(df_copy):
                    future_df.loc[i, 'volume_lag_3'] = df_copy['volume'].iloc[idx]
                else:
                    # If we don't have enough history, use the mean
                    future_df.loc[i, 'volume_lag_3'] = df_copy['volume'].mean()
            
            # Update moving average
            if i >= 3:
                future_df.loc[i, 'volume_ma_3'] = future_df.loc[i-3:i, 'volume'].mean()
            elif i > 0:
                # Use a mix of original and predicted values
                last_n = min(3, len(df_copy) + i)
                values = list(df_copy['volume'].iloc[-last_n+i:].values) + list(future_df.loc[:i-1, 'volume'].values)
                future_df.loc[i, 'volume_ma_3'] = np.mean(values[-3:])
            
            # Update exponential smoothing
            if i > 0:
                alpha = 0.5  # Smoothing factor
                future_df.loc[i, 'volume_exp_smoothed'] = alpha * future_df.loc[i-1, 'volume'] + (1-alpha) * future_df.loc[i-1, 'volume_exp_smoothed']
            
            # Make prediction
            current_df = future_df.iloc[[i]].copy()
            pred = predict_ensemble(models, feature_names, current_df)
            future_df.loc[i, 'volume'] = pred[0]
            
            # Store prediction
            predictions.append({
                'date': future_df['date'].iloc[i],
                'year': int(future_df['year'].iloc[i]),
                'month': int(future_df['month'].iloc[i]),
                'volume': float(pred[0])
            })
    
    return predictions, future_df

# File management functions - now optional since we do everything in memory
# Keeping these functions in case they're needed for future expansion
def save_dataframe(file_id, df):
    """Save dataframe to disk"""
    file_path = os.path.join(DATA_DIR, f"{file_id}.csv")
    df.to_csv(file_path, index=False)
    return file_path

def load_dataframe(file_id):
    """Load dataframe from disk"""
    file_path = os.path.join(DATA_DIR, f"{file_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
    return pd.read_csv(file_path)

def save_model(file_id, models, feature_names):
    """Save trained models to disk"""
    model_path = os.path.join(MODELS_DIR, f"{file_id}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'models': models, 'feature_names': feature_names}, f)
    return model_path

def load_model(file_id):
    """Load trained models from disk"""
    model_path = os.path.join(MODELS_DIR, f"{file_id}_model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model with ID {file_id} not found")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['models'], model_data['feature_names']

# API endpoint for complete workflow
@app.post("/forecast/", response_model=ForecastResponse)
async def upload_train_forecast(months_ahead: int, file: UploadFile = File(...)):
    """Upload data, train model, and generate forecast in one request"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    if months_ahead < 1 or months_ahead > 24:
        raise HTTPException(status_code=400, detail="months_ahead must be between 1 and 24")
    
    try:
        # Read CSV content
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(buffer)
        
        # Check required columns
        required_cols = ['year', 'month', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain the following columns: {', '.join(required_cols)}"
            )
        
        # Check if data is valid for forecasting
        if len(df) < 3:
            raise HTTPException(
                status_code=400, 
                detail="Data must contain at least 3 data points for forecasting"
            )
        
        # Prepare the data
        df_prepared = load_and_prepare_data(df)
        
        # Train the model
        models, feature_names = train_ensemble_model(df_prepared)
        
        # Generate forecast
        forecast_results, future_df = forecast_recursive(
            models, 
            feature_names, 
            df_prepared, 
            months_ahead
        )
        
        # Prepare response
        response = {
            "forecast": forecast_results
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("prediction_api:app", host="0.0.0.0", port=8000, reload=True)