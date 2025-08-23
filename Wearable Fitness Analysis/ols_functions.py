import os
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import kaggle
import kagglehub
from kagglehub import KaggleDatasetAdapter
import math

def ols_model(x, y, return_details=False):
    """
    Creates and evaluates an OLS linear regression model.
    
    Parameters:
    - x: Feature matrix
    - y: Target variable
    - return_details: If True, returns detailed results; if False, returns test R²
    
    Returns:
    - If return_details=False: Test R² score
    - If return_details=True: Dictionary with train R², test R², and model
    """
    # Train/test split (80:20) + center the dataset
    # Prepare data
    
    encoded_features = []
    categorical_cols = []
    if isinstance(x, pd.DataFrame):
        # print("x is a Dataframe")
        x_model = x.copy().dropna()
        already_encoded = any(col.endswith('_encoded') for col in x_model.columns)
    
        if already_encoded:
            print("Data appears to be already encoded, using as-is")
            X_processed = x_model
    
        # Encode categorical variables
        else:
            for feature in x_model.columns:
                if x_model[feature].dtype == 'object':
                    le = LabelEncoder()
                    x_model[f'{feature}_encoded'] = le.fit_transform(x_model[feature].astype(str))
                    encoded_features.append(f'{feature}_encoded')
                    categorical_cols.append(feature)
                else:
                    encoded_features.append(feature)
        # Drop original categorical columns and use encoded ones
        #display(x_model.head())
        X_processed = x_model.drop(columns=categorical_cols) if categorical_cols else x_model
    else: # x is a series
        # print("x is a Series")
    
        # Handle Series case
        if x.dtype == 'object':
            print("Series contains categorical data, encoding...")
            le = LabelEncoder()
            x_clean = x.dropna()
            X_processed = pd.DataFrame({
                f'{x_clean.name}_encoded' if x_clean.name else 'feature_encoded': le.fit_transform(x_clean.astype(str))
            }, index=x_clean.index)        
        elif isinstance(x, pd.Series):
            print("Series contains numerical data, using as-is")
            x_clean = x.dropna()
            X_processed = pd.DataFrame({
                x_clean.name if x_clean.name else 'feature': x_clean.values
            }, index=x_clean.index)
        else:
            raise TypeError("x must be a DataFrame or Series.")
    
        print(f"Converted Series to DataFrame with shape {X_processed.dropna().shape}")

    X_processed = X_processed.dropna()
    X_train, X_test, y_train, y_test = train_test_split(StandardScaler(with_std=False).fit_transform(X_processed), 
                                                        y.loc[X_processed.index], 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    # Train model
    model = LinearRegression().fit(X_train, y_train) # we are fitting the OLS model on the training data
    
    # Calculate scores (R2)
    train_score = model.score(X_train, y_train) # calculate the R2 score of the training data
    test_score = model.score(X_test, y_test) # calculate the R2 score of the test data
    
    if return_details:
        return {
            'model': model,
            'train_r2': train_score,
            'test_r2': test_score,
            'train_rmse': np.sqrt(mean_squared_error(y_train, model.predict(X_train))),
            'test_rmse': np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        }
    else:
        return test_score

def remove_correlated_variables(df, features, target, correlation_threshold=0.8):
    """
    Remove variables that are highly correlated with each other
    """
    print(f"REMOVE HIGHLY CORRELATED VARIABLES")
    print("="*60)
    
    # Prepare data
    df_work = df[features + [target]].dropna()
    df_encoded = df_work.copy()
    
    # Encode categorical variables
    encoded_features = []
    for feature in features:
        if df[feature].dtype == 'object':
            le = LabelEncoder()
            df_encoded[f'{feature}_encoded'] = le.fit_transform(df_work[feature].astype(str))
            encoded_features.append(f'{feature}_encoded')
        else:
            encoded_features.append(feature)
    
    # Calculate correlations
    correlation_matrix = df_encoded[encoded_features].corr()
    sns.heatmap(data=correlation_matrix, cmap='RdYlGn', linecolor='gray', linewidths=0.5)   
    plt.show()
    # Find variables to remove
    to_remove = set()
    removed_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            
            if abs(corr_value) > correlation_threshold:
                # Keep the variable with higher individual R² with target
                r2_var1 = LinearRegression().fit(df_encoded[[var1]], df_encoded[target]).score(df_encoded[[var1]], df_encoded[target])
                r2_var2 = LinearRegression().fit(df_encoded[[var2]], df_encoded[target]).score(df_encoded[[var2]], df_encoded[target])
                
                if r2_var1 >= r2_var2:
                    to_remove.add(var2)
                    removed_pairs.append(f"Removed {var2.replace('_encoded', '')} (kept {var1.replace('_encoded', '')}) - correlation: {corr_value:.3f}")
                else:
                    to_remove.add(var1)
                    removed_pairs.append(f"Removed {var1.replace('_encoded', '')} (kept {var2.replace('_encoded', '')}) - correlation: {corr_value:.3f}")
    
    # Final feature set
    final_features = [f for f in encoded_features if f not in to_remove]
    
    print(f"Correlation threshold: {correlation_threshold}")
    print(f"Variables removed: {len(to_remove)}")
    print(f"Variables remaining: {len(final_features)}")
    
    if removed_pairs:
        print(f"\nRemoved pairs:")
        for pair in removed_pairs:
            print(f"   {pair}")
    
    print(f"\nFinal feature set:")
    for feature in final_features:
        print(f"   - {feature.replace('_encoded', '')}")
    
    # Test performance
    X = df_encoded[final_features]
    y = df_encoded[target]
    
    r2_after_removal = ols_model(X, y, return_details=True)
    
    print(f"\nModel performance after removing correlated variables:")
    for key, value in r2_after_removal.items():
        print(f"{key}: {value}")
    
    return final_features, r2_after_removal