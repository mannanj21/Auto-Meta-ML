import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset(df, target_name):
    """Preprocess dataset for meta-feature extraction"""
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Separate features and target
    X = df.drop(columns=[target_name])
    y = df[target_name]
    
    # Encode categorical target
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        y = y.astype('category').cat.codes
    
    # Encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Impute missing values
    X = X.fillna(X.median(numeric_only=True))
    
    return X, y
