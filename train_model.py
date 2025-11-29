import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from skmultilearn.adapt import MLkNN
import sklearn
from sklearn.neighbors import NearestNeighbors

# Fix MLkNN compatibility with newer scikit-learn
if sklearn.__version__ >= "1.2.0":
    original_init = NearestNeighbors.__init__
    def fixed_init(self, n_neighbors=None, **kwargs):
        if n_neighbors is not None and 'n_neighbors' not in kwargs:
            kwargs['n_neighbors'] = n_neighbors
        elif n_neighbors is None and 'n_neighbors' not in kwargs:
            kwargs['n_neighbors'] = 5  # default value
        original_init(self, **kwargs)
    NearestNeighbors.__init__ = fixed_init

def train_and_save_model():
    print("🚀 Starting Meta-Learning Model Training...")
    
    # Load training data
    print("📥 Loading training data...")
    X_train = pd.read_csv("data/meta_features_final.csv", index_col=0)
    y_train = pd.read_csv("data/top3_labels_cc18_core_clean.csv", index_col=0)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Handle infinite values
    print("🔧 Preprocessing data...")
    inf_count = np.isinf(X_train.values).sum()
    print(f"Found {inf_count} infinite values")
    
    if inf_count > 0:
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        
        # Handle extremely large values
        large_threshold = 1e10
        large_mask = X_train > large_threshold
        if large_mask.sum().sum() > 0:
            print(f"Replacing {large_mask.sum().sum()} extremely large values (>1e10) with NaN...")
            X_train[large_mask] = np.nan
    
    # Fill missing values first (before feature selection)
    X_filled = X_train.fillna(X_train.mean())
    
    # Feature Selection - Select top 20 features
    print("🎯 Performing feature selection...")
    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X_filled, y_train.values.argmax(axis=1))
    selected_features = X_filled.columns[selector.get_support()].tolist()
    
    print(f"Selected top {len(selected_features)} features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
    
    # Create imputer for the selected features only
    print("🔄 Creating imputer for selected features...")
    imputer = SimpleImputer(strategy="mean")
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    X_imputed = imputer.fit_transform(X_selected_df)
    
    # Train MLkNN model
    print("🤖 Training MLkNN model...")
    clf = MLkNN(k=3, s=1.0)
    clf.fit(X_imputed, y_train.values)
    
    print("✅ Model training completed!")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save model components
    print("💾 Saving model files...")
    
    joblib.dump(clf, data_dir / "mlknn_model.pkl")
    print("✓ Saved mlknn_model.pkl")
    
    joblib.dump(selected_features, data_dir / "selected_features.pkl")
    print("✓ Saved selected_features.pkl")
    
    joblib.dump(imputer, data_dir / "imputer.pkl")
    print("✓ Saved imputer.pkl")
    
    print("\n🎉 All model files saved successfully!")
    print(f"📁 Model files location: {data_dir.absolute()}")
    
    # Verify files exist
    model_path = data_dir / "mlknn_model.pkl"
    features_path = data_dir / "selected_features.pkl"
    imputer_path = data_dir / "imputer.pkl"
    
    if model_path.exists() and features_path.exists() and imputer_path.exists():
        print("✅ All required files created successfully!")
        return True
    else:
        print("❌ Some files failed to save!")
        return False

if __name__ == "__main__":
    success = train_and_save_model()
    if success:
        print("\n🚀 Your meta-learning recommender is now ready!")
        print("Run your FastAPI backend and test with: python -m backend.api.main")
    else:
        print("\n❌ Training failed. Check the error messages above.")