import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from core.config import settings
from core.meta_features import extract_meta_features

class MetaLearningPredictor:
    def __init__(self):
        self.model = None
        self.imputer = None
        self.selected_features = None
        self.algorithm_names = [
            "Naive Bayes",
            "Reduced Error Pruning Tree (REPTree)", 
            "Bayesian Network (K2)",
            "Decision Tree (J48/C4.5)",
            "Logistic Regression",
            "PART Rule Learner",
            "Random Forest",
            "Decision Stump"
        ]
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            if settings.MODEL_PATH.exists():
                self.model = joblib.load(settings.MODEL_PATH)
                self.selected_features = joblib.load(settings.FEATURES_PATH)
                self.imputer = joblib.load(settings.IMPUTER_PATH)
                print("✓ Models loaded successfully")
            else:
                print("⚠ Warning: Pre-trained models not found. Using fallback mode.")
        except Exception as e:
            print(f"⚠ Error loading models: {e}")
    
    def is_loaded(self):
        return self.model is not None
    
    def predict(self, csv_path: str):
        """
        Predict best algorithms for a dataset
        Returns list of recommendations
        """
        # Extract meta-features
        meta_features, n_rows, n_cols = extract_meta_features(csv_path)
        
        # Convert to DataFrame
        X_test = pd.DataFrame([meta_features])
        
        # Align with training features
        for col in self.selected_features:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[self.selected_features]
        
        # Handle infinite values
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Impute missing values
        X_test_processed = self.imputer.transform(X_test)
        
        # Predict
        y_pred = self.model.predict(X_test_processed).toarray()[0]
        
        # Get top 3 recommendations
        top_indices = np.argsort(y_pred)[-3:][::-1]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            if y_pred[idx] > 0.3:  # Confidence threshold
                recommendations.append({
                    "algorithm": self.algorithm_names[idx],
                    "confidence": float(y_pred[idx]),
                    "rank": rank,
                    "explanation": self.get_explanation(self.algorithm_names[idx]),
                    "characteristics": self.get_characteristics(self.algorithm_names[idx]),
                    "dataset_rows": n_rows,
                    "dataset_cols": n_cols
                })
        
        return recommendations
    
    def get_explanation(self, algorithm_name):
        """Get algorithm explanation"""
        explanations = {
            "Naive Bayes": "Fast and effective for high-dimensional data. Works well with categorical features and assumes feature independence.",
            "Random Forest": "Robust ensemble method that handles complex non-linear relationships. Excellent for most classification tasks.",
            "Decision Tree (J48/C4.5)": "Highly interpretable algorithm that creates human-readable decision rules. Good for understanding feature importance.",
            "Logistic Regression": "Simple yet powerful linear classifier. Best when relationships are linear and interpretability is important.",
        }
        return explanations.get(algorithm_name, "General purpose machine learning algorithm.")
    
    def get_characteristics(self, algorithm_name):
        """Get algorithm characteristics"""
        return {
            "speed": "Fast" if "Naive" in algorithm_name or "Stump" in algorithm_name else "Medium",
            "interpretability": "High" if "Tree" in algorithm_name or "Decision" in algorithm_name else "Medium",
            "handles_non_linear": "Tree" in algorithm_name or "Forest" in algorithm_name,
            "ensemble": "Forest" in algorithm_name
        }
    
    def get_supported_algorithms(self):
        """Return list of supported algorithms"""
        return self.algorithm_names
