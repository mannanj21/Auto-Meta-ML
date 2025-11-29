import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from pymfe.mfe import MFE
from scipy.stats import entropy, kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

# OpenML-CC18 exact columns (MUST match training)
OPENML_COLUMNS = [
    "dataset_id", "AutoCorrelation", "CfsSubsetEval_DecisionStumpAUC", "CfsSubsetEval_DecisionStumpErrRate",
    "CfsSubsetEval_DecisionStumpKappa", "CfsSubsetEval_NaiveBayesAUC", "CfsSubsetEval_NaiveBayesErrRate",
    "CfsSubsetEval_NaiveBayesKappa", "CfsSubsetEval_kNN1NAUC", "CfsSubsetEval_kNN1NErrRate",
    "CfsSubsetEval_kNN1NKappa", "ClassEntropy", "DecisionStumpAUC", "DecisionStumpErrRate", "DecisionStumpKappa",
    "Dimensionality", "EquivalentNumberOfAtts", "J48.00001.AUC", "J48.00001.ErrRate", "J48.00001.Kappa",
    "J48.0001.AUC", "J48.0001.ErrRate", "J48.0001.Kappa", "J48.001.AUC", "J48.001.ErrRate", "J48.001.Kappa",
    "MajorityClassPercentage", "MajorityClassSize", "MaxAttributeEntropy", "MaxKurtosisOfNumericAtts",
    "MaxMeansOfNumericAtts", "MaxMutualInformation", "MaxNominalAttDistinctValues", "MaxSkewnessOfNumericAtts",
    "MaxStdDevOfNumericAtts", "MeanAttributeEntropy", "MeanKurtosisOfNumericAtts", "MeanMeansOfNumericAtts",
    "MeanMutualInformation", "MeanNoiseToSignalRatio", "MeanNominalAttDistinctValues", "MeanSkewnessOfNumericAtts",
    "MeanStdDevOfNumericAtts", "MinAttributeEntropy", "MinKurtosisOfNumericAtts", "MinMeansOfNumericAtts",
    "MinMutualInformation", "MinNominalAttDistinctValues", "MinSkewnessOfNumericAtts", "MinStdDevOfNumericAtts",
    "MinorityClassPercentage", "MinorityClassSize", "NaiveBayesAUC", "NaiveBayesErrRate", "NaiveBayesKappa",
    "NumberOfBinaryFeatures", "NumberOfClasses", "NumberOfFeatures", "NumberOfInstances",
    "NumberOfInstancesWithMissingValues", "NumberOfMissingValues", "NumberOfNumericFeatures", "NumberOfSymbolicFeatures",
    "PercentageOfBinaryFeatures", "PercentageOfInstancesWithMissingValues", "PercentageOfMissingValues",
    "PercentageOfNumericFeatures", "PercentageOfSymbolicFeatures", "Quartile1AttributeEntropy",
    "Quartile1KurtosisOfNumericAtts", "Quartile1MeansOfNumericAtts", "Quartile1MutualInformation",
    "Quartile1SkewnessOfNumericAtts", "Quartile1StdDevOfNumericAtts", "Quartile2AttributeEntropy",
    "Quartile2KurtosisOfNumericAtts", "Quartile2MeansOfNumericAtts", "Quartile2MutualInformation",
    "Quartile2SkewnessOfNumericAtts", "Quartile2StdDevOfNumericAtts", "Quartile3AttributeEntropy",
    "Quartile3KurtosisOfNumericAtts", "Quartile3MeansOfNumericAtts", "Quartile3MutualInformation",
    "Quartile3SkewnessOfNumericAtts", "Quartile3StdDevOfNumericAtts", "REPTreeDepth1AUC", "REPTreeDepth1ErrRate",
    "REPTreeDepth1Kappa", "REPTreeDepth2AUC", "REPTreeDepth2ErrRate", "REPTreeDepth2Kappa", "REPTreeDepth3AUC",
    "REPTreeDepth3ErrRate", "REPTreeDepth3Kappa", "RandomTreeDepth1AUC", "RandomTreeDepth1ErrRate",
    "RandomTreeDepth1Kappa", "RandomTreeDepth2AUC", "RandomTreeDepth2ErrRate", "RandomTreeDepth2Kappa",
    "RandomTreeDepth3AUC", "RandomTreeDepth3ErrRate", "RandomTreeDepth3Kappa", "StdvNominalAttDistinctValues",
    "kNN1NAUC", "kNN1NErrRate", "kNN1NKappa"
]

def preprocess_dataset(df, target_name):
    """Preprocess dataset for meta-feature extraction (MUST match training)"""
    df.columns = df.columns.str.strip().str.replace('"', '')
    df = df.drop_duplicates()

    if target_name not in df.columns:
        raise ValueError(f"Target '{target_name}' not found in columns: {df.columns.tolist()}")

    X = df.drop(columns=[target_name])
    y = df[target_name]

    # Encode categorical target
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        print(f"Encoding categorical target: {y.unique()[:5]}")
        y = y.astype('category').cat.codes

    # Encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Impute missing values
    X = X.fillna(X.median(numeric_only=True))

    return X, y

def quartiles(series):
    if len(series) == 0:
        return 0, 0, 0
    return np.percentile(series, 25), np.percentile(series, 50), np.percentile(series, 75)

def compute_manual_features(X, y):
    """Manual features (MUST match training exactly)"""
    features = {}
    
    # AutoCorrelation
    def autocorrelation(X):
        X_num = X.select_dtypes(include=np.number)
        if X_num.empty:
            return 0
        autocorrs = [X_num[col].autocorr() for col in X_num.columns]
        return np.mean([0 if np.isnan(v) else v for v in autocorrs])
    features["AutoCorrelation"] = autocorrelation(X)

    X_num = X.select_dtypes(include=[np.number])
    X_cat = X.select_dtypes(exclude=[np.number])

    # Class stats
    class_counts = y.value_counts()
    features["MajorityClassPercentage"] = class_counts.max() / len(y)
    features["MajorityClassSize"] = class_counts.max()
    features["MinorityClassPercentage"] = class_counts.min() / len(y)
    features["MinorityClassSize"] = class_counts.min()
    features["NumberOfClasses"] = y.nunique()
    features["ClassEntropy"] = entropy(class_counts / len(y), base=2)

    # Numeric stats
    stats = {}
    if not X_num.empty:
        stats['mean'] = X_num.mean()
        stats['std'] = X_num.std(ddof=0)
        stats['kurtosis'] = X_num.kurtosis()
        stats['skewness'] = X_num.apply(lambda col: skew(col, bias=False))
        stats['min'] = X_num.min()
        stats['max'] = X_num.max()
        stats['median'] = X_num.median()
        stats['range'] = X_num.max() - X_num.min()
        
        discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform', random_state=42)
        X_num_discrete = discretizer.fit_transform(X_num)
        stats['mutual_info'] = mutual_info_classif(X_num_discrete, y, random_state=42)
    else:
        stats = {k: pd.Series() for k in ['mean', 'std', 'kurtosis', 'skewness', 'min', 'max', 'median', 'range', 'mutual_info']}

    # Quartiles
    for stat_name, series in stats.items():
        q1, q2, q3 = quartiles(series)
        features[f"Quartile1{stat_name.capitalize()}"] = q1
        features[f"Quartile2{stat_name.capitalize()}"] = q2
        features[f"Quartile3{stat_name.capitalize()}"] = q3

    features["MeanNoiseToSignalRatio"] = np.mean(X_num.var() / (y.var() if y.var() > 0 else 1)) if not X_num.empty else 0

    # Categorical stats
    if not X_cat.empty:
        distinct_vals = X_cat.nunique()
        features["MaxNominalAttDistinctValues"] = distinct_vals.max()
        features["MeanNominalAttDistinctValues"] = distinct_vals.mean()
        features["MinNominalAttDistinctValues"] = distinct_vals.min()
        features["StdvNominalAttDistinctValues"] = np.std(distinct_vals)
    else:
        features["MaxNominalAttDistinctValues"] = 0
        features["MeanNominalAttDistinctValues"] = 0
        features["MinNominalAttDistinctValues"] = 0
        features["StdvNominalAttDistinctValues"] = 0

    # Counts
    features["NumberOfBinaryFeatures"] = (X.nunique() == 2).sum()
    features["NumberOfFeatures"] = X.shape[1]
    features["NumberOfInstances"] = X.shape[0]
    features["NumberOfInstancesWithMissingValues"] = X.isna().any(axis=1).sum()
    features["NumberOfMissingValues"] = X.isna().sum().sum()
    features["NumberOfNumericFeatures"] = X_num.shape[1]
    features["NumberOfSymbolicFeatures"] = X_cat.shape[1]

    # Percentages
    features["PercentageOfBinaryFeatures"] = features["NumberOfBinaryFeatures"] / X.shape[1] if X.shape[1] > 0 else 0
    features["PercentageOfInstancesWithMissingValues"] = features["NumberOfInstancesWithMissingValues"] / X.shape[0] if X.shape[0] > 0 else 0
    features["PercentageOfMissingValues"] = features["NumberOfMissingValues"] / (X.shape[0] * X.shape[1]) if X.shape[0] * X.shape[1] > 0 else 0
    features["PercentageOfNumericFeatures"] = features["NumberOfNumericFeatures"] / X.shape[1] if X.shape[1] > 0 else 0
    features["PercentageOfSymbolicFeatures"] = features["NumberOfSymbolicFeatures"] / X.shape[1] if X.shape[1] > 0 else 0

    features["Dimensionality"] = X.shape[0] / X.shape[1] if X.shape[1] > 0 else 0
    
    # Equivalent Number of Attributes
    cond_entropy = 0.0
    for col in X.columns:
        try:
            for val, subset_idx in X.groupby(col).groups.items():
                subset_y = y.iloc[list(subset_idx)]
                probs = subset_y.value_counts(normalize=True)
                cond_entropy += (len(subset_y) / len(y)) * entropy(probs, base=2)
        except Exception:
            continue
    features["EquivalentNumberOfAtts"] = float(np.exp(cond_entropy))

    # Attribute entropy
    if not X_num.empty:
        discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        X_num_discrete = pd.DataFrame(discretizer.fit_transform(X_num), columns=X_num.columns)
        attribute_entropies = []
        for col in X_num_discrete.columns:
            value_counts = X_num_discrete[col].value_counts()
            probs = value_counts / len(X_num_discrete)
            ent = entropy(probs, base=2)
            attribute_entropies.append(ent)
        features["MaxAttributeEntropy"] = np.max(attribute_entropies) if attribute_entropies else 0

    return features

def compute_landmarkers(X, y):
    """Landmarkers (MUST match training)"""
    features = {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    X_np = X.values
    y_np = y.values

    def safe_auc(clf, X, y):
        try:
            if len(np.unique(y)) == 2:
                return cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean()
            else:
                y_bin = label_binarize(y, classes=np.unique(y))
                aucs = []
                for i in range(y_bin.shape[1]):
                    auc = cross_val_score(clf, X, y_bin[:, i], cv=cv, scoring='roc_auc').mean()
                    aucs.append(auc)
                return np.mean(aucs)
        except:
            return 0

    def safe_kappa(clf, X, y):
        y = np.array(y)
        preds = np.zeros_like(y)
        for train_idx, test_idx in cv.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            preds[test_idx] = clf.predict(X[test_idx])
        return cohen_kappa_score(y, preds)

    def cfs_subset_eval(X, y, clf):
        if X.shape[1] < 1:
            return 0, 0, 0
        X_sel = SelectKBest(score_func=f_classif, k=min(5, X.shape[1])).fit_transform(X, np.array(y))
        auc = safe_auc(clf, X_sel, np.array(y))
        err = 1 - cross_val_score(clf, X_sel, np.array(y), cv=cv, scoring='accuracy').mean()
        kappa = safe_kappa(clf, X_sel, np.array(y))
        return auc, err, kappa

    ds = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=42)
    nb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=1)

    features["CfsSubsetEval_DecisionStumpAUC"], features["CfsSubsetEval_DecisionStumpErrRate"], features["CfsSubsetEval_DecisionStumpKappa"] = cfs_subset_eval(X, y, ds)
    features["CfsSubsetEval_NaiveBayesAUC"], features["CfsSubsetEval_NaiveBayesErrRate"], features["CfsSubsetEval_NaiveBayesKappa"] = cfs_subset_eval(X, y, nb)
    features["CfsSubsetEval_kNN1NAUC"], features["CfsSubsetEval_kNN1NErrRate"], features["CfsSubsetEval_kNN1NKappa"] = cfs_subset_eval(X, y, knn)

    features["DecisionStumpErrRate"] = 1 - cross_val_score(ds, X_np, y_np, cv=cv, scoring='accuracy').mean()
    features["DecisionStumpAUC"] = safe_auc(ds, X_np, y_np)
    features["DecisionStumpKappa"] = safe_kappa(ds, X_np, y_np)

    features["NaiveBayesErrRate"] = 1 - cross_val_score(nb, X_np, y_np, cv=cv, scoring='accuracy').mean()
    features["NaiveBayesAUC"] = safe_auc(nb, X_np, y_np)
    features["NaiveBayesKappa"] = safe_kappa(nb, X_np, y_np)

    features["kNN1NErrRate"] = 1 - cross_val_score(knn, X_np, y_np, cv=cv, scoring='accuracy').mean()
    features["kNN1NAUC"] = safe_auc(knn, X_np, y_np)
    features["kNN1NKappa"] = safe_kappa(knn, X_np, y_np)

    for depth in range(1, 4):
        dt = DecisionTreeClassifier(max_depth=depth, criterion='entropy', random_state=42)
        features[f"REPTreeDepth{depth}ErrRate"] = 1 - cross_val_score(dt, X_np, y_np, cv=cv, scoring='accuracy').mean()
        features[f"REPTreeDepth{depth}AUC"] = safe_auc(dt, X_np, y_np)
        features[f"REPTreeDepth{depth}Kappa"] = safe_kappa(dt, X_np, y_np)

        dt_random = DecisionTreeClassifier(max_depth=depth, splitter='random', criterion='entropy', random_state=42)
        features[f"RandomTreeDepth{depth}ErrRate"] = 1 - cross_val_score(dt_random, X_np, y_np, cv=cv, scoring='accuracy').mean()
        features[f"RandomTreeDepth{depth}AUC"] = safe_auc(dt_random, X_np, y_np)
        features[f"RandomTreeDepth{depth}Kappa"] = safe_kappa(dt_random, X_np, y_np)

    alphas = [1e-5, 1e-4, 1e-3]
    for alpha in alphas:
        dt = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha, random_state=42)
        features[f"J48.{int(alpha * 1e5):05d}.AUC"] = safe_auc(dt, X_np, y_np)
        features[f"J48.{int(alpha * 1e5):05d}.ErrRate"] = 1 - cross_val_score(dt, X_np, y_np, cv=cv, scoring='accuracy').mean()
        features[f"J48.{int(alpha * 1e5):05d}.Kappa"] = safe_kappa(dt, X_np, y_np)

    return features

def extract_meta_features(csv_path: str, target_name: str = None):
    """
    Extract meta-features matching training data format
    """
    try:
        df = pd.read_csv(csv_path)
        
        if target_name is None:
            target_name = auto_detect_target(df)
        
        print(f"Processing dataset: {csv_path}")
        print(f"Target column: {target_name}")
        
        X, y = preprocess_dataset(df, target_name)
        
        print(f"Dataset shape: {X.shape[0]} rows × {X.shape[1]} features")
        
        # Extract all features
        meta_dict = {}
        meta_dict.update(compute_manual_features(X, y))
        meta_dict.update(compute_landmarkers(X, y))
        
        # Add PyMFE features (limited set)
        mfe = MFE(groups=["general"], summary="mean")
        mfe.fit(X.values, y.values)
        ft_names, ft_values = mfe.extract()
        if ft_names and ft_values:
            meta_dict.update(dict(zip(ft_names, ft_values)))
        
        # Ensure all OPENML columns exist
        meta_dict["dataset_id"] = csv_path
        for col in OPENML_COLUMNS:
            if col not in meta_dict:
                meta_dict[col] = 0
        
        # Return only OPENML columns in correct order
        meta_dict = {col: meta_dict[col] for col in OPENML_COLUMNS}
        
        print(f"Extracted {len(meta_dict)} meta-features")
        return meta_dict, len(X), X.shape[1]
        
    except Exception as e:
        print(f"Error extracting meta-features: {e}")
        import traceback
        traceback.print_exc()
        raise

def auto_detect_target(df):
    """Auto-detect target column"""
    possible_targets = ['class', 'target', 'label', 'outcome', 'y']
    
    for col in df.columns:
        if col.lower() in possible_targets:
            print(f"Auto-detected target: '{col}'")
            return col
    
    target_col = df.columns[-1]
    print(f"Using last column as target: '{target_col}'")
    return target_col