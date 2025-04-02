import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import warnings

def prepare_features_and_labels(df, group_cols, target_col):
    """
    从透视后的 DataFrame 中提取特征和标签。
    """
    feature_cols = [col for col in df.columns if col not in group_cols]
    X = df[feature_cols].copy()
    X = X.fillna(X.mean())
    y = df[target_col].copy()
    return X, y, feature_cols

def train_evaluate_rf(X, y, label_encoder, title_suffix=""):
    """
    训练 Random Forest 并输出混淆矩阵和准确率。
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix{title_suffix} (Accuracy: {accuracy:.2f})")
    plt.tight_layout()
    plt.show()

    return rf, accuracy, cm

def univariate_analysis(data, variable, is_continuous):
    if variable not in data.columns:
        raise ValueError(f"Variable {variable} not found in the dataframe.")

    df = data[[variable, 'Group']].dropna()

    if len(df) < 2:
        warnings.warn(f"Not enough data for variable: {variable}")
        return {
            "variable": variable,
            "statistic": np.nan,
            "p_value": np.nan,
            "test_type": "N/A",
            "reason_for_nan": "Not enough data"
        }

    try:
        if is_continuous:
            group1 = df[df['Group'] == df['Group'].cat.categories[0]][variable]
            group2 = df[df['Group'] == df['Group'].cat.categories[1]][variable]

            if len(group1) == 0 or len(group2) == 0:
                return {
                    "variable": variable,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "test_type": "Mann-Whitney",
                    "reason_for_nan": "One group has no valid data"
                }

            stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
            return {
                "variable": variable,
                "statistic": stat,
                "p_value": p,
                "test_type": "Mann-Whitney",
                "reason_for_nan": None
            }
        else:
            contingency = pd.crosstab(df[variable], df['Group'])

            if contingency.shape[0] < 2 or (contingency < 5).any().any():
                warnings.warn(f"Using Fisher's Exact Test for variable: {variable}")
                if contingency.shape == (2, 2):
                    _, p = fisher_exact(contingency)
                    return {
                        "variable": variable,
                        "statistic": np.nan,
                        "p_value": p,
                        "test_type": "Fisher's Exact",
                        "reason_for_nan": "Fisher's test does not return statistic"
                    }
                else:
                    return {
                        "variable": variable,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "test_type": "Fisher's Exact",
                        "reason_for_nan": "Fisher's test only supports 2x2 tables"
                    }
            else:
                stat, p, _, _ = chi2_contingency(contingency)
                return {
                    "variable": variable,
                    "statistic": stat,
                    "p_value": p,
                    "test_type": "Chi-square",
                    "reason_for_nan": None
                }
    except PerfectSeparationError:
        return {
            "variable": variable,
            "statistic": np.nan,
            "p_value": np.nan,
            "test_type": "Unknown",
            "reason_for_nan": "Perfect separation or model failure"
        }
