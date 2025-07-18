from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from sklearn.metrics import roc_auc_score
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.api import Logit
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

def check_multiclass_variables(data, var_list):
    print("\n以下是非二分类（unique值多于2个）的变量：")
    multi_class_vars = []
    for var in var_list:
        if var not in data.columns:
            print(f" 变量 {var} 不存在于数据中，跳过")
            continue
        unique_vals = data[var].dropna().unique()
        if len(unique_vals) > 2:
            print(f" {var}: {len(unique_vals)} 个唯一值 -> {sorted(unique_vals)}")
            multi_class_vars.append(var)
    if not multi_class_vars:
        print("所有变量都是二分类或数值型")
    return multi_class_vars

def show_ordinal_level_counts(data, ordinal_vars):
    """
    显示每个 ordinal 变量的等级数和各等级的样本数
    """
    for var in ordinal_vars:
        if var not in data.columns:
            print(f" {var} 不在数据中，跳过")
            continue
        print(f"\n 变量: {var}")
        value_counts = data[var].value_counts().sort_index()
        print(f"共有 {value_counts.shape[0]} 个等级:")
        print(value_counts)

def check_if_meet_ordinal_requirements(data, ordinal_vars, target_col="Group", min_counts=5):
    """
    检查每个 ordinal variable 是否满足构建有序逻辑回归的基本条件：
    - 样本量是否充足
    - 是否是有序变量（>=2个唯一值）
    - 在Group=0和Group=1中是否都有等级分布（避免perfect separation）
    """
    results = []

    # 如果 Group 是分类类型，先转换为 0/1
    if data[target_col].dtype.name == "category":
        data[target_col] = data[target_col].cat.codes

    for var in ordinal_vars:
        if var not in data.columns:
            results.append({
                "variable": var,
                "status": "变量不存在",
                "unique_levels": None,
                "group0_levels": None,
                "group1_levels": None
            })
            continue

        df = data[[var, target_col]].dropna()
        levels = df[var].nunique()
        group0 = df[df[target_col] == 0][var].unique()
        group1 = df[df[target_col] == 1][var].unique()

        if levels < 2:
            status = "等级数量不足"
        elif len(group0) < 2 or len(group1) < 2:
            status = "分组等级不均衡（某组只有一个等级）"
        elif df.shape[0] < min_counts * levels:
            status = f"总样本量偏少（{df.shape[0]} 条，建议 ≥ {min_counts * levels}）"
        else:
            status = "满足要求"

        results.append({
            "variable": var,
            "status": status,
            "unique_levels": levels,
            "group0_levels": list(sorted(group0)),
            "group1_levels": list(sorted(group1))
        })

    return pd.DataFrame(results)

def binarize_ordinal_vars(data, ordinal_vars, threshold=3):
    """
    二值化所有 ordinal_vars（大于等于 threshold 为 1，其他为 0）
    返回：
        - data_copy：带有二值化列的新 DataFrame
        - new_var_names：所有新变量名列表
    """
    data_copy = data.copy()
    new_var_names = []

    for var in ordinal_vars:
        if var in data.columns:
            new_var = f"{var}_bin"
            data_copy[new_var] = (data_copy[var] >= threshold).astype(int)
            new_var_names.append(new_var)
        else:
            warnings.warn(f"{var} 不在数据中，跳过。")

    return data_copy, new_var_names

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
            if df['Group'].dtype.name == "category":
                group0_label = df['Group'].cat.categories[0]
                group1_label = df['Group'].cat.categories[1]
            else:
                sorted_labels = sorted(df['Group'].unique())
                group0_label = sorted_labels[0]
                group1_label = sorted_labels[1]

            group1 = df[df['Group'] == group0_label][variable]
            group2 = df[df['Group'] == group1_label][variable]

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


def univariate_logistic_regression(data, variable_list, target_col="Group"):
    """
    Perform univariate logistic regression for each variable in variable_list.
    Returns a DataFrame with OR, 95% CI, p-value, and AUC for each variable.
    """
    results = []

    # Convert target to binary 0/1
    y = data[target_col].dropna()
    if y.dtype.name == "category":
        y = y.cat.codes
    data[target_col] = y

    for var in variable_list:
        if var not in data.columns:
            warnings.warn(f"{var} not found in data, skipped.")
            continue

        df = data[[var, target_col]].dropna()

        if len(df[target_col].unique()) != 2:
            warnings.warn(f"{var} skipped: target must have exactly 2 classes.")
            continue

        try:
            X = sm.add_constant(df[[var]])
            model = sm.Logit(df[target_col], X).fit(disp=0)

            coef = model.params[var]
            OR = np.exp(coef)
            conf = model.conf_int().loc[var]
            CI_lower = np.exp(conf[0])
            CI_upper = np.exp(conf[1])
            pval = model.pvalues[var]

            # AUC calculation
            y_true = df[target_col]
            y_score = model.predict(X)
            auc = roc_auc_score(y_true, y_score)

            results.append({
                "variable": var,
                "OR": OR,
                "CI_lower": CI_lower,
                "CI_upper": CI_upper,
                "p_value": pval,
                "AUC": auc
            })

        except Exception as e:
            warnings.warn(f"{var} failed: {e}")
            results.append({
                "variable": var,
                "OR": np.nan,
                "CI_lower": np.nan,
                "CI_upper": np.nan,
                "p_value": np.nan,
                "AUC": np.nan
            })

    return pd.DataFrame(results)

from statsmodels.miscmodels.ordinal_model import OrderedModel

def univariate_ordinal_logit(data, ordinal_vars, target_col="Group"):
    """
    对每个有序变量（ordinal_vars）做有序logit回归（Ordinal Logistic Regression）。
    要求 target 是二分类（0/1）。
    返回 summary DataFrame（包含 coef, OR, p_value）
    """
    results = []

    # 处理目标变量为 0/1
    y = data[target_col].dropna()
    if y.dtype.name == "category":
        y = y.cat.codes
    data[target_col] = y

    for var in ordinal_vars:
        if var not in data.columns:
            warnings.warn(f"{var} 不在数据中，跳过。")
            continue

        df = data[[var, target_col]].dropna()
        if len(df[target_col].unique()) != 2:
            warnings.warn(f"{var} skipped: target must have 2 classes.")
            continue

        try:
            # 建立模型
            model = OrderedModel(
                endog=df[target_col],
                exog=df[[var]],
                distr='logit'
            )
            res = model.fit(method='bfgs', disp=0)

            coef = res.params[var]
            pval = res.pvalues[var]
            OR = np.exp(coef)
            conf_int = res.conf_int().loc[var]
            CI_lower = np.exp(conf_int[0])
            CI_upper = np.exp(conf_int[1])

            results.append({
                "variable": var,
                "coef": coef,
                "OR": OR,
                "CI_lower": CI_lower,
                "CI_upper": CI_upper,
                "p_value": pval
            })

        except Exception as e:
            warnings.warn(f"{var} 计算失败: {e}")
            results.append({
                "variable": var,
                "coef": np.nan,
                "OR": np.nan,
                "CI_lower": np.nan,
                "CI_upper": np.nan,
                "p_value": np.nan
            })

    return pd.DataFrame(results)

def check_collinearity_and_plot(data, variable_list, title="Selected Variables"):
    """
    计算共线性指标（VIF）并可视化相关性矩阵热图。

    参数:
    - data: 原始 DataFrame
    - variable_list: 要检查共线性的变量名列表
    - title: 热图标题（默认值 "Selected Variables"）

    返回:
    - vif_df: VIF 表格（DataFrame）
    """
    vif_data = data[variable_list].dropna()
    X = sm.add_constant(vif_data)
    vif_df = pd.DataFrame()
    vif_df["variable"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\n Variance Inflation Factors (VIF):")
    print(vif_df.sort_values(by="VIF", ascending=False))
    plt.figure(figsize=(10, 8))
    sns.heatmap(vif_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(f"Correlation Matrix of {title}, Pearson r method")
    plt.tight_layout()
    plt.show()
    return vif_df[vif_df["variable"] != "const"]



def stepwise_logit_forward_backward(data, candidate_vars, target, threshold_in=0.05, threshold_out=0.10, verbose=True):
    """
    前向选择 + 后向删除 的 stepwise 多变量逻辑回归变量筛选。

    参数:
    - data: 包含候选变量和目标变量的 DataFrame
    - candidate_vars: 所有候选变量的列表
    - target: 目标变量列名（必须是0/1或category）
    - threshold_in: p值进入模型的阈值
    - threshold_out: p值移除模型的阈值
    - verbose: 是否打印过程

    返回:
    - selected_vars: 入选变量名列表
    - final_model: 拟合的逻辑回归模型对象
    """
    import statsmodels.api as sm

    selected_vars = []
    remaining = candidate_vars.copy()
    changed = True

    while changed:
        changed = False
        pvals_in = []
        for var in remaining:
            try_vars = selected_vars + [var]
            df_sub = data[try_vars + [target]].dropna()
            if df_sub.empty:
                continue
            y = df_sub[target].cat.codes if df_sub[target].dtype.name == "category" else df_sub[target]
            X = sm.add_constant(df_sub[try_vars])
            try:
                model = sm.Logit(y, X).fit(disp=0)
                pval = model.pvalues[var]
                pvals_in.append((var, pval))
            except:
                continue

        if pvals_in:
            best_var, best_p = min(pvals_in, key=lambda x: x[1])
            if best_p < threshold_in:
                selected_vars.append(best_var)
                remaining.remove(best_var)
                changed = True
                if verbose:
                    print(f"Adding '{best_var}' with p = {best_p:.4f}")

        if selected_vars:
            df_sub = data[selected_vars + [target]].dropna()
            y = df_sub[target].cat.codes if df_sub[target].dtype.name == "category" else df_sub[target]
            X = sm.add_constant(df_sub[selected_vars])
            try:
                model = sm.Logit(y, X).fit(disp=0)
                pvals = model.pvalues.drop("const")
                worst_var = pvals.idxmax()
                worst_p = pvals.max()
                if worst_p > threshold_out:
                    selected_vars.remove(worst_var)
                    remaining.append(worst_var)
                    changed = True
                    if verbose:
                        print(f" Dropping '{worst_var}' with p = {worst_p:.4f}")
            except:
                pass

    df_final = data[selected_vars + [target]].dropna()
    y_final = df_final[target].cat.codes if df_final[target].dtype.name == "category" else df_final[target]
    X_final = sm.add_constant(df_final[selected_vars])
    final_model = sm.Logit(y_final, X_final).fit()

    return selected_vars, final_model

def try_logit_increasing_vars(data, var_list, group_col="Group"):
    for i in range(1, len(var_list)+1):
        vars_subset = var_list[:i]
        print(f"\n Testing with {i} variables: {vars_subset}")
        df_test = data[vars_subset + [group_col]].dropna()
        if df_test[group_col].dtype.name == "category":
            df_test[group_col] = df_test[group_col].cat.codes
        X_test = sm.add_constant(df_test[vars_subset])
        y_test = df_test[group_col]
        try:
            model = Logit(y_test, X_test).fit(disp=0)
            print(" Model fit successful.")
        except Exception as e:
            print(f" Fit failed with error: {e}")
            break

def print_var_distribution(data, group_col, vars_to_check):
    for var in vars_to_check:
        print(f"\n{var} 分布情况:")
        print(pd.crosstab(data[group_col], data[var]))

def fit_logit_model_and_get_summary(data, vars_list, group_col="Group"):
    df = data[vars_list + [group_col]].dropna()
    if df[group_col].dtype.name == 'category':
        df[group_col] = df[group_col].cat.codes
    X = sm.add_constant(df[vars_list])
    y = df[group_col]
    model = sm.Logit(y, X).fit()
    return get_logit_summary_table(model)

def get_logit_summary_table(model):
    summary = model.summary2().tables[1]
    summary["OR"] = np.exp(summary["Coef."])
    summary["CI_lower"] = np.exp(summary["Coef."] - 1.96 * summary["Std.Err."])
    summary["CI_upper"] = np.exp(summary["Coef."] + 1.96 * summary["Std.Err."])
    summary = summary.drop("const", errors="ignore")
    return summary[["OR", "CI_lower", "CI_upper", "P>|z|"]].rename(columns={"P>|z|": "p_value"})

def run_logit_with_dropped_vars(data, base_vars, vars_to_drop, title_suffix, standardize=True):
    """
    从 base_vars 中剔除 vars_to_drop，计算 VIF、拟合 logit 模型。
    可选：自动标准化连续变量（默认启用）。
    """
    filtered_vars = [v for v in base_vars if v not in vars_to_drop]
    data_copy = data.copy()
    if standardize:
        # 只对非二分类变量进行标准化（例如 dummy/二值变量跳过）
        for col in filtered_vars:
            if col not in data_copy.columns:
                continue
            unique_vals = data_copy[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                continue  # 跳过二分类变量
            # z-score 标准化
            mean = data_copy[col].mean()
            std = data_copy[col].std()
            if std != 0:
                data_copy[col] = (data_copy[col] - mean) / std

    vif_df = check_collinearity_and_plot(data_copy, filtered_vars, title=f"p < 0.2 Variables {title_suffix}")
    df_results = fit_logit_model_and_get_summary(data_copy, filtered_vars)
    print(f"\n Variables used: {filtered_vars}")
    print(df_results)
    return data_copy, vif_df, df_results

def add_degeneration_presence_columns(df):
    """添加 degeneration presence 变量（是否存在任一类型的退变）"""

    modic_vars = ["Modic_change_L3", "Modic_change_L4", "Modic_change_L5", "Modic_change_S1"]
    schmorls_vars = ["Schmorls_nodes_L3", "Schmorls_nodes_L4", "Schmorls_nodes_L5"]
    hiz_vars = ["HIZ_L3_4", "HIZ_L4_5", "HIZ_L5_S1"]
    facet_tropism_vars = ["Facet_tropism_L3_4", "Facet_tropism_L4_5", "Facet_tropism_L5_S1"]
    facet_degeneration_vars = [
        "facet_degeneration_L3_4_right", "facet_degeneration_L3_4_left",
        "facet_degeneration_L4_5_right", "facet_degeneration_L4_5_left",
        "facet_degeneration_L5_S1_right", "facet_degeneration_L5_S1_left"
    ]
    pfirrmann_vars = [
        "Pffirmann_L3_4_grading", "Pffirmann_L4_5_grading", "Pffirmann_L5_S1_grading"
    ]

    all_vars = modic_vars + schmorls_vars + hiz_vars + facet_tropism_vars + facet_degeneration_vars + pfirrmann_vars
    df[all_vars] = df[all_vars].fillna(0)

    df["modic_change_presence"] = df[modic_vars].gt(0).any(axis=1).astype(int)
    df["schmorls_nodes_presence"] = df[schmorls_vars].gt(0).any(axis=1).astype(int)
    df["hiz_presence"] = df[hiz_vars].gt(0).any(axis=1).astype(int)
    df["facet_tropism_presence"] = df[facet_tropism_vars].gt(0).any(axis=1).astype(int)
    df["facet_degeneration_presence"] = df[facet_degeneration_vars].gt(0).any(axis=1).astype(int)
    df["pfirrmann_presence"] = df[pfirrmann_vars].gt(1).any(axis=1).astype(int)  # I=正常，II及以上为退变

    presence_cols = [
        "modic_change_presence", "schmorls_nodes_presence", "hiz_presence",
        "facet_tropism_presence", "facet_degeneration_presence", "pfirrmann_presence"
    ]
    df["degeneration_score"] = df[presence_cols].sum(axis=1)
    df["degeneration_presence"] = (df["degeneration_score"] > 0).astype(int)
    df["total_degeneration_presence"] = df[presence_cols].sum(axis=1)

    return df


def add_degeneration_score_columns(df):
    """添加 degeneration score 总分列（表示严重程度）"""
    modic_vars = ["Modic_change_L3", "Modic_change_L4", "Modic_change_L5", "Modic_change_S1"]
    schmorls_vars = ["Schmorls_nodes_L3", "Schmorls_nodes_L4", "Schmorls_nodes_L5"]
    hiz_vars = ["HIZ_L3_4", "HIZ_L4_5", "HIZ_L5_S1"]
    facet_tropism_vars = ["Facet_tropism_L3_4", "Facet_tropism_L4_5", "Facet_tropism_L5_S1"]
    facet_degeneration_vars = [
        "facet_degeneration_L3_4_right", "facet_degeneration_L3_4_left",
        "facet_degeneration_L4_5_right", "facet_degeneration_L4_5_left",
        "facet_degeneration_L5_S1_right", "facet_degeneration_L5_S1_left"
    ]
    pfirrmann_vars = [
        "Pffirmann_L3_4_grading", "Pffirmann_L4_5_grading", "Pffirmann_L5_S1_grading"
    ]

    df["modic_score"] = df[modic_vars].sum(axis=1)
    df["schmorls_score"] = df[schmorls_vars].sum(axis=1)
    df["hiz_score"] = df[hiz_vars].sum(axis=1)
    df["facet_tropism_score"] = df[facet_tropism_vars].sum(axis=1)
    df["facet_degeneration_score"] = df[facet_degeneration_vars].sum(axis=1)
    df["pfirrmann_score"] = df[pfirrmann_vars].sum(axis=1)

    score_cols = [
        "modic_score", "schmorls_score", "hiz_score",
        "facet_tropism_score", "facet_degeneration_score", "pfirrmann_score"
    ]
    df["global_degeneration_score"] = df[score_cols].sum(axis=1)

    return df


def run_backward_stepwise_logit(data, base_vars, y_col='outcome', p_threshold=0.05, verbose=True):
    """
    自动进行 backward stepwise logistic regression。

    参数:
    - data: 数据框（包含 y 和 X）
    - base_vars: 初始变量列表
    - y_col: 因变量的列名
    - p_threshold: 剔除变量的 p 值阈值
    - verbose: 是否打印过程

    返回:
    - final_model: 最后拟合的logit模型
    - remaining_vars: 最后保留的变量列表
    - summary_df: 各步p值变化记录
    """
    remaining_vars = base_vars.copy()
    steps = []

    while True:
        X = data[remaining_vars]
        X = sm.add_constant(X)
        y = data[y_col]

        model = sm.Logit(y, X).fit(disp=0)
        pvalues = model.pvalues.drop("const")
        max_p = pvalues.max()
        worst_var = pvalues.idxmax()

        steps.append(pvalues.to_frame(name="p_value").T)

        if verbose:
            print(f"Max p-value: {max_p:.4f} ({worst_var})")

        if max_p > p_threshold:
            if verbose:
                print(f"Dropping '{worst_var}'")
            remaining_vars.remove(worst_var)
        else:
            break

    summary_df = pd.concat(steps).reset_index(drop=True)
    final_model = sm.Logit(y, sm.add_constant(data[remaining_vars])).fit()

    return final_model, remaining_vars, summary_df

import seaborn as sns
import matplotlib.pyplot as plt

def plot_density_heatmap(
    df,
    x: str,
    y: str,
    hue: str = None,
    cmap: str = 'viridis',
    bw_adjust: float = 1.5,
    levels: int = 100,
    figsize=(6, 5)
):
    """
    Plots KDE-based 2D density heatmaps with optional group-wise subplots.

    Parameters:
        df (pd.DataFrame): Input data.
        x (str): Column for x-axis (continuous).
        y (str): Column for y-axis (continuous).
        hue (str, optional): Column to split subplots by group.
        cmap (str): Colormap to use (default: 'viridis').
        bw_adjust (float): KDE bandwidth scaling (default: 1.5 for smoother edges).
        levels (int): Number of contour levels.
        figsize (tuple): Size of each subplot.
    """
    xlim = (df[x].min(), df[x].max())
    ylim = (df[y].min(), df[y].max())

    if hue:
        unique_hues = df[hue].dropna().unique()
        fig, axes = plt.subplots(1, len(unique_hues), figsize=(figsize[0] * len(unique_hues), figsize[1]), constrained_layout=True)
        if len(unique_hues) == 1:
            axes = [axes]
        for ax, group in zip(axes, unique_hues):
            sub = df[df[hue] == group]
            sns.kdeplot(
                data=sub,
                x=x, y=y,
                fill=True,
                cmap=cmap,
                ax=ax,
                thresh=0,
                levels=levels,
                bw_adjust=bw_adjust,
                clip=(xlim, ylim)
            )
            ax.set_title(f"{hue} = {group}")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
    else:
        plt.figure(figsize=figsize)
        sns.kdeplot(
            data=df,
            x=x, y=y,
            fill=True,
            cmap=cmap,
            thresh=0,
            levels=levels,
            bw_adjust=bw_adjust,
            clip=(xlim, ylim)
        )
        plt.title("2D Density Heatmap")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xlim(xlim)
        plt.ylim(ylim)

    plt.show()


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr


def point_biserial_analysis(data: pd.DataFrame, binary_col: str, continuous_col: str, output_folder: str,
                            title: str = ""):
    """
    Performs point-biserial correlation analysis, shows and saves a grouped scatter + line plot.

    Parameters:
    - data: pandas DataFrame containing the binary and continuous variable
    - binary_col: column name of the binary variable (e.g., 'Exam')
    - continuous_col: column name of the continuous variable (e.g., 'Maths Test')
    - output_folder: path to save the plot image
    - title: optional plot title
    """

    # Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)

    # Convert binary to numeric if necessary
    binary_map = {v: i for i, v in enumerate(data[binary_col].unique())}
    numeric_binary = data[binary_col].map(binary_map)

    # Correlation analysis
    corr, pval = pointbiserialr(numeric_binary, data[continuous_col])
    print(f"Point-biserial correlation (r): {corr:.4f}, p-value: {pval:.4g}")

    # Plot
    plt.figure(figsize=(6, 5))
    sns.stripplot(x=binary_col, y=continuous_col, data=data, jitter=True, color='black', alpha=0.6)

    means = data.groupby(binary_col)[continuous_col].mean()
    plt.plot(means.index, means.values, color='black', linewidth=1.5)

    plot_title = title or f'Point-Biserial Correlation: r = {corr:.2f}, p = {pval:.3g}'
    plt.title(plot_title)
    plt.xlabel(binary_col)
    plt.ylabel(continuous_col)
    plt.grid(False)
    plt.tight_layout()

    # Save and show
    filename = f"point_biserial_{binary_col}_vs_{continuous_col}.png"
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Plot saved to: {save_path}")


