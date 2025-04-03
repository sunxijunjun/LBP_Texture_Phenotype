from defs import *
import os

os.makedirs("outputs", exist_ok=True)
file_path = "inputs/GSTAT_Univariate.xlsx"
data = pd.read_excel(file_path)
data.columns = data.columns.str.strip()
print(data.columns)
data['Group'] = data['Group'].astype('category')
continuous_vars = ["Age", "Height", "Weight", "BMI"]
dichotomous_vars = ["Gender", "Education", "Occupation_LBP_Baseline_Nominal",
 "Marital_status_LBP_Baseline", "Smoking_Baseline", "Alcohol_baseline",
 "Modic_change_L3", "Modic_change_L4", "Modic_change_L5", "Modic_change_S1",
 "Schmorls_nodes_L3", "Schmorls_nodes_L4", "Schmorls_nodes_L5",
 "HIZ_L3_4", "HIZ_L4_5", "HIZ_L5_S1",
 "Facet_tropism_L3_4", "Facet_tropism_L4_5", "Facet_tropism_L5_S1"]
ordinal_vars = ["Pffirmann_L3_4_grading", "Pffirmann_L4_5_grading", "Pffirmann_L5_S1_grading",
 "facet_degeneration_L3_4_right", "facet_degeneration_L3_4_left",
 "facet_degeneration_L4_5_right", "facet_degeneration_L4_5_left",
 "facet_degeneration_L5_S1_right", "facet_degeneration_L5_S1_left"]
show_ordinal_level_counts(data, ordinal_vars)
check_ordinal_result = check_if_meet_ordinal_requirements(data, ordinal_vars)


all_vars = continuous_vars + dichotomous_vars + ordinal_vars
multi_level_vars_logit = check_multiclass_variables(data, all_vars)

continuous_results = [univariate_analysis(data, var, True) for var in continuous_vars]
dichotomous_results = [univariate_analysis(data, var, False) for var in dichotomous_vars]
ordinal_results = [univariate_analysis(data, var, False) for var in ordinal_vars]
univariate_statistical_results= pd.DataFrame(continuous_results + dichotomous_results + ordinal_results)
print(univariate_statistical_results)
univariate_statistical_results.to_csv("outputs/univariate_statistical_results.csv", index=False)

logit_results = univariate_logistic_regression(data, continuous_vars + dichotomous_vars)
logit_results.to_csv("outputs/univariate_logit_results.csv", index=False)
print(logit_results)
logit_sig_vars = logit_results[logit_results["p_value"] < 0.2]["variable"].tolist()
print(f"Selected {len(logit_sig_vars)} variables with p < 0.2:")
print(logit_sig_vars)

ordinal_logit_results = univariate_ordinal_logit(data, ordinal_vars)
ordinal_logit_results.to_csv("outputs/univariate_ordinal_logit_results.csv", index=False)
print(ordinal_logit_results)

#制作二分类变量
data_copy, ordinal_bin_vars = binarize_ordinal_vars(data, ordinal_vars, threshold=2)

vif_vars = logit_sig_vars + ordinal_bin_vars
vif_df = check_collinearity_and_plot(data_copy, vif_vars, title="p < 0.2 Variables")


_, vif_df_1, df_multiv_results_1 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=vif_vars,
    vars_to_drop=["Weight", "Height"],
    title_suffix="without Weight and Height"
)

_, vif_df_2, df_multiv_results_2 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=vif_vars,
    vars_to_drop=["Weight", "Height","Pffirmann_L3_4_grading_bin"],
    title_suffix="without Weight, Height, Pffirmann_L3_4_grading_bin"
)

_, vif_df_3, df_multiv_results_3 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=vif_vars,
    vars_to_drop=["Weight", "Height","Pffirmann_L3_4_grading_bin","facet_degeneration_L3_4_right_bin","facet_degeneration_L3_4_left_bin"],
    title_suffix="without Weight, Height,Pffirmann_L3_4_grading_bin,facet_degeneration_L3_4_right_bin,facet_degeneration_L3_4_left_bin"
)


data_copy = add_degeneration_presence_columns(data_copy)
data_copy = add_degeneration_score_columns(data_copy)

degen_presence_vars = [
    "modic_change_presence",
    "schmorls_nodes_presence",
    "hiz_presence",
    "facet_tropism_presence",
    "facet_degeneration_presence",
    "pfirrmann_presence"
]

degen_score_vars = [
    "modic_score",
    "schmorls_score",
    "hiz_score",
    "facet_tropism_score",
    "facet_degeneration_score",
    "pfirrmann_score"
]

logit_results1 = univariate_logistic_regression(data_copy, degen_presence_vars + degen_score_vars )
new_logit_sig_vars = logit_results1[logit_results1["p_value"] < 0.2]["variable"].tolist()
print(f"Selected {len(new_logit_sig_vars)} variables with p < 0.2 from newly created vars:")
print(new_logit_sig_vars)

_, vif_df_4, df_multiv_results_4 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=new_logit_sig_vars,
    vars_to_drop=["facet_tropism_presence"],
    title_suffix="without facet_tropism_presence "
)

_, vif_df_5, df_multiv_results_5 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=new_logit_sig_vars,
    vars_to_drop=["facet_tropism_score"],
    title_suffix="without facet_tropism_score "
)

sigscorevars =  ['facet_tropism_score', 'facet_degeneration_score', 'pfirrmann_score']

_, vif_df_6, df_multiv_results_6 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=sigscorevars,
    vars_to_drop=[],
    title_suffix="without  "
)

sigpresencevars = ['schmorls_nodes_presence', 'hiz_presence', 'facet_tropism_presence', 'pfirrmann_presence']
_, vif_df_7, df_multiv_results_7 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=sigpresencevars,
    vars_to_drop=[],
    title_suffix="without  "
)
agengendervar = ["Age", "Gender"]
co_vars = ["Age", "Height", "Weight", "BMI","Gender", "Education", "Occupation_LBP_Baseline_Nominal",
 "Marital_status_LBP_Baseline", "Smoking_Baseline", "Alcohol_baseline"]

_, vif_df_8, df_multiv_results_8 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=sigpresencevars + agengendervar,
    vars_to_drop=[],
    title_suffix="without  "
)

_, vif_df_9, df_multiv_results_9 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=sigscorevars + agengendervar,
    vars_to_drop=[],
    title_suffix="without  "
)

_, vif_df_10, df_multiv_results_10 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=sigscorevars + agengendervar + sigpresencevars,
    vars_to_drop=['facet_tropism_presence'],
    title_suffix="without facet_tropism_presence "
)

_, vif_df_11, df_multiv_results_11 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=new_logit_sig_vars + agengendervar,
    vars_to_drop=["facet_tropism_score"],
    title_suffix="without facet_tropism_score "
)

_, vif_df_12, df_multiv_results_12 = run_logit_with_dropped_vars(
    data_copy,
    base_vars=new_logit_sig_vars + co_vars,
    vars_to_drop=["facet_tropism_score","Height", "Weight"],
    title_suffix="without facet_tropism_score "
)

df_dimension = pd.read_csv("inputs/merged_GSTAT.csv")

