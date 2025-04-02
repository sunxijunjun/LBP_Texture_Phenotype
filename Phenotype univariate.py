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
ordinal_logit_results = univariate_ordinal_logit(data, ordinal_vars)
ordinal_logit_results.to_csv("outputs/univariate_ordinal_logit_results.csv", index=False)
print(ordinal_logit_results)


logit_sig_vars = logit_results[logit_results["p_value"] < 0.2]["variable"].tolist()
print(f"Selected {len(logit_sig_vars)} variables with p < 0.2:")
print(logit_sig_vars)
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




