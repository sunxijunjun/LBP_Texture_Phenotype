from defs import *
import os
file_path = "inputs/GSTAT_Univariate.xlsx"
data = pd.read_excel(file_path)
data.columns = data.columns.str.strip()
print(data.columns)
data['Group'] = data['Group'].astype('category')
continuous_vars = ["Age", "Height", "Weight", "BMI"]
dichotomous_vars = [
    "Gender", "Education", "Occupation_LBP_Baseline_Nominal",
    "Marital_status_LBP_Baseline", "Smoking_Baseline",
    "Alcohol_baseline", "Modic_change_L3", "Modic_change_L4",
    "Modic_change_L5", "Modic_change_S1", "Schmorls_nodes_L3",
    "Schmorls_nodes_L4", "Schmorls_nodes_L5", "HIZ_L3_4",
    "HIZ_L4_5", "HIZ_L5_S1", "Pffirmann_L3_4", "Pffirmann_L4_5",
    "Pffirmann_L5_S1", "Facet_tropism_L3_4", "Facet_tropism_L4_5",
    "Facet_tropism_L5_S1", "facet_degeneration_L3_4_right",
    "facet_degeneration_L3_4_left", "facet_degeneration_L4_5_right",
    "facet_degeneration_L4_5_left", "facet_degeneration_L5_S1_right",
    "facet_degeneration_L5_S1_left"
]
ordinal_vars = [
    "Pffirmann_L3_4_grading",
    "Pffirmann_L4_5_grading",
    "Pffirmann_L5_S1_grading"
]

continuous_results = [univariate_analysis(data, var, True) for var in continuous_vars]
dichotomous_results = [univariate_analysis(data, var, False) for var in dichotomous_vars]
ordinal_results = [univariate_analysis(data, var, False) for var in ordinal_vars]
results_df = pd.DataFrame(continuous_results + dichotomous_results + ordinal_results)
print(results_df)
os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/univariate_results.csv", index=False)
