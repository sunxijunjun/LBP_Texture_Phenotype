# Vertebral Texture & Phenotype Analysis Report

---
## 0. Data Cleaning
- The data consisted of dichotomous, ordered categorical, and continuous variables, which were first distinguished.

## 1. Texture Feature Classification

- Data source: combined_texture_data_reordered.xlsx
- Model: Random Forest
- Target: folder_index
- Result: overall accuracy 0.85, see confusion matrix below
- TODO: This random forest model is built using features extracted from the gray-level co-occurrence matrix (GLCM). Feature selection and importance evaluation are currently in progress to assess the model's performance and enhance its interpretability.
![image](https://github.com/user-attachments/assets/bf81ed5a-b655-4bdf-9909-b0d5c0ac4223)

## 2. Phenotype Statistical Test

- Data source: GSTAT_Univariate.xlsx 
- Statistical methods: Mann-Whitney U, Chi-square, Fisher's Exact
- See table below for significant associations.
  ![image](https://github.com/user-attachments/assets/99eff9c9-36eb-4299-a2a2-42c7d64849c1)

## 3. Phenotype Univariate logistic regression
- Data source: GSTAT_Univariate.xlsx
- Statistical methods: univariate logistic regression for continuous_vars + dichotomous_vars, and ordinal logistic regression for ordinal vars.
### Univariate logistic regression results for continuous_vars + dichotomous_vars:
![image](https://github.com/user-attachments/assets/b389918d-832e-45ab-8964-40eade17c472)

### Ordinal logistic regression (ORL) for ordinal vars:
- Prior to constructing the OLR model, all variables were evaluated for compliance with the model’s assumptions. All variables met the prerequisites.
- Results:
![image](https://github.com/user-attachments/assets/cf1f2722-b330-44c0-929f-21b263059d15)

  


## 4. Phenotype Multivariate logistic regression
- Data source: GSTAT_Univariate.xlsx, Selected 7 * (dichotomous + continuous) variables with p < 0.2: ['Height', 'Weight', 'BMI', 'Education', 'Marital_status_LBP_Baseline', 'HIZ_L3_4', 'Facet_tropism_L5_S1'] ; as well as 9 * ordinal bin vars (ordinal converted to binary).
- VIF:
- ![image](https://github.com/user-attachments/assets/e8c02ec9-bc55-4084-b4e8-166435cf7970)


### DELETED high VIF feature Heigh and Weight, kept BMI, rechecked VIF and collinearity
- ![image](https://github.com/user-attachments/assets/717dcf72-d5fa-4dde-9e94-d84f89d12674)

- ![image](https://github.com/user-attachments/assets/d8ae4585-3958-459c-b600-8bb56027af40)

### Multivariate logistic regression (MLR), standardized continuous_vars using z-score.
- After multiple rounds of feature selection and model fitting, the following set of vars was finalized for MLR. The figure below illustrates their intercorrelations.
- ![image](https://github.com/user-attachments/assets/f2e37b1b-2be9-49ab-9444-0b841d471057)

- MLR Results: 
- ![image](https://github.com/user-attachments/assets/76c6beb9-d4b6-467b-86aa-ad1339f33609)
- Further validation and interpretation of the results are in progress.


# Updates made on 3rd, April:

## 5. reconstructed the feature columns
- Reconstructed the feature columns representing various degenerative phenotypes by summarizing the presence and overall severity of each phenotype, regardless of vertebral level. Before constructing these features, it was observed that the same degenerative phenotype often appeared at multiple vertebral levels within the same patient, leading to data collinearity.
- A univariate analysis was conducted again on the reconstructed feature columns, and the results are as follows:
- ![image](https://github.com/user-attachments/assets/3b50523f-ae7f-4977-acd6-c42bc948734e)
- Selected 7 variables with p < 0.2 from newly created vars:
['schmorls_nodes_presence', 'hiz_presence', 'facet_tropism_presence', 'pfirrmann_presence', 'facet_tropism_score', 'facet_degeneration_score', 'pfirrmann_score']
- 'Facet_tropism_presence' and 'facet_tropism_score' were highly correlated, so only one was used for modeling to reduce collinearity. Both yielded similar MLR results, and the model using 'facet_tropism_score' is presented here.
- ![image](https://github.com/user-attachments/assets/7c43ec82-a243-4aa4-86f6-15a6c415fd96)
- MLR Adjusted for age and gender:
- ![image](https://github.com/user-attachments/assets/322d2cdb-7cb0-4561-ae73-ca81c18950b8)
- MLR Adjusted for age, gender and other co var including ["BMI", "Education", "Occupation_LBP_Baseline_Nominal", "Marital_status_LBP_Baseline", "Smoking_Baseline", "Alcohol_baseline"]:
- ![image](https://github.com/user-attachments/assets/d649a44b-85ed-41e9-8f5c-4c584175cf07)
- The current analysis reveals that the Pfirrmann score differs significantly between CLBP patients and asymptomatic individuals.
- TODO: adjust MLR by other vertebral body dimension. Since both vertebral dimensions and degeneration labels are level-specific, the next step is to explore how dimensions of each vertebra relates to localized degenerative changes.





