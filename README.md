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




