from defs import *
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
df = pd.read_excel("inputs/combined_texture_data_reordered.xlsx")
df.rename(columns={'Unnamed: 0': 'texture_features'}, inplace=True)
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/combined_texture_data_reordered.csv", index=False)
group_cols = ['folder_index', 'sub_index', 'spine_index']
value_cols = [col for col in df.columns if col not in group_cols and col != 'texture_features']
df_pivot = df.pivot_table(index=group_cols, columns='texture_features', values=value_cols, aggfunc='first')
df_pivot.columns = [f"{texture}_{col}" for col, texture in df_pivot.columns]
df_pivot.reset_index(inplace=True)
os.makedirs("outputs", exist_ok=True)
df_pivot.to_csv("outputs/texture_feature_pivot.csv", index=False)

target_col = 'folder_index'
X, y_raw, feature_cols = prepare_features_and_labels(df_pivot, group_cols, target_col)

le = LabelEncoder()
y = le.fit_transform(y_raw)
rf_model, accuracy, cm = train_evaluate_rf(X, y, le, title_suffix=" for Folder Index")
