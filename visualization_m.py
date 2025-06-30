from defs import*
import pandas as pd

df = pd.read_excel("inputs/GSTAT original.xlsx")
print(df.dtypes)

SAVE_FOLDER = "outputs/biserial"
os.makedirs(SAVE_FOLDER, exist_ok=True)
point_biserial_analysis(df, 'Modic_change_L3', 'SIWeightedCentroid_XCOR',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L3', 'SIWeightedCentroid_YCOR',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L3', 'Mean',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L3', 'Coeff of Variation',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L3', 'SIWeightedCentroid_YCOR',SAVE_FOLDER)

point_biserial_analysis(df, 'Modic_change_L4', 'SIWeightedCentroid_XCOR',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L4', 'SIWeightedCentroid_YCOR',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L4', 'Mean',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L4', 'Coeff of Variation',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L4', 'SIWeightedCentroid_YCOR',SAVE_FOLDER)

point_biserial_analysis(df, 'Modic_change_L5', 'SIWeightedCentroid_XCOR',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L5', 'SIWeightedCentroid_YCOR',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L5', 'Mean',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L5', 'Coeff of Variation',SAVE_FOLDER)
point_biserial_analysis(df, 'Modic_change_L5', 'SIWeightedCentroid_YCOR',SAVE_FOLDER)


