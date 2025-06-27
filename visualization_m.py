from defs import*
import pandas as pd


df = pd.read_csv("inputs/merged_GSTAT.csv")
print(df.dtypes)

plot_density_heatmap(df, x='WeightedCentroid_XCOR', y='WeightedCentroid_YCOR', hue='folder_index')
plot_density_heatmap(df, x='SIWeightedCentroid_XCOR', y='SIWeightedCentroid_YCOR', hue='folder_index')


