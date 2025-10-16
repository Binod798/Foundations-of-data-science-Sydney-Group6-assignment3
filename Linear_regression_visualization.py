import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path


# Load datasets
bat = pd.read_csv("dataset1.csv")
rat = pd.read_csv("dataset2.csv")

print("Bat dataset shape:", bat.shape)
print("Rat dataset shape:", rat.shape)

# Clean + Basic inspection
bat = bat.dropna(subset=["risk", "reward", "season", "bat_landing_to_food"])
rat = rat.dropna(subset=["rat_arrival_number", "food_availability"])