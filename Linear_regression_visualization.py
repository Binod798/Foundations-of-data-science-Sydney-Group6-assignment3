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

# Convert season to categorical labels
season_map = {0: "Winter", 1: "Spring"}
bat["season_label"] = bat["season"].map(season_map)
bat["season_numeric"] = bat["season"]  # for regression

# Descriptive statistics
print("\nBat behaviour by season:")
print(bat.groupby("season_label")[["risk", "reward", "bat_landing_to_food"]].describe())

print("\nBat landings count by season:")
print(bat.groupby("season_label")["bat_landing_to_food"].count())

print("\nRat activity summary:")
print(rat[["rat_arrival_number", "food_availability"]].describe())

# t-tests for seasonal differences
winter = bat[bat["season_label"] == "Winter"]
spring = bat[bat["season_label"] == "Spring"]

# Risk
risk_ttest = ttest_ind(winter["risk"], spring["risk"], equal_var=False)
# Reward
reward_ttest = ttest_ind(winter["reward"], spring["reward"], equal_var=False)
# Landing-to-food delay
delay_ttest = ttest_ind(winter["bat_landing_to_food"], spring["bat_landing_to_food"], equal_var=False)

print("\nT-test Results:")
print(f"Risk difference p-value = {risk_ttest.pvalue:.4f}")
print(f"Reward difference p-value = {reward_ttest.pvalue:.4f}")
print(f"Landing-to-food delay p-value = {delay_ttest.pvalue:.4f}")