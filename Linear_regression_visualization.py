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
# Linear regression for Bat behaviours vs Season

# Risk
risk_model = smf.ols("risk ~ season_numeric", data=bat).fit()
# Reward
reward_model = smf.ols("reward ~ season_numeric", data=bat).fit()
# Landing-to-food delay
delay_model = smf.ols("bat_landing_to_food ~ season_numeric", data=bat).fit()

print("\nLinear Regression Summaries:")
print("Risk ~ Season")
print(risk_model.summary())
print("Reward ~ Season")
print(reward_model.summary())
print("Landing-to-food delay ~ Season")
print(delay_model.summary())

# Number of landings per season
landing_counts = bat.groupby("season_label")["bat_landing_to_food"].count().reset_index()
landing_counts.rename(columns={"bat_landing_to_food": "landing_count"}, inplace=True)

plt.figure(figsize=(6,5))
sns.barplot(x="season_label", y="landing_count", data=landing_counts, palette="muted")
plt.title("Number of Bat Landings by Season")
plt.text(0, landing_counts["landing_count"].max()*0.95,
         f"Winter: {landing_counts['landing_count'][0]}, Spring: {landing_counts['landing_count'][1]}",
         color='blue', fontsize=12)
plt.savefig("bat_landings_by_season.png")
plt.close()

# Visualizations with regression and annotations
sns.set(style="whitegrid", palette="muted")

# Risk
plt.figure(figsize=(10,5))
sns.boxplot(x="season_label", y="risk", data=bat, showfliers=False)
sns.stripplot(x="season_label", y="risk", data=bat, color="black", alpha=0.5, jitter=True)
sns.regplot(x="season_numeric", y="risk", data=bat, scatter=False, color="red")
plt.title("Bat Risk by Season")
plt.text(0.5, bat["risk"].max()*0.95, f"t-test p = {risk_ttest.pvalue:.4f}\nCoef = {risk_model.params['season_numeric']:.2f}",
         ha='center', color='blue', fontsize=12)
plt.savefig("bat_risk_by_season.png")
plt.close()

# Reward
plt.figure(figsize=(10,5))
sns.boxplot(x="season_label", y="reward", data=bat, showfliers=False)
sns.stripplot(x="season_label", y="reward", data=bat, color="black", alpha=0.5, jitter=True)
sns.regplot(x="season_numeric", y="reward", data=bat, scatter=False, color="red")
plt.title("Bat Reward by Season")
plt.text(0.5, bat["reward"].max()*0.95, f"t-test p = {reward_ttest.pvalue:.4f}\nCoef = {reward_model.params['season_numeric']:.2f}",
         ha='center', color='blue', fontsize=12)
plt.savefig("bat_reward_by_season.png")
plt.close()

# Landing-to-food delay
plt.figure(figsize=(10,5))
sns.boxplot(x="season_label", y="bat_landing_to_food", data=bat, showfliers=False)
sns.stripplot(x="season_label", y="bat_landing_to_food", data=bat, color="black", alpha=0.5, jitter=True)
sns.regplot(x="season_numeric", y="bat_landing_to_food", data=bat, scatter=False, color="red")
plt.title("Bat Landing-to-Food Delay by Season")
plt.text(0.5, bat["bat_landing_to_food"].max()*0.95, f"t-test p = {delay_ttest.pvalue:.4f}\nCoef = {delay_model.params['season_numeric']:.2f}",
         ha='center', color='blue', fontsize=12)
plt.savefig("bat_landing_to_food_by_season.png")
plt.close()