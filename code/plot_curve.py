import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

this_filepath = os.path.abspath(__file__)

# Load data
datapath = os.path.join(os.path.dirname(this_filepath), "..", "data", "results.xls")

references = [10, 50, 80]

frames = []
for ref in references:
    temp_df = pd.read_excel(
        datapath,
        sheet_name=f"{ref}",
    )
    temp_df["Ref"] = ref
    temp_df["AbsDiff"] = np.abs(temp_df["Real"] - temp_df["Read"])
    frames.append(temp_df)

data = pd.concat(frames, ignore_index=True)
# Calculate the mean and standard deviation for 'Difference' grouped by 'Real' and 'DataSet'
stats_df = data.groupby(["Real", "Ref"])["AbsDiff"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x="Real",
    y="mean",
    hue="Ref",
    data=stats_df,
    palette="viridis",
    ci="sd",
    capsize=3,
)

plt.title("Absolute difference between Real values and Estimates")
plt.xlabel("Real")
plt.ylabel("Absolute difference")
plt.legend(title="reference")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()


print("done!")
