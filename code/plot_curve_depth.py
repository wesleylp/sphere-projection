import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_theme(font_scale=1.7, rc={"text.usetex": True})
sns.set_style("white")

this_filepath = os.path.abspath(__file__)

language = "pt-br"

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
# Calculate the mean and standard deviation for 'Read' grouped by 'Real' and 'Ref'
data_processed = (
    data.groupby(["Real", "Ref"])["Read"].agg(["mean", "std"]).reset_index()
)

# Calculate the mean and standard deviation for 'Absdiff' grouped by 'Real' and 'Ref'
stats_df = data.groupby(["Real", "Ref"])["AbsDiff"].agg(["mean", "std"]).reset_index()


references = data_processed["Ref"].unique()
# Colors for different datasets (if you want specific colors)
colors = ["blue", "green", "red"]
color_dict = dict(zip(references, colors))

#####
# Plotting each reference
plt.figure(figsize=(10, 6))
for reference in references:
    subset = data_processed[stats_df["Ref"] == reference]
    plt.errorbar(
        subset["Real"],
        subset["mean"],
        yerr=subset["std"],
        label=reference,
        fmt="-o",
        capsize=5,
        color=color_dict[reference],
    )

# Add reference line y = x
# Determine the range of x-values
x_values = np.linspace(0, 100, 400)
plt.plot(
    x_values, x_values, "-", color=(0.5, 0.5, 0.5), label="y = x"
)  # Plot the reference line y = x in gray

if language == "pt-br":
    plt.xlabel(r"Profundidade real $z$[cm]")
    plt.ylabel(r"Média da estimativa de profundidade $\hat{z}$ [cm]")
    plt.legend(title=r"Referência ($z^\ast$)")
else:
    plt.xlabel(r"Real depth $z$[cm]")
    plt.ylabel(r"Mean depth estimate $\hat{z}$ [cm]")
    plt.legend(title=r"Reference ($z^\ast$)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.xlim(5, 85)
plt.ylim(5, 85)

# Set the plot to log scale for both axes
# plt.gca().set_xscale("log")
# plt.gca().set_yscale("log")

plt.savefig(f"trend_mean_estimate_{language}.png", dpi=300, bbox_inches="tight")
plt.show()
####


# Plotting each reference
plt.figure(figsize=(10, 6))
for reference in references:
    subset = stats_df[stats_df["Ref"] == reference]
    plt.errorbar(
        subset["Real"],
        subset["mean"],
        yerr=subset["std"],
        label=reference,
        fmt="-o",
        capsize=5,
        color=color_dict[reference],
    )

# plt.title("Trend of Difference Between Real and Read Values by Data Source")
if language == "pt-br":
    plt.xlabel(r"Profundidade real $z$[cm]")
    plt.ylabel(r"Média do erro absoluto (MAE) $|z-\hat{z}|$ [cm]")
    plt.legend(title=r"Referência ($z^\ast$)")
else:
    plt.xlabel(r"Real depth $z$[cm]")
    plt.ylabel(r"Mean absolute error (MAE) $|z-\hat{z}|$ [cm]")
    plt.legend(title=r"Reference ($z^\ast$)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.savefig(f"trend_mean_abs_error_{language}.png", dpi=300, bbox_inches="tight")
plt.show()
####


plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x="Real",
    y="mean",
    hue="Ref",
    data=stats_df,
    palette=color_dict,
    errorbar="sd",
    capsize=3,
)

# plt.title("Mean absolute error between Real and Estimates Measures")
if language == "pt-br":
    plt.xlabel(r"Profundidade real $z$ [cm]")
    plt.ylabel(r"Média do erro absoluto (MAE) $|z-\hat{z}|$ [cm]")
    plt.legend(title=r"Referência ($z^\ast$)")
else:
    plt.xlabel(r"Real depth $z$ [cm]")
    plt.ylabel(r"Mean absolute error (MAE) $|z-\hat{z}|$ [cm]")
    plt.legend(title=r"Reference ($z^\ast$)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.savefig(f"mean_abs_error_{language}.png", dpi=300, bbox_inches="tight")
plt.show()


print("done!")
