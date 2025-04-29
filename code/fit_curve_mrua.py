import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score

sns.set_theme(font_scale=1.7, rc={"text.usetex": True})
sns.set_style("white")
this_filepath = os.path.abspath(__file__)

# input data
logging_filepath = os.path.join(
    os.path.dirname(this_filepath),
    "..",
    "data",
    "logging_trilho_acelerado1743623025.8807564.txt",
)

initial_time = 6
final_time = 10

# read file
with open(logging_filepath, "r") as f:
    lines = f.readlines()
    # Skip the first line (header)
    lines = lines[1:]
    data = []
    for line in lines:
        parts = line.strip().split("\t")
        timestamp = float(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        r = float(parts[3])
        data.append((timestamp, x, y, r))
df = pd.DataFrame(data, columns=["timestamp", "x", "y", "r"])

# Convert timestamp to seconds
df["timestamp"] = df["timestamp"] - df["timestamp"].min()
# Convert to milliseconds
# df["timestamp"] = df["timestamp"] * 1000

### filter data ###
# Remove rows with negative radius
df = df[df["r"] > 0]

# Remove rows with negative x or y
df = df[(df["x"] >= 0) & (df["y"] >= 0)]
### filter data ###

# plot data
plt.figure(figsize=(8, 7))
plt.scatter(
    df["timestamp"], df["x"], label="Experimental data", color="red", marker="o", s=15
)

# Remove outliers based on IQR of the radius
# Calculate IQR
q1 = df["r"].quantile(0.25)
q3 = df["r"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df["r"] >= lower_bound) & (df["r"] <= upper_bound)]

# crop
df = df[df["timestamp"] > initial_time]
df = df[df["timestamp"] < final_time]


# fit model
coef = np.polyfit(df["timestamp"], df["x"], 2)  # first-order fit (linear)
x_o, v_o, a_half = coef[2], coef[1], coef[0]
a = 2 * a_half
x_pred = np.polyval(coef, df["timestamp"])
print(f"x0: {x_o:.4f} px")
print(f"v: {v_o:.4f} px/s")
print(f"a: {a:.4f} px/s²")

R2 = r2_score(df["x"], x_pred)
print(f"R2: {R2:.4f}")

# plot data
plt.scatter(
    df["timestamp"],
    df["x"],
    label="Experimental data (filtered)",
    color="blue",
    marker="^",
    s=15,
)

# plot model
t_model_plot = np.arange(df["timestamp"].min(), df["timestamp"].max() + 0.3, 0.1)
x_model_plot = np.polyval(coef, t_model_plot)
plt.plot(
    t_model_plot,
    x_model_plot,
    color="black",
    label=f"Model: $x(t) = {x_o:.2f} + {v_o:.2f}t  {a_half:.2f}t^2$\n$R^2 = {R2:.4f}$",
)

# plot radius
# plt.scatter(df["timestamp"], df["r"], label="radius", color="green", marker="o")

plt.xlabel("Time [s]")
plt.ylabel("position [pixel]")
# plt.title("Position and Radius Over Time")
plt.grid(linestyle="--", alpha=0.7)
plt.legend(fontsize=18)

plt.savefig(
    os.path.join(os.path.dirname(this_filepath), "mrua_model.png"),
    bbox_inches="tight",
    dpi=600,
)

plt.savefig(
    os.path.join(os.path.dirname(this_filepath), "mrua_model.pdf"),
    bbox_inches="tight",
    dpi=600,
)

plt.show()
print("done")
