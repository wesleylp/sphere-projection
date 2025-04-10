import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(font_scale=1.7, rc={"text.usetex": True})
sns.set_style("white")
this_filepath = os.path.abspath(__file__)

logging_filepath = os.path.join(
    os.path.dirname(this_filepath),
    "..",
    "data",
    "logging_trilho1743622371.80657.txt",
)


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
df["timestamp"] = df["timestamp"] * 1000

# Remove rows with negative radius
df = df[df["r"] > 0]

# Remove rows with negative x or y
df = df[(df["x"] >= 0) & (df["y"] >= 0)]

# plot the data
plt.figure(figsize=(10, 6))
plt.plot(df["timestamp"], df["x"], label="position", color="red", marker="o")


# pass median filter in x
# df["x"] = df["x"].rolling(window=5, center=True).median()
# plt.plot(
#     df["timestamp"], df["x"], label="position (filtered)", color="blue", marker="^"
# )


# Remove outliers based on IQR of the radius
# Calculate IQR
q1 = df["r"].quantile(0.25)
q3 = df["r"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df["r"] >= lower_bound) & (df["r"] <= upper_bound)]


plt.plot(
    df["timestamp"], df["x"], label="position (filtered)", color="blue", marker="^"
)


# plt.plot(df["timestamp"], df["y"], label="y", color="green", marker="o")
plt.plot(df["timestamp"], df["r"], label="radius", color="green", marker="o")
plt.xlabel("Timestamp (ms)")
plt.ylabel("value [pixel]")
plt.title("Position and Radius Over Time")
plt.legend()
plt.grid()


plt.savefig(
    os.path.join(os.path.dirname(this_filepath), "mru.png"),
    bbox_inches="tight",
)

plt.show()
print("done")
