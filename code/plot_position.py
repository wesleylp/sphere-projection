import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# sns.set_theme(font_scale=1.7, rc={"text.usetex": True})
sns.set_style("white")
this_filepath = os.path.abspath(__file__)

logging_filepath = os.path.join(
    os.path.dirname(this_filepath),
    "..",
    "logging_trilho_acelerado1743623025.8807564.txt",
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
plt.plot(df["timestamp"], df["x"], label="x", color="blue", marker="o")
plt.plot(df["timestamp"], df["y"], label="y", color="green", marker="o")
plt.plot(df["timestamp"], df["r"], label="r", color="red", marker="o")
plt.xlabel("Timestamp (ms)")
plt.ylabel("pixels")
plt.title("Position and Radius Over Time")
plt.legend()
plt.grid()

plt.show()
plt.savefig(
    os.path.join(os.path.dirname(this_filepath), "position.png"),
    bbox_inches="tight",
)
