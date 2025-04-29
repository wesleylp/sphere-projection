import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.fft import fft


sns.set_theme(font_scale=1.7, rc={"text.usetex": True})
sns.set_style("white")
this_filepath = os.path.abspath(__file__)


def model_mhs(t, A, omega, phi, C):
    """Model MHS: x(t) = A * cos(omega*t + phi) + C"""
    return A * np.cos(omega * t + phi) + C


# input data
logging_filepath = os.path.join(
    os.path.dirname(this_filepath),
    "..",
    "data",
    "logging_massa_mola1743619723.9843779.txt",
)

initial_time = 5
final_time = 30

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
plt.figure(figsize=(10, 6))
plt.scatter(
    df["timestamp"], df["x"], label="Experimental data", color="red", marker="o", s=15
)

# Remove outliers based on IQR of the radius
# Calculate IQR
q1, q3 = df["r"].quantile([0.25, 0.75])
iqr = q3 - q1
df = df[df["r"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]


# crop
df = df[df["timestamp"] > initial_time]
df = df[df["timestamp"] < final_time]


# Estimate omega via FFT
dt = np.mean(np.diff(df["timestamp"]))
freq = np.fft.fftfreq(len(df), dt)
fft_val = np.abs(fft(df["x"] - np.mean(df["x"])))

# find the most significative peak (ignore too low frequencies)
mask = freq > 0.1  # Filter too-low frequencies
omega_guess = 2 * np.pi * freq[mask][np.argmax(fft_val[mask])]


print(f"Angular frequency estimate: w = {omega_guess:.2f} rad/s")


X = np.column_stack(
    [
        np.cos(omega_guess * df["timestamp"]),
        np.sin(omega_guess * df["timestamp"]),
        np.ones_like(df["timestamp"]),
    ]
)

coef = np.linalg.lstsq(X, df["x"], rcond=None)[0]
# coef = np.linalg.solve(X.T @ X, X.T @ df["x"])
alpha, beta, gamma = coef

A_linear = np.sqrt(alpha**2 + beta**2)
phi_linear = np.arctan2(-beta, alpha)
C_linear = gamma
params_linear = A_linear, omega_guess, phi_linear, C_linear

x_pred_linear = model_mhs(df["timestamp"], *params_linear)
R2_linear = r2_score(df["x"], x_pred_linear)

print(f"\nResults linear fit:")
print(f"Amplitude (A): {A_linear:.2f} pixels")
print(f"Frequência angular (omega): {omega_guess:.2f} rad/s")
print(f"Fase (phi): {phi_linear:.2f} rad")
print(f"Offset (C): {C_linear:.2f} pixels")
print(f"R2: {R2_linear:.4f}")

# Non-linear (fit using the linear params as initial guess)
params, _ = curve_fit(
    model_mhs,
    df["timestamp"],
    df["x"],
    p0=[A_linear, omega_guess, phi_linear, C_linear],
    bounds=([0, 0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf]),
)
A, omega, phi, C = params


x_pred = model_mhs(df["timestamp"], A, omega, phi, C)
R2 = r2_score(df["x"], x_pred)


print(f"\nResults non-linear fit:")
print(f"Amplitude (A): {A:.2f} pixels")
print(f"Frequência angular (omega): {omega:.2f} rad/s")
print(f"Fase (phi): {phi:.2f} rad")
print(f"Offset (C): {C:.2f} pixels")
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
t_plot = np.linspace(df["timestamp"].min(), df["timestamp"].max(), 500)

plt.plot(
    t_plot,
    model_mhs(t_plot, *params_linear),
    "--",
    color="green",
    label=f"Model: $x(t) = {A_linear:.2f} \cos({omega_guess:.2f}t  {phi_linear:.2f}) + {C_linear:.2f}$\n"
    f"$R^2 = {R2_linear:.4f}$",
)

plt.plot(
    t_plot,
    model_mhs(t_plot, *params),
    "--",
    color="black",
    label=f"Model: $x(t) = {A:.2f} \cos({omega:.2f}t  {phi:.2f}) + {C:.2f}$ (refined)\n"
    f"$R^2 = {R2:.4f}$",
)


# plot radius
# plt.scatter(df["timestamp"], df["r"], label="radius", color="green", marker="o")

plt.xlabel("Time [s]")
plt.ylabel("position [pixel]")
# plt.title("Position and Radius Over Time")
plt.grid(linestyle="--", alpha=0.7)
plt.legend(fontsize=18)
plt.ylim(340, 400)


plt.savefig(
    os.path.join(os.path.dirname(this_filepath), "mhs_model.png"),
    bbox_inches="tight",
    dpi=600,
)
plt.savefig(
    os.path.join(os.path.dirname(this_filepath), "mhs_model.pdf"),
    bbox_inches="tight",
    dpi=600,
)

plt.show()
print("done")


print("done")
