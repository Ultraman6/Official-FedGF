# Simulating reading from an Excel file as a DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Manually creating the DataFrame to simulate the data from the provided content
df = pd.DataFrame({
    "dirichlet": [0, 0.05, 0.1, 0.2, 0.5, 1, 100],
    "Div Perturb": [12.1, 9.43, 7.76, 5.21, 3.4638, 2.965, 2.8421],
    "Div Update": [16.9732, 21.6551, 22.611, 6.95, 5.08, 3.71, 1.74],
    "Cov Update and Perturb": [21.016, 18.736, 39, 8.95, 5.08, 3.244, 3.47],
    "Div Residual": [1.394293177, 1.461694908, 1.245546915, 1.329944524, 1.557764119, 1.697307712, 1.356985313],
    "Val Acc%": [33.74, 28.8, 44.46, 45.88, 37.3, 44.36, 41.76],
    "Test Acc%": [12.65, 15.04, 17.83, 22.8, 27.64, 32.93, 43.24]
})

# Set a display-friendly font globally
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Create an array of equal spacing for the x-axis based on the length of the data
equal_spacing_x = np.arange(len(df["dirichlet"]))

# Save the plot with larger fonts and export as a PDF
output_pdf_path = 'E:\Github\Official-FedGF/tools\draw/enlarged_fonts_metrics_plot.pdf'

width_in_inches = 496 / 72
height_in_inches = 279 / 72
# Regenerate the plot with adjusted legend position
fig, ax1 = plt.subplots(figsize=(width_in_inches, height_in_inches))

# Plot left-axis metrics
ax1.set_xlabel("Heterogeneity Coefficient (dirichlet)", fontsize=20, color="black")
ax1.set_ylabel("Metrics", fontsize=20, color="black")
ax1.plot(equal_spacing_x, df["Div Perturb"], label="Div Perturb", color="blue", linestyle='-')
ax1.plot(equal_spacing_x, df["Div Update"], label="Div Update", color="cyan", linestyle='--')
ax1.plot(equal_spacing_x, df["Cov Update and Perturb"], label="Cov Update and Perturb", color="green", linestyle='-.')
ax1.plot(equal_spacing_x, df["Div Residual"], label="Div Residual", color="purple", linestyle=':')
ax1.set_ylim(0, 10)  # Left y-axis range: 0 to 10
ax1.tick_params(axis="y", labelcolor="black", labelsize=16)
ax1.tick_params(axis="x", labelcolor="black", labelsize=16)
ax1.set_xticks(equal_spacing_x)  # Set x-ticks to the new equal spacing
ax1.set_xticklabels(df["dirichlet"])  # Label the ticks with the actual dirichlet values

# Add a legend for the left-axis metrics with fully transparent background
ax1.legend(loc="upper left", fontsize=18, framealpha=0.5, prop={'weight': 'bold'})  # Fully transparent legend

# Plot right-axis metrics
ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy (%)", fontsize=20, color="black")
ax2.plot(equal_spacing_x, df["Val Acc%"], label="Val Acc%", color="red", linestyle='-')
ax2.plot(equal_spacing_x, df["Test Acc%"], label="Test Acc%", color="orange", linestyle='--')
ax2.set_ylim(0, df[["Val Acc%", "Test Acc%"]].max().max())  # Right y-axis range: 0 to max value
ax2.tick_params(axis="y", labelcolor="black", labelsize=16)

# Add a legend for the right-axis metrics, moved to the center of the vertical axis
ax2.legend(loc="center right", fontsize=18, framealpha=0.5, prop={'weight': 'bold'})  # Fully transparent legend

# Modify the top border to a black dashed line
ax1.spines["top"].set_linestyle("--")
ax1.spines["top"].set_color("black")

# Remove the title and add gridlines
plt.grid(True)
plt.tight_layout()  # Ensure proper spacing for the plot

# Save the figure as a PDF
plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight')
plt.close()