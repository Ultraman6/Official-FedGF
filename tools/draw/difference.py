import pandas as pd
from matplotlib import pyplot as plt
columns_to_visualize = [
    "Div of Update standard",
    "Div of Perturb standard",
    "Cov of Update and Perturb",
    "Mean of Residual",
    "Std of Residual"
]
# Export the plot with larger text and thicker lines
enhanced_plot_pdf_path = '/mnt/data/enhanced_plot.pdf'
width_in_inches = 496 / 72
height_in_inches = 279 / 72
large_window_size = 50
# Reload the newly uploaded data
file_path_new = '/mnt/data/Abs ov of Update and Perturb.xlsx'
new_data = pd.read_excel(file_path_new)
# Inspect the content of the uploaded file
new_data.head()
# Create the plot with matched dimensions, enlarged text, and thicker lines
plt.figure(figsize=(width_in_inches, height_in_inches))
# Apply smoothing (moving average) to each column
window_size = 10  # Set window size for smoothing
smoothed_data = new_data[columns_to_visualize].rolling(window=window_size, min_periods=1).mean()
extremely_smoothed_data = smoothed_data.rolling(window=large_window_size, min_periods=1).mean()
# Plot each curve with thicker lines and custom color for "Div of Update standard"
for column in columns_to_visualize:
    if column == "Div of Update standard":
        plt.plot(new_data["Step"], extremely_smoothed_data[column], label=column, color='darkgreen', linewidth=2)
    else:
        plt.plot(new_data["Step"], extremely_smoothed_data[column], label=column, linewidth=2)

# Customize the axis labels and ticks with larger, bold fonts
plt.xlabel('Communication Rounds', fontsize=20, weight='bold')
plt.ylabel('Difference', fontsize=20, weight='bold')
plt.xticks(ticks=[0, 50, 100], labels=[0, 50, 100], fontsize=18, weight='bold')
plt.yticks(ticks=[extremely_smoothed_data.min().min(),
                  (extremely_smoothed_data.min().min() + extremely_smoothed_data.max().max()) / 2,
                  extremely_smoothed_data.max().max(),
                  (extremely_smoothed_data.max().max() - extremely_smoothed_data.min().min()) / 4],
           fontsize=18, weight='bold')

# Adjust the legend with larger, bold text and transparency
plt.legend(fontsize=18, framealpha=0.5, prop={'weight': 'bold'})  # Use prop for text weight
plt.grid(True)

# Save the enhanced plot as a PDF
plt.savefig(enhanced_plot_pdf_path, format='pdf', bbox_inches='tight')
plt.close()