import matplotlib.pyplot as plt
import seaborn as sns
import json

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('results_file', type=str,
                    help='Path to the results file')
args = parser.parse_args()

# Read data from JSON file
results_file = args.results_file
with open(results_file, 'r') as file:
    data = json.load(file)

# Extract the required data
max_erase = [int(key.split(': ')[1].replace('}', '')) for key in data.keys()]
percent_safe = [value['percent_safe'] for value in data.values()]
time_per_prompt = [value['time_per_prompt'] for value in data.values()]

# Plotting
sns.set_style("darkgrid", {"grid.color": ".85"})
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot percent_safe on left y-axis
ax1.set_xlabel('Max Erased Tokens (= Certified Length)', fontsize=14, labelpad=12)
ax1.set_ylabel('Detcted Safe (%, empirical)', color='tab:green', fontsize=14)
ax1.plot(max_erase, percent_safe, '-', color='tab:green', label='Detcted Safe (%, empirical)', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:green', labelsize=14)
ax1.set_ylim(-0.5, 100.5)

ax1.axhline(y=93.6, color='tab:blue', linewidth=2, linestyle='--')

# Set the ticks for the left y-axis
ax1.set_yticks(range(0, 101, 20))
ax1.grid(linewidth=2)

# Set x-axis range and ticks
ax1.set_xlim(-0.05, 12.05)
ax1.set_xticks(range(0, 13, 4))
# ax1.set_xlim(-0.1, 30.1)        # ax1.set_xlim(-0.1, 50.2)
# ax1.set_xticks(range(0, 31, 10))
ax1.tick_params(axis='x', labelsize=14)

ax1.text(0.1, 88, 'Certified Harmful (93.6%)', fontsize=14, color='tab:blue')
# ax1.text(9, 88, 'Certified Harmful (93.6%)', fontsize=14, color='tab:blue')

# Create a second y-axis for time_per_prompt
ax2 = ax1.twinx()
ax2.set_ylabel('Time per Prompt (sec)', color='tab:brown', fontsize=14, labelpad=12)
ax2.plot(max_erase, time_per_prompt, '-', color='tab:brown', label='Time per Prompt (sec)', linewidth=2)
ax2.tick_params(axis='y', labelcolor='tab:brown', labelsize=14)
ax2.set_ylim(-0.009, 15.01)
# ax2.set_ylim(-0.009, 2.01)

# Set the ticks for the right y-axis
# ax2.set_yticks(np.arange(0, 2.1, 0.4))
ax2.set_yticks(np.arange(0, 15.1, 3))
ax2.grid(linewidth=2)

# Save the plot to a PNG file
output_file = results_file.replace('.json', '.png')
# plt.title('Comparison of percent_safe and time_per_prompt vs max_erase')
plt.savefig(output_file, bbox_inches='tight', dpi=300)
