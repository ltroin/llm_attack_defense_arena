import json
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('results_file', type=str,
                    help='Path to the results file')
args = parser.parse_args()

# Read data from JSON file
results_file = args.results_file
with open(results_file, 'r') as file:
    data = json.load(file)

# Extract data
max_erase_values = []
percent_safe_values = []

for key, value in data.items():
    max_erase = int(key.split(': ')[1].replace('}', ''))
    max_erase_values.append(max_erase)
    percent_safe_values.append(value['percent_safe'])

# Plotting
sns.set_style("darkgrid", {"grid.color": ".85"})
plt.figure(figsize=(7, 6))
# plt.plot(max_erase_values, percent_safe_values, '-', color='tab:green', label='% Safe Prompts Labeled Safe', linewidth=2)
plt.bar(max_erase_values[0:4], percent_safe_values[0:4], color='tab:green', width=(0.2 * max_erase_values[3]), alpha=0.7)
# Write the values on top of the bars
for i, v in enumerate(percent_safe_values[0:4]):
    if i == 0:
        plt.text(max_erase_values[i] - (0.08 * max_erase_values[3]),  v - 5, f'{v:.1f}%', color='white', fontsize=14)
    elif i == 1:
        plt.text(max_erase_values[i] - (0.07 * max_erase_values[3]),  v - 5, f'{v:.1f}%', color='white', fontsize=14)
    else:
        plt.text(max_erase_values[i] - (0.07 * max_erase_values[3]),  v - 5, f'{v:.1f}%', color='white', fontsize=14)
        # plt.text(max_erase_values[i] - (0.07 * max_erase_values[3]),  v + 1, f'{v:.1f}%', color='gray', fontsize=14)
# plt.axhline(y=93.6, color='tab:blue', linewidth=2, linestyle='--', label='Certified Harmful Prompts (93.6%)')
plt.xlabel('Max Erase Length (= Certified Length)', fontsize=18, labelpad=10)
plt.ylabel('% Safe Prompts Labeled Safe', fontsize=18)
# plt.xlim(-0.05, 12.05)
# plt.xticks(range(0, 13, 4))
# plt.xlim(-0.1, 30.1)
# plt.xticks(range(0, 31, 10))
plt.xticks(max_erase_values[0:4])
plt.ylim(0, 100)
# plt.grid(True, linewidth=2)

# plt.legend(loc="lower right",fontsize=18)
# Remove grid lines
plt.gca().yaxis.grid(False)
plt.gca().xaxis.grid(False)

plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

# Save the figure
plot_file = results_file.replace('.json', '_acc.png')
plt.savefig(plot_file, dpi=300)
plt.close()
