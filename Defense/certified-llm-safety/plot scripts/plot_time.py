import json
import matplotlib.pyplot as plt
import seaborn as sns

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

# Extract data
max_erase_values = []
time_per_prompt_values = []

for key, value in data.items():
    max_erase = int(key.split(': ')[1].replace('}', ''))
    max_erase_values.append(max_erase)
    time_per_prompt_values.append(value['time_per_prompt'])

# Plotting
sns.set_style("darkgrid", {"grid.color": ".85"})
plt.figure(figsize=(7, 6))
# plt.plot(max_erase_values, time_per_prompt_values, linewidth=2)
plt.bar(max_erase_values[0:4], time_per_prompt_values[0:4], color='tab:blue', alpha=0.7, width=(0.2 * max_erase_values[3]))
# Write the values on top of the bars
for i, v in enumerate(time_per_prompt_values[0:4]):
    plt.text(max_erase_values[i] - (0.05 * max_erase_values[3]),  v + 0.02, f'{v:.2f}', color='gray', fontsize=14)
    # plt.text(max_erase_values[i] - (0.06 * max_erase_values[3]),  v + 0.3, f'{v:.2f}', color='gray', fontsize=14)
    # plt.text(max_erase_values[i] - (0.05 * max_erase_values[3]),  v + 0.2, f'{v:.2f}', color='gray', fontsize=14)
plt.xlabel('Max Erase Length', fontsize=18, labelpad=10)
plt.ylabel('Time per Prompt (sec)', fontsize=18, labelpad=10)

# plt.xlim(-0.05, 12.04)
# plt.xticks(range(0, 13, 4))
plt.ylim(0, 1.5)
plt.yticks(np.arange(0, 1.51, 0.3))
# plt.ylim(0, 15)
# plt.yticks(np.arange(0, 15.1, 3))
# plt.ylim(0, 80)
# plt.yticks(np.arange(0, 81, 20))
# plt.ylim(0, 1.8)
# plt.yticks(np.arange(0, 1.81, 0.3))

# plt.xlim(-0.1, 30.1)
plt.xticks(max_erase_values[0:4])


# plt.grid(True, linewidth=2)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

# Remove grid lines
plt.gca().yaxis.grid(False)
plt.gca().xaxis.grid(False)

# Save the figure
plot_file = results_file.replace('.json', '_time.png')
plt.savefig(plot_file, dpi=300)
plt.close()
