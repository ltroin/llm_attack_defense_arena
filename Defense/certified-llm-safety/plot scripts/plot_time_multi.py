import json
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('results_file', type=str,
                    help='Path to the results file')
args = parser.parse_args()

# Read data from JSON file
results_file = args.results_file
with open(results_file, 'r') as file:
    data = json.load(file)

# Extract the data for plotting
num_adv_values = list(data.keys())
max_erase_values = list(data[num_adv_values[0]].keys())

plot_data = []
for num_adv in num_adv_values:
    for max_erase in max_erase_values:
        plot_data.append({
            # 'num_adv': int(num_adv.split(': ')[1][:-1]),
            'model': num_adv.split(': ')[1][:-1],
            'max_erase': int(max_erase.split(': ')[1][:-1]),
            'time_per_prompt': data[num_adv][max_erase]['time_per_prompt']
        })

# Convert the data to a DataFrame
df = pd.DataFrame(plot_data)

# Set the seaborn style
sns.set_style("darkgrid", {"grid.color": ".85"})

# Create the plot
colors = ['tab:blue', 'tab:purple']
plt.figure(figsize=(7, 6))
# Bar plot
sns.barplot(x='max_erase', y='time_per_prompt',
            # hue='num_adv',
            hue='model',
            data=df, palette=colors, alpha=0.7)
# for num_adv in [1, 2]:
#     subset = df[df['num_adv'] == num_adv]
#     plt.plot(subset['max_erase'], subset['time_per_prompt'], label=f'# Adv Prompts = {num_adv}', linewidth=2, color=colors[num_adv - 1])

# Set the labels, title, and legend
plt.xlabel("Max Erase Length", fontsize=14)
plt.ylabel("Time per Prompt (sec)", fontsize=14)
# plt.legend(loc='upper left', fontsize=14)
# change labels for the bar plots in the legend
handles, labels = plt.gca().get_legend_handles_labels()
# labels = ['# Insertions = 1', '# Insertions = 2']
labels = ['Llama 2', 'DistilBERT']
plt.legend(handles, labels, loc='upper left', fontsize=14)
# plt.xlim(-0.02, 6.02)
# plt.xticks(range(0, 7, 2), fontsize=14)
plt.ylim(0, 1.5)
plt.yticks(np.arange(0, 1.51, 0.3), fontsize=14)
# plt.ylim(0, 15)
# plt.yticks(np.arange(0, 15.1, 3), fontsize=14)
# plt.ylim(0, 180)
# plt.yticks(range(0, 181, 45), fontsize=14)
plt.grid(False)

plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

# Save the plot to a PNG file
plot_file = results_file.replace('.json', '_time.png')
plt.savefig(plot_file, bbox_inches='tight', dpi=300)
plt.show()
