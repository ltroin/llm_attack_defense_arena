import json
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

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
            'percent_safe': data[num_adv][max_erase]['percent_safe']
        })

# Convert the data to a DataFrame
import pandas as pd
df = pd.DataFrame(plot_data)

# Set the seaborn style
sns.set_style("darkgrid", {"grid.color": ".85"})

# Create the plot
colors = ['tab:green', 'tab:olive']
plt.figure(figsize=(7, 6))
# Bar plot
sns.barplot(x='max_erase', y='percent_safe',
            # hue='num_adv',
            hue='model',
            data=df, palette=colors, alpha=0.7)
# for num_adv in [1, 2]:
#     subset = df[df['num_adv'] == num_adv]
#     # plt.plot(subset['max_erase'], subset['percent_safe'], label=f'# Adv Prompts = {num_adv}', linewidth=2, color=colors[num_adv - 1])
#     plt.bar(subset['max_erase'], subset['percent_safe'], color=colors[num_adv - 1], alpha=0.5, width=1, label=f'# Adv Prompts = {num_adv}')

# Set the labels, title, and legend
plt.xlabel("Max Erase Length (= Certified Length)", fontsize=14)
plt.ylabel("% Safe Prompts Labeled Safe", fontsize=14)
# plt.legend(loc='lower right', fontsize=14)
# change labels for the bar plots in the legend
handles, labels = plt.gca().get_legend_handles_labels()
# labels = ['# Insertions = 1', '# Insertions = 2']
labels = ['Llama 2', 'DistilBERT']
plt.legend(handles, labels, loc='lower right', fontsize=14)

plt.xticks(fontsize=14)
# plt.xlim(-0.02, 6.02)
# plt.xticks(df['max_erase'])
plt.ylim(0, 100)
plt.yticks(fontsize=14)
plt.grid(False)

# Save the plot to a PNG file
plot_file = results_file.replace('.json', '_acc.png')
plt.savefig(plot_file, bbox_inches='tight', dpi=300)
plt.show()
