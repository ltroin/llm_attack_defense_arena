import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('results_file', type=str,
                    help='Path to the results file')
args = parser.parse_args()

# Read the JSON data from a file
filename = args.results_file    # "results/smoothing_30.json"
with open(filename, "r") as file:
    data = json.load(file)

# Set the seaborn style
sns.set_style("darkgrid")

# Create the plot
plt.figure(figsize=(6, 5))

# Draw the horizontal line at 90% accuracy
plt.axhline(y=92.6, label="Our certificate", color="tab:blue", linewidth=2)

colors = ["tab:orange", "tab:green", "tab:brown", "tab:purple", "tab:olive", "tab:cyan"]

for key, value in data.items():
    max_erase = int(key.split(":")[1].strip().split("}")[0])
    plt.plot(value["certified_accuracy"], label=f"Max erase: {max_erase}", linewidth=2, linestyle='--', color=colors.pop(0))

# Set the plot properties
plt.xlabel('Certified Length', fontsize=14, labelpad=10)
plt.ylabel('Certified Accuracy', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 100)
plt.xlim(0, 30)
plt.legend(fontsize=14)
plt.tight_layout()

# Save the plot to a PNG file
plt.savefig(filename.replace(".json", ".png"), dpi=300)
