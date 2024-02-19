import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse

# Function to read data from a json file and return a DataFrame
def read_data_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    plot_data = []
    for max_erase_key, adv_toks in data.items():
        max_erase_value = json.loads(max_erase_key.replace("'", "\""))['num_iters']
        # max_erase_value = float(json.loads(max_erase_key.replace("'", "\""))['sampling_ratio'])
        # max_erase_value = int(json.loads(max_erase_key.replace("'", "\""))['max_erase'])
        for adv_tok_key, metrics in adv_toks.items():
            adv_tok_value = int(json.loads(adv_tok_key.replace("'", "\""))['adv_tok'])
            plot_data.append({
                'sampling_ratio': max_erase_value,
                # 'max_erase': max_erase_value,
                'adv_tok': adv_tok_value,
                'percent_harmful': metrics['percent_harmful']
            })

    return pd.DataFrame(plot_data)

# Plotting function that saves the plot to a file
def plot_data(df, filename='plot.png'):
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))
    plot = sns.lineplot(data=df, x='adv_tok', y='percent_harmful', hue='sampling_ratio', marker='o', palette='tab10')
    # plot = sns.lineplot(data=df, x='adv_tok', y='percent_harmful', hue='max_erase', marker='o', palette='tab10')

    # plot.set_title('Percent Harmful vs. Adversarial Tokens', fontsize=16)
    plot.set_xlabel('Adversarial Sequence Length (in tokens)', fontsize=14)
    plot.set_ylabel('Percent Harmful', fontsize=14)
    plot.set_xticks(range(0, 21, 4))
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(title='# Iterations', title_fontsize=14, fontsize=14)
    # plt.legend(title='Sampling Ratio', title_fontsize=14, fontsize=14)
    # plt.legend(title='Max Erase', title_fontsize=14, fontsize=14)
    
    # Save the figure
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the figure to avoid displaying it

# Main script execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot empirical results.')
    parser.add_argument('json_file_path', type=str, default='results/empirical_suffix_120_clf_rand.json', help='Path to json file')

    args = parser.parse_args()

    json_file_path = args.json_file_path

    df = read_data_from_json(json_file_path)
    plot_file = json_file_path.replace('.json', '.png')
    plot_data(df, plot_file)  # The plot will be saved as 'percent_harmful_vs_adv_tok.png'
