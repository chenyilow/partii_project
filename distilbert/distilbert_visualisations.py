# Copyright 2025 [Your Name]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 16})

col = "ref"
reduced = True
if col == "ref" or col == "test":
    labels = ['pred/prop', 'spec', 'nonspec']
elif col == "def":
    labels = ["def", "indef"]
elif col == "Hawkins":
    labels = ["indef", "situational", "explanatory", "anaphoric"]

df = pd.read_csv(f"cross_validation_results_{col.lower()}.csv")

runs = []
for _, row in df.iterrows():
    runs.append({
        "eval_accuracy": row["Accuracy"],
        "eval_precision": ast.literal_eval(row["Precision"]),
        "eval_recall": ast.literal_eval(row["Recall"]),
        "eval_f1": ast.literal_eval(row["F1 Score"])
    })

# Define base colors for each subplot
base_colors = ['Blues', 'Reds', 'pink', 'Greens']  # Blue, Red, Yellow-Green, Green

# Create a 2x2 grid of subplots with a larger figure size
if reduced:
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Data for plotting
if reduced:
    metrics = ['eval_accuracy', 'eval_f1']
else:
    metrics = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']

for i, (metric, cmap) in enumerate(zip(metrics, base_colors)):
    if reduced:
        ax = axes[i % 2]
    else:
        ax = axes[i // 2, i % 2]  # Access the correct subplot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if cmap == 'pink':
        colors = [plt.get_cmap(cmap)(0.1 + 0.1 * j) for j in range(len(labels))]  # Reverse the colormap
    else:
        colors = [plt.get_cmap(cmap)(0.9 - 0.1 * j) for j in range(len(labels))]  # Darker shades
    colors_acc = [plt.get_cmap("Blues")(0.5 + 0.1*j) for j in range(10)]
    
    bar_width = 0.5/len(labels) # Adjust for spacing
    epoch_positions = np.arange(1, 4)
    num_bars_per_epoch = 5
    confidence_level = 0.95
    n = 10

    if metric != "eval_accuracy":
        for ann in range(len(labels)):
            for epoch in range(3):
                mean_array = []
                for fold in range(10):
                    mean_array.append(runs[fold * 3 + epoch][metric])
                x_position = epoch + 0.75 + 0.25/len(labels) + ann * bar_width
                y_values = [i[ann + 2] for i in mean_array]
                mean_val = np.mean(y_values)
                err = stats.t.ppf((1 + confidence_level) / 2, n-1) * np.std(y_values) / np.sqrt(n)
                if epoch == 0:
                    ax.bar(x_position, mean_val, width=bar_width, color=colors[ann], label = labels[ann], yerr=err, capsize=5)
                else:
                    ax.bar(x_position, mean_val, width=bar_width, color=colors[ann], yerr=err, capsize=5)
                print(f"Epoch: {epoch+1}, Metric: {metric}, Mean: {mean_val:.2f}, Std Dev: {err:.2f}, Label: {labels[ann]}")
    else:
        for fold in range(10):
            ax.plot(range(1, 4), [epoch[metric] for epoch in runs[fold * 3: (fold + 1) * 3]], color=colors_acc[fold], label = f"Fold {fold+1}")
        epoch_mean = []
        epoch_std = []
        for j in range(3):
            fold_data = []
            for k in range(j, 30, 3):
                fold_data.append(runs[k][metric])
            mean_val = np.mean(fold_data)
            margin_of_error = stats.t.ppf((1 + confidence_level) / 2., n-1) * (np.std(fold_data) / np.sqrt(n))
            epoch_mean.append(mean_val)
            epoch_std.append(margin_of_error)
        ax.errorbar(range(1, 4), epoch_mean, yerr=epoch_std, fmt='-o', color="black", capsize=5, elinewidth=2)
        for l in range(3):
            print(f"Epoch: {l+1}, Metric: {metric}, Mean: {epoch_mean[l]:.2f}, Std Dev: {epoch_std[l]:.2f}")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    ax.set_title(f'{metric.capitalize()}', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    if i > 0:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max)

plt.tight_layout()
filename = "visualisations/distilbert_visualisations_" + col
if reduced:
    filename += "_reduced"
plt.savefig(filename)
plt.show()