import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def plot_group_differences(data, column, groups, group_labels, features, title="Group Comparison"):

    df = data[data[column].isin(groups)]
    # Calculate the number of rows and columns for the subplots
    num_cols = 2
    num_rows = (len(features) + 1) // num_cols

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=False)
    axes = axes.flatten()

    # Iterate through each feature and create the density plots
    for idx, feature in enumerate(features):
        for ix, group in enumerate(groups):
            group_data = df[df[column] == group]
            if '_000s' in feature:
                sns.kdeplot(group_data[feature], ax=axes[idx], label=group_labels[ix], clip=(0, 1000))
            else:    
                sns.kdeplot(group_data[feature], ax=axes[idx], label=group_labels[ix], clip=(0, 1000000000))
        
        axes[idx].set_title(f'Distribution of {feature}')
        axes[idx].legend(title=column)
    
        # Remove unused subplots
    for idx in range(len(features), len(axes)):
        fig.delaxes(axes[idx])

    # Set the main title and show the plot
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()