import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def plot_density_differences(data, column, groups, group_labels, features, title="Group Comparison"):

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



def plot_histogram_differences(data, column, X, groups, group_labels, features, title="Group Comparison"):

    df = data[data[column].isin(groups)]

    num_cols = 2
    num_rows = (len(features) + 2) // num_cols

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=False)
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        approval_dict = {}
        for group in groups:
            group_data = df[df[column] == group]
            approval_dict[group] = group_data[X].value_counts(normalize=True)

        # Create bar plot for each race
        for i, race in enumerate(groups):
            axes[idx].bar(i, approval_dict[race][0]*100, label=f'{race}')

        axes[idx].set_title(f'{feature} by Race')
        axes[idx].set_xticks(range(len(groups)))
        axes[idx].set_xticklabels(group_labels)
        axes[idx].set_ylim([0, 100])
        axes[idx].set_ylabel('Percentage')

    # Create count plot of each race
    sns.countplot(x=column, data=data, order=groups, ax=axes[-1])
    for p in axes[-1].patches:
        axes[-1].annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha = 'center', va = 'bottom', xytext = (0, 10), textcoords = 'offset points')
    axes[-1].set_title('Count of Each Race')
    axes[-1].set_xlabel('')
    axes[-1].set_ylabel('Count')
    axes[-1].set_xticks(range(len(groups)))
    axes[-1].set_xticklabels(group_labels)

    for idx in range(len(features) + 1, len(axes)):
        fig.delaxes(axes[idx])

    # Set the main title and show the plot
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    


def approval_rates(data):
    group_labels = ["American Indian", "Asian", "African American", "Hawaiian/Pacific", "White", "Latino"]

    data = data[data.applicant_co_applicant_sex.isin(['1_2', '2_1', '1_1', '2_2'])]
    data = data[data.race_ethnicity.isin([1, 2, 3, 4, 5, 9])]
    data['approved'] = [1 if x == 'Approved' else 0 for x in data['action_taken']]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

    for i, group in enumerate(['low', 'middle', 'high']):
        subset = data[data.income_group == group]
        approval_rate = subset.groupby(['race_ethnicity', 'applicant_co_applicant_sex'])['approved'].mean() * 100

        ax = axes[i]
        approval_rate.unstack().plot(kind='bar', position=.5, width=0.5, ax=ax)

        ax.set_title(f'Loan Approval Rates for {group.capitalize()} Income Group')
        ax.set_xticklabels(group_labels, rotation=45)
        ax.set_xlabel("")
        ax.set_ylabel('Percentage of Approved Loans')

        if i == 0:
            ax.legend(title='Gender', loc='upper left', labels=["male-female", "female-male", "male-male", "female-female"])
        else:
            ax.get_legend().remove()

    fig.text(0.5, 0.04, 'Race / Ethnicity', ha='center', va='center')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.25)
    plt.show()