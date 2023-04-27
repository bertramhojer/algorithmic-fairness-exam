import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import os

def data_loader(num=10000):
    # load csv file processed_data if it exists. Check if it exists with os module
    print('Loading data...')
    if os.path.exists('data/processed_data.csv'):
        print('processed_data.csv exists. Loading data from file.')
        df = pd.read_csv('data/processed_data.csv')
    else:
        print('processed_data.csv does not exist. Loading data from original file.')
        df = process_data(num)
    return df

def process_data(num):
    df = pd.read_csv('data/hmda_2017_nationwide_all-records_codes.csv').sample(num, random_state=42)

    # Creating a combined variable of race and ethnicity
    # specifically to divide white and latino people 
    df['race_Ethnicity'] = df['applicant_race_1'] 
    index = df.loc[(df['applicant_race_1'] == 5) & (df['applicant_ethnicity'] == 1)].index 
    df.loc[index, 'race_ethnicity'] = 9 # 9 is a new category for people of latino ethnicity and of white race

    df['applicant_co_applicant_sex'] = df['applicant_sex'].astype(str) + '_' + df['co_applicant_sex'].astype(str)

    # filter DataFrame based on 'action_taken' column
    df = df[df['action_taken'].isin([1, 3])]
    df['action_taken'] = df['action_taken'].replace({1: 'Approved', 3: 'Denied'})
    df.to_csv('data/processed_data.csv', index=False)
    return df



def compare_race_ethnicity_groups(df, race_ethnicity_col, groups, features, title="Comparison of Race/Ethnicity Groups"):
    # Create a filtered DataFrame containing only the specified groups
    filtered_df = df[df[race_ethnicity_col].isin(groups)]

    # Determine the number of subplots needed based on the number of features
    num_subplots = len(features)

    # Calculate the number of rows and columns for the subplots
    num_cols = 2
    num_rows = (num_subplots + 1) // num_cols

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=False)
    axes = axes.flatten()

    # Iterate through each feature and create the density plots
    for idx, feature in enumerate(features):
        for group in groups:
            group_data = filtered_df[filtered_df[race_ethnicity_col] == group]
            if '_000s' in feature:
                sns.kdeplot(group_data[feature], ax=axes[idx], label=group, clip=(0, 1000))
            else:    
                sns.kdeplot(group_data[feature], ax=axes[idx], label=group, clip=(0, 100000000000))
        
        axes[idx].set_title(f'Distribution of {feature}')
        axes[idx].legend(title=race_ethnicity_col)
    
    # Remove unused subplots
    for idx in range(num_subplots, len(axes)):
        fig.delaxes(axes[idx])

    # Set the main title and show the plot
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
