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
    df['race_ethnicity'] = df['applicant_race_1'] 
    index = df.loc[(df['applicant_race_1'] == 5) & (df['applicant_ethnicity'] == 1)].index 
    df.loc[index, 'race_ethnicity'] = 9 # 9 is a new category for people of latino ethnicity and of white race

    df['applicant_co_applicant_sex'] = df['applicant_sex'].astype(str) + '_' + df['co_applicant_sex'].astype(str)

    # filter DataFrame based on 'action_taken' column
    df = df[df['action_taken'].isin([1, 3])]
    df['action_taken'] = df['action_taken'].replace({1: 'Approved', 3: 'Denied'})

    df = add_income_group_column(df)

    df.to_csv('data/processed_data.csv', index=False)

    return df

def add_income_group_column(data):
    # Calculate the income percentiles
    income_percentiles = pd.Series(data['applicant_income_000s']).rank(pct=True)

    # Define the income groups
    low_income_mask = income_percentiles <= 0.3
    middle_income_mask = (income_percentiles > 0.3) & (income_percentiles <= 0.7)
    high_income_mask = income_percentiles > 0.7

    # Create a new column indicating the income group
    data.loc[low_income_mask, 'income_group'] = 'low'
    data.loc[middle_income_mask, 'income_group'] = 'middle'
    data.loc[high_income_mask, 'income_group'] = 'high'

    return data