import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

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

    df['joint_sex'] = df['applicant_sex'].astype(str) + '_' + df['co_applicant_sex'].astype(str)

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

def preprocess(df, features):
    df['action_taken'].replace({'Approved': 1, 'Denied': 0}, inplace=True)
    # remove rows with "applicant_sex" values of 3, 4 or 5
    print(f'Sample size BEFORE filtering {df.shape[0]}')
    df = df[df['applicant_sex'].isin([1, 2])] # men and female
    df = df[df['race_ethnicity'].isin([3, 5])] # black and white 
    df['applicant_sex'].replace({2: 0}, inplace=True) # men 1, women 0 
    df['race_ethnicity'].replace({3: 1, 5: 0}, inplace=True) # black 1, white 0
    print(f'Sample size AFTER filtering {df.shape[0]}')
    x_train, x_test, y_train, y_test = train_test_split(df, df['action_taken'], test_size=0.2, random_state=42)

    # convert df['applicant_sex'] and df['race_ethnicity'] to numpy arrays
    train_groups = np.column_stack([
        x_train['applicant_sex'].to_numpy(),
        x_train['race_ethnicity'].to_numpy()
    ])
    test_groups = np.column_stack([
        x_test['applicant_sex'].to_numpy(),
        x_test['race_ethnicity'].to_numpy()
    ])
    # filter columns to only include columns in the features list above
    print(f'Num features BEFORE filtering {df.shape[0]}')
    x_train = x_train[features]
    x_test = x_test[features]
    print(f'Num features AFTER filtering {df.shape[0]}')

    # print shapes
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", x_test.shape)

    # Replace nan values with median value for that column 
    x_train = x_train.fillna(x_train.median())
    x_test = x_test.fillna(x_train.median())
    
    # print number of nan values in each column
    # print(x_train.isna().sum())

    # Standardize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, train_groups, test_groups