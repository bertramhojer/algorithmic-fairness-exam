# Description: This file contains the code for the explainability of the model
# https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454
# For binary classification SHAP works with the log-odds of a class.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
shap.initjs()

from LR_pt import LogisticRegression
from NN import SimpleNN
from data_loader import *

race_cols = ['Race_American_Indian_Alaska_Native',
        'Race_Asian',
        'Race_Black_African_American',
        'Race_Native_Hawaiian_Pacific_Islander',
        'Race_White',
        'Race_White_Latino']

features = ['loan_amount_000s', 'loan_type', 
       'property_type','applicant_income_000s', 'purchaser_type', 'hud_median_family_income',
       'tract_to_msamd_income', 'number_of_owner_occupied_units', 
       'number_of_1_to_4_family_units', 'race_ethnicity', 'state_code', 'county_code',
       'joint_sex', "minority_population", 'lien_status']

features.remove('race_ethnicity')

clean_features = ['Loan Amount', 'Loan Type', 'Property Type', 'Applicant Income', 'Purchaser Type',
                'Family Income', 'Income Ratio', 'Occupied Units', '1-4 Family Units', 
                'Race/Ethnicity', 'State Code', 'County Code', 'Joint Sex', 'Minority Population', 
                'Lien Status']

data = data_loader(race_cols, num=100000, one_hot=True)
x_train, x_val, x_test, y_train, y_val , y_test, train_groups, val_groups, test_groups = preprocess(data, features, race_cols)
X = torch.from_numpy(x_test).float()
y = torch.from_numpy(np.array(y_test)).long().view(-1, 1)

# model
model_lr_state = torch.load("models/LRmodel_S:1000000_E:10_F:False.pt")
model_lr = LogisticRegression(x_train.shape[1])
model_lr.weights = model_lr_state['linear.weight']
model_lr.bias = model_lr_state['linear.bias']

model_nn_state = torch.load("models/NN_pca:False_E:20_lr:0.001_bs:512.pt")
model_nn = SimpleNN(X.shape[1], 2)
model_nn.load_state_dict(model_nn_state)

def get_shap(model, X, feature_names):
    
        X_np = X.detach().numpy()
        explainer = shap.Explainer(model, X_np, feature_names=feature_names)
        shap_values = explainer(X_np)
        ## Individual instance shap vals
        # force plot
        shap.plots.force(shap_values[0])

        ## Overall shap vals
        # mean SHAP value for each feature
        shap.plots.bar(shap_values[:,:6], max_display=16)

        # Beeswarm plot
        shap.plots.beeswarm(shap_values[:,:6], xlim=(-.2, .2), max_display=16)

get_shap(model_nn, X, clean_features)
"""
We could potentially do dependence plots as well but we need to
consider how much space we have and how many things we can ideally
focus on.
"""






