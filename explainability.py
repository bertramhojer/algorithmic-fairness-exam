# Description: This file contains the code for the explainability of the model
# https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454
# For binary classification SHAP works with the log-odds of a class.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch

from data_loader import *

race_cols = ['Race_American_Indian_Alaska_Native',
        'Race_Asian',
        'Race_Black_African_American',
        'Race_Native_Hawaiian_Pacific_Islander',
        'Race_White',
        'Race_White_Latino']

data = data_loader(race_cols, num=10000)

# data
X = ""
# model
model = ""

explainer = shap.Explainer(model)
shap_values = explainer(X)

## Individual instance shap vals
# waterfall plot - a specific instance
shap.plots.waterfall(shap_values[0])

# force plot (condensed waterfall) - a specific instance
shap.plots.force(shap_values[0])

## Overall shap vals
# mean SHAP value for each feature
shap.plots.bar(shap_values)

# Beeswarm plot
shap.plots.beeswarm(shap_values)


"""
We could potentially do dependence plots as well but we need to
consider how much space we have and how many things we can ideally
focus on.
"""






