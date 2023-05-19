# Algorithmic Fairness - Exam Project
## Uncovering and Correcting Biases in Mortgage Approval Algorithms

We work with the HMDA dataset to analyze bias in mortgage lending in the US, and apply a variety of methods to mitigate said biases.

## Repository structure
```bash
.
├── data
├── models
├── plots
└── src
    ├── LR_pt.py
    ├── NN.py
    ├── PCA.py
    ├── data_loader.py
    ├── model_eval.ipynb
    ├── model_helper.py
    ├── models.py
    ├── visualization.ipynb
    └── visualization.py
```

Most of the functionality is hidden away in helper python files. Functions are then imported into various notebooks to run analysis and classification models.

## Documentation
### Data
- The data folder holds the original HDMA data and a processed csv. If the csv file isn't already in the data folder it will be loaded and processed from the HMDA data file when running the data_loader.

### Notebooks
- 'visualization.ipynb' imports plotting functionality from the visualization python file and when run produces all plots and tables for the EDA, statistical analysis, PCA & explainability using SHAP.

- 'model_eval.ipynb' imports all trained models and evaluates them and plots the evaluation metrics for all models together.

### Helper files
- 'data_loader.py' handles loading the data, preprocessing it and creating the training splits using a specific seed.

- 'visualization.py' comprises the code for creating the main visualizations plotted in the visualization notebook. The functionality of this script is called in the visualization notebook, this is done to keep the notebooks clean.

- 'LR_pt.py' comprises all code related to training logistic regression models. We have implemented logistic regression to be able to modify the loss function to use an implementation of a 'Fair loss' function. It was done in pytorch to speed up the training process.

- 'NN.py' comprises all code related to training and evaluating the neural network models. The models are implemented in pytorch.

- 'PCA.py' comprises our own implementation of PCA and FairPCA as well as the visualization scripts used to plot various PCA metrics such as explained variance, reconstruction loss as well as the correlation matrix for the normal PCA and the fair PCA.

- 'models.py' comprises all code for model training. This script can train each of the five different models trained for the project. To train a different type of model the parameters in the call to the main() function must simply be changed.

- 'model_helper.py' comprises helper functions for the model training. These are e.g. the various implemented loss function as well as functionality for parameter tuning of lambda and gamma values.

