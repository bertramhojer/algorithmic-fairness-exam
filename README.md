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
    ├── Bootstrap_and_eval.py
    ├── LR_pt.py
    ├── NN.py
    ├── PCA.py
    ├── data_loader.py
    ├── explainability.ipynb
    ├── model_eval.ipynb
    ├── model_helper.py
    ├── models.ipynb
    ├── models.py
    ├── visualization.ipynb
    └── visualization.py
```
The data folder is where all data is saved. If the csv file isn't already in the data folder it will be loaded and processed from the HMDA data file.
