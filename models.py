#%%
from data_loader import data_loader, preprocess
from model_helper import get_tuned_gamma, train, tune_lambda, plot_lambda_tuning
import numpy as np
from PCA import project_and_plot_PCA, plot_reconstruction_loss, fair_PCA, corr_plot
import pandas as pd
from Bootstrap_and_eval import eval, bootstrap_eval_one, plot_violin_metrics_with_ci_single
from NN import train_and_evaluate_nn, evaluate_model
import time
from LR_pt import train_lr

one_hot_cols = ['Race_American_Indian_Alaska_Native',
        'Race_Asian',
        'Race_Black_African_American',
        'Race_Native_Hawaiian_Pacific_Islander',
        'Race_White',
        #'Race_Info_Not_Provided',
        #'Race_Not_applicable',
        #'Race_No_co_applicant', 
        'Race_White_Latino']
one_hot = True

num_samples = 1_000_000
df = data_loader(one_hot_cols, num=num_samples)

# filter columns to only include columns in the features list below
features = ['loan_amount_000s', 'loan_type', #'owner_occupancy', 
       'property_type','applicant_income_000s', 'purchaser_type', 'hud_median_family_income',
       'tract_to_msamd_income', 'number_of_owner_occupied_units', 
       'number_of_1_to_4_family_units', 'race_ethnicity', 'state_code', 'county_code',
       'joint_sex', "minority_population", 'lien_status']
if one_hot:
       # remove  'race_ethnicity' from features
       features.remove('race_ethnicity')
x_train, x_val, x_test, y_train,y_val , y_test, train_groups, val_groups,test_groups = preprocess(df, features, one_hot_cols)
print(f'All rows in train_groups sum to 1: {np.allclose(np.sum(train_groups, axis=1), 1)}')

# convert y_train, y_val, y_test to numpy arrays
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()

print()
print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'x_val shape: {x_val.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_test shape: {y_test.shape}')
print(f'train_groups shape: {train_groups.shape}')
print()
#%%
#Find_best_gamma(x_train, y_train, train_groups)
best_gamma = 0.325
best_lambda = 0.001
#lambda_vals = [0.001, 1]
lambda_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 1]


def find_best_lambda(x_train, y_train, test_groups, train_groups, x_test, y_test, best_gamma, one_hot_cols, lambda_vals):
    performance_metrics = tune_lambda(x_train, y_train, test_groups, train_groups, x_test, y_test, best_gamma, one_hot_cols, lambda_vals)
    plot_lambda_tuning(performance_metrics, lambda_vals, one_hot_cols)


#%%
def Find_best_gamma(x_train, y_train, train_groups):
    start_time = time.perf_counter()
    best_gamma = get_tuned_gamma(np.linspace(0.1, 1, 5), x_train, y_train, train_groups, num_folds=5, verbose=False)

    print(f'best_gamma: {best_gamma}')
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print("get_tuned_gamma execution time: {:.4f} seconds".format(execution_time))

def Train_bare_lr(num_samples, x_train, x_val, y_train, y_val, train_groups, val_groups):
    fair_loss_ = 'NO l2'
    start_time = time.perf_counter()
    model, fig, axs = train_lr(x_train, y_train, x_val, y_val, train_groups, val_groups, num_epochs = 1000, fair_loss_= fair_loss_, num_samples=num_samples)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print("Execution time: {:.4f} seconds".format(execution_time))


#Train_bare_lr(num_samples, x_train, x_val, y_train, y_val, train_groups, val_groups)


def LR_l2(x_train, x_test, y_train, y_test, train_groups, best_gamma):
    fair_loss_ = False
    model, fig, axs = train_lr(x_train, y_train, x_val, y_val, train_groups, val_groups, num_epochs = 1000, fair_loss_= fair_loss_, num_samples=num_samples)

#LR_l2(x_train, x_test, y_train, y_test, train_groups, best_gamma) 

#####################

def LR_L2_fairloss(x_train, x_val, y_train, y_val, train_groups, val_groups, fair_loss_= True):
    fair_loss_ = True

    model, fig, axs = train_lr(x_train, y_train, x_val, y_val, train_groups, val_groups, num_epochs = 1000, fair_loss_= fair_loss_, num_samples=num_samples)

#LR_L2_fairloss(x_train, x_val, y_train, y_val, train_groups, val_groups, fair_loss_= True, num_samples=num_samples)) 
#####################

# train NN model 
def Train_NN(x_train, x_val, x_test, y_train, y_val, y_test, train_groups, num_epochs=1000, batch_size=32, lr=0.1, plot_loss=True, seed=4206942):
    start_time = time.perf_counter()
    pca = False
    model, accuracy = train_and_evaluate_nn(x_train, x_val, x_test, y_train, y_val, y_test, train_groups, pca, num_epochs=num_epochs, batch_size=batch_size, 
                                            lr=lr, plot_loss=plot_loss, seed=seed, f1_freq_=1)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print("Execution time: {:.4f} seconds".format(execution_time))

Train_NN(x_train, x_val, x_test, y_train, y_val, y_test, train_groups, num_epochs=2, batch_size=512, lr=0.001, plot_loss=True, seed=2)


model_path = "models/NN_pca:False_E:2_lr:0.001_bs:512.pt"
pca_state = False
print()
print("evaluate_model on test set")
evaluate_model(model_path,x_train ,x_test, y_test, x_test.shape[1], 2, pca=pca_state, train_groups=train_groups)
print("evaluate_model on train set")
print()
evaluate_model(model_path,x_train ,x_train, y_train, x_test.shape[1], 2, pca=pca_state, train_groups=train_groups)
print()
print("evaluate_model on val set")
evaluate_model(model_path,x_train ,x_val, y_val, x_test.shape[1], 2, pca=pca_state, train_groups=train_groups)
