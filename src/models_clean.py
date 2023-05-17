from data_loader import data_loader, preprocess
from model_helper import get_tuned_gamma, tune_lambda, plot_lambda_tuning, timer
import numpy as np
from NN import train_and_evaluate_nn, evaluate_model
from LR_pt import train_lr

one_hot_cols = ['Race_American_Indian_Alaska_Native', 'Race_Asian', 'Race_Black_African_American', 
                'Race_Native_Hawaiian_Pacific_Islander', 'Race_White', 'Race_White_Latino']
# filter columns to only include columns in the features list below
features = ['loan_amount_000s', 'loan_type', 'property_type','applicant_income_000s', 
            'purchaser_type', 'hud_median_family_income', 'tract_to_msamd_income', 
            'number_of_owner_occupied_units', 'number_of_1_to_4_family_units', #'race_ethnicity', 
            'state_code', 'county_code', 'joint_sex', "minority_population", 'lien_status']

def main(find_gamma=False, find_lambda=False, train_bare_lr=False, train_LR_l2=False, 
         train_LR_L2_fairloss=False, train_NN=False, hyp_params=False, Train_NN_fairpca_ = False):

    # The common setup code...

    num_samples = 1_000_000
    df = data_loader(one_hot_cols, num=num_samples)

    # filter columns to only include columns in the features list below
    features = ['loan_amount_000s', 'loan_type', 'property_type','applicant_income_000s', 
                'purchaser_type', 'hud_median_family_income', 'tract_to_msamd_income', 
                'number_of_owner_occupied_units', 'number_of_1_to_4_family_units', #'race_ethnicity',  'joint_sex', "minority_population"
                'state_code', 'county_code', 'lien_status']

    x_train, x_val, x_test, y_train, y_val, y_test, train_groups, val_groups, test_groups = preprocess(df, features, one_hot_cols)

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    # If hyp_params is set to True, compute gamma and lambda from scratch
    if hyp_params:
        best_gamma = get_tuned_gamma(np.linspace(0.1, 10, 5), x_train, y_train, train_groups, num_folds=5, verbose=False)
        lambda_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 1]
        find_best_lambda(x_train, y_train, test_groups, train_groups, x_test, y_test, best_gamma, one_hot_cols, lambda_vals)

    else:
        best_gamma = 0.325
        lambda_vals = [0.001]

    if find_gamma:
        Find_best_gamma(x_train, y_train, train_groups)

    if find_lambda:
        find_best_lambda(x_train, y_train, test_groups, train_groups, x_test, y_test, best_gamma, one_hot_cols, lambda_vals)
        
    if train_bare_lr:
        Train_bare_lr(x_train, x_val, y_train, y_val, train_groups, val_groups)

    if train_LR_l2:
        LR_l2(x_train, x_val, y_val, y_train, train_groups, val_groups, best_gamma)

    if train_LR_L2_fairloss:
        LR_L2_fairloss(x_train, x_val, y_train, y_val, train_groups, val_groups, fair_loss_= True)

    if train_NN:
        Train_NN(x_train, x_val, x_test, y_train, y_val, y_test, train_groups, num_epochs=2, batch_size=512, lr=0.001, plot_loss=True, seed=2)
    if Train_NN_fairpca_:
        Train_NN(x_train, x_val, x_test, y_train, y_val, y_test, train_groups, num_epochs=2, batch_size=512, lr=0.001, plot_loss=True, seed=2)

@timer
def find_best_lambda(x_train, y_train, test_groups, train_groups, x_test, y_test, best_gamma, one_hot_cols, lambda_vals):
    performance_metrics = tune_lambda(x_train, y_train, test_groups, train_groups, x_test, y_test, best_gamma, one_hot_cols, lambda_vals)
    plot_lambda_tuning(performance_metrics, lambda_vals, one_hot_cols)

@timer
def Find_best_gamma(x_train, y_train, train_groups):
    best_gamma = get_tuned_gamma(np.linspace(0.1, 1, 5), x_train, y_train, train_groups, num_folds=5, verbose=False)
    print(f'best_gamma: {best_gamma}')

@timer
def Train_bare_lr(x_train, x_val, y_train, y_val, train_groups, val_groups):
    fair_loss_ = 'NO l2'
    model, fig, axs = train_lr(x_train, y_train, x_val, y_val, train_groups, val_groups, num_epochs=1000, fair_loss_=fair_loss_)

@timer
def LR_l2(x_train, x_val, y_val, y_train, train_groups, val_groups, best_gamma):
    fair_loss_ = False
    model, fig, axs = train_lr(x_train, y_train, x_val, y_val, train_groups, val_groups, num_epochs=1000, fair_loss_=fair_loss_, gamma=best_gamma)

@timer
def LR_L2_fairloss(x_train, x_val, y_train, y_val, train_groups, val_groups, fair_loss_=True):
    fair_loss_ = True
    model, fig, axs = train_lr(x_train, y_train, x_val, y_val, train_groups, val_groups, num_epochs=1000, fair_loss_=fair_loss_)

@timer
def Train_NN(x_train, x_val, x_test, y_train, y_val, y_test, train_groups, num_epochs=100, batch_size=512, lr=0.0001, plot_loss=True, seed=4206942):
    pca = False
    model, input_size, model_path = train_and_evaluate_nn(x_train, x_val, x_test, y_train, y_val, y_test, train_groups, pca, num_epochs=num_epochs, batch_size=batch_size,
                                            lr=lr, plot_loss=plot_loss, seed=seed, f1_freq_=1)
    evaluate_model(model_path, x_train, x_test, y_test, input_size, num_classes=2, pca=False, train_groups=None)
@timer
def Train_NN_fairpca(x_train, x_val, x_test, y_train, y_val, y_test, train_groups, num_epochs=100, batch_size=512, lr=0.001, plot_loss=True, seed=4206942):
    pca = True
    model, input_size, model_path = train_and_evaluate_nn(x_train, x_val, x_test, y_train, y_val, y_test, train_groups, pca, num_epochs=num_epochs, batch_size=batch_size,
                                            lr=lr, plot_loss=plot_loss, seed=seed, f1_freq_=1)
    evaluate_model(model_path, x_train, x_test, y_test, input_size, num_classes=2, pca=True, train_groups=train_groups)

if __name__ == '__main__':
    main(find_gamma=False, find_lambda=False, train_bare_lr=False, train_LR_l2=False, train_LR_L2_fairloss=False, 
         train_NN=True, hyp_params=False, Train_NN_fairpca_ = False)
