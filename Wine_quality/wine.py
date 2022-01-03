# packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sts
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, classification_report


###################################################################################
## functions

def histogram_boxplot(dataset, target_column, figsize = (14, 5), bins = 'auto', color = None, title = 'Plot'):
    '''
    This function generates paired plots for ease of comparison. The left plot is a histogram and the right plot is a boxplot.
    
    Args:
        dataset: pandas DataFrame
            
    '''

    # understanding outliers on each column
    ## fixed acidity
    plt.figure(figsize = figsize)

    ## histogram
    ### main
    ax1 = plt.subplot(1, 2, 1)
    sns.histplot(data = dataset, x = target_column, color = color, kde = True, stat = 'density', alpha = 0.5, ax = ax1, bins = bins)
    plt.title(title)
    plt.xlabel(target_column, fontsize = 12)
    plt.ylabel('Density', fontsize = 12)
    
    ### descriptive statistics
    mean_r = dataset[target_column].mean()
    plt.axvline(mean_r, ls = '--', c = 'red', lw = 1.5, label = 'mean')

    median_r = dataset[target_column].median()
    plt.axvline(median_r, ls = '--', c = 'black', lw = 1.5, label = 'median')

    mode_r = dataset[target_column].mode()[0]
    plt.axvline(mode_r, ls = '--', c = 'orange', lw = 1.5, label = 'mode')

    qt1_r = dataset[target_column].quantile(0.25)
    plt.axvline(qt1_r, ls = '--', c = '#35ccc9', lw = 1.5, label = '1st quartil')

    qt3_r = dataset[target_column].quantile(0.75)
    plt.axvline(qt3_r, ls = '--', c = '#2bad41', lw = 1.5, label = '3rd quartil')

    plt.legend()

    ## boxplot
    ### main
    ax2 = plt.subplot(1, 2, 2)
    sns.boxplot(data = dataset, x = target_column, color = color)
    plt.title(title)
    plt.xlabel(target_column, fontsize = 12)
        
    ######################                  
    plt.tight_layout()
    plt.show()

######################################################################################


def comparative_histogram_descriptive_statistics(df_red, df_white, target_column, figsize = (14, 5), bins_r = 'auto', bins_w = 'auto'):

    # understanding outliers on each column
    ## fixed acidity
    plt.figure(figsize = figsize)

    ## red wine
    ### main
    ax1 = plt.subplot(1, 2, 1)
    sns.histplot(data = df_red, x = target_column, color = '#ad241a', kde = True, stat = 'density', alpha = 0.5, ax = ax1, bins = bins_r)
    plt.title("Red wine")
    plt.xlabel(target_column, fontsize = 12)
    plt.ylabel('Density', fontsize = 12)
    ### descriptive statistics
    mean_r = df_red[target_column].mean()
    plt.axvline(mean_r, ls = '--', c = 'red', lw = 1.5, label = 'mean')

    median_r = df_red[target_column].median()
    plt.axvline(median_r, ls = '--', c = 'black', lw = 1.5, label = 'median')

    mode_r = df_red[target_column].mode()[0]
    plt.axvline(mode_r, ls = '--', c = 'orange', lw = 1.5, label = 'mode')

    qt1_r = df_red[target_column].quantile(0.25)
    plt.axvline(qt1_r, ls = '--', c = '#35ccc9', lw = 1.5, label = '1st quartil')

    qt3_r = df_red[target_column].quantile(0.75)
    plt.axvline(qt3_r, ls = '--', c = '#2bad41', lw = 1.5, label = '3rd quartil')

    plt.legend()

    ## white wine
    ### main
    ax2 = plt.subplot(1, 2, 2)
    sns.histplot(data = df_white, x = target_column, color = '#ded721', kde = True, stat = 'density', alpha = 0.5, ax = ax2, bins = bins_w)
    plt.title("White wine")
    plt.xlabel(target_column, fontsize = 12)
    plt.ylabel('Density', fontsize = 12)
    ### descriptive statistics
    mean_w = df_white[target_column].mean()
    plt.axvline(mean_w, ls = '--', c = 'red', lw = 1.5, label = 'mean')

    median_w = df_white[target_column].median()
    plt.axvline(median_w, ls = '--', c = 'black', lw = 1.5, label = 'median')

    mode_w = df_white[target_column].mode()[0]
    plt.axvline(mode_w, ls = '--', c = 'orange', lw = 1.5, label = 'mode')

    qt1_w = df_white[target_column].quantile(0.25)
    plt.axvline(qt1_w, ls = '--', c = '#35ccc9', lw = 1.5, label = '1st quartil')

    qt3_w = df_white[target_column].quantile(0.75)
    plt.axvline(qt3_w, ls = '--', c = '#2bad41', lw = 1.5, label = '3rd quartil')

    plt.legend()
    plt.tight_layout()
    plt.show()
    
########################################################################################################

def f_test_two_samples(x, y, ddof=1, confidence_interval=0.90, alternative = 'two_tail'):
    '''
    This function calculates the f-score for a given confidence interval.
    
    Args:
        x, y: array-like
            The array must not contain missing values.
        ddof: int, default = 1
            Used to estimate sample variance. If population variance is aimed at this value must be 0.
        confidence_interval: float, default = 0.90
            Confidence interval is used to return the f critical value. Alpha equals to (1 - confidence_interval). If alternative ==
            'two_tail', however, alpha equals to (1 - confidence_interval) / 2. For example, if alpha == 0.05 is aimed at, interval 
            confidence must be 0.90 so each tail will be searched at 0.05 (0.10 / 2).
        alternative: string, default = 'two_tail'
            This refers as to the tail on the distribution that will be considered.
                two_tail: Consider the accumulated probability below the lower and above the upper tails
                lower: Consider the accumulated probability from the confidence interval backwards.
                upper: Consider the accumulated probability from the confidence interval forward.
    
    Return:
        This function will return the F score calculated, F critical value and p_value, respectively.
        
    Interpretation:
        If p_value < alpha, variance between the two populations are not statistically different from one another and one can infer
        that both populations are the same or derived from the same.
    
    Auxiliary packages: scipy stats and numpy
    
    '''
    #########################################################################################################################
    ## choosing numerator and denominator for the f-score equation
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    s = sorted([vx, vy], reverse = True)
    if vx == s[0]:
        pass
    else:
        x, y = y, x    
    
    ## sample variances -> ddof = 1
    var_num = np.var(x, ddof=1)
    var_denom = np.var(y, ddof=1)
    
    ## sample sizes
    n_num = len(x)
    n_denom = len(y)
    
    ## degrees of freedom
    dof_num = n_num - 1
    dof_denom = n_denom - 1
    
    ## f_calculated
    f_calculated = var_num / var_denom
        
    ## finding critical point
    if alternative == 'lower':
        less = (1 - confidence_interval)
        f_critical = sts.f.ppf(less, dof_num, dof_denom)
    elif alternative == 'upper':
        f_critical = sts.f.ppf(confidence_interval, dof_num, dof_denom)
    else:
        less = (1 - confidence_interval)
        f_critical = (sts.f.ppf(less, dof_num, dof_denom), sts.f.ppf(confidence_interval, dof_num, dof_denom))
    
    ## p_value    
    if alternative == 'lower':
        p_value = 1 - (sts.f.cdf(f_calculated, dof_num, dof_denom))
    elif alternative == 'upper':
        p_value = sts.f.cdf(f_calculated, dof_num, dof_denom)
    else:
        p_value = (1 - (sts.f.cdf(f_calculated, dof_num, dof_denom))) * 2
    
    
    return f_calculated, f_critical, p_value


#############################################################################


def interquantile_range_calculator(array, factor = 1.5):
    '''
    This functions calculates the values of inferior and superior whiskers for a boxplot.
    
    Args:
        Array: array-like
            A single array is passed in for which IQR will be calculated
        factor: numeric
            A factor which corrects the inferior and superior whiskers. Calculation is factor * IQR.
    
    Return:
        This functions returns a tuple with the inferior and superior whiskers values, respectively
        
    Auxiliary modules: numpy
    '''
    # finding first and third quantiles
    q25 = np.quantile(array, 0.25)
    q75 = np.quantile(array, 0.75)
    # finding IQR
    IQR = q75 - q25
    # calculations on interquantil range
    inferior_value =  np.around(q25 - (factor * IQR), 2)
    superior_value =  np.around(q75 + (factor * IQR), 2)
    
    return inferior_value, superior_value


################################################################################

#def replace_values(arrays, list_keys, list_values, cutoff):
def replace_values(arrays, d, cutoff, replace_superior = True):
    '''
    Args:
        arrays = arrays-like
            Arrays with the target column and the column from which mapping is going to be based on for replacement.
        d: Dictionary
            A dictionary containing the values for mapping the replacement
        cutoff: numeric
            Standpoint from which replacement will take place.
        replace_superior: boolean, default = True
            If True, outliers above the cutoff point are replaced. Conversely, outliers below the cutoff point are replaced.
    
    Return:
        This function returns a value that may be used to replace the current one according to the dictionary passed in.
        
    Auxiliary modules: pandas
    '''
    acidity = arrays[0]
    quality = arrays[1]
    
    if replace_superior:
        if acidity > cutoff:
            return d[quality]
        else:
            return acidity
    else:
        if acidity < cutoff:
            return d[quality]
        else:
            return acidity
        
#############################################################################################################################

def test_models_metrics(X, y, model_list, col_model_name, col_model, scaler='standard_scaler', test_size = 0.3, cross_validation=True,
                        cross_validation_split_number = 3, random_state = 42, save_fig=False, stratify=None):
    '''
    Disclaimer: This function was created by Sandro Saorin, teacher of machine learning at Let's Code Academy, Brazil, and modified by
    Diego Batista.
    
    Purpose: This function takes in classifier models and prints out metrics related to each model in an organized fashion.
    
    Args:
        X, y:
            X stands for the features that will be used to develop the model, whereas y stands for the target variable.
        model_list: <class list>
            A list of dictionaries. Each dictionary contains the model's names and the estimator with arguments for instantiation:
                Example: list_models = [{'model_name': 'Logistic regression',
                                         'estimator' : LogisticRegression(max_iter = 300)}]
        col_model_name: <class str>
            Refers to the string in the dictionary that brings up the model's name. In the example it is 'model_name'.
        col_model: <class str>
            Refers to the string in the dictionary that brings up the classifier estimator. In the example it is 'estimator'.
        scaler: <class str>
            Scaler that will be used to standardize the X features for the model. It can be {'standard_scaler', 'robust_scaler', 'minmax_scale'}.
        test_size: <class float>
            Choose the amount of data that will be spared to train and to validate/test the model. Values must be more than 0 
            and less than 1. Default is 0.3.
        cross_validation: <class boolean>
            If True, subsets of the dataset will be used to develop models and compare metrics, such as accuracy, precision,
            recall, global F1-score and f1-scores for each target class and ROC curve. It is highly recommended in case there
            are outliers in the dataset. Default is True.
        cross_validation_split_number: <class int>
            It stands for how many subsets will be assembled in order to test the model. Deafult is 3 subsets.
        random_state: <class int>
            A random seed number for the sake of reproductibility. Default is 42.
    
    Return:
        This function returns a graph comparing ROC values for every estimator tested and metrics for classification models, that is:
        Accuracy, precision, recall, global F1-score and f1-scores for each target class.
        
    '''
    # splitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)
    
    # data standardization
    scaler_methods = {'standard_scaler': StandardScaler, 'robust_scaler': RobustScaler, 'minmax_scaler': MinMaxScaler}
    sc = scaler_methods[scaler]()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # cross-validation
    if cross_validation:
        skf = StratifiedKFold(n_splits=cross_validation_split_number)
        cv_indexes = skf.get_n_splits(X_train)
        
        for mdl in model_list:
            model = mdl[col_model]
            cross = np.around(cross_val_score(model, X_train, y_train, scoring='accuracy', cv = cv_indexes), 4)
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            cr = classification_report(y_test, y_predict)
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
            auc = roc_auc_score(y_test, model.predict(X_test))
            plt.plot(fpr, tpr, label='%s ROC (AUC = %0.2f)' % (mdl[col_model_name], auc))
            print(f'Cross-validation: {cross}\nMean: {np.around(np.mean(cross), 4)}\nStandard deviation: {np.around(np.std(cross), 4)}')
            print("Model      : %s" % mdl[col_model_name])
            print("Accuracy   : %0.4f " %  accuracy_score(y_test, y_predict))
            print("Precision  : %0.4f " % precision_score(y_test, y_predict, average='weighted'))
            print("Recall     : %0.4f " % recall_score(y_test, y_predict, average='weighted'))
            print("F1 - Score : %0.4f " % f1_score(y_test, y_predict, average='weighted'))
            print("F1 - Score for class 0: {}\nF1 - Score for class 1: {}".format(cr[94:98], cr[148:152]))            
            print("ROC - AUC  : %0.4f " % auc)
            print("======================")
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC curve')
        plt.legend(loc="lower right")
        if save_fig == True:
            plt.savefig('ROC_curve.jpg', dpi=300)
            plt.show()
        else:
            plt.show()
        
    else:
        for mdl in model_list:
            model = mdl[col_model]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            cr = classification_report(y_test, y_predict)
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
            auc = roc_auc_score(y_test, model.predict(X_test))
            plt.plot(fpr, tpr, label='%s ROC (AUC = %0.2f)' % (mdl[col_model_name], auc))
            print("Model      : %s" % mdl[col_model_name])
            print("Accuracy   : %0.4f " %  accuracy_score(y_test, y_predict))
            print("Precision  : %0.4f " % precision_score(y_test, y_predict, average='weighted'))
            print("Recall     : %0.4f " % recall_score(y_test, y_predict, average='weighted'))
            print("F1 - Score : %0.4f " % f1_score(y_test, y_predict, average='weighted'))
            print("F1 - Score for class 0: {}\nF1 - Score for class 1: {}".format(cr[94:98], cr[148:152]))
            print("ROC - AUC  : %0.4f " % auc)
            print("======================")
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC curve')
        plt.legend(loc="lower right")
        #plt.legend(loc="lower right")
        if save_fig == True:
            plt.savefig('ROC_curve.jpg', dpi=300)
            plt.show()
        else:
            plt.show()

########################################################################################################################

def best_classification_split_score(dataset, threshold_begin=None, threshold_increment=0.1, threshold_end=1.0, colum_probabilities='proba', colum_target='target'):
    '''
    Disclaimer: This function was created by Sandro Saorin, teacher of machine learning at Let's Code Academy, Brazil.
    
    Purpose: This function takes in a structured dataset containing binary information on a column (colum_target) and probabilities on
    another (colum_probabilities) calculated by a classifier of choice. It returns metrics to assist in the decision of which cutoff
    should be used on the dataset.
        
    Args:
        dataset: <pandas.core.frame.DataFrame>.
            This dataset may be obtained by joining results from the target column with probabilities from a classifier 
            model.predict_proba(X_test)[:, 1].
        threshold_begin: float, default == None.
            A floating point number between 0 and 1, which is the starting number from which the model will look for better fits.
            If None, threshold_begin == threshold_increment.
        threshold_increment: float, default == 0.1.
            A floating point number between 0 and 1 that will increase the threshold at each new round of iteration.
        threshold_end: float, default == None.
            A floating point number between 0 and 1 that must be greater than the threshold_begin. It is the final number to end the
            iteration.
        colum_probabilities: str, default == 'proba'
            It refers as to the name of column on the dataset that contains the probabilities found by the classifier for a given dataset.
        colum_target: str, default == 'target'
            It refers as to the name of column on the dataset that contains the values for validation (y_test).
     
     Return:
         This function will return a <pandas.core.frame.DataFrame> containig:
             threshold: The threshold value of the round.
             TN: True negative values based on the given threshold.
             FN: False negative values based on the given threshold.
             FP: False positive values based on the given threshold.
             TP: True positive values based on the given threshold.
             precision: General precision calculated on the 1 values.
             recall: General recall calculated on the 1 values.
             accuracy: General accuracy calculated on the 1 values.
             f0_score: Harmonic mean between precision and recall for 0 values.
             f1_score: Harmonic mean between precision and recall for 1 values.
    '''
    

    if threshold_begin == None:
        threshold_begin = threshold_increment
    else:
        threshold_begin = threshold_begin


    # Para cada threshold, no looping calcula TN, FN, FP, TP e outras métricas

    list_threshold  = []
    list_TN         = []
    list_FN         = []
    list_FP         = []
    list_TP         = []
    list_precision  = []
    list_recall     = []
    list_accuracy   = []
    list_f0_score   = []
    list_f1_score   = []

    for i in np.arange(threshold_begin, threshold_end + threshold_increment, threshold_increment):
        i_threshold = np.round(i, 2)
        print(str(i_threshold) + ' ', end = '')

        dataset['y_pred'] = dataset[colum_probabilities].apply(lambda x: 1 if x >= i_threshold else 0)
        dataset['flag_TN'] = np.where((dataset['y_pred'] == 0) & (dataset[colum_target] == 0), 1, 0)
        dataset['flag_FN'] = np.where((dataset['y_pred'] == 0) & (dataset[colum_target] == 1), 1, 0)
        dataset['flag_TP'] = np.where((dataset['y_pred'] == 1) & (dataset[colum_target] == 1), 1, 0)
        dataset['flag_FP'] = np.where((dataset['y_pred'] == 1) & (dataset[colum_target] == 0), 1, 0)

        TN = dataset['flag_TN'].sum()
        FN = dataset['flag_FN'].sum()
        TP = dataset['flag_TP'].sum()
        FP = dataset['flag_FP'].sum()

        # reference == 0
        precision = np.where((TN + FN) > 0, TN / (TN + FN), 0)
        recall = np.where((TN + FP) > 0, TN / (TN + FP), 0)
        accuracy = np.where((TP + FN + TN + FN) > 0, 
                             (TN + TP)/(TP + FP + TN + FN), 0)
        f0_score = np.where((precision + recall) > 0, (2 * precision * recall)/(precision + recall), 0)

        # reference == 1
        precision = np.where((TP + FP) > 0, TP / (TP + FP), 0)
        recall = np.where((TP + FN) > 0, TP / (TP + FN), 0)
        accuracy = np.where((TN + FN + TP + FP) > 0, 
                             (TP + TN)/(TN + FN + TP + FP), 0)
        f1_score = np.where((precision + recall) > 0, (2 * precision * recall)/(precision + recall), 0)

        list_threshold.append(i_threshold)
        list_TN.append(TN)
        list_FN.append(FN)
        list_FP.append(FP)
        list_TP.append(TP)
        list_precision.append(np.round(precision, 4))
        list_recall.append(np.round(recall, 4))
        list_accuracy.append(np.round(accuracy, 4))
        list_f0_score.append(np.round(f0_score, 4))
        list_f1_score.append(np.round(f1_score, 4))

    #---------------------
    dict_output = {
      "threshold" : list_threshold, 
      "TN" : list_TN,
      "FN" : list_FN,
      "FP" : list_FP,
      "TP" : list_TP,
      "precision" : list_precision,
      "recall" : list_recall,
      "accuracy" : list_accuracy,
      "f0_score" : list_f0_score,
      "f1_score" : list_f1_score
    }

    df_results = pd.DataFrame(dict_output)
    
    return df_results