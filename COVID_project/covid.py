
# modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sts
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression


###########################################################################################################################################

# functions
def values_before_after(dataset, target_column, age_categories_column, normalize=False):
    '''
    This function uses unique information on the age category column to recover the counts of binary information on the target column.
    
    Args:
        dataset: Provide the name of the dataset from which information will be retrieved.
        target_column: Provide the name of a single column containig the binary data of interest. <class str>
        age_categories_column: Provide the of a single column that contains age categorized. If it does not exist, please create it. <class str>
    
    Return: A dictionary compiling the information retrieved by age and by presence (1) \ absence (0).
    
    '''

    dict_before = {}

    for age in dataset[age_categories_column].unique():
        
        values = dataset[target_column][dataset[age_categories_column] == age].value_counts(normalize=normalize)
        
        if values.index.tolist() == [0, 1]:
            dict_before[age] = {values.index[0]: values.values[0], values.index[1]: values.values[1]}
        else:
            if values.index.tolist() == [0]:                
                dict_before[age] = {values.index[0]: values.values[0], 1.0: 0.0}            
            else:                
                dict_before[age] = {0.0: 0.0, values.index[0]: values.values[0]}
    
    return dict_before



def index_fill(index_missing, name_classes, proportion_classes):
    '''
    This function creates a Pandas Series to fill the NaN values on the original dataset by column.
    
    Args:
        index_missing: Suply a list with the indexes for the missing values on the original Series.
        name_classes: Name of the classes that will be used to fill in the gaps. A list must be provided.
        proportion_classes: Provide a list of values that will drive the generation of classes according to the given proportion.
        
    Return: Return a Dataframe with the index where there were missing values and randomly chosen values for replacement.
    
    '''
    
    np.random.seed(42)
    raffled_indexes = np.random.choice(name_classes, size = index_missing.size, p = proportion_classes)
    
    return raffled_indexes



def plot_percentage_subplots(dataset, target_column, hue=None, normalize=True, ax=None, legend = True, legend_label= None,
                    x_rotation=90, color=None, column_name=None, legend_names=None):
    '''
    This function will return a pandas barplot on the passed in data.
    Args:
        dataset: <pandas.core.frame.DataFrame>
            Provide the dataset from which data will be retrieved for plotting. 
        target_column: <class str>
            Provide the name of the column on the dataset that will be the source of data for plotting.
        hue: <class str>, default == None.
            A class used to stratify the target_column into subclasses.
        normalize: <class boolean>, default == True.
            Provide total frequency by class ignoring NaN values. If True, output will be relative frequency by class.
        ax: <matplotlib.axes._subplots.AxesSubplot>, default == None.
            Create subplots and them provide the ax for splitting the data on the screen as planned.
        legend: <class boolean>, default == True.
            If True, legend is displayed on the graph. If False, no legend is showed.
        legend_label: <class list>, default == None.
            A list with names to replace the automatically generated titles on the legend chart. Default is None meaning that the
            original names will be used on the legend chart.
        x_rotation: <class int>, default == 90.
            Degrees to turn the names on axis x to.
        color: <class list>, default == None.
            A list of strings with the names of the colors for each bar class. Number of colors passed in must match the numbers
            of columns on the graph. Default is None, meaning the colors will be automatically generated.
        column_name: <class list>, default == None.
            A list of strings to change the names of labels on the x axis. Default is None, meaning that the original labels will
            be used.
        
    Return: Return a single pandas bar plot with customized features. Iteration might be useful if many plots must be generated
            from the same basic characteristics.
    
    '''

    col = []
    value_0 = []
    value_1 = []

    if hue == None:
        values = dataset[target_column].value_counts(normalize=normalize)
        col.extend([target_column])        
        value_0.extend([values[0]])
        value_1.extend([values[1]])
        
    else:        
        for x in sorted(dataset[hue].unique()):
            values = dataset[target_column][dataset[hue] == x].value_counts(normalize=normalize)
            col.extend([x])
            value_0.extend([values[0]])
            value_1.extend([values[1]])

    
    if normalize == True:
        if column_name == None:
            column_name = col
        data = pd.DataFrame(data = [[(x * 100) for x in value_0], [(y * 100) for y in value_1]],
                            columns=column_name, index = [0, 1]).T
    else:
        if column_name == None:
            column_name = col
        data = pd.DataFrame(data = [value_0, value_1], columns=column_name, index = [0, 1]).T
    
    #display(target_column, data)
    
    # plotting
    data.plot(kind = 'bar', ax = ax, legend = legend, alpha = 0.5, color = color, rot=x_rotation)
    ax.set_title(target_column, fontsize = 18)
    #ax.set_xlabel(hue, fontsize = 13)
    if normalize == True:
        ax.set_ylabel('Relative frequency (%)', fontsize = 13)
    else:
        ax.set_ylabel('Absolute frequency', fontsize = 13)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(visible = True, which = "major", axis = "y", color="#ddede1")
    ax.set_axisbelow(True)
    if legend == True and legend_label == None:
        ax.legend(bbox_to_anchor=(1.2, 1.0))
    elif legend == True:
        ax.legend(labels = legend_label, bbox_to_anchor=(1.2, 1.0))
    elif legend == False and legend_label == True:
        print('Legend_label only possible if legend is True.')
    else:
        pass
    
    
    #plt.show()
    
    
    
def test_models_metrics(X, y, model_list, col_model_name, col_model, scaler='standard_scaler', test_size = 0.3, cross_validation=True,
                        cross_validation_split_number = 3, random_state = 42, save_fig=False, stratify=None):
    '''
    Disclaimer: This function was partially created by Sandro Saorin, teacher of machine learning at Let's Code Academy, Brazil.
    
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

        

def feature_select(X, y, score_func='chi_square', k='all', ascending=False, scaler=None, test_size=0.3, random_state=42, stratify=None):
    '''
    This function selects the best features for the model based on statistics.
    
    Args:
        X: <pandas.core.frame.DataFrame>
            Features that will be tested.
        y: <pandas.core.series.Series>
            Target column.
        score_func: <class str>
            It refers as to the statistical test that will be used to find the best features. Default is chi2. Options are:
                a) chi_square: Chi-squared stats of non-negative features for classification tasks.
                b) f_classification: ANOVA F-value between label/feature for classification tasks.
                c) f_regression: F-value between label/feature for regression tasks.                
        k: <class int>
            Number of best features. Default is 'all', meaning all the features available in X.
        ascending: <class boolean>
            Number on the pandas Series will be organized in increasing (True) or decreasing (False) order. Default is False.
        scaler: <class str>
            Method to standardize the data if necessary. Available methods are: 'max_abs_scaler', 'quantile_transform', 'minmax_scaler'.
            For more information, please visit <https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html>
        test_size: <class float>
            Choose the amount of data that will be spared to train and to validate/test the model. Values must be more than 0 
            and less than 1. Default is 0.3.
        random_state: <class int>
            A random seed number for the sake of reproductibility. Default is 42.
            
     Return:
         This function returns a pandas Series with the scores in the column and the corresponding features as indexes.
         Interpretation: Higher scores point to best features.
         
    '''
    
    score_function = {'f_classification':f_classif, 'chi_square':chi2, 'f_regression':f_regression}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    
    if scaler == None:        
        # running SelectKBest on data not scaled
        fs = SelectKBest(score_func=score_function[score_func], k=k)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)

        # what are scores for the features
        cols = fs.get_support(indices=True)
        features_df_new = X.iloc[:,cols].columns
        values = pd.Series(data = fs.scores_, index=features_df_new)
        values.sort_values(ascending=ascending, inplace=True)
    
    else:
        # scaling data
        scaler_methods = {'max_abs_scaler': MaxAbsScaler, 'quantile_transform': QuantileTransformer, 'minmax_scaler': MinMaxScaler}
        sc = scaler_methods[scaler]()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)        
        # running SelectKBest on the scaled data
        fs = SelectKBest(score_func=score_function[score_func], k=k)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)

        # what are scores for the features
        cols = fs.get_support(indices=True)
        features_df_new = X.iloc[:,cols].columns
        values = pd.Series(data = fs.scores_, index=features_df_new)
        values.sort_values(ascending=ascending, inplace=True)
    
    return values


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

def f_test_two_samples(x, y, ddof=1, confidence_interval=0.90, alternative = 'two_tail'):
    '''
    This function calculates the f-score for a given confidence interval.
    
    Args:
        x, y: numpy.ndarray or pandas array
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


    
    
    
    