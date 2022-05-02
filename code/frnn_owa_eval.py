import numpy as np

from frlearn.uncategorised.weights import Weights as OWAOperator
from frlearn.uncategorised.weights import LinearWeights as additive
from frlearn.classifiers import FRNN
from frlearn.feature_preprocessors import VectorSizeNormaliser
from frlearn.vector_size_measures import MinkowskiSize

from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support

def frnn_owa_method(train_data, y, test_data, vector_name, NNeighbours, lower, upper):
    '''
    This function implements FRNN OWA classification method 
    
    It uses "numpy" library as "np" to create zero matrix and calculate argmax.
    It uses "frlearn" library to run FRNN OWA method.
    
    Input:  train_data - train data in form of pandas DataFrame 
            y - the list of train_data golden labels 
            test_data - train data in form of pandas DataFrame 
            vector_name - string, the name of vector with features in train_data and test_data
            NNeighbours - the int number of neighbours for FRNN OWA method
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA) aggregation operators
    Output: conf_scores - the list of confidence scores, y_pred - the list of predicted labels
    '''
    # Transform data into numpy matrix 
    train_ind = list(train_data.index)
    test_ind = list(test_data.index)
    X = np.zeros((len(train_data), len(train_data[vector_name][train_ind[0]])))
    for m in range(len(train_ind)):
        for j in range(len(train_data[vector_name][train_ind[m]])):
            X[m][j] = train_data[vector_name][train_ind[m]][j]
    X_test = np.zeros((len(test_data), len(test_data[vector_name][test_ind[0]])))
    for k in range(len(test_data)):
        for j in range(len(test_data[vector_name][test_ind[0]])):
            X_test[k][j] = test_data[vector_name][test_ind[k]][j]
    
    # Define FRNN OWA model 
    OWA = OWAOperator()(NNeighbours)
    clf = FRNN(
    dissimilarity=MinkowskiSize(p=2, unrooted=True),
        preprocessors=(VectorSizeNormaliser('euclidean'), ), lower_weights=lower, lower_k=NNeighbours, upper_k=NNeighbours)
    
    # Train the model 
    cl = clf.construct(X, y)    
    # Extract confident scores 
    conf_scores = cl.query(X_test)
    # Calculate labels 
    y_pred = np.argmax(conf_scores, axis=1)
    
    return conf_scores, y_pred
  
def weights_sum_test(conf_scores, class_num, alpha=0.8):    
    '''
    This function performs rescaling and softmax transformation of confidence scores 
    
    It uses "numpy" library as "np" to calculate softmax. 
    
    Input:  conf_scores - the list of confidence scores
            class_num - the integer number of classes
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8 
    Output: the list of transformed confidence scores
    '''
    conf_scores_T = conf_scores.T
    # Rescale confident scores 
    conf_scores_T_rescale = [[(conf_scores_T[k][i]-0.5)/(alpha) for i in range(len(conf_scores_T[k]))] for k in range(class_num)]
    conf_scores_T_rescale_sum = [sum(conf_scores_T_rescale[k]) for k in range(class_num)]
    # Normalize confident scores 
    res = [np.exp(conf_scores_T_rescale_sum[k])/sum([np.exp(conf_scores_T_rescale_sum[k]) for k in range(class_num)]) for k in range(class_num)]
    return res
    
def test_ensemble_confscores(train_data, y, test_data, vector_names, NNeighbours, lower, upper, alpha=0.8):
    '''
    This function performs ensemble of FRNN OWA methods based on confidence scores outputs.
    
    It uses "numpy" library as "np" to operate with vectors ans matrixes.
    It uses "frnn_owa_method" function to calculate FRNN OWA method.
    It uses "weights_sum_test" function to rescale conference scores.
    
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances 
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            vector_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that will be used 
                to perform FRNN OWA classification method for the corresponded feature vector. Lenghts of 'vector_names' and 'NNeighbours' lists should be equal. 
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA) aggregation operators
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8 
    Output: y_pred_res - the list of predicted labels
    '''
    # Calculate number of classes
    class_num = len(set(y))
    # Create and fill 3D array
    conf_scores_all = np.zeros((len(vector_names), len(test_data), class_num))
    for j in range(len(vector_names)):    
        # Calculate confidence scores for each feature vector 
        result = frnn_owa_method(train_data, y, test_data, vector_names[j], NNeighbours[j], lower, upper)[0]
        # Check for NaNs 
        for k in range(len(result)):
            if np.any(np.isnan(result[k])):
                result[k] = [0 for i in range(class_num)]              
        conf_scores_all[j] = (result)
    # Rescale obtained confidence scores 
    rescaled_conf_scores = np.array([weights_sum_test(conf_scores_all[:, k, :], class_num, alpha) for k in range(len(conf_scores_all[0]))])
    # Use the mean voting function to obtain the predicted label 
    y_pred_res = [np.round(np.average(k, weights=list(set(y)))) for k in rescaled_conf_scores]
    return y_pred_res

def test_ensemble_labels(train_data, y, test_data, features_names, NNeighbours, lower, upper):
    '''
    This function performs ensemble of FRNN OWA methods based on labels as output
    
    It uses "numpy" library as "np" to take a mean. 
    It uses "frnn_owa_method" function to calculate FRNN OWA method.
    
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances 
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            features_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'NNeighbours' that 
                will be used to perform FRNN OWA classification method for the corresponded feature vector. Lenghts of 'vector_names' and 'NNeighbours' lists should be equal. 
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA) aggregation operators
    Output: y_pred_res - the list of predicted labels
    '''
    y_pred = []
    for j in range(len(features_names)):       
        y_pred.append(frnn_owa_method(train_data, y, test_data, features_names[j], NNeighbours[j], lower, upper)[1])
    # The voting function to obtain the ensembled label - we used mean 
    y_pred_res = np.mean(y_pred, axis=0)
    return [round(i) for i in y_pred_res]

def cross_validation_ensemble_owa(df, features_names, class_name, K_fold, k, lower, upper, method, evaluation, alpha=0.8):
    '''
    This function performs cross-validation evaluation for FRNN OWA ensemble.
    
    It uses "numpy" library as "np" for random permutation of list.
    It uses "scipy" library to calculate Pearson Correlation Coefficient.
    It uses "sklearn" library to calculate F1-score.    
    It uses "test_ensemble_labels" function to calculate FRNN OWA method's predictions as labels.
    It uses "test_ensemble_confscores" function to calculate FRNN OWA method's predictions as confidence scores.
    
    Input:  df - pandas DataFrame with features to evaluate 
            features_names - the list of strings, names of features vectors in df
            class_name - the string name of the column of df that contains classes of instances 
            K_fold - the number of folds of cross-validation, we used K_fold = 5
            k - the list of int numbers, where each number represent amount of neighbours 'k' that will be used 
                to perform FRNN OWA classification method for the corresponded feature vector. Lenghts of 'vector_names' and 'k' lists should be equal. 
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA) aggregation operators
            method - this string variable defines the output of FRNN OWA approach, it can be 'labels' or 'conf_scores'
            evaluation - the evaluation method's name: could be 'pcc' for Pearson Correlation Coefficient or 'f1' for F1-score
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8 
    Output: The evaluation score as float number: either PCC or F1-score             
    '''
    # Create column for results
    df[method] = None

    # Cross-validation
    random_indices = np.random.permutation(df.index)
    for i in range(K_fold): 
        # Split df on train and test data
        test_data = df.loc[df.index.isin(random_indices[i*len(df.index)//K_fold:(i+1)*len(df.index)//K_fold])]
        train_data = df[~df.index.isin(test_data.index)]
        y = train_data[class_name]
        y_true = test_data[class_name]       
        # Apply FRNN OWA method for each feature vector depends on specified output type 
        if method == 'labels':
            # Solution for labels calculation 
            y_pred_res = test_ensemble_labels(train_data, y, test_data, features_names, k, lower, upper)
        elif method == 'conf_scores':
            # Solution for confidence scores calculation 
            y_pred_res = test_ensemble_confscores(train_data, y, test_data, features_names, k, lower, upper, alpha)
        else:
            print('Wrong output type was specified! Choose "labels" or "conf_scores".')
        df[method].loc[test_data.index] = y_pred_res
    
    # Evaluation with F1-score or Pearson Correlation Coefficient 
    if evaluation == 'f1':
        p, r, res, support = precision_recall_fscore_support(df[class_name].to_list(), df[method].to_list(), average = "macro")
    elif evaluation == 'pcc':
        res = pearsonr(df[class_name].to_list(), df[method].to_list())[0]
    else:
        print('Wrong evaluation metric was specified! Choose "pcc" or "f1".')
    return res