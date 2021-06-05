from frlearn.neighbours import FRNN, KDTree, NNSearch
from frlearn.utils.owa_operators import OWAOperator, additive, strict, exponential, mean, invadd
import numpy as np


def least_indices_and_values(a, m, axis=-1):
    '''
    The helper-function for the Cosine relation class that sorts distances
    '''
    ia = np.argpartition(a, m - 1, axis=axis)
    a = np.take_along_axis(a, ia, axis=axis)
    take_this = np.arange(m)
    ia = np.take(ia, take_this, axis=axis)
    a = np.take(a, take_this, axis=axis)
    i_sort = np.argsort(a, axis=axis)
    ia = np.take_along_axis(ia, i_sort, axis=axis)
    a = np.take_along_axis(a, i_sort, axis=axis)
    return ia, a


class Cosine(NNSearch):
    '''
    This class defines the cosine similarity relation that can be used into FRNN OWA classification method
    Cosine metric: cos(A, B) = (A * B)(||A|| x ||B||),
                    where A, B - embedding vectors of tweets, * is a scalar product, and ||.|| is the vector norm
    Cosine similarity: cos_similarity(A, B) = (1 + cos(A, B))/2.
    '''
    class Model(NNSearch.Model):
        X_T: np.ndarray
        X_T_norm: np.ndarray
            
        def _query(self, X, m_int: int):
            distances = 1 - 0.5 * X @ self.X_T / np.linalg.norm(X, axis=1)[:, None] / self.X_T_norm
            return least_indices_and_values(distances, m_int, axis=-1)
        
    def construct(self, X) -> Model:
        model: NNSearch.Model = super().construct(X)
        model.X_T = np.transpose(X)
        model.X_T_norm = np.linalg.norm(model.X_T, axis=0)
        return model


def frnn_owa_method(train_data, y, test_data, vector_name, NNeighbours, lower, upper):
    '''
    This function implements FRNN OWA classification method
    Input:  train_data - train data in form of pandas DataFrame
            y - the list of train_data golden labels
            test_data - train data in form of pandas DataFrame
            vector_name - string, the name of vector with features in train_data and test_data
            NNeighbours - the int number of neighbours for FRNN OWA method
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA)
                aggregation operators possible options: strict(), exponential(), invadd(), mean(), additive()
    Output: conf_scores - the list of confidence scores, y_pred - the list of predicted labels
    '''
    train_ind = list(train_data.index)
    test_ind = list(test_data.index)
    X = np.zeros((len(train_data), len(train_data[vector_name][train_ind[0]])))
    for m in range(len(train_ind)):
        for j in range(len(train_data[vector_name][train_ind[m]])):
            X[m][j] = train_data[vector_name][train_ind[m]][j]
    vector_size = len(test_data[vector_name][test_ind[0]])
    X_test = np.zeros((len(test_data), vector_size))
    for k in range(len(test_data)):
        for j in range(vector_size):
            X_test[k][j] = test_data[vector_name][test_ind[k]][j]
    OWA = OWAOperator(NNeighbours)
    nn_search = Cosine()
    clf = FRNN(nn_search=nn_search, upper_weights=upper, lower_weights=lower, lower_k=NNeighbours, upper_k=NNeighbours)
    cl = clf.construct(X, y)
    # confidence scores
    conf_scores = cl.query(X_test)
    # labels
    y_pred = np.argmax(conf_scores, axis=1)
    return conf_scores, y_pred


def weights_sum_test(conf_scores, alpha, classes = 4):
    '''
    This function performs rescaling and softmax transformation of confidence scores
    Input:  conf_scores - the list of confidence scores
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8
            classes - the amount of classes for which confidence scores where calculated
    Output: the list of transformed confidence scores
    '''
    conf_scores_T = conf_scores.T
    conf_scores_T_rescale = [[(conf_scores_T[k][i]-0.5)/(alpha)
                              for i in range(len(conf_scores_T[k]))] for k in range(classes)]
    conf_scores_T_rescale_sum = [sum(conf_scores_T_rescale[k]) for k in range(classes)]
    res = [np.exp(conf_scores_T_rescale_sum[k])/sum([np.exp(conf_scores_T_rescale_sum[m])
                                                     for m in range(classes)]) for k in range(classes)]
    return res


def test_ensemble_confscores(train_data, y, test_data, vector_names, NNeighbours, lower, upper, alpha):
    '''
    This function performs ensemble of FRNN OWA methods based on confidence scores outputs
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            vector_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number is amount of neighbours 'k' that will be used
                to perform FRNN OWA classification method for the corresponded feature vector.
                Lenghts of 'vector_names' and 'NNeighbours' lists should be equal.
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA)
                aggregation operators with possible options: strict(), exponential(), invadd(), mean(), additive()
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8
    Output: y_pred_res - the list of predicted labels
    '''
    # Create and fill 3D array with dimentions: number of vectors, number of test instances, number of classes
    conf_scores_all = np.zeros((len(vector_names), len(test_data), 4))
    for j in range(len(vector_names)):
        # Calculate confidence scores for each feature vector
        result = frnn_owa_method(train_data, y, test_data, vector_names[j], NNeighbours[j], lower, upper)[0]
        # Check for NaNs
        for k in range(len(result)):
            if np.any(np.isnan(result[k])):
                result[k] = [0, 0, 0, 0]
        conf_scores_all[j] = (result)
    # Rescale obtained confidence scores
    rescaled_conf_scores = np.array([weights_sum_test(conf_scores_all[:, k, :], alpha) 
                                     for k in range(len(conf_scores_all[0]))])
    # Use the mean voting function to obtain the predicted label
    y_pred_res = [np.round(6*np.average(k, weights=[0, 1, 2, 3])) for k in rescaled_conf_scores]
    return y_pred_res


def test_ensemble_labels(train_data, y, test_data, vector_names, NNeighbours, lower, upper):
    '''
    This function performs ensemble of FRNN OWA methods based on labels as output
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            vector_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that will
                be used to perform FRNN OWA classification method for the corresponded feature vector.
                Lenghts of 'vector_names' and 'NNeighbours' lists should be equal. 
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA)
                aggregation operators with possible options: strict(), exponential(), invadd(), mean(), additive()
    Output: y_pred_res - the list of predicted labels
    '''
    y_pred = []
    for j in range(len(vector_names)):
        y_pred.append(frnn_owa_method(train_data, y, test_data, vector_names[j], NNeighbours[j], lower, upper)[1])
    # Use voting function to obtain the ensembled label - we used mean
    y_pred_res = np.mean(y_pred, axis=0)
    return y_pred_res


def cross_validation_ensemble_owa(df, vector_names, K_fold, NNeighbours, lower, upper, method, alpha=0.8):
    '''
    This function performs cross-validation evaluation for FRNN OWA ensemble
    Input:  df - pandas DataFrame with features to evaluate
            vector_names - the list of strings, names of features vectors in df
            K_fold - the number of folds of cross-validation, we used K_fold = 5
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that
                will be used to perform FRNN OWA classification method for the corresponded feature vector.
                Lenghts of 'vector_names' and 'NNeighbours' lists should be equal.
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA)
                aggregation operators with possible options: strict(), exponential(), invadd(), mean(), additive()
            method - this string variable defines the output of FRNN OWA approach, it can be 'labels' or 'conf_scores'
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8
    Output: Pearson Correlation Coefficient (PCC) as float number
            PCC = (sum_i(x_i-mean(x))(y_i-mean(y)))/(sqrt(sum_i(x_i-mean(x))^2*sum_i(y_i-mean(y))^2)),
            where x_i and y_i present the i-th components of vectors x and y
    '''
    pearson = []
    random_indices = np.random.permutation(df.index)
    for i in range(K_fold):
        # Split df on train and test data
        test_data = df.loc[df.index.isin(random_indices[i*len(df.index)//K_fold:(i+1)*len(df.index)//K_fold])]
        train_data = df[~df.index.isin(test_data.index)]
        y = train_data['Class']
        y_true = test_data['Class']
        # Apply FRNN OWA method for each feature vector depends on specified output type
        if method == 'labels':
            # Solution for labels calculation
            y_pred_res = test_ensemble_labels(train_data, y, test_data, vector_names, NNeighbours, lower, upper)
        elif method == 'conf_scores':
            # Solution for confidence scores calculation
            y_pred_res = test_ensemble_confscores(train_data, y, test_data, vector_names, NNeighbours, lower, upper, alpha)
        else:
            print('Wrong output type was specified!')
        # Calculate PCC
        pearson.append(pearsonr(y_true, y_pred_res)[0])
    return np.mean(pearson)
