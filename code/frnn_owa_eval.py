from frlearn.neighbours import FRNN, KDTree, NNSearch
from frlearn.utils.owa_operators import OWAOperator, additive, strict, exponential, mean, invadd
import numpy as np
from scipy.stats import pearsonr

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
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA) aggregation operators
                            possible options: strict(), exponential(), invadd(), mean(), additive()
    Output: conf_scores - the list of confidence scores, y_pred - the list of predicted labels
    '''
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
    OWA = OWAOperator(NNeighbours)
    nn_search = Cosine()
    clf = FRNN(nn_search=nn_search, upper_weights=upper, lower_weights=lower, lower_k=NNeighbours, upper_k=NNeighbours)
    cl = clf.construct(X, y)     
    # confidence scores
    conf_scores = cl.query(X_test)
    # labels 
    y_pred = np.argmax(conf_scores, axis=1)
    return conf_scores, y_pred
    
def test_ensemble(train_data, y, test_data, vector_names, NNeighbours, lower, upper):
    '''
    This function performs ensemble of FRNN OWA methods based on labels as output
    
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances 
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            vector_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that will be used to perform    
                           FRNN OWA classification method for the corresponded feature vector. Lenghts of 'vector_names' and 'NNeighbours' lists should be equal. 
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA) aggregation operators
                            possible options: strict(), exponential(), invadd(), mean(), additive()
    Output: y_pred_res - the list of predicted labels
    '''
    y_pred = []
    for j in range(len(vector_names)):       
        y_pred.append(frnn_owa_method(train_data, y, test_data, vector_names[j], NNeighbours[j], lower, upper)[1])
    # Use voting function to obtain the ensembled label - we used mean 
    y_pred_res = np.mean(y_pred, axis=0)
    return y_pred_res

def cross_validation_ensemble_owa(df, vector_names, class_name, K_fold, NNeighbours, lower, upper):
    '''
    This function performs cross-validation evaluation for FRNN OWA ensemble 
    
    Input:  df - pandas DataFrame with features to evaluate 
            vector_names - the list of strings, names of features vectors in df
            class_name - name of the column in df that corresponds to labels 
            K_fold - the number of folds of cross-validation, we used K_fold = 5
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that will be used to 
                perform FRNN OWA classification method for the corresponded feature vector. Lenghts of 'vector_names' and 'NNeighbours' lists should be equal. 
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA) aggregation operators
                            possible options: strict(), exponential(), invadd(), mean(), additive()
    Output: df as pandas DataFrame, identical to input df but with additional column 'Labels' that contains predictions
    '''
    df['Labels'] = None
    random_indices = np.random.permutation(df.index)
    for i in range(K_fold): 
        # Split df on train and test data
        test_data = df.loc[df.index.isin(random_indices[i*len(df.index)//K_fold:(i+1)*len(df.index)//K_fold])]
        train_data = df[~df.index.isin(test_data.index)]
        y = train_data[class_name]    
        # Apply FRNN OWA method 
        y_pred_res = test_ensemble(train_data, y, test_data, vector_names, NNeighbours, lower, upper)
        df['Labels'].loc[test_data.index] = y_pred_res
    return df

def cosine_relation(tweet1, tweet2):
    '''
    This function calculates cosine similarity between two vectors
    
    It uses "numpy" package as "np"
    
    Input: tweet1, tweet2 as arrays of numbers
    Output: float number - cosine similarity between tweet1 and tweet2
    '''
    return 0.5 * (1 + np.dot(tweet1, tweet2)/(np.linalg.norm(tweet1)*np.linalg.norm(tweet2)))

def get_neigbours(test_vector, df_train_vectors, feature, k, text_column, class_column): 
    '''
    This function calculates k neirest neighbours to the test_vector using cosine similarity
    
    Input: test_vector - array of numbers
           df_train_vectors - DataFrame with train instances
           feature - name of a column in df_train_vectors with texts' embedding vectors
           k - a number of neirest neighbours 
           text_column - name of a column in df_train_vectors with texts
           class_column - name of a column in df_train_vectors with texts' classes
    Output: list of k neirest neighbours' texts and list with their classes
    '''
    distances = df_train_vectors[feature].apply(lambda x: cosine_relation(x, test_vector))
    top_k = distances.sort_values(ascending=False)[:k]
    df_top_k = df_train_vectors.loc[top_k.index]
    return df_top_k[text_column].to_list(), df_top_k[class_column].to_list()
    
def weighted_knn(df, vector, class_vector, sample, K):   
    '''
    This function performs the weighted k neirest neighbours classification
    
    Input: df - DataFrame with train instances
           vector - name of a column in df with texts' embedding vectors
           class_vector - name of a column in df with texts' classes
           sample - the test instance as array of numbers 
           K - a number of neirest neighbours 
           
    Output: list of k neirest neighbours' texts and list with their classes
    ''' 
    class_range = list(set(df[class_vector].to_list()))
    distances = df[vector].apply(lambda x: cosine_relation(x, sample))
    top_k = distances.sort_values(ascending=False)[:K]
    return np.argmax([sum(top_k[df.loc[top_k.index, class_vector] == i]) for i in class_range])
    
def cross_validation_ensemble_wknn(df, vector_name, class_vector, K_fold, k):
    '''
    This function performs cross-validation evaluate of the weighted k neirest neighbours classification
    
    Input: df - DataFrame with train instances
           vector_name - name of a column in df with texts' embedding vectors
           class_vector - name of a column in df with texts' classes
           K_fold - number of folds of cross-validation
           K - a number of neirest neighbours 
           
    Output: df as pandas DataFrame, identical to input df but with additional column 'Prediction' that contains predictions
    ''' 
    random_indices = np.random.permutation(df.index)
    df['Prediction'] = None
    for i in range(K_fold): 
        # Split df on train and test data
        test_data = df.loc[df.index.isin(random_indices[i*len(df.index)//K_fold:(i+1)*len(df.index)//K_fold])]
        train_data = df[~df.index.isin(test_data.index)]
        y = train_data[class_vector]
        y_true = test_data[class_vector]  
        # Apply wkNN
        res = test_data[vector_name].apply(lambda x: weighted_knn(train_data, vector_name, x, k))
        df['Prediction'].loc[test_data.index] = res
    return df