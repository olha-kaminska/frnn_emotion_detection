import preprocessor as p
import functools
import operator
import pandas as pd
import re
from emoji import *
from nltk.corpus import stopwords

def tweet_cleaning(tweet):
    '''
    This function performs cleaning of the tweet. 
    
    It uses "preprocessor" library for deleting URLs, numbers, and usertags. 
    It uses "emoji" library to transform emojis from unicodes to their textual descriptions.
    It uses "re" library to delete extra whitespaces.
    
    Input: tweet as a string, example: '@xandraaa5 @amayaallyn6 shut up hashtags are cool #offended'
    Output: cleaned tweet as a string, example: 'shut up hashtags are cool offended'
    '''
    # delete #
    tweet = tweet.replace('#', '')
    # delete \n
    tweet = tweet.replace('\\n', ' ')
    # &amp; replace with &
    tweet = tweet.replace('&amp;', 'and')
    # delere URLs, numbers, and usertags
    p.set_options(p.OPT.URL, p.OPT.NUMBER, p.OPT.MENTION)
    tweet = p.tokenize(tweet)
    # transform old-style smilies to their textual descriptions
    tweet = tweet.replace(':)', 'smiley').replace(':-)', 'smiley').replace(':(', 'sad').replace(':/', 'skeptical').replace(':D', 'laughing').replace(':o', 'surprise').replace(':O', 'surprise')
    # transform emojis to their textual descriptions
    tweet = ' '.join([i for k in functools.reduce(operator.concat, [substr.split() 
                for substr in get_emoji_regexp().split(tweet)]) 
                    for i in demojize(k).replace(":", "").replace("_", " ").split(" ")])
    # delete extra whitespaces
    tweet = re.sub(' +',' ',tweet)
    return tweet
    
def delete_stopwords(tweet):
    '''
    This function delete stop-words from the tweet. 
    
    It uses the stop-words list from the "nltk" library. 
    
    Input: tweet as a string, example: '@xandraaa5 @amayaallyn6 shut up hashtags are cool #offended'
    Output: tweet without stopwords as a string, example: '@xandraaa5 @amayaallyn6 shut hashtags cool #offended'
    '''
    # get the list of stopwords
    stop_words = list(set(stopwords.words('english')))
    # delete stopwords
    tweet = ' '.join([i for i in tweet.split(' ') if i not in stop_words])
    return tweet

def transform_data(file_name, columns, sep):
    '''
    This function transforms the original dataset into a proper DataFrame.
    
    It uses "pandas" library to read dataset as a DataFrame.
    It uses "tweet_cleaning" and "delete_stopwords" functions to preprocess tweets.
    
    Input: a string with a path to the dataset;
           columns - the list of columns of dataframes that corresponds to their id, text, and class columns;
           sep - string that identifies separator in the saved file, for example: ','.
    Output: Pandas DataFrame with columns: index, 'ID' (as string), 
                                           'Tweet' (original tweet), 
                                           'Cleaned_tweet' (preprocessed tweet), 
                                           'Cleaned_tweet_wt_stopwords' (preprocessed tweets after stop-words removing), 
                                           'Class' (as integer).	
    '''
    # read dataset
    data = pd.read_csv(file_name, sep)
    # clean the text
    data["Cleaned_tweet"] = data[columns[1]].apply(lambda x: tweet_cleaning(x))
    # filter stop-words
    data['Cleaned_tweet_wt_stopwords'] = data['Cleaned_tweet'].apply(lambda x: delete_stopwords(x))
    # in case class columns contains not only class number - filter it 
    # it was added for emotion datsets that have class column in a format as '1: low amount of anger can be inferred'
    if ' ' in str(data[columns[2]][0]):
        data["Class"] = data[columns[2]].apply(lambda x: int(x.split(':')[0]))
    return data    
    
def upload_datasets(file_train, file_dev, file_test, columns, sep):
    '''
    This function extracts datasets by pathes and composes them in DataFrames suitable for the further experiments.
    
    It uses "transform_data" function to upload one dataset as a DataFrame.
    It uses "pandas" library to combine datasets.
    
    Input: two (train and test) or three (if there are development) strings with pathes to datasets;
           columns - the list of columns of dataframes that corresponds to their id, text, and class columns;
           sep - string that identifies separator in the saved file, for example: ','.
    Output: for two inputs: two DataFrames - for train and test datasets; 
            for four inputs: four DataFrames - for train, development, train concatenated with development, and test datasets.
    '''
    # upload train and test datasets
    train_data = transform_data(file_train, columns, sep)
    test_data = transform_data(file_test, columns, sep)
    # if there any - upload development dataset and merge it with train 
    if file_dev!='':
        dev_data = transform_data(file_dev, columns, sep)
        data = pd.concat([train_data, dev_data], ignore_index=True)
        return train_data, dev_data, data, test_data
    else:
        return train_data, test_data
