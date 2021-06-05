import pandas as pd
import regex as re
import functools
import operator
from emoji import *
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))


def get_class(row):
    '''
    The helper function for 'transform_data(file)' which saves the class as an integer value
    Input: string from 'Intensity Class' column of original emotion data,
            example: '1: low amount of anger can be inferred'
    Output: class as an integer, example: '1'
    '''
    return int(row.split(':')[0])


def tweet_cleaning(tweet):
    '''
    This function performs the general preprocessing of the tweet, without stop-words removing step
    Input: tweet as a string, example: '@xandraaa5 @amayaallyn6 shut up hashtags are cool #offended'
    Output: cleaned tweet as a string, example: 'shut up hashtags are cool offended'
    '''
    # delete \n
    tweet = tweet.replace('\\n', ' ')
    # delete # symbol from hashtags
    tweet = tweet.replace(' #', ' ')
    # &amp; replace with &
    tweet = tweet.replace('&amp;', 'and')
    # replace dots with white spaces
    tweet = re.sub('\.+', ' ', tweet)
    # delete account tags
    tweet = re.sub('@[^\s]+', '', tweet)
    # delere URL-s
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # delete numbers
    tweet = ''.join([i for i in tweet if not i.isdigit()])
    # transform old-style smilies
    smilies = [(':)', 'smiley'),
               (':-)', 'smiley'),
               (':(', 'sad'),
               (':/', 'skeptical'),
               (':D', 'laughing'),
               (':o', 'surprise'),
               (':O', 'surprise')]
    for smile in smilies:
        tweet = tweet.replace(smile[0], smile[1])
    # transform emojis to their textual descriptions
    new_text = ' '.join([i for k in functools.reduce(operator.concat, [substr.split()
                for substr in get_emoji_regexp().split(tweet)])
                    for i in demojize(k).replace(":", "").replace("_", " ").split(" ")])
    # delete punctuation
    new_text = ''.join([i.lower() if i.isalpha() or i == ' ' else ' ' for i in new_text])
    # delete extra whitespaces
    new_text = re.sub(' +', ' ', new_text)
    return new_text


def transform_data(file_name):
    '''
    This function transform the original dataset provided by competition into a DataFrame
    Input: string with a path to the dataset
    Output: Pandas DataFrame with columns: index, 'ID' (as string), 'Tweet' (original tweet), 'Cleaned_tweet' (preprocessed tweet),
            'Cleaned_tweet_wt_stopwords' (preprocessed tweets after stop-words removing), 'Class' (as integer)
    '''
    train_data = pd.read_csv(file_name, sep='\t')
    # Clean the text
    train_data["Cleaned_tweet"] = train_data["Tweet"].apply(lambda x: tweet_cleaning(x))
    # Filter stop-words
    train_data['Cleaned_tweet_wt_stopwords'] = train_data['Cleaned_tweet'].apply(
        lambda x: ' '.join([i for i in x.split(' ') if i not in stop_words]))
    # Obtain clean class
    train_data["Class"] = train_data["Intensity Class"].apply(lambda x: get_class(x))
    train_data = train_data.drop(['Intensity Class', 'Affect Dimension'], axis=1)
    return train_data


def upload_datasets(file_train, file_dev, file_test):
    '''
    This function composes emotion datasets in four DataFrames suitable for the further experiments
    Input: three strings with pathes to the train, development and test original emotion datasets
    Output: four DataFrames - for train, development, train concatenated with development, and test datasets
    '''
    train_data = transform_data(file_train)
    dev_data = transform_data(file_dev)
    data = pd.concat([train_data, dev_data], ignore_index=True)
    test_data = transform_data(file_test)
    return train_data, dev_data, data, test_data


def namestr(obj, namespace):
    '''
    This function return the name of 'obj' variable
    Input: obj - any variable, namespace - the namespace setup, we used 'globals()'
    Output: string - the name of 'obj' variable
    '''
    return [name for name in namespace if namespace[name] is obj]
