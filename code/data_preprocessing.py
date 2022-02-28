import functools
import operator
from emoji import *

def tweet_preprocessing(tweet):
    '''
    This function performs the general preprocessing of the tweet, without stop-words removing step
    
    It uses packages "functools", "operator", and "emoji" to transform emojis to text
    
    Input: tweet as a string
    Output: preprocessed tweet as a string
    '''
    # transform emojis to their textual descriptions
    tweet = ' '.join([i for k in functools.reduce(operator.concat, [substr.split() 
                for substr in get_emoji_regexp().split(tweet)]) 
                    for i in demojize(k).replace(":", "").replace("_", " ").split(" ")])
    # transform old-style smilies
    tweet = tweet.replace(':)', 'smiley').replace(':-)', 'smiley').replace(':(', 'sad').replace(':/', 'skeptical').replace(':D', 'laughing').replace(':o', 'surprise').replace(':O', 'surprise')
    # delete '#' in hashtags
    tweet = tweet.replace('#', '')
    return tweet