from __future__ import print_function, division, unicode_literals
import json
# imports for torchmoji
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_feature_encoding
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from transformers import AutoTokenizer, AutoModel, TFAutoModel
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
# imports for roBERTa model
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from gensim.models import KeyedVectors
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import torch
from transformers import BertTokenizer, BertModel
# Load pre-trained model tokenizer (vocabulary)
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
# Upload BERT model
model_bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
# path to the pre-loaded roBERTa model
MODEL_path_roberta = r"../model/twitter-roberta-base-emotion"
tokenizer_roberta = AutoTokenizer.from_pretrained(MODEL_path_roberta)
# path to the pre-loaded Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin', binary=True)
# uload the big Universal Sentence Encoder model from HTTPS domain
model_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
# upload Sentence-BERT model from the 'sentence_transformers' package
model_sbert = SentenceTransformer('distilbert-base-nli-mean-tokens')


def get_vector_bert(text):
    '''
    This function provides BERT embedding for the tweet
    Input: tweet as a string
    Output: 768-dimentional vector as a list
    '''
    marked_text = "[CLS] " + text + " [SEP]"
    # Tokenize tweet with the BERT tokenizer
    tokenized_text = tokenizer_bert.tokenize(marked_text)
    indexed_tokens = tokenizer_bert.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Make vector
    with torch.no_grad():
        _, _, hidden_states = model_bert(tokens_tensor, segments_tensors)
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding.cpu().detach().numpy()


def get_vector_roberta(text):
    '''
    This function provides Twitter-roBERTa-based embedding for the tweet
    Input: tweet as a string
    Output: 768-dimentional vector as a list
    '''
    encoded_input = tokenizer_roberta(text, return_tensors='tf')
    model = TFAutoModel.from_pretrained(MODEL_path_roberta)
    features = model(encoded_input)
    features = features[0].numpy()
    features_mean = np.mean(features[0], axis=0)
    return features_mean


def get_vectors_deepmoji(data, tweets_column):
    '''
    This function provides DeepMoji (torchmoji) embedding for the tweet
    Input: tweet as a string
    Output: 2304-dimentional vector as a list
    '''
    maxlen = 30
    batch_size = 32
    tweets = list(data[tweets_column])
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    tokenized, _, _ = st.tokenize_sentences(tweets)
    model = torchmoji_feature_encoding(PRETRAINED_PATH)
    encoding = model(tokenized)
    vectors = []
    for i in range(len(encoding)):
        vectors.append(encoding[i, :])
    return vectors


def get_vector_w2v(tweet):
    '''
    This function provides Word2Vec embedding for the tweet
    Input: Word2Vec model imported with KeyedVectors, tweet as a string
    Output: 300-dimentional vector as a list
    '''
    vectors = []
    for w in tweet.split():
        if w in word2vec_model.wv.vocab:
            vectors.append(word2vec_model.wv.__getitem__(w))
        else:
            vectors.append(np.zeros(300))
    return np.mean(vectors, axis=0)


def get_vector_use(tweet):
    '''
    This function provides Universal Sentence Encoder embedding for the tweet
    Input: tweet as a string
    Output: 512-dimentional vector as a list
    '''
    message_embeddings = model_use([tweet])
    mess_list = np.array(message_embeddings).tolist()
    return mess_list[0]


def get_vector_sbert(tweet):
    '''
    This function provides Sentence-BERT embedding for the tweet
    Input: tweet as a string
    Output: 768-dimentional vector as a list
    '''
    return model_sbert.encode(tweet)
