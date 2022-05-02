from __future__ import print_function, division, unicode_literals
import json
import numpy as np
import tensorflow_hub as hub
import torch

from scipy.spatial.distance import cosine
from collections import defaultdict

# torchmoji 
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_feature_encoding 
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from transformers import AutoTokenizer, AutoModel, TFAutoModel
# roBERTa model 
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
# Word2Vec
from gensim.models import KeyedVectors
# SBERT
from sentence_transformers import SentenceTransformer
# BERT
from transformers import BertTokenizer, BertModel

# Uncomment to preload all models here
'''
# path to the pre-loaded roBERTa model 
MODEL_path_roberta = r"..\model\twitter-roberta-base-emotion"
tokenizer_roberta = AutoTokenizer.from_pretrained(MODEL_path_roberta)
model_roberta = TFAutoModel.from_pretrained(MODEL_path_roberta)
# load pre-trained model tokenizer (vocabulary)
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
# upload BERT model 
model_bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
# upload Sentence-BERT model from the 'sentence_transformers' package 
model_sbert = SentenceTransformer('distilbert-base-nli-mean-tokens')
# upload the big Universal Sentence Encoder model from HTTPS domain 
model_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
# path to the pre-loaded Word2Vec model 
w2v_path = '../model/GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
'''

def get_vector_roberta(text, tokenizer_roberta, model_roberta):
    '''
    This function provides Twitter-roBERTa-based embedding for the tweet.
    
    Input: tweet as a string, preloaded tokenizer_roberta and model_roberta
    Output: 768-dimentional vector  
    '''
    encoded_input = tokenizer_roberta(text, return_tensors='tf')
    features = model_roberta(encoded_input)
    features = features[0].numpy()
    features_mean = features[0][-1]
    return features_mean
    
def get_vector_bert(text, tokenizer_bert, model_bert):
    '''
    This function provides BERT embedding for the tweet.
    
    It uses "torch" library to work with tensors.
    
    Input: tweet as a string, preloaded tokenizer_bert and model_bert
    Output: 768-dimentional vector as 
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
        outputs = model_bert(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding.cpu().detach().numpy()
    
def get_vector_sbert(tweet, model_sbert):
    '''
    This function provides Sentence-BERT embedding for the tweet.
    
    Input: tweet as a string, preloaded model_sbert
    Output: 768-dimentional vector as a list 
    '''
    return model_sbert.encode(tweet)

def get_vectors_deepmoji(tweets):
    '''
    This function provides DeepMoji (torchmoji) embedding for the tweet.
    
    It uses "json" library to upload DeepMoji vocabulary.
    
    Input: tweets - a list of strings
    Output: 2304-dimentional vector as a list 
    '''
    maxlen = 30
    batch_size = 32
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    tokenized, _, _ = st.tokenize_sentences(tweets)
    model = torchmoji_feature_encoding(PRETRAINED_PATH)
    encoding = model(tokenized)
    vectors = []
    for i in range(len(encoding)):
        vectors.append(encoding[i,:])
    return vectors
    
def get_vector_use(tweet, model_use):
    '''
    This function provides Universal Sentence Encoder embedding for the tweet.
    
    It uses "numpy" library as "np" to process vectors.
    
    Input: tweet as a string, preloaded model_use
    Output: 512-dimentional vector as a list 
    '''
    message_embeddings = model_use([tweet])
    mess_list = np.array(message_embeddings).tolist()
    return mess_list[0]
    
def get_vector_w2v(tweet, w2v_model):
    '''
    This function provides Word2Vec embedding for the tweet.
    
    It uses "numpy" library as "np" to create zeros vectors.
    
    Input: tweet as a string, preloaded Word2Vec model
    Output: 300-dimentional vector as a list 
    '''
    vectors = []
    for w in tweet.split():
        if w in w2v_model.wv.vocab:
            vectors.append(w2v_model.wv.__getitem__(w))
        else:
            vectors.append(np.zeros(300))
    return np.mean(vectors, axis = 0)