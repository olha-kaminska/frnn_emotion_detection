# frnn_emotion_detection
Code for the paper "Fuzzy-Rough Nearest Neighbour Approaches for Emotion Detection in Tweets" by O. Kaminska, Ch. Cornelis and V. Hoste presented at IJCRS 2021 conference. 

The task is based on [SemEval-2018 Task 1: Affect in Tweets](https://competitions.codalab.org/competitions/17751) competition. We  chose the ordinal classification Task EI-oc: Detecting Emotion Intensity.

### Repository Overview ###
- The **code** directory contains .py files with different functions:
  - *data_preprocessing.py* - functions for data uploading and preperation;
  - *frnn_owa_eval.py* - functions for FRNN-OWA approach and cross-validation;
  - *tweets_embedding.py* - functions for tweets embeddingswith different methods.
- The **data** directory contains *README_data_download.md* file with instruction how to upload nesessary dataset files, that should be saved in the *data* folder.
- The **model** directory contains *README_model_download.md* file with instruction how to upload nesessary models, that should be saved in the *model* folder.
- The file **Test.ipynb** provides an overview of all function and their usage. It built as a pipeline describes in the paper with corresponded results. 
- The file *requirements.txt* contains the list of all nesessary packages and their versions, that couls be used with Python 3.7.4 environment.


### Abstract:###
*Social media are an essential source of meaningful data that can be used in different tasks such as sentiment analysis and emotion recognition. Mostly, these tasks are solved with deep learning methods. Due to the fuzzy nature of textual data, we consider using classification methods based on fuzzy rough sets.
Specifically, we develop an approach for the SemEval-2018 emotion detection task, based on the fuzzy rough nearest neighbour (FRNN) classifier enhanced with ordered weighted average (OWA) operators. We use tuned ensembles of FRNN--OWA models based on different text embedding methods. Our results are competitive with the best SemEval solutions based on more complicated deep learning methods.*
