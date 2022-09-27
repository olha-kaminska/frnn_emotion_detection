## Fuzzy-Rough Nearest Neighbour Approaches for Emotion Detection in Tweets
Code for the paper written by [Olha Kaminska](https://scholar.google.com/citations?hl=en&user=yRgJkEwAAAAJ), [Chris Cornelis](https://scholar.google.com/citations?hl=en&user=ln46HlkAAAAJ), and [Veronique Hoste](https://scholar.google.com/citations?hl=en&user=WxOsW3IAAAAJ) and presented at [IJCRS 2021](http://ifsa-eusflat2021.eu/ijcrs_conf.html) conference, organized jointly with [IFSA-EUSFLAT 2021](http://ifsa-eusflat2021.eu/). 

The task is based on [SemEval-2018 Task 1: Affect in Tweets](https://competitions.codalab.org/competitions/17751) competition. We chose the ordinal classification Task EI-oc: Detecting Emotion Intensity.

### Repository Overview ###
- The **code** directory contains .py files with different functions:
  - *data_preprocessing.py* - functions for data uploading and preperation;
  - *frnn_owa_eval.py* - functions for FRNN-OWA approach and cross-validation;
  - *tweets_embedding.py* - functions for tweets embeddings with different methods.
- The **data** directory contains *README_data_download.md* file with instruction on uploading necessary dataset files that should be saved in the *data* folder.
- The **model** directory contains *README_model_download.md* file with instruction on uploading necessary models that should be saved in the *model* folder.
- The file **Test.ipynb** provides an overview of all function and their usage. It is built as a pipeline describes in the paper with corresponded results. 
- The file *requirements.txt* contains the list of all necessary packages and versions used with the Python 3.7.4 environment.

### Arxiv link ### 
https://arxiv.org/abs/2107.05392  

### BibTex citation ###
 >@inproceedings{kaminska2021fuzzy,
  title={Fuzzy-Rough Nearest Neighbour Approaches for Emotion Detection in Tweets},
  author={Kaminska, Olha and Cornelis, Chris and Hoste, Veronique},
  booktitle={International Joint Conference on Rough Sets},
  pages={231--246},
  year={2021},
  organization={Springer}
}
  
### Abstract ###
*Social media are an essential source of meaningful data used in different tasks such as sentiment analysis and emotion recognition. Mostly, these tasks are solved with deep learning methods. Due to the fuzzy nature of textual data, we consider using classification methods based on fuzzy rough sets.*

*Specifically, we develop an approach for the SemEval-2018 emotion detection task, based on the fuzzy rough nearest neighbour (FRNN) classifier enhanced with ordered weighted average (OWA) operators. We use tuned ensembles of FRNN-OWA models based on different text embedding methods. Our results are competitive with the best SemEval solutions based on more complicated deep learning methods.*
