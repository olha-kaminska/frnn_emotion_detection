## Fuzzy Rough Nearest Neighbour Methods for Detecting Emotions, Hate Speech and Irony
Code for the paper written by [Olha Kaminska](https://scholar.google.com/citations?hl=en&user=yRgJkEwAAAAJ), [Chris Cornelis](https://scholar.google.com/citations?hl=en&user=ln46HlkAAAAJ), and [Veronique Hoste](https://scholar.google.com/citations?hl=en&user=WxOsW3IAAAAJ). 

### Repository Overview ###
- The **code** directory contains .py files with different functions:
  - *preprocessing.py* - functions for data uploading and preperation;
  - *frnn_owa_eval.py* - functions for FRNN-OWA approach and cross-validation;
  - *tweets_embedding.py* - functions for tweets embeddings with different methods.
- The **data** directory contains *README_data_download.md* file with instruction on uploading necessary dataset files that should be saved in the *data* folder.
- The **model** directory contains *README_model_download.md* file with instruction on uploading necessary models that should be saved in the *model* folder.
- The file **Example.ipynb** provides an overview of all function and their usage. It is built as a pipeline describes in the paper with corresponded results. As an example it uses one out of seven datases that we considered - [Hate Speech SemEval 2019 Task 5 dataset](https://competitions.codalab.org/competitions/19935).
- The file *requirements.txt* contains the list of all necessary packages and versions used with the Python 3.7.4 environment.

### Abstract ###
*Due to the ever-expanding volumes of information available on social media, the need for reliable and efficient automated text understanding mechanisms becomes evident. Unfortunately, most current approaches rely on black-box solutions rooted in deep learning technologies. In order to provide a more transparent and interpretable framework for extracting intrinsic text characteristics like emotions, hate speech and irony, we propose the integration of fuzzy rough set techniques and text embeddings. We apply our methods to different classification problems originating from Semantic Evaluation (SemEval) competitions, and demonstrate that their accuracy is on par with leading deep learning solutions.*
