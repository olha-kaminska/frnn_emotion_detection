## LT3 at SemEval-2022 Task 6: Fuzzy-Rough Nearest neighbor Classification for Sarcasm Detection
Code for the paper written by [Olha Kaminska](https://scholar.google.com/citations?hl=en&user=yRgJkEwAAAAJ), [Chris Cornelis](https://scholar.google.com/citations?hl=en&user=ln46HlkAAAAJ), and [Veronique Hoste](https://scholar.google.com/citations?hl=en&user=WxOsW3IAAAAJ) for [SemEval 2022 - Task 6 (iSarcasmEval):](https://codalab.lisn.upsaclay.fr/competitions/1340) **Intended Sarcasm Detection In English and Arabic**. We participated in the *SubTask A:* Given a text, determine whether it is sarcastic or non-sarcastic.

### Repository Overview ###
- The **code** directory contains .py files with different functions:
  - *data_preprocessing.py* - functions for data uploading and preperation;
  - *frnn_owa_eval.py* - functions for FRNN-OWA approach and cross-validation;
  - *tweets_embedding.py* - functions for tweets embeddings with different methods.
- The **data** directory contains *README_data_download.md* file with instruction on uploading necessary dataset files that should be saved in the *data* folder.
- The **model** directory contains *README_model_download.md* file with instruction on uploading necessary models that should be saved in the *model* folder.
- The file **iSarcasmEval.ipynb** provides an overview of our pipeline during the problem solving. 
- The file **test_labels.txt** is the final file with test labels that we provided to the organizers.
- The file **requirements.txt** contains the list of all necessary packages and versions used with the Python 3.7.4 environment.

### Abstract ###
*This paper describes the approach developed by the LT3 team in the Intended Sarcasm Detection task at SemEval-2022 Task 6. We considered the binary classification subtask A for English data. The presented system is based on the fuzzy-rough nearest neighbor classification method using various text embedding techniques. Our solution reached 9th place in the official leader-board for English subtask A.*

### BibTeX citation ###
>@inproceedings{kaminska2022lt3,
  title={LT3 at SemEval-2022 Task 6: Fuzzy-Rough Nearest Neighbor Classification for Sarcasm Detection},
  author={Kaminska, Olha and Cornelis, Chris and Hoste, Veronique},
  booktitle={Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
  pages={987--992},
  year={2022}
}
