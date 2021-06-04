This folder should contain two models:

1. **GoogleNews-vectors-negative300.bin** Model for Word2Vec embedding. It can be upload via:
```python
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gzip -d GoogleNews-vectors-negative300.bin.gz
```

2. **twitter-roberta-base-emotion** Folder with roBERTa model. It can be upload via:
```python
git lfs install
git clone https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion
```
Or you can try to run it without preloading (it was an issue for me: https://github.com/cardiffnlp/tweeteval/issues/1). To do it, replace this code in 'code/tweets_embedding.py':
```python 
MODEL_path_roberta = r"..\model\twitter-roberta-base-emotion"
```
with this code:
```python 
MODEL_path_roberta = f"cardiffnlp/twitter-roberta-base-emotion"
```
