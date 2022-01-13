# Extremely Low-Resource Neural Machine Translation: A Case Study of Cantonese

This repo provides the implentation scripts in the project, as well as the synthetic data generated via bitext mining.

The development of NLP applications for Cantonese, a language with over 85 million speakers, is lagging compared to other languages with a similar number of speakers. In this paper, we present, to our best knowledge, the first benchmark of multiple neural machine translation (NMT) systems between Cantonese and Mandarin Chinese. Secondly, we performed parallel sentence mining as data augmentation for the extremely low resource language pair and increased the number of sentence pairs by 3480% (1,002 to 35,877). Results show that with the parallel sentence mining technique, the best performing model (BPE-level bidirectional LSTM) scored 11.98 BLEU better than the vanilla baseline and 9.93 BLEU higher than our strong baseline. Thirdly, we evaluated the quality of the translated texts using modern texts and historical texts to investigate the models' ability to translate historical texts. Finally, we provide the first large-scale parallel training dataset of the language pair (post-sentence mining) as well as an evaluation dataset comprising Cantonese, Mandarin, and Literary Chinese for future research.

## Key implementations in the project
1. Bidirectional LSTM (BiLSTM) MT
    - word-level
    - **BPE-level**
2. Transofrmer MT
    - word-level
    - BPE-level
3. Unsupervised NMT via Language Model Pretraining and Transfer Learning

For the baseline models, we only used existing parallel data of Cantonese and Mandarin Chinese (UD data), while for the experimental models we used both the UD data and synthetic data generated via bitext mining.

## Bitext mining
The scripts for bitext mining is in here. [add link]
It will generate a pickle file comprised of the UD data and synthetic data from the mining.

## Model training
### Preliminary
1. Split data into training, evaluation, and test sets
```
!mkdir /rd_nmt/bitext_mining/data/bitext_and_ud/split
!python3 split_data.py
```
2. Install dependencies for model training
```
# apex
!git clone https://github.com/NVIDIA/apex
!cd apex
!pip install -v --disable-pip-version-check --no-cache-dir ./
!cd ~

# faiseq
!git clone https://github.com/pytorch/fairseq
%cd fairseq
!pip install --editable ./
!cd ~
```
### Implentation
Scripts of the model parameters can be found in /rd_nmt/scripts/training
##### BiLSTM or Transformer based (via JoeyNMT)
""" TODO: rename the variables inside the JoeyNMT yaml script 
"""
Train a model by running the command below
```
!python3 -m joeynmt train {config.yaml}
```
##### Transfer learning (via XLM)
```


```
### Evaluation
##### BiLSTM or Transformer based (via JoeyNMT)
Perform evaluation of the model on the test set. 
This will return the sacrebleu score on the validation and test set based on the highest validation score the model got during training.
```
!python -m joeynmt test {modelname_config.yaml} --output_path /models/modelname_predictions
```

### Inference
#### BiLSTM or Transformer based (via JoeyNMT)
```
# file translation
!python3 -m joeynmt translate {modelname_config.yaml} < literary_goldref_zh_bpe.txt --output_path eval_literary_translated_yue.txt
```
