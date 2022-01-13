# Extremely Low-Resource Neural Machine Translation: A Case Study of Cantonese
[Result](#result) | [Data](#data) | [Parallel Sentence Mining](#parallel-sentence-mining) | [Model Training](#nmt-model-training)

This repo provides the implentation scripts in the project, as well as the synthetic data generated via bitext mining.

The development of NLP applications for Cantonese, a language with over 85 million speakers, is lagging compared to other languages with a similar number of speakers. This project is, to my best knowledge, the first benchmark of multiple neural machine translation (NMT) systems of Cantonese. Secondly, I performed parallel sentence mining as data augmentation for the extremely low resource language pair (Cantonese-Mandarin) and increased the number of sentence pairs by 3480% (1,002 to 35,877). Results show that with the parallel sentence mining technique, the best performing model (BPE-level bidirectional LSTM) scored 11.98 BLEU better than the vanilla baseline and 9.93 BLEU higher than our strong baseline. Thirdly, we evaluated the quality of the translated texts using modern texts and historical texts to investigate the models' ability to translate historical texts. Finally, we provide the first large-scale parallel training dataset of the language pair (post-sentence mining) as well as an evaluation dataset comprising Cantonese, Mandarin, and Literary Chinese for future research.

## Key implementations in the project
1. Data augumentation via [Parallel Sentence Mining (PSM)](#parallel-sentence-mining)
2. [NMT models training](#nmt-model-training)
    1. Bidirectional LSTM (BiLSTM) MT
          - word represenation
          - **BPE represenation**
    2. Transoformer MT
          - word represenation
          - BPE represenation
    3. Unsupervised NMT via Language Model Pre-training and Transfer Learning

## Result
|  Model          | SacreBLEU  | 
| --------------  | ---------- |
| BiLSTM (Vanilla baseline)  |  1.24 |
| BiLSTM<sub>t</sub> (Strong baseline)   | 3.29 |
| BiLSTM<sub>t</sub> +PSM | 12.37 |
| BiLSTM<sub>bpe</sub> +PSM | 13.22 |
| Transformer<sub>word</sub> +PSM | 3.56 |
| Transformer<sub>bpe</sub> +PSM | 11.66 |
| RELMadap + PSM | 1.85 |


##  Parallel Sentence Mining
The scripts for Parallel Sentence Mining (PSM) (also known as bitext mining) can be found [here](https://github.com/evelynkyl/yue_nmt/tree/main/bitext_mining).
It will perform PSM from Wikipedia backup files, concatenate the UD synthetic data & the synthetic dataset, and finally generate a pickle file of the combined dataset.

## Data
|  Model   | Data    |  Size (Sentence pair)   |  Ratio (Train/Validation/Test) |
| --------  | ------- | ------------------------| ------------------------------ |
| Baseline |  [Parallel data of Cantonese and Mandarin Chinese](https://github.com/UniversalDependencies/UD_Cantonese-HK) (UD) | 1,002 | 80/10/10 |
| Experimental models | UD+PSM | 35,877 | 68/15/17 |

## NMT Model Training
### Preliminary
1. Split data into training, evaluation, and test sets
```
mkdir /yue_nmt/bitext_mining/data/bitext_and_ud/split
python3 split_data.py
```
2. Install dependencies for model training
```
# apex (for fp16 training)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ~

# faiseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ~
```
*note: It is HIGHLY recommended to use half precision (using Apex) by simply adding --fp16 True --amp 1 flags to each running command. Without it, you might run out of memory.

### Implentation
##### BiLSTM or Transformer based (via [JoeyNMT](https://github.com/joeynmt/joeynmt))
Scripts of the model parameters can be found in [/yue_nmt/scripts/training](https://github.com/evelynkyl/yue_nmt/scripts/training).
To train a model, run the command below
```
python3 -m joeynmt train {config.yaml}
```
##### Transfer learning (via [RELM](https://github.com/alexandra-chron/relm_unmt))
Please refer to [UNMT_via_RELM](https://github.com/evelynkyl/yue_nmt/UNMT_via_RELM).

### Evaluation
##### BiLSTM or Transformer based (via [JoeyNMT](https://github.com/joeynmt/joeynmt))
Perform evaluation of the model on the test set. 
This will return the sacrebleu score on the validation and test set based on the highest validation score the model got during training.
```
!python -m joeynmt test {modelname_config.yaml} --output_path /yue_nmt/models/modelname_predictions
```

### Inference
#### BiLSTM or Transformer based (via [JoeyNMT](https://github.com/joeynmt/joeynmt))
```
# file translation
!python3 -m joeynmt translate {modelname_config.yaml} < literary_goldref_zh_bpe.txt --output_path eval_literary_translated_yue.txt
```
