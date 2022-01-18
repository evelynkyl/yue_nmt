# Extremely Low-Resource Neural Machine Translation: A Case Study of Cantonese
[Result](#result) | [Data](#data) | [Parallel Sentence Mining](#parallel-sentence-mining) | [Model Training](#nmt-model-training)

This repo provides the implentation scripts in the project, as well as the synthetic data generated via bitext mining.

The development of NLP applications for Cantonese, a language with over 85 million speakers, is lagging compared to other languages with a similar number of speakers. This project is, to my best knowledge, the first benchmark of multiple neural machine translation (NMT) systems of Cantonese. Secondly, I performed parallel sentence mining as data augmentation for the extremely low resource language pair (Cantonese-Mandarin) and increased the number of sentence pairs by 3480% (1,002 to 35,877). Results show that with the parallel sentence mining technique, the best performing model (BPE-level bidirectional LSTM) scored 11.98 BLEU better than the vanilla baseline and 9.93 BLEU higher than my strong baseline. Thirdly, I evaluated the quality of the translated texts using modern texts and historical texts to investigate the models' ability to translate historical texts. Finally, I provide the first large-scale parallel training dataset of the language pair (post-sentence mining) as well as an evaluation dataset comprising Cantonese, Mandarin, and Literary Chinese for future research.

## Key implementations in the project
1. Data augumentation via [Parallel Sentence Mining (PSM)](#parallel-sentence-mining)
2. [NMT models training](#nmt-model-training)
    1. Bidirectional LSTM (BiLSTM) MT
          - word represenation
          - **BPE represenation** (highest BLEU score)
    2. Transoformer MT
          - word represenation
          - **BPE represenation** (best translation quality)
    3. Unsupervised NMT via Language Model Pre-training and Transfer Learning

*Note: The script of finetuning mBART can be found [here](https://github.com/evelynkyl/yue_nmt/failed_attempt_mnmt/mbart_finetune_yue.sh); however it should be noted that this approach failed to perform on the unseen language (Cantonese) and resulted in a 0 BLEU score.

## Result
|  Model          | SacreBLEU  | 
| --------------  | ---------- |
| BiLSTM (Vanilla baseline)  |  1.24 |
| BiLSTM<sub>t</sub> (Strong baseline)   | 3.29 |
| BiLSTM<sub>t</sub> +PSM | 12.37 |
| BiLSTM<sub>bpe</sub> +PSM | 13.22 |
| Transformer<sub>word</sub> +PSM | 3.56 |
| Transformer<sub>bpe</sub> +PSM | 11.66 |
| RELM<sub>adap</sub> + PSM | 1.85 |

## Pretrained models
- [BiLSTM<sub>bpe</sub> +PSM](https://github.com/evelynkyl/yue_nmt/tree/main/models/bilstm)
- [Transformer<sub>bpe</sub> +PSM](https://github.com/evelynkyl/yue_nmt/tree/main/models/transformer)

##  Parallel Sentence Mining
The scripts for Parallel Sentence Mining (PSM) (also known as bitext mining) can be found [here](https://github.com/evelynkyl/yue_nmt/bitext_mining).
It will perform PSM from Wikipedia backup files, concatenate the UD data & the synthetic dataset, and finally generate a pickle file of the combined dataset.

## Data
|  Model   | Data    |  Size (Sentence pair)   |  Ratio (Train/Validation/Test) |
| --------  | ------- | ------------------------| ------------------------------ |
| Baseline |  [Cantonese and Mandarin Chinese Parallel Corpus](https://github.com/UniversalDependencies/UD_Cantonese-HK) (UD) | 1,002 | 80/10/10 |
| Experimental models | [UD+PSM](https://github.com/evelynkyl/yue_nmt/blob/main/data/ud_and_bitext/yue_zh_combined36k.pkl) | 35,877 | 68/15/17 |

## NMT Model Training
### Preliminary
1. Clone current repo
```
git clone https://github.com/evelynkyl/yue_nmt
```
2. Split data into training, evaluation, and test sets
```
mkdir /yue_nmt/bitext_mining/data/bitext_and_ud/split
python3 split_data.py
```
3. Install dependencies for model training
```
# apex (for fp16 training)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ~

# faiseq (for machine translation)
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ~
```
*Note: It is HIGHLY recommended to use half precision (using Apex) by simply adding --fp16 True --amp 1 flags to each running command. Without it, you might run out of memory.

### Implentation
#### BiLSTM or Transformer (via [JoeyNMT](https://github.com/joeynmt/joeynmt))
Scripts of the model parameters can be found in [/yue_nmt/scripts/training](https://github.com/evelynkyl/yue_nmt/scripts/training).
To train a model, run the command below
```
python3 -m joeynmt train {config.yaml}
```
#### Unsupervised NMT by Transfer Learning (via [RELM](https://github.com/alexandra-chron/relm_unmt))
Please refer to [UNMT_via_RELM](https://github.com/evelynkyl/yue_nmt/UNMT_via_RELM).

### Evaluation
#### BiLSTM or Transformer (via [JoeyNMT](https://github.com/joeynmt/joeynmt))
Perform evaluation of the model on the test set. 
This will return the sacrebleu score on the validation and test set based on the highest validation score the model got during training.
```
python3 -m joeynmt test {modelname_config.yaml} --output_path /yue_nmt/models/modelname_predictions
```

### Inference
#### BiLSTM or Transformer (via [JoeyNMT](https://github.com/joeynmt/joeynmt))
```
# file translation
python3 -m joeynmt translate {modelname_config.yaml} < literary_goldref_zh_bpe.txt --output_path eval_literary_translated_yue.txt
```
## License
Yue_NMT is BSD-licensed, as found in the LICENSE file in the root directory of this source tree.

## Acknowledgement
- The UD dataset is downloaded from [UD Cantonese](https://github.com/UniversalDependencies/UD_Cantonese-HK) based on the [Universal Dependecies Project](https://universaldependencies.org).
- The Literary-Modern Chinese evaluation dataset is manually translated based on [Ancient-Modern Chinese Translation with a New Large Training Dataset](https://github.com/dayihengliu/a2m_chineseNMT?utm_source=catalyzex.com).
- Our code of bitext mining is based on [LASER](https://github.com/facebookresearch/LASER). 
- Our code of unsupverised NMT (RELM) is based on [RELM](https://github.com/alexandra-chron/relm_unmt). 

We thank the authors for sharing their great work.
