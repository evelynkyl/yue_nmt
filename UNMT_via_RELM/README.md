# Unsupervised Neural Machine Translation (UNMT) via Pretraining Language Model and Transfer Learning using RELM
[Original Paper](https://www.aclweb.org/anthology/2020.emnlp-main.214/) | [Original Repo](https://github.com/alexandra-chron/relm_unmt)

## Preliminary steps:
### 1. Clone the following repositories
```
git clone https://github.com/evelynkyl/yue_nmt/UNMT_via_RELM
cd UNMT_via_RELM
git clone https://github.com/alexandra-chron/relm_unmt.git
cd relm_unmt
```
### 2. Install dependencies
```
!pip3 install -r /yue_nmt/UNMT_via_RELM/requirements.txt
```
### 3. Preprocessing data
Run the following bash script
```
./preprocessing.sh
```

## Implementation of RELM + adapters
### 1. Train a monolingual LM
Train the monolingual masked LM (BERT without the next-sentence prediction task) on the monolingual (Zh) data:
```
python3  /yue_nmt/UNMT_via_RELM/relm_unmt/train.py \
--exp_name mono_mlm_zh \
--dump_path /yue_nmt/UNMT_via_RELM/models/ \
--data_path /yue_nmt/UNMT_via_RELM/data/zh/  \
--lgs 'zh' \
--mlm_steps 'zh' \
--emb_dim 512 \
--n_layers 3 \
--n_heads 4 \
--dropout '0.1' \
--attention_dropout '0.1' \
--gelu_activation true  \
--batch_size 32 \
--bptt 256 \
--optimizer 'adam,lr=0.0001' \
--epoch_size 200000  \
--validation_metrics _valid_zh_mlm_ppl \
--stopping_criterion '_valid_zh_mlm_ppl,10' 
```
### Preprocessing for fine-tuning (and UNMT)
Run the following bash script
```
./preprocessing_for_finetune.sh
```
### 2. Fine-tune part of the model on the target language only using adapters
The monolingual LM model checkpoint (mono_mlm_zh) can be found [here](https://drive.google.com/file/d/1IVAXJ8abpQ7rW3KNxl5-9MZuYMztF5QL/view?usp=sharing)
```
python3  /yue_nmt/UNMT_via_RELM/relm_unmt/train.py \
--exp_name finetune_zh_mlm_yue_adapters \
--dump_path /yue_nmt/UNMT_via_RELM/models/ \
--reload_model '/yue_nmt/UNMT_via_RELM/models/mono_mlm_zh/checkpoint.pth' \
--data_path /yue_nmt/UNMT_via_RELM/data/yue-zh/  \
--lgs 'zh-yue' \
--clm_steps '' \
--mlm_steps 'yue' \
--mlm_eval_steps 'zh' \
--emb_dim 512 \
--n_layers 3 \
--n_heads 4 \
--dropout '0.1' \
--attention_dropout '0.1' \
--gelu_activation true \
--use_adapters True \
--batch_size 32 \
--bptt 256 \
--optimizer 'adam,lr=0.0001' \
--epoch_size 100000  \
--validation_metrics _valid_yue_mlm_ppl \
--stopping_criterion '_valid_yue_mlm_ppl,10' \
--increase_vocab_for_lang zh \
--increase_vocab_from_lang yue \
--increase_vocab_by 8389
```

### 3. Train a UNMT model (encoder and decoder initialized with RE-LM + adapters)
The fine-tuned model checkpoint (finetune_zh_mlm_yue_adapters) can be found [here](https://drive.google.com/file/d/1I939PPedFsC6IxtnZasY51bY6tVcr7Et/view?usp=sharing)

```
python3  /yue_nmt/UNMT_via_RELM/relm_unmt/train.py \
--exp_name unsupMT_ft_yue \
--dump_path /yue_nmt/UNMT_via_RELM/models/ \
--reload_model '/yue_nmt/UNMT_via_RELM/models/finetune_zh_mlm_yue_adapters/checkpoint.pth','/yue_nmt/UNMT_via_RELM/models/finetune_zh_mlm_yue_adapters/checkpoint.pth' \
--data_path /yue_nmt/UNMT_via_RELM/data/yue-zh/  \
--lgs 'zh-yue' \
--ae_steps zh,yue \
--bt_steps zh-yue-zh,yue-zh-yue \
--word_shuffle 3 \
--word_dropout 0.1 \
--word_blank 0.1 \
--lambda_ae 0:1,100000:0.1,300000:0  \
--encoder_only False \
--emb_dim 512 \
--n_layers 3 --n_heads 4 \
--dropout '0.1' \
--attention_dropout '0.1' \
--gelu_activation true \
--use_adapters True \
--tokens_per_batch 1000 \
--batch_size 32 --bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 100000  \
--eval_bleu true \
--beam_size 5 \ 
--stopping_criterion valid_yue-zh_mt_bleu,10 \
--validation_metrics valid_yue-zh_mt_bleu \
--increase_vocab_for_lang zh \
--increase_vocab_from_lang yue \
--increase_vocab_by 8389
```
