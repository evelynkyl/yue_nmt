#!/bin/bash
# Fine-tuning on Multilingual NMT
# This is to finetune mBART cc25 model on zh-yue language pair (unseen in training)
# *Note -- It requires packages: [simpletransformers, transformers, fairseq]
#For tokenization
pip install sentencepiece -q
pip install omegaconf==2.0.6

# Download and unzip model
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz
tar -xzvf mbart.CC25.tar.gz

sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev

FAIRSEQ="/content/fairseq"
DATA="/rd_nmt/bitext_mining/data/bitext_and_ud/split"
SRC="zh"
TGT="yue"

# Preprocessing
TRAIN="train"
VALID="valid"
TEST="test"
DEST="/checkpoint"
NAME="zh-yue"
DICT="/mbart.cc25/dict.txt"

python ${FAIRSEQ}/fairseq_cli/preprocess.py \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${DATA}/${TRAIN}$.spm \
--validpref ${DATA}/${VALID}$.spm \
--testpref ${DATA}/${TEST}$.spm  \
--destdir ${DEST}/${NAME}$ \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT}$ \
--tgtdict ${DICT}$ \
--workers 70


# Model training
PRETRAIN="/mbart.cc25/model.pt"
langs="ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN"
SAVEDIR="checkpoint_zh-yue" 

python ${FAIRSEQ}$/train.py ${DATA}$ \
  --encoder-normalize-before --decoder-normalize-before \
  --arch "mbart_large" --task "translation_from_pretrained_bart" \
  --source-lang ${SRC}$ --target-lang ${TGT}$ \
  --criterion "label_smoothed_cross_entropy" \
  --label-smoothing 0.2  --dataset-impl "mmap" \
  --optimizer "adam" --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler "polynomial_decay" --lr 3e-05 \
  --warmup-updates 2500 --max-update 80000 --dropout 0.3 \
  --attention-dropout 0.1  --weight-decay 0.0 --max-tokens 768 \
  --update-freq 2 --save-interval 1 --save-interval-updates 8000 \
  --keep-interval-updates 10 --no-epoch-checkpoints --seed 222 \
  --log-format "simple" --log-interval 2 --reset-optimizer \
  --reset-meters --reset-dataloader --reset-lr-scheduler \
  --restore-file ${PRETRAIN}$ --langs ${langs}$ --layernorm-embedding --ddp-backend "no_c10d" --save-dir ${SAVEDIR}$