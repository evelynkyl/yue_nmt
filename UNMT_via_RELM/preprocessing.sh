#!/bin/bash

# For training monolingual LM on the high-resource language
mkdir /yue_nmt/UNMT_via_RELM/data/zh

# clone & install fastBPE
git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast

# learn bpe codes on the training set (or only use a subset of it)
./fast learnbpe 30000 /yue_nmt/bitext_mining/data/bitext_and_ud/split/train.zh > /yue_nmt/UNMT_via_RELM/data/zh/codes

# apply BPE tokenization to train files
./fast applybpe /yue_nmt/UNMT_via_RELM/data/zh/train.zh /yue_nmt/bitext_mining/data/bitext_and_ud/split/train.zh /yue_nmt/UNMT_via_RELM/data/zh/codes

# extract source vocabulary
./fast getvocab /yue_nmt/UNMT_via_RELM/data/zh/train.zh > /yue_nmt/UNMT_via_RELM/data/zh/vocab.zh

# binarize the trained data
python3 /yue_nmt/UNMT_via_RELM/relm_unmt/preprocess.py /yue_nmt/UNMT_via_RELM/data/zh/vocab.zh /yue_nmt/UNMT_via_RELM/data/zh/train.zh

# apply BPE tokenization to validation & test files:
./fast applybpe /yue_nmt/UNMT_via_RELM/data/zh/valid.zh /yue_nmt/bitext_mining/data/bitext_and_ud/split/valid.zh /yue_nmt/UNMT_via_RELM/data/zh/codes /yue_nmt/UNMT_via_RELM/data/zh/vocab.zh

./fast applybpe /yue_nmt/UNMT_via_RELM/data/zh/test.zh /yue_nmt/bitext_mining/data/bitext_and_ud/split/test.zh /yue_nmt/UNMT_via_RELM/data/zh/codes /yue_nmt/UNMT_via_RELM/data/zh/vocab.zh

# binarize the valid/test data
python3 /yue_nmt/UNMT_via_RELM/relm_unmt/preprocess.py /yue_nmt/UNMT_via_RELM/data/zh/vocab.zh /yue_nmt/UNMT_via_RELM/data/zh/valid.zh

python3 /yue_nmt/UNMT_via_RELM/relm_unmt/preprocess.py /yue_nmt/UNMT_via_RELM/data/zh/vocab.zh /yue_nmt/UNMT_via_RELM/data/zh/test.zh
