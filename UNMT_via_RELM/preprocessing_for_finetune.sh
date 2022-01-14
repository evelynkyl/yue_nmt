#!/bin/bash

# Bash script for preprocessing the high-resource and low-resource language pair 
# in preparation for fine-tuning and UNMT
# *note: make sure fastBPE is installed

mkdir /yue_nmt/UNMT_via_RELM/data/yue-zh

# learn BPE codes on the concatenation of the source and target datasets
./fast learnbpe 30000 /yue_nmt/bitext_mining/data/bitext_and_ud/split/train.zh /yue_nmt/bitext_mining/data/bitext_and_ud/split/train.yue > /yue_nmt/UNMT_via_RELM/data/yue-zh/codes.yue-zh

# apply BPE codes to the target language
./fast applybpe /yue_nmt/UNMT_via_RELM/data/yue-zh/train.yue /yue_nmt/bitext_mining/data/bitext_and_ud/split/train.yue /yue_nmt/UNMT_via_RELM/data/yue-zh/codes.yue-zh

# extract target vocabulary
./fast getvocab /yue_nmt/UNMT_via_RELM/data/yue-zh/train.yue > /yue_nmt/UNMT_via_RELM/data/yue-zh/vocab.yue

# compute full vocabulary
python3 /yue_nmt/UNMT_via_RELM/relm_unmt/add_vocabs.py zh yue #get [vocab.yue-zh-ext-by-8389]

# binarize data
# SRC data
python3 /yue_nmt/UNMT_via_RELM/relm_unmt/preprocess.py /yue_nmt/UNMT_via_RELM/data/yue-zh/vocab.yue-zh-ext-by-8389 /yue_nmt/UNMT_via_RELM/data/zh/train.zh &
# TGT data
python3 /yue_nmt/UNMT_via_RELM/relm_unmt/preprocess.py /yue_nmt/UNMT_via_RELM/data/yue-zh/vocab.yue-zh-ext-by-8389 /yue_nmt/UNMT_via_RELM/data/yue-zh/train.yue &


# apply BPE to valid and test files
# SRC data
./fast applybpe /yue_nmt/UNMT_via_RELM/data/yue-zh/valid.zh /yue_nmt/bitext_mining/data/bitext_and_ud/split/valid.zh /yue_nmt/UNMT_via_RELM/data/zh/codes &
./fast applybpe /yue_nmt/UNMT_via_RELM/data/yue-zh/test.zh /yue_nmt/bitext_mining/data/bitext_and_ud/split/test.zh /yue_nmt/UNMT_via_RELM/data/zh/codes &
# TGT data
./fast applybpe /yue_nmt/UNMT_via_RELM/data/yue-zh/valid.yue /yue_nmt/bitext_mining/data/bitext_and_ud/split/valid.yue /yue_nmt/UNMT_via_RELM/data/yue-zh/codes.yue-zh &
./fast applybpe /yue_nmt/UNMT_via_RELM/data/yue-zh/test.yue /yue_nmt/bitext_mining/data/bitext_and_ud/split/test.yue /yue_nmt/UNMT_via_RELM/data/yue-zh/codes.yue-zh &


# binarize the valid/test data
# SRC data
python3 /yue_nmt/UNMT_via_RELM/relm_unmt/preprocess.py /yue_nmt/UNMT_via_RELM/data/yue-zh/vocab.yue-zh-ext-by-8389 /yue_nmt/UNMT_via_RELM/data/yue-zh/valid.zh &
python3 /yue_nmt/UNMT_via_RELM/relm_unmt/preprocess.py /yue_nmt/UNMT_via_RELM/data/yue-zh/vocab.yue-zh-ext-by-8389 /yue_nmt/UNMT_via_RELM/data/yue-zh/test.zh &
# TGT data
python3 /yue_nmt/UNMT_via_RELM/relm_unmt/preprocess.py /yue_nmt/UNMT_via_RELM/data/yue-zh/vocab.yue-zh-ext-by-8389 /yue_nmt/UNMT_via_RELM/data/yue-zh/valid.yue &
python3 /yue_nmt/UNMT_via_RELM/relm_unmt/preprocess.py /yue_nmt/UNMT_via_RELM/data/yue-zh/vocab.yue-zh-ext-by-8389 /yue_nmt/UNMT_via_RELM/data/yue-zh/test.yue &


# Link monolingual validation and test data to parallel data, also link SRC train set to this folder
cd /yue_nmt/UNMT_via_RELM/data/yue-zh/
# SRC data
sudo ln -sf valid.zh.pth valid.zh-yue.zh.pth &
sudo ln -sf test.zh.pth test.zh-yue.zh.pth &
# TGT data
sudo ln -sf valid.yue.pth valid.zh-yue.yue.pth &
sudo ln -sf test.yue.pth test.zh-yue.yue.pth &


