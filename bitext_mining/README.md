# Bitext Mining (Parallel Sentence Mining)

## Preliminary steps:
### 1. Clone the repository
### 2. Install dependencies
```
!pip3 install -r /rd_nmt/bitext_mining/scripts/requirements.txt
```
### 3. Extract the downloaded Wikipedia data
```
!mkdir ~/bitext_mining/data
!tar -xzvf XLM_wiki_zh_yue_text.tar.gz /rd_nmt/bitext_mining/data
```
## Implementation
The following steps show the steps to perform the parallel sentence mining in the project.

### 1. Extract monolingual corpora & data preprocessing, including script conversion
This will load the extracted Wikipedia data, parse it, convert the Chinese scripts from Simiplified to Traditional (HK) and save the final as .txt files.
```
!python3 get_and_preprocess.py
```
### 2. Bitext mining
```
!mkdir ~/bitext_mining/data/mining_results
!python3 mined_bitext.py
```
### 3. Combine with existing parallel corpus as the new training data
Output the concatenated data as a pickle file
```
# unzip the mined file
!gunzip -k parallel-sentences-out.tsv.gz
# load the existing parallel corpus (UD)
!tar -xvzf /rd_nmt/data/ud/zh-yue_data.tar.gz  /rd_nmt/bitext_mining/data/UD
!mkdir /rd_nmt/bitext_mining/data/bitext_and_ud/
!python3 combine_data.py
```
