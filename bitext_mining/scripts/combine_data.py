import pandas as pd
import re

def read_and_assign_list(infile): # as list of sentences
  with open(datadir+infile, 'r') as file:
      info = file.read().splitlines()
  return info


def main():
    # 1. Load the mined sentence zip file
    tsv_data = pd.read_csv("/rd_nmt/bitext_mining/data/mining_results/parallel-sentences-out.tsv", sep='\t', header=None)
    tsv_data1 = tsv_data.rename(columns={0: 'score', 1: 'zh', 2: 'yue'})

    # Filter out low quality sentence, the threshold is set to 1.1286
    # after manual examinating the sentecnes 
    # (the recommend threshold to get high quality sentences is 1.2 according to the paper)
    tsv_data_high_quality = tsv_data1.drop(tsv_data1[tsv_data1.score <= 1.1286].index)
    tsv_data_high_quality = tsv_data_high_quality.drop(0)

    # get a list of the sentences for each language
    yue_mined = tsv_data_high_quality["yue"].tolist()
    zh_mined = tsv_data_high_quality["zh"].tolist()
    assert len(yue_mined) == len(zh_mined)
    
    
    # 2. Load the UD data
    datadir="/rd_nmt/bitext_mining/data/UD"
    test_yue = read_and_assign_list("test.yue") 
    test_zh = read_and_assign_list("test.zh") 
    val_yue = read_and_assign_list("val.yue")
    val_zh = read_and_assign_list("val.zh")
    train_yue = read_and_assign_list("train.yue")
    train_zh = read_and_assign_list("train.zh")
    
    # 3. Concatenate the data (UD + sentences from bitext mining) 
    all_yue = test_yue + val_yue + train_yue + yue_mined
    all_zh = test_zh + val_zh + train_zh + zh_mined

    assert len(all_yue) == len(all_zh)
    print(f"After concatenating, there are {len(all_yue)} sentences in the YUE/ZH training data.")
    
    cated_df = pd.DataFrame(list(zip(all_yue, all_zh)), columns = ['yue', 'zh'])

    # pickle the data
    cated_df.to_pickle("/rd_nmt/bitext_mining/data/bitext_and_ud/yue_zh_combined36k.pkl")
    
if __name__ == '__main__':
    main()