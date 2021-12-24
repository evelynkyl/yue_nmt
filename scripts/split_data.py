import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split

target_dir = "/rd_nmt/bitext_mining/data/bitext_and_ud/"

def df_to_txt(some_df, lang, targetdir, filename="train.yue"):
    """ Save the training and test set as .txt to make a training/valid/test set """
    save_as = os.path.join(targetdir, filename)
    return pd.DataFrame(some_df[lang]).to_csv(save_as,index=False, encoding='utf-8', header=False)

def main():
    # load the combined data set
    datapath = target_dir + 'yue_zh_combined36k.pkl'
    with open(datapath, "rb") as f:
        df = pickle.load(f)
    
    # split into train, valid, and test sets
    for_split_df = df.copy()
    train, test = train_test_split(for_split_df, test_size=0.15, random_state=1)
    train_t, valid = train_test_split(train, test_size=0.2, random_state=1)
    print(f"The data is split into {len(train_t)} of training sentences, {len(valid)} of validation sentences, and {len(test)} of test sentences.")

    # save as text files
    save_to_dir = os.path.join(target_dir, 'split')
    df_to_txt(train_t, "target_text", save_to_dir, "train.yue")
    df_to_txt(train_t, "input_text", save_to_dir, "train.zh")

    df_to_txt(test, "target_text", save_to_dir, "test.yue")
    df_to_txt(test, "input_text", save_to_dir, "test.zh")

    df_to_txt(valid, "target_text", save_to_dir, "valid.yue")
    df_to_txt(valid, "input_text", save_to_dir, "valid.zh")
    
if __name__ == '__main__':
    main()