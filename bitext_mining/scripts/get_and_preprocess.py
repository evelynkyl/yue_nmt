import os
import pickle
import re
import pandas as pd
import opencc

def read_txt_to_list_of_string(filepath):
    file_list = []
    for filename in os.listdir(filepath):
        with open(os.path.join(filepath, filename), 'r', encoding='utf-8', errors='ignore') as f:
            file_read = f.read().split('\n')
            file_list.append(file_read)
    return file_list


def convert_scripts():
    """ Convert the Chinese scripts from Simplified to Traditional (Hong Kong variant) """
    converter = opencc.OpenCC('s2hk.json')
    return converter.convert


def sent_segemenaion(para):
    """ segement a document into sentences"""
    for sent in re.findall(u'[^!?。\.\，\！\？\!\?]+[!?。\.\!\?]?', para, flags=re.U):
        yield sent


def df_to_txt(some_df, lang="yue", save_as="comparable_wiki.txt"):
    """ Save the training and test set as .txt to make a training/test set """
    return pd.DataFrame(some_df[lang]).to_csv(save_as,index=False, encoding='utf-8', header=False)         
      
            
def main():
    datapath = "/yue_nmt/bitext_mining/data"
    # load the data
    yue_wiki = read_txt_to_list_of_string(datapath)
    zh_wiki = read_txt_to_list_of_string(datapath)
    # flatten the nested lists
    flat_yue_list = [item for sublist in yue_wiki for item in sublist]
    flat_zh_list = [item for sublist in zh_wiki for item in sublist]

    # build dataframe, where each row contains the content of a wikipedia page
    df_yue = pd.DataFrame((flat_yue_list), columns = ['doc_yue'])
    df_zh = pd.DataFrame((flat_zh_list), columns = ['doc_zh'])

    # convert the scripts from Simplified to Traditional (HK variant)
    df_zh['doc_zh'] = df_zh['doc_zh'].apply(lambda x: convert_scripts(x)) #converter.convert

    # Merge the two dataframes side-by-side
    finalDF = df_yue.reset_index(drop=True).merge(df_zh.reset_index(drop=True), left_index=True, right_index=True)
    
    # segement the sentences
    finalDF['sents_yue'] = finalDF['doc_yue'].apply(lambda x: list(sent_segemenaion(x)))
    finalDF['sents_zh'] = finalDF['doc_zh'].apply(lambda x: list(sent_segemenaion(x)))
    finalDF = finalDF.explode('sents_yue')
    finalDF = finalDF.explode('sents_zh')
    
    # pickle the processed data
   # pic_path = datapath + "/CW_yue_zh_sent.pkl"
  #  finalDF.to_pickle(pic_path)

    # also save as .txt
    final_yue_sents = datapath + "/CW_yue_sents.txt"
    final_zh_sents = datapath + "/CW_zh_sents.txt"
    df_to_txt(finalDF, "sents_yue", final_yue_sents)
    df_to_txt(finalDF, "sents_zh", final_zh_sents)
    
    
if __name__ == '__main__':
    main()
