#!/bin/bash

# To download and extract monolingual corpora of Cantonese as well as Traditional Chinese #

## Downloading and Extracting data sets

# Cantonese (Yue)
!wget http://dumps.wikimedia.org/zh_yuewiki/20211020/zh_yuewiki-20211020-page.sql.gz
!wget http://dumps.wikimedia.org/zh_yuewiki/20211020/zh_yuewiki-20211020-langlinks.sql.gz

# Traditional Chinese (Zh_tra)
!wget https://dumps.wikimedia.org/zhwiki/20211020/zhwiki-20211020-pages-articles.xml.bz2

### Make corpora
!python make_wiki_corpus.py zh_yuewiki-20211020-pages-articles.xml.bz2 wiki_yue.txt
!python make_wiki_corpus.py zhwiki-20211020-pages-articles.xml.bz2 wiki_zh.txt

# !tar -czvf zhwiki_corpus.tar.gz wiki_zh.txt 

echo "" &&
/utils.py

## Clean the corpora
