# @title tfidf作成（ファジー）
# Tf-idf calculation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity # cosine calculation
# from sklearn.metrics.pairwise import linear_kernel # kernel density estimation quick ver of cosine
import pandas as pd
import streamlit as st

from wakati import wakati_proc

# 分析対象の列指定
# strt_rdy = df['f1_']
# strt_rdy = dfs_20["IPC"].str.replace("\(.*?\)| +","").str.replace(","," ").str.replace("/","_")


@st.cache_data
def get_tfidfword(dataframe, colname):
    strt_rdy = wakati_proc(dataframe[colname].astype(str))

    vectorizer = TfidfVectorizer(
        stop_words='english',
        # token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b'
        # token_pattern='^[a-z][0-9]{2}[a-z].*/'
        ngram_range=(1, 1), min_df=2, lowercase=True)

    # fit the vectorizer to the dataframe and generate 1038380 texts tf-idf matrix table
    tfidf_matrix = vectorizer.fit_transform(strt_rdy)

    tfidf_matrix.mean(axis=0)  # axis=1で行方向（column）
    tfidf_df = pd.concat([pd.DataFrame(vectorizer.get_feature_names_out(), columns=["word"]), pd.DataFrame(
        tfidf_matrix.mean(axis=0).T)], axis=1).sort_values(by=0, ascending=False).head(100)

    return tfidf_df
