from collections import defaultdict
import gensim
from gensim import corpora
import streamlit as st
from wakati import wakati_proc
import streamlit as st


@st.cache_data
def get_lda_topics(dataframe, options):
    documents = wakati_proc(dataframe[options].tolist())

    stop_words = set(
        'for a an of the and to in be are or is by then can at on that this の および 及び その あの を た それ 、 。'.split())
    texts = [[word for word in document.lower().split() if word not in stop_words]
             for document in documents]
    frequency = defaultdict(int)

    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    dictionary = corpora.Dictionary(texts)
    # dictionary.save('/tmp/deerwester.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = gensim.models.ldamodel.LdaModel(
        corpus=corpus, num_topics=10, id2word=dictionary, random_state=100)

    ldfdic = lda.show_topics(formatted=False, num_topics=10)

    topics_word_list = []
    for ldatopic in ldfdic:
        word_list = []
        for wd in ldatopic[1]:
            word_list.append(wd[0])
        topics_word_list.append("|".join(word_list))

    return topics_word_list

    # [0][1][[0].tolist()]

    # return topi_list
