from pycaret.classification import *
import streamlit as st
from pycaret.datasets import get_data


@st.cache_data
def learning(data):
    setup(data=data,  target='Purchase', html=False)
    compare_models()
    best_model_results = pull()  # 比較結果の取得
    st.write(best_model_results)  # 比較結果の表示

    st.write(get_leaderboard())


@st.cache_data
def learning2(data):
    clf1 = setup(data=data, target='Class variable')

    # creating a model
    lr = create_model('lr')

    # plot model
    plot_model(lr, plot='confusion_matrix', plot_kwargs={'percent': True})


output = st.empty()
with st.expander("結果"):
    learning(data=get_data('juice'))
    #learning2(data=get_data('diabetes'))
