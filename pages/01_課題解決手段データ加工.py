import streamlit as st
import pandas as pd
import numpy as np

from tfidfvectorize import get_tfidfword
from streamlit_app import check_password
from lda import get_lda_topics
# from wakati import to_wakati_proc


def dummy_update():
    pass


@st.cache_data
def read_file(uploaded_file):
    try:
        dataframe = pd.read_table(uploaded_file, encoding="cp932")
    except Exception as e:
        dataframe = pd.read_csv(uploaded_file, encoding="cp932")
    return dataframe


if 'selected_cols' not in st.session_state:
    st.session_state["selected_cols"] = ["要約"]
selected_cols = st.session_state["selected_cols"]

if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = pd.DataFrame()


if check_password():
    try:
        df_dic = st.session_state["data"]
        colnames = st.session_state["colnames"]
        select_ind = st.session_state["select_ind"]
    except Exception as e:
        df_dic = {"row1": [""]}
        colnames = ["col1"]
        select_ind = 0

    with st.expander("STEP1：ファイルアップロード&指定"):
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            dataframe = read_file(uploaded_file)
            st.session_state["dataframe"] = dataframe
        else:
            dataframe = st.session_state["dataframe"]
            st.dataframe(dataframe, use_container_width=True)

    with st.expander("STEP2：param指定"):
        paramcol1, paramcol2 = st.columns(2)
        with paramcol1:
            cols = dataframe.columns.tolist()
            # options = st.selectbox(
            #    '分析対象の列を指定して下さい',
            #    cols, index=cols.index(selected_cols))

            optest = st.multiselect(
                '分析対象の列を～～',
                cols, cols[0]
            )

            dataframe["analyze_col"] = ""
            for opcol in optest:
                dataframe["analyze_col"] = dataframe['analyze_col'] + \
                    "" + dataframe[opcol].astype(str)

            options = "analyze_col"

            # st.write(to_wakati_proc(dataframe["analyze_col"].tolist()))

            extract_ops = st.multiselect(
                'ワード初期抽出方法を指定して下さい',
                ['特徴語(tfidf)', '坂地の手法', 'LDA', '分野別辞書']
            )

            if "特徴語(tfidf)" in extract_ops:
                with st.sidebar.expander("特徴語抽出結果"):
                    tfidfdf = get_tfidfword(dataframe, options)
                    searchtxt = st.text_input("", placeholder="hydrogen")
                    st.dataframe(tfidfdf[tfidfdf["word"].str.contains(
                        searchtxt)], use_container_width=True)

            if "分野別辞書" in extract_ops:
                with st.sidebar.expander("分野別課題解決手段辞書"):
                    st.write("作成中")

            if "LDA" in extract_ops:
                with st.sidebar.expander("LDA抽出結果"):
                    # st.write("作成中")
                    # ldadf = pd.DataFrame(
                    #    get_lda_topics(dataframe, options))
                    # st.dataframe(ldadf)
                    st.write(pd.DataFrame(
                        get_lda_topics(dataframe, options)))
            if "坂地の手法" in extract_ops:
                with st.sidebar.expander("坂地の手法抽出結果"):
                    st.write("作成中")

        with paramcol2:
            st.write("分析対象データ")
            st.dataframe(dataframe[options], use_container_width=True)

    with st.expander("STEP3：データ加工"):
        init_df = pd.DataFrame.from_dict(
            df_dic, columns=colnames, orient="index")

        if st.button('行列交換'):
            init_df = init_df.T

        map_df2 = st.experimental_data_editor(
            init_df, num_rows="dynamic", use_container_width=True,
            on_change=dummy_update()
        )

        rownames = map_df2.index.tolist()
        colnames = map_df2.columns.tolist()
        for col in colnames:
            for row in rownames:
                fillna_df = dataframe.fillna("-")
                cross_count = fillna_df[fillna_df[options].astype(str).str.lower().str.contains(
                    col)][fillna_df[options].astype(str).str.lower().str.contains(row)].shape[0]
                # st.write(cross_count)
                map_df2.loc[row, col] = cross_count

        temp_df2_dic = map_df2.reset_index().to_dict(orient="records")
        # st.write(temp_df2_dic)
        tempdf = pd.DataFrame.from_dict(temp_df2_dic)
        # st.write(tempdf)
        # st.write(pd.DataFrame.from_dict(temp_df2_dic).set_index("index"))

        # editableで編集するとなぜか１行重複行が発生する場合があるdrop_duplicatesで重複削除
        try:
            df_dic = map_df2.to_dict(orient="index")
            # st.write(df_dic)
        except:
            df_dic = map_df2.reset_index().drop_duplicates(
            ).set_index("index").to_dict(orient="index")
            # st.write(df_dic)

        # st.write(colnames)
        # st.write(select_ind)
        st.session_state["data"] = df_dic
        st.session_state["colnames"] = colnames
        st.session_state["select_ind"] = select_ind
        st.session_state["agg_datframe"] = map_df2
        st.session_state["selected_cols"] = options
        st.session_state["dataframe"] = dataframe
