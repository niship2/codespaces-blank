import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from streamlit_app import check_password


if check_password():
    with st.expander("ファイルアップロード&指定"):
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file, encoding="cp932")
            # To read file as bytes:
            # bytes_data = uploaded_file.getvalue()
            # st.write(bytes_data)

            # To convert to a string based IO:
            # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # st.write(stringio)

            # To read file as string:
            # string_data = stringio.read()
            # st.write(string_data)
    with st.expander("param指定"):
        # Can be used wherever a "file-like" object is accepted:

        edited_df = st.experimental_data_editor(dataframe)
        # st.write(dataframe)

    with st.expander("データ加工"):
        num_rows = st.slider('行数指定', 2, 25, 2)
        num_cols = st.slider('列数指定', 2, 25, 2)

        init_list = [[0] * num_cols] * num_rows
        init_rownames = [str(idx) + word for idx,
                         word in enumerate(["ka"]*num_rows)]
        init_colnames = [str(idx) + word for idx,
                         word in enumerate(["ko"]*num_cols)]

        init_df = pd.DataFrame(init_list, index=init_colnames)

        option = st.selectbox('編集対象選択', ('行', '列'))

        map_df = st.experimental_data_editor(init_df, num_rows="dynamic")

        if option == '行':
            edf = st.experimental_data_editor(map_df, num_rows="dynamic")
        else:
            edf = st.experimental_data_editor(map_df.T, num_rows="dynamic")

        kadai_dic = map_df.to_dict(orient="records")
        st.write(kadai_dic)
