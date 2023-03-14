import streamlit as st
import pandas as pd
from io import StringIO
from streamlit_app import check_password


if check_password():
    with st.expander("ファイルアップロード"):
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            # To read file as bytes:
            # bytes_data = uploaded_file.getvalue()
            # st.write(bytes_data)

            # To convert to a string based IO:
            # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # st.write(stringio)

            # To read file as string:
            # string_data = stringio.read()
            # st.write(string_data)

            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(uploaded_file, encoding="cp932")
            edited_df = st.experimental_data_editor(dataframe)
            # st.write(dataframe)

            st.write("データ加工")
            init_df = pd.DataFrame([[""], [""]])
            map_df = st.experimental_data_editor(init_df)

            kadai_dic = map_df.index.to_dict(orient="records")
            st.write(kadai_dic)
