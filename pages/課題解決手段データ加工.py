import streamlit as st
import pandas as pd
from io import StringIO
from streamlit_app import check_password


if check_password():
    with st.expander("ファイルアップロード&指定"):
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
    with st.expander("param指定"):
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file, encoding="cp932")
        edited_df = st.experimental_data_editor(dataframe)
        # st.write(dataframe)

    with st.expander("データ加工")
        num_rows = st.slider('行数指定', 2, 25, 2)
        num_cols = st.slider('列数指定', 2, 25, 2)

        init_df = pd.DataFrame(np.arange(num_rows * num_cols).reshape(num_rows, num_cols))
        map_df = st.experimental_data_editor(init_df, num_rows="dynamic")

        kadai_dic = map_df.to_dict(orient="records")
        st.write(kadai_dic)
