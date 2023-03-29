import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pydot
import graphviz

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_text as text
import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers


dataframe = st.session_state["dataframe"]

# if 'dataframe' not in st.session_state:
#    st.session_state["dataframe"] = pd.DataFrame()


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '、|。', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]|【.*?】' % re.escape(
                                        string.punctuation),
                                    '')

# def segment_by_spm(input_data):
#    text = s2.tokenize(input_data)
#    return text


with st.expander("データ"):
    st.dataframe(dataframe)

with st.expander("param指定"):
    paramcol1, paramcol2 = st.columns(2)
    with paramcol1:
        cols = dataframe.columns.tolist()
        selected_col = st.selectbox(
            '分析対象の列を指定して下さい',
            cols
        )

    # dataframe["analyze_col"] = dataframe[selected_col]
    with paramcol2:
        dataframe["y_train"] = 0
        st.write("分析対象データ")

        analyze_df = dataframe[[selected_col, "y_train"]]
        st.experimental_data_editor(analyze_df, use_container_width=True)

# Functionalな書き方。
# A text input.


@st.cache_resource
def getmoel(dataframe):
    text_dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe[selected_col].astype(str).tolist(), dataframe['y_train'].tolist()))
    text_dataset2 = tf.data.Dataset.from_tensor_slices(
        dataframe[selected_col].astype(str).tolist())

    # Model constants.
    max_features = 3000
    embedding_dim = 128
    max_len = 100
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        split="character",
        # split=segment_by_spm,
        max_tokens=max_features,
        ngrams=(1, 1),
        # output_mode='tf-idf',
        # output_mode='int',
        output_mode='count',
        # output_sequence_length=max_len
        # output_mode = 'multi_hot',
        # pad_to_max_tokens=True,
        # sparse=True,

    )

    vectorize_layer.adapt(text_dataset2)

    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
    x = vectorize_layer(text_input)
    x = layers.Embedding(max_features + 1, embedding_dim)(x)
    x = layers.Dropout(0.5)(x)
    # Conv1D + global max pooling
    x = layers.Conv1D(256, 7, padding='valid', activation='relu', strides=3)(x)
    x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(1, activation='sigmoid', name='predictions')(x)

    model = tf.keras.Model(text_input, predictions)

    model.compile(
        # loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # with st.expander("グラフ構造"):
    #    st.write(tf.keras.utils.plot_model(
    #    model, to_file=dot_img_file, show_shapes=True))
    #    st.image(dot_img_file)
    #    st.write(model.summary())

    # @title 学習
    batch_size = 20
    epochs = 2
    # Fit the model using the train and test datasets.

    model.fit(
        text_dataset.batch(batch_size),
        validation_data=text_dataset.batch(batch_size),
        epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta=0, patience=20,
                                                    verbose=1)])
    return model


exec_button = st.button("学習実行！")
if exec_button:
    dot_img_file = "result.png"
    model = getmoel(dataframe)
