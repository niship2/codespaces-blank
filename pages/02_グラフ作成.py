import streamlit as st
map_df2 = st.session_state["agg_datframe"]

st.dataframe(map_df2)
