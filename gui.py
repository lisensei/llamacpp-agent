import streamlit as st
from flask import request
import requests
if "messages" not in st.session_state:
    st.session_state.messages=[]

for m in st.session_state.messages:
    if m["role"]=="user":
        with st.chat_message("user"):
            st.write(m["content"])
    if m["role"]=="assistant":
        with st.chat_message("assistant"):
            st.write(m["content"])
prompt=st.chat_input("Type a question...")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
        url="http://localhost:6000/api/v1"
        st.session_state.messages.append({"role":"user","content":prompt})
    output=requests.get(url,json={"question":prompt},stream=True).json()["output"]
    def stream(data):
        for char in list(data):
            yield char
    with st.chat_message("assistant"):
        out=st.write_stream(stream(output))
        st.session_state.messages.append({"role":"assistant","content":output})