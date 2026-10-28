import time

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_lottie import st_lottie

from visualization import data_visualization
from prediction_model import prediction_model_pipeline
from src.util import read_file_from_streamlit
from util import welcome_message, stream_data, load_lottie, introduction_message

_notes="""
initialized:
    None:页面第一次加载
    True:要流式输出文字
    False:不要流式输出文字
welcome_message:
lottie:
    None:页面第一次加载
    True:已加载动画
lottie_url1:
lottie_url2:
DF_uploaded:
    None:还没上传文件
    有值:上传了文件
is_file_empty:
    None:还没上传文件
    True:上传了空文件
button_clicked:
    None:页面第一次加载
    False:可以点击
    True:禁止点击
"""

def disable_button():
    st.session_state.button_clicked=True

st.set_page_config(page_title="Streamline Analyst",page_icon=":rocket:",layout="wide")

with st.container():
    st.subheader("Hello there 👋")
    st.title("Welcome to Streamline Analyst")
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    #首次加载时才播放一次流式输出置
    if st.session_state.initialized:
        st.session_state.welcome_message=welcome_message()
        st.write(stream_data(st.session_state.welcome_message))
        time.sleep(0.5)
        st.session_state.initialized=False
    else:
        st.write(st.session_state.welcome_message)

with st.container():
    st.divider()
    if 'lottie' not in st.session_state:
        st.session_state.lottie_url1,st.session_state.lottie_url2=load_lottie()
        st.session_state.lottie=True

    left_column_r1,right_column_r1=st.columns([6,4])
    with left_column_r1:
        st.header("What can Streamline Analyst do？")
        st.write(introduction_message()[0])
    with right_column_r1:
        if st.session_state.lottie:
            st_lottie(st.session_state.lottie_url1,height=280,key="animation1")

    left_column_r2,_,right_column_r2=st.columns([6,1,5])
    with left_column_r2:
        if st.session_state.lottie:
            st_lottie(st.session_state.lottie_url2,height=200,key="animation2")
    with right_column_r2:
        st.header("Simple to Use")
        st.write(introduction_message()[1])

with st.container():
    st.divider()
    st.header("Let's Get Started")
    left_column,right_column=st.columns([6,4])
    with left_column:
        API_KEY=st.text_input("Your API Key won't be stored or shared",placeholder="Enter your API Key here...")
        st.write("👆Your OpenAI API key")
        uploaded_file:UploadedFile=st.file_uploader("Choose a data file. Your data won't be stored as well!",accept_multiple_files=False,type=['csv','json','xls','xlsx'])
        #上传了文件
        if uploaded_file:
            #文件有大小
            if uploaded_file.getvalue():
                st.session_state.DF_uploaded=read_file_from_streamlit(uploaded_file)
                st.session_state.is_file_empty=False
            #文件0kb
            else:
                st.session_state.is_file_empty=True

        with right_column:
            SELECTED_MODEL=st.selectbox("Which OpenAI model do you want to use?",('ollama:qwen2.5:7b','deepseek:deepseek-v4-flash'))
            MODE=st.selectbox('Select proper data analysis mode',('Predictive Classification','Clustering Model','Regression Model','Data Visualization'))

            st.write(f"Model selected: :green[{SELECTED_MODEL}]")
            st.write(f"Data analysis mode: :green[{MODE}]")

        is_proceed_enable = uploaded_file is not None and (API_KEY != "" or MODE == 'Data Visualization')
        if 'button_clicked' not in st.session_state:
            st.session_state.button_clicked=False
        #点击一次后就会禁止点击了，防止重复点击
        st.button('start Analysis',disabled=(not is_proceed_enable) or st.session_state.button_clicked,type='primary',on_click=disable_button)
        if 'is_file_empty' in st.session_state and st.session_state.is_file_empty:
            st.caption("Your data file is empty")

        if st.session_state.button_clicked:
            with st.container():
                if "DF_uploaded" not in st.session_state:
                    st.error("File is empty!")
                else:
                    if MODE=='Predictive Classification':
                        prediction_model_pipeline(st.session_state.DF_uploaded,API_KEY,SELECTED_MODEL)
                    elif MODE=='Clustering Model':
                        cluster_model_pipeline(st.session_state.DF_uploaded,API_KEY,SELECTED_MODEL)
                    elif MODE=='Regression Model':
                        regression_model_pipeline(st.session_state.DF_uploaded,API_KEY,SELECTED_MODEL)
                    elif MODE=='Data Visualization':
                        data_visualization(st.session_state.DF_uploaded)