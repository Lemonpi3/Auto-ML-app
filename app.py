import streamlit as st
import FileReader
import os

import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret import regression, classification

st.set_page_config(layout="wide",page_title='Auto ML app 🤖',page_icon='🤖')

with st.sidebar:
    st.image("assets/imgs/305129231.png")
    st.title("Auto ML app")
    choice = st.radio("Navigation",["Load Dataset", "Profiling", "Clean Dataset", "ML", "Download",])
    st.info("This application speeds up the data cleaning process as well as allowing you to build an Auto ML pipeline.")

if choice == "Load Dataset":
    st.title("Load the data to process")
    st.info("Suports .csv .json .xlsx")
    up_choice = st.radio("How do you want to load the dataset",["Upload", "Url",])
    
    if up_choice == "Upload":                            
        file = st.file_uploader("Upload your Dataset here")
        if file:
            termination = file.name.split('.')[-1]
            FileReader.read_file(file,termination)

    if up_choice == "Url":
        file_url = st.text_input("Paste the url to your Dataset here (only suports github raw)","https://github.com/Lemonpi3/datasets-coderhouse/blob/main/telecom_customer_churn.csv?raw=true")
        file_format = st.selectbox('Choose file type',['csv','json','xlsx'],0)
        if file_url:
            FileReader.read_from_url(file_url, file_format)

if choice == "Profiling":
    if os.path.exists("./assets/data/data.csv"):
        df = pd.read_csv("./assets/data/data.csv")
        profile_report = df.profile_report()
        st_profile_report(profile_report)        
    else:
        st.warning("No dataset loaded")

if choice == "Clean Dataset":
    pass

if choice == "ML":
    st.title('Model Training')
    if os.path.exists("./assets/data/data.csv"):
        df = pd.read_csv("./assets/data/data.csv")
        model_type = st.selectbox('Model Type',['Regression','Classification'])
        target = st.selectbox('Select target column', df.columns)
        train_size = st.slider('Train size',min_value=0.,max_value=1. ,value=0.8)
        train_model = st.button('Train model')
        if train_model:
            if model_type == 'Regression':
                regression.setup(df,target=target,silent=True)
                setup_df = regression.pull()
                st.info("Loaded settings")
                st.dataframe(setup_df)
                best_model = regression.compare_models()
                compare_df = regression.pull()
                st.info("Best Model")
                st.dataframe(compare_df)
                best_model
                regression.save_model(best_model,"best_model")

            if model_type == 'Classification':
                classification.setup(df,target=target,silent=True)
                setup_df = classification.pull()
                st.info("Loaded settings")
                st.dataframe(setup_df)
                best_model = classification.compare_models()
                compare_df = classification.pull()
                st.info("Best Model")
                st.dataframe(compare_df)
                best_model
                classification.save_model(best_model,"best_model")

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download Model", f, "best_model.pkl")
