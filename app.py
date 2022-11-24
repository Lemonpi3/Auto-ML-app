import streamlit as st
import FileReader
import DataCleaner
import os
from modelselectioninterface import ModelSelection
import numpy as np
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret import regression, classification
from imblearn.over_sampling import SMOTENC, SMOTE, RandomOverSampler, ADASYN, KMeansSMOTE, SVMSMOTE

st.set_page_config(layout="wide",page_title='Auto ML app 🤖',page_icon='🤖')

with st.sidebar:
    st.image("assets/imgs/305129231.png")
    st.title("Auto ML app")
    choice = st.radio("Navigation",["Load Dataset", "Profiling", "ML", "Download",])
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
        file_url = st.text_input("Paste the url to your Dataset here","https://github.com/Lemonpi3/datasets-coderhouse/blob/main/telecom_customer_churn.csv?raw=true")
        file_format = st.selectbox('Choose file type',['csv','json','xlsx'],0)
        if file_url:
            FileReader.read_from_url(file_url, file_format)

if choice == "Profiling":
    if os.path.exists("./assets/data/data.csv"):
        df = pd.read_csv("./assets/data/data.csv")
        st.info('Profiles the dataset with pandas_profiling')
        if st.button('Profile the dataset'):
            profile_report = df.profile_report()
            st_profile_report(profile_report)        
    else:
        st.warning("No dataset loaded")

if choice == "Clean Dataset":
    if os.path.exists("./assets/data/data.csv"):
        st.title('Data cleaning')
        clean_mode = st.selectbox('Select clean mode',['auto','manual'],0)
        if clean_mode == 'auto':
            DataCleaner.auto_clean()
        if clean_mode == 'manual':
            DataCleaner.manual_clean()
    else:
        st.warning("No dataset loaded")

if choice == "ML":
    st.title('Model Training')
    ModelSelection()

if choice == "Download":
    if os.path.exists("./best_model.pkl"):
        st.header("Download Model")
        with open("best_model.pkl", 'rb') as f:
            st.download_button("Download Model", f, "best_model.pkl")
    else:
        st.warning("There isn't a model aviable to download")