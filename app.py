import streamlit as st
import FileReader
import DataCleaner
import os

import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret import regression, classification

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
    if os.path.exists("./assets/data/data.csv"):
        df = pd.read_csv("./assets/data/data.csv")
        model_type = st.selectbox('Model Type',['Regression','Classification'])
        target = st.selectbox('Select target column', df.columns)
        train_size = st.slider('Train size',min_value=0.,max_value=1. ,value=0.8)
        use_gpu = st.checkbox('Use GPU')
        preprocess = st.checkbox('Preprocess')
        if preprocess:
            #Categorical features
            st.header('Categorical Columns Setup')
            st.info('Categorical features that are not listed as ordinal will be one hot encoded')

            categorical_features = st.multiselect('Input Categorical features',[col for col in df.columns if col != target])
            if categorical_features:
                categorical_imputation = st.selectbox('How would missing categorical features be imputed',['constant','mode'])

                ordinal_features = st.multiselect('Input Ordinal columns',categorical_features)
                ordinal_features_setted = {}
                if ordinal_features:
                    for feature in ordinal_features:
                        ordinal_features_setted[feature] = st.multiselect(f'Select the order of values for {feature} starting from lower to higher',df[feature].unique())
                        if len(ordinal_features_setted[feature]) < len(df[feature].unique()):
                            st.warning('There are some values missing, if its null ignore this warning')
                        if len(ordinal_features_setted[feature]) > len(df[feature].unique()):
                            st.warning('There are some extra values')

                high_cardinality_features = st.multiselect('Input High cardinality features',categorical_features)
                if high_cardinality_features:
                    high_cardinality_method = st.selectbox("Select how will high cardinalty features get imputed",['frequency','clustering'])

            #Numerical Features
            st.header('Numerical Columns Setup')
            num_features = st.multiselect('Input Numerical features',[col for col in df.columns if col not in categorical_features])
            if num_features:
                numeric_imputation = st.selectbox('How would missing numerical features be inputed',['mean','median','zero'])
                normalize = st.checkbox('Normalize')
                if normalize:
                    st.info("* zscore: is calculated as z = (x - u) / s\n\n* minmax: scales and translates each feature individually such that it is in the range of 0 - 1.\n\n* maxabs: scales and translates each feature individually such that the maximal absolute value of each feature will be 1.0.\n    It does not shift/center the data, and thus does not destroy any sparsity.\n\n* robust: scales and translates each feature according to the Interquartile range.\n  When the dataset contains outliers, robust scaler often gives better results.")
                    normalize_method = st.selectbox('Normalize method',['zscore','minmax','maxabs','robust'])
                    
            #Date Features
            st.header('Date Features')
            date_features = st.multiselect('Input datetime format columns',[col for col in df.columns if (col not in categorical_features) and (col not in num_features)])
            #ignore cols
            st.header('Features to Ignore')
            ignore_features = st.multiselect('Input columns to ignore during model training',list(df.columns))
            if target in ignore_features:
                st.error(f'The target feature "{target}" is going to be ignored')
            for feature in categorical_features:
                if feature in ignore_features:
                    st.error(f'The categorical feature {feature} is going to be ignored')
            for feature in num_features:
                if feature in ignore_features:
                    st.error(f'The numerical feature {feature} is going to be ignored')
            null_rows_to_drop = st.multiselect('Drop rows that contain null values in cols:',list(df.columns))

        train_model = st.button('Train model')

        if train_model:
            if model_type == 'Regression':
                regression.setup(df,target=target,silent=True,use_gpu=use_gpu)
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
                classification.setup(df,target=target,silent=True,use_gpu=use_gpu)
                setup_df = classification.pull()
                st.info("Loaded settings")
                st.dataframe(setup_df)
                best_model = classification.compare_models()
                compare_df = classification.pull()
                st.info("Best Model")
                st.dataframe(compare_df)
                best_model
                classification.save_model(best_model,"best_model")
    else:
        st.warning("No dataset Loaded")

if choice == "Download":
    if os.path.exists("./best_model.pkl"):
        st.header("Download Model")
        with open("best_model.pkl", 'rb') as f:
            st.download_button("Download Model", f, "best_model.pkl")
    else:
        st.warning("There isn't a model aviable to download")
