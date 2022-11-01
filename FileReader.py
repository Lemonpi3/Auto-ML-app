import pandas as pd
import streamlit as st
import json
import requests
import io

def read_file(file,termination):
    if termination == 'csv':
        process_csv(file=file)
    if termination == 'xlsx':
        st.warning('TBD')
        process_xlsx(file=file)
    if termination == 'json':
        process_json(file=file)

def read_from_url(url,format_type):
    if format_type == 'csv':
        st.info('Make sure that the url leads directly to the csv')
        process_csv(url=url)
    if format_type == 'xlsx':
        st.warning('TBD')
        process_xlsx(url=url)
    if format_type == 'json':
        st.info('Make sure that the url leads directly to the json contents Example URL: http://api.steampowered.com/ISteamWebAPIUtil/GetSupportedAPIList/v0001/')
        process_json(url=url)

def process_csv(file = None, url = None):
    if file or url:
        separator = st.text_input('please input separator',',')
        heading = st.number_input('header_row',value=0)
        
        load = st.button('Read Dataframe')
        if load:
            df = None
            if file:
                df = pd.read_csv(file,sep=separator,header=heading)

            if url:
                file = requests.get(url).text
                df = pd.read_csv(io.StringIO(file),sep=separator,header=heading)

            if len(df):
                st.dataframe(df)
                df.to_csv('./assets/data/data.csv',index=False)

def process_xlsx(file = None, url = None):
    pass

def process_json(file = None, url = None):
    key = st.text_input('if it is a nested json place the json key you want to stract, leave blank if you want it as it comes','')

    load = st.button('Read Dataframe')
    if load:
        if file:
            if key:
                df = pd.read_json(file)
                df = pd.DataFrame(df[key])
            else:
                df = pd.read_json(file)
        if url:
            file = requests.get(url).text
            if key:
                df = pd.DataFrame(json.loads(file)[key])
            else:
                df = pd.DataFrame([json.loads(file)])
        if len(df):
                st.dataframe(df)
                df.to_csv('./assets/data/data.csv',index=False)
