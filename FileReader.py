import pandas as pd
import streamlit as st

def read_file(file,termination):
    if termination == 'csv':
        process_csv(file=file)
    if termination == 'xlsx':
        process_xlsx(file=file)
    if termination == 'json':
        process_json(file=file)
    if termination == 'pdf':
        process_pdf(file=file)
    if termination == 'html':
        process_html(file=file)
    if termination == 'xml':
        process_xml(file=file)

def read_from_url(url):
    termination = url.split('.')[-1].strip('?raw=true')
    if termination == 'csv':
        process_csv(url=url)
    if termination == 'xlsx':
        process_xlsx(url=url)
    if termination == 'json':
        process_json(url=url)
    if termination == 'pdf':
        process_pdf(url=url)
    if termination == 'html':
        process_html(url=url)
    if termination == 'xml':
        process_xml(url=url)

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
                df = pd.read_csv(url,sep=separator,header=heading)

            if len(df):
                st.dataframe(df)
                df.to_csv('./assets/data/data.csv',index=False)

def process_xlsx(file = None, url = None):
    pass

def process_json(file = None, url = None):
    pass

def process_pdf(file = None, url = None):
    pass

def process_html(file = None, url = None):
    pass

def process_xml(file = None, url = None):
    pass