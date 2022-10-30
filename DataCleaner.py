import pandas as pd
import streamlit as st
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import re


def column_menu(df,column):
    selected_col_idx = list(df.columns).index(column)

def auto_clean():
    df = pd.read_csv('./assets/data/data.csv')
    st.header('Null settings')
    null_thresh_hold = st.slider('Min Null % Count to drop',min_value=0.,max_value=100. ,value=50.)
    st.info(f'It will drop any column that has >= {null_thresh_hold}% of nulls')
    cols_to_drop = st.text_input('Columns you want to drop (like metadata , undisired ones)')
    st.info('split each column name with ", " \n example: col 1, col 2')
    
def manual_clean():
    df = pd.read_csv('./assets/data/data.csv')
    st.warning('Work in progress')
    st.header('Select column to clean')
    selected_col = st.selectbox('Column',list(df.columns))
    profile_report = pd.DataFrame(df[selected_col]).profile_report()
    st_profile_report(profile_report,height=500)
    choice = st.selectbox('What do you want to do',['Drop the Column', 'Deal with nulls', 'Deal with duplicates', 'Deal with outliers'],0)

    if choice == 'Drop the Column':
        summit = st.button('Drop the Column')
        if summit:
            del df[selected_col]
            df.to_csv('./assets/data/data.csv',index=False)
            df = pd.read_csv('./assets/data/data.csv')

    if choice == 'Deal with nulls':
        st.header('Null management')
        choice = st.selectbox('What do you want to do',['Drop null rows', 'Fill values'],0)
        df, finished_step = manual_null_management(selected_col, choice)
        if finished_step:
            df.to_csv('./assets/data/data.csv',index=False)
            df = pd.read_csv('./assets/data/data.csv')

def manual_null_management(col,choice):
    df = pd.read_csv('./assets/data/data.csv')

    if choice == 'Drop null rows':
        drop_mode = st.selectbox('Drop mode',['All','Condition'],0)
        if drop_mode == 'All':
            summit = st.button('Drop all null rows for this col')
            if summit:
                df[col].dropna(inplace=True)
                return df , True
        if drop_mode == 'Condition':
            st.info('* If you select a value type as column input the column name\n* For imputing lists split each value with ", " \n example: item1, item2\n* For re match the string variable must be in value one and the regexp must be in value 2')
            n_condtions = st.number_input('Number of conditions',value=0)
            df = filter_df_by_conditions(df,n_condtions)
            summit = st.button('Drop all null rows for this col that meet the conditions')
            if summit:
                df.dropna(inplace=True)
                return df , True

    if choice == 'Fill values':
        fill_mode = st.selectbox('Fill mode',['most frequent','average','min','max',],0)
    
    return df, False

def filter_df_by_conditions(df,n_condtions):
    conditions = []
    for i in range(n_condtions):
        condition_i = []
        st.text(f'cond {i+1}')
        cols = st.columns(5, gap="small")
        with cols[0]:
            value_1 = st.text_input(f'cond {i+1} Value 1')
        with cols[1]:
            value_1_type = st.selectbox(f'cond {i+1} Value 1 type',['Column','String','List','Float','Int'],1)
            if value_1_type == 'Column':
                condition_i.append(df[value_1])
            if value_1_type == 'String':
                condition_i.append(value_1)
            if value_1_type == 'List':
                condition_i.append(value_1.split(', '))
            if value_1_type == 'Float':
                condition_i.append(float(value_1))
            if value_1_type == 'int':
                condition_i.append(int(value_1))   

        with cols[2]:
            operand = st.selectbox(f'cond {i+1} Operand',['<','>','==','<=','>=','not','isna','in','re match'],0)
            condition_i.append(operand)

        with cols[3]:
            value_2 = st.text_input(f'cond {i+1} Value 2',)

        with cols[4]:
            value_2_type = st.selectbox(f'cond {i+1} Value 2 type',['Column','String','List','Float','Int'],1)

            if value_2_type == 'Column':
                condition_i.append(df[value_2])
            if value_2_type == 'String':
                condition_i.append(value_2)
            if value_2_type == 'List':
                condition_i.append(value_2.split(', '))
            if value_2_type == 'Float':
                condition_i.append(float(value_2))
            if value_2_type == 'int':
                condition_i.append(int(value_2))    
        
        conditions.append(condition_i)

        # df_filter = []
        # for condition in conditions:
        #     if condition[1] == '<':
        #         df_filter.append(condition)
        
    