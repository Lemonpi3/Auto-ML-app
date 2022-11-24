import streamlit as st
import numpy as np
import pandas as pd

from pycaret import classification
from scripts.model import Model
from imblearn.over_sampling import SMOTENC, SMOTE, RandomOverSampler, ADASYN, KMeansSMOTE, SVMSMOTE

class ClassifierModel(Model):
    def __init__(self, df:pd.DataFrame,target:str, **kwargs)-> None:
        super().__init__(self, **kwargs)
        self.df = df
        self.target=target
        self.balance_ds = kwargs.get('balance_ds',False)
        self.comparison_metric = kwargs.get('comparison_metric','Accuracy')
        self.balance_select = kwargs.get('balance_select','SMOTENC')
        
        if self.balance_select == 'SMOTENC':
            self.balance_method = SMOTENC(self.categorical_features,random_state=self.seed)
        if self.balance_select == 'SMOTE':
            self.balance_method = SMOTE(random_state=self.seed)
        if self.balance_select == 'RandomOverSampler':
            self.balance_method = RandomOverSampler(random_state=self.seed)
        if self.balance_select == 'ADASYN':
            self.balance_method = ADASYN(random_state=self.seed)
        if self.balance_select == 'KMeansSMOTE':
            self.balance_method = KMeansSMOTE(random_state=self.seed)
        if self.balance_select == 'SVMSMOTE':
            self.balance_method = SVMSMOTE(random_state=self.seed)
        
        self.train_model()

    def train_model(self):
        # balance_ds hard setted to false due to a pycaret bug.
        classification.setup(
                        self.df, target= self.target, silent= True, use_gpu= self.use_gpu, preprocess= self.preprocess,
                        categorical_features= self.categorical_features, categorical_imputation= self.categorical_imputation, 
                        ignore_low_variance= self.ignore_low_variance, combine_rare_levels= self.combine_rare_levels, 
                        rare_level_threshold= self.rare_level_threshold,ordinal_features= self.ordinal_features,
                        high_cardinality_features= self.high_cardinality_features, high_cardinality_method= self.high_cardinality_method, 
                        numeric_features= self.num_features, numeric_imputation= self.numeric_imputation, normalize= self.normalize,
                        normalize_method= self.normalize_method, remove_outliers=self.remove_outliers,
                        outliers_threshold= self.outliers_threshold, date_features= self.date_features, ignore_features=self.ignore_features,
                        handle_unknown_categorical= self.handle_unknown_categorical, unknown_categorical_method = self.unknown_categorical_method,
                        train_size=self.train_size,balance_ds=False,balance_method=self.balance_method
                        )
        classification.set_config('seed', self.seed)
        setup_df = classification.pull()
        st.info("Loaded settings")
        st.dataframe(setup_df)
        best_model = classification.compare_models(n_select=5, sort=self.comparison_metric)
        compare_df = classification.pull()
        st.info("Best Model")
        st.dataframe(compare_df)
        best_model
        classification.save_model(best_model,"best_model")