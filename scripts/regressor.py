import streamlit as st
import numpy as np
import pandas as pd

from pycaret import regression
from scripts.model import Model

class RegressorModel(Model):
    def __init__(self, df:pd.DataFrame,target:str, **kwargs)-> None:
        super().__init__(self, **kwargs)
        self.df = df
        self.target=target
        self.comparison_metric = kwargs.get('comparison_metric','R2')
        self.train_model()
    
    def train_model(self):
        regression.setup(
                            self.df, target= self.target, silent= True, use_gpu= self.use_gpu, preprocess= self.preprocess,
                            categorical_features= self.categorical_features, categorical_imputation= self.categorical_imputation,
                            ignore_low_variance= self.ignore_low_variance, combine_rare_levels= self.combine_rare_levels,
                            rare_level_threshold= self.rare_level_threshold,ordinal_features= self.ordinal_features,
                            high_cardinality_features= self.high_cardinality_features, high_cardinality_method= self.high_cardinality_method,
                            numeric_features= self.num_features, numeric_imputation= self.numeric_imputation,
                            normalize= self.normalize, normalize_method= self.normalize_method, remove_outliers=self.remove_outliers,
                            outliers_threshold= self.outliers_threshold, date_features= self.date_features, 
                            ignore_features=self.ignore_features, handle_unknown_categorical= self.handle_unknown_categorical,
                            unknown_categorical_method = self.unknown_categorical_method,
                            train_size= self.train_size, 
                            )
        setup_df = regression.pull()
        st.info("Loaded settings")
        st.dataframe(setup_df)
        best_model = regression.compare_models(n_select=5, sort=self.comparison_metric)
        compare_df = regression.pull()
        st.info("Best Model")
        st.dataframe(compare_df)
        best_model
        regression.save_model(best_model,"best_model")