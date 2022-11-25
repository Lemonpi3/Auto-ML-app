import streamlit as st
import numpy as np
import pandas as pd

from pycaret import clustering
from scripts.model import Model

class ClusteringModel(Model):
    def __init__(self, df:pd.DataFrame, **kwargs)-> None:
        super().__init__(self, **kwargs)
        self.df = df
        self.ground_truth = kwargs.get('ground_truth',None) #ground_truth
        self.num_clusters = kwargs.get('num_clusters',4)
        self.model = kwargs.get('model', 'kmeans')
        self.feature_choice = kwargs.get('feature_choice',self.df.columns[0])
        self.plot_choice = kwargs.get('plot_choice','cluster')
        self.train_model()

    def train_model(self):
        clustering.setup(
                            self.df, silent= True, use_gpu= self.use_gpu, preprocess= self.preprocess,
                            categorical_features= self.categorical_features, categorical_imputation= self.categorical_imputation,
                            ignore_low_variance= self.ignore_low_variance, combine_rare_levels= self.combine_rare_levels,
                            rare_level_threshold= self.rare_level_threshold,ordinal_features= self.ordinal_features,
                            high_cardinality_features= self.high_cardinality_features, high_cardinality_method= self.high_cardinality_method,
                            numeric_features= self.num_features, numeric_imputation= self.numeric_imputation,
                            normalize= self.normalize, normalize_method= self.normalize_method,  date_features= self.date_features, 
                            ignore_features=self.ignore_features, handle_unknown_categorical= self.handle_unknown_categorical,
                            unknown_categorical_method = self.unknown_categorical_method,
                            )
        clustering.set_config('seed', self.seed)
        model = clustering.create_model(self.model, ground_truth=self.ground_truth, num_clusters=self.num_clusters)
        st.header('Plot Model')
        clustering.save_model(model,"best_model")
        try:
            st.text('plotting this may take a while...')
            clustering.plot_model(model, plot = self.plot_choice,feature=self.feature_choice,display_format='streamlit')
        except:
            st.error("There was an error with the graph, probably the type graph chosen it's not suported (See: pycaret.clustering.plot_mode https://pycaret.readthedocs.io/en/stable/api/clustering.html)")
            raise