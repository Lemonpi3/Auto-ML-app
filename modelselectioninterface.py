import streamlit as st 
import pandas as pd
import numpy as np
import os

from scripts.classifer import ClassifierModel
from scripts.regressor import RegressorModel
from scripts.clusterer import ClusteringModel

class ModelSelection():
    def __init__(self) -> None:
        self.model_selection_interface()

    def model_selection_interface(self):
        if os.path.exists("./assets/data/data.csv"):
            df = pd.read_csv("./assets/data/data.csv")
            model_type = st.selectbox('Model Type',['Regression','Classification','Clustering'])
            target = st.selectbox('Select target column',[None] + list(df.columns), 0)
            train_size = st.slider('Train size',min_value=0.,max_value=1. ,value=0.8)
            use_gpu = st.checkbox('Use GPU')
            preprocess = st.checkbox('Preprocess')

            if model_type == 'Clustering':
                st.header('Clustering Settings\n---------------------')
                st.info('Target its treatead as Ground Truth')
                num_clusters = st.number_input('Number of clusters',value=4, step=1)
                options = ['K-means Clustering', 'Affinity Propagation', 'Mean shift Clustering', 'Spectral Clustering',
                            'Agglomerative Clustering', 'Density-Based Spatial Clustering', 'OPTICS Clustering', 'Birch Clustering',
                            'K-Modes Clustering']
                model_names = ['kmeans','ap','meanshift','sc','hclust','dbscan','optics','birch','kmodes']
                model = st.selectbox('Model to use',options,0)
                st.markdown('### Model Plotting options')
                plot_choice = st.selectbox('Plot type',['elbow','cluster','tsne','silhouette','distance','distribution'],1)
                feature_choice = st.selectbox('Feature to plot',df.columns,0)

                for option in options:
                    if model == option:
                        model = model_names[options.index(option)]
            elif model_type == 'Classification':
                comparison_metric = st.selectbox('Metric to compare models',['Accuracy','AUC','F1','Recall','Prec.'])

            elif model_type == 'Regression':
                comparison_metric = st.selectbox('Metric to compare models',['R2','MAE','MSE','RMSE','MAPE'])
                
            
            
            

            #Preprocessing
            if preprocess:
                st.header('Preprocessing\n---------------------')
                seed = st.number_input('Seed',0)
                #Categorical features
                st.header('Categorical Columns Setup')
                st.info('* Categorical features that are not listed as ordinal will be one-hot encoded. \n* Drops duplicates automatically.')

                categorical_features = st.multiselect('Input Categorical features',[col for col in df.columns if col != target])
                categorical_imputation = st.selectbox('How would missing categorical features be imputed',['constant','mode'],0)
                ignore_low_variance = st.checkbox('Ignore low variance')
                st.info('When set to True, all categorical features with insignificant variances are removed from the data. The variance is calculated using the ratio of unique values to the number of samples, and the ratio of the most common value to the frequency of the second most common value.')
                
                combine_rare_levels = st.checkbox('combine features with that are below a certain threshold')    
                rare_level_threshold = st.slider('rare features threshold',0.,1.,0.1)
                #ordinal
                ordinal_features = st.multiselect('Input Ordinal columns',categorical_features)
                ordinal_features_setted = {}
                if ordinal_features:
                    for feature in ordinal_features:
                        ordinal_features_setted[feature] = st.multiselect(f'Select the order of values for {feature} starting from lower to higher',df[feature].unique())
                        if len(ordinal_features_setted[feature]) < len(df[feature].unique()):
                            missing = []
                            for i in ordinal_features_setted[feature]:
                                if (i != np.nan) and (i not in df[feature].unique()):
                                    missing.append(i)
                            if len(missing)>0:
                                st.warning('There are some values missing')
                        if len(ordinal_features_setted[feature]) > len(df[feature].unique()):
                            st.warning('There are some extra values')

                high_cardinality_features = st.multiselect('Input High cardinality features',categorical_features)
                high_cardinality_method = st.selectbox("Select how will high cardinalty features get imputed",['frequency','clustering'],0)
                
                #Numerical Features
                st.header('Numerical Columns Setup')
                num_features = st.multiselect('Input Numerical features',[col for col in df.columns if col not in categorical_features])
                numeric_imputation = st.selectbox('How would missing numerical features be inputed',['mean','median','zero'],0)
                normalize = st.checkbox('Normalize')
                st.info("* zscore: is calculated as z = (x - u) / s\n\n* minmax: scales and translates each feature individually such that it is in the range of 0 - 1.\n\n* maxabs: scales and translates each feature individually such that the maximal absolute value of each feature will be 1.0.\n    It does not shift/center the data, and thus does not destroy any sparsity.\n\n* robust: scales and translates each feature according to the Interquartile range.\n  When the dataset contains outliers, robust scaler often gives better results.")
                normalize_method = st.selectbox('Normalize method',['zscore','minmax','maxabs','robust'],0)
                if model_type != 'Clustering':
                    remove_outliers = st.checkbox('remove outliers')
                    st.info('Removes outliers using the Singular Value Decomposition.')  
                    outliers_threshold = st.slider('Outlier threshold',0.,1.,0.05)
                    
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

                #Resto de las columnas
                st.header('Handle unkown columns')
                st.info('For the columns that not chosen in any of the sections above')
                handle_unknown_categorical = st.checkbox('Handle unknown categorical', True)
                unknown_categorical_method = st.selectbox('Method',['least_frequent','most_frequent'],0)

                #Model specific preprocessing
                if model_type == 'Classification':
                    st.header('Classification specific settings')
                    balance_ds = st.checkbox('Balance Dataset')
                    st.warning("Balancing feature it's disabled due to a library bug https://stackoverflow.com/questions/73362136/unboundlocalerror-local-variable-fix-imbalance-model-name-referenced-before-a")
                    balance_select = st.selectbox('Balance method',['SMOTENC', 'SMOTE', 'RandomOverSampler', 'ADASYN', 'KMeansSMOTE', 'SVMSMOTE'],0)
                    if balance_select == 'SMOTENC':
                        st.info("Synthetic Minority Over-sampling Technique for Nominal and Continuous.\nUnlike SMOTE, SMOTE-NC is for a dataset containing numerical and categorical features. However, it is not designed to work with only categorical features.")
                    if balance_select == 'SMOTE':
                        st.info("Only accepts numerical variables.")
                    if balance_select == 'RandomOverSampler':
                        st.info("Object to over-sample the minority class(es) by picking samples at random with replacement. The bootstrap can be generated in a smoothed manner.")
                    if balance_select == 'ADASYN':
                        st.info("This method is similar to SMOTE but it generates different number of samples depending on an estimate of the local distribution of the class to be oversampled.")
                    if balance_select == 'KMeansSMOTE':
                        st.info("Apply a KMeans clustering before to over-sample using SMOTE.")
                    if balance_select == 'SVMSMOTE':
                        st.info("Variant of SMOTE algorithm which use an SVM algorithm to detect sample to use for generating new synthetic samples")
        
                
            #Train Models
            if st.button('Train model'):
                if preprocess:
                    df = df.drop_duplicates()
                    df[null_rows_to_drop] = df[null_rows_to_drop].dropna()

                if model_type == 'Classification' and preprocess:
                    ClassifierModel(df, target= target, use_gpu= use_gpu, preprocess= preprocess,
                            categorical_features= categorical_features, categorical_imputation= categorical_imputation, ignore_low_variance= ignore_low_variance,
                            combine_rare_levels= combine_rare_levels, rare_level_threshold= rare_level_threshold,ordinal_features= ordinal_features_setted,
                            high_cardinality_features= high_cardinality_features, high_cardinality_method= high_cardinality_method, numeric_features= num_features,
                            numeric_imputation= numeric_imputation, normalize= normalize, normalize_method= normalize_method, remove_outliers=remove_outliers,
                            outliers_threshold= outliers_threshold, date_features= date_features, ignore_features=ignore_features,
                            handle_unknown_categorical= handle_unknown_categorical, unknown_categorical_method = unknown_categorical_method,seed=seed,
                            balance_select=balance_select, balance_ds= balance_ds,train_size=train_size,comparison_metric=comparison_metric)
                
                elif model_type == 'Regression' and preprocess:
                    RegressorModel(df, target= target, silent= True, use_gpu= use_gpu, preprocess= preprocess,
                            categorical_features= categorical_features, categorical_imputation= categorical_imputation,
                            ignore_low_variance= ignore_low_variance, combine_rare_levels= combine_rare_levels,
                            rare_level_threshold= rare_level_threshold,ordinal_features= ordinal_features_setted,
                            high_cardinality_features= high_cardinality_features, high_cardinality_method= high_cardinality_method,
                            numeric_features= num_features, numeric_imputation= numeric_imputation,
                            normalize= normalize, normalize_method= normalize_method, remove_outliers=remove_outliers,
                            outliers_threshold= outliers_threshold, date_features= date_features, 
                            ignore_features=ignore_features, handle_unknown_categorical= handle_unknown_categorical,
                            unknown_categorical_method = unknown_categorical_method,seed=seed,
                            train_size= train_size, comparison_metric= comparison_metric,
                    )
                elif model_type == 'Clustering' and preprocess:
                    ClusteringModel(
                            df, silent= True, use_gpu= use_gpu, preprocess= preprocess,
                            categorical_features= categorical_features, categorical_imputation= categorical_imputation,
                            ignore_low_variance= ignore_low_variance, combine_rare_levels= combine_rare_levels,
                            rare_level_threshold= rare_level_threshold,ordinal_features= ordinal_features_setted,
                            high_cardinality_features= high_cardinality_features, high_cardinality_method= high_cardinality_method,
                            numeric_features= num_features, numeric_imputation= numeric_imputation,
                            normalize= normalize, normalize_method= normalize_method, date_features= date_features, 
                            ignore_features=ignore_features, handle_unknown_categorical= handle_unknown_categorical,
                            unknown_categorical_method = unknown_categorical_method,seed=seed,
                            ground_truth=target,plot_choice=plot_choice,feature_choice=feature_choice,
                            num_clusters=num_clusters,model=model
                    )
                elif model_type == 'Classification' and not preprocess:
                    ClassifierModel(df, target= target, use_gpu= use_gpu,train_size= train_size,comparison_metric=comparison_metric, preprocess= False)
                elif model_type == 'Regression' and not preprocess:
                    RegressorModel(df, target= target, use_gpu= use_gpu,train_size= train_size,comparison_metric=comparison_metric, preprocess= False)
                elif model_type == 'Clustering' and not preprocess:
                    ClusteringModel(df,use_gpu= use_gpu, ground_truth=target,num_clusters=num_clusters,model=model, preprocess= False)
                
        else:
            st.warning("No dataset Loaded")