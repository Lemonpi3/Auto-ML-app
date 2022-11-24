import pandas as pd

class Model():
    '''Contains the parameters shared by all models'''
    def __init__(self,*args, **kwargs) -> None:
        #general settings
        self.use_gpu = kwargs.get('use_gpu',False)
        self.preprocess = kwargs.get('preprocess',False)
        self.seed = kwargs.get('seed',0)
        self.train_size = kwargs.get('train_size', 0.7)
        #categorical settings
        self.categorical_features = kwargs.get('categorical_features',[])
        self.ignore_low_variance = kwargs.get('ignore_low_variance',False)
        self.combine_rare_levels = kwargs.get('combine_rare_levels',False)
        self.rare_level_threshold = kwargs.get('rare_level_threshold',0.1)
        self.ordinal_features = kwargs.get('ordinal_features',{})
        self.high_cardinality_features = kwargs.get('high_cardinality_features',[])
        self.high_cardinality_method = kwargs.get('high_cardinality_method','frequency')
        self.categorical_imputation = kwargs.get('categorical_imputation','constant')

        #numeric settings
        self.num_features = kwargs.get('num_features',[])
        self.numeric_imputation = kwargs.get('numeric_imputation','mean')
        self.normalize = kwargs.get('normalize',False)
        self.normalize_method = kwargs.get('normalize_method','zscore')
        self.remove_outliers = kwargs.get('remove_outliers',False)
        self.outliers_threshold = kwargs.get('outliers_threshold',0.05)

        #others, unknown, nulls settings
        self.date_features = kwargs.get('date_features',[])
        self.ignore_features = kwargs.get('ignore_features',[])
        self.null_rows_to_drop = kwargs.get('null_rows_to_drop',[])
        self.handle_unknown_categorical = kwargs.get('handle_unknown_categorical',True)
        self.unknown_categorical_method =kwargs.get('unknown_categorical_method','least_frequent')