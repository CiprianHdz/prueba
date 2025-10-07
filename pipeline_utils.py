import pandas as pd
from sklearn.pipeline import Pipeline
class PandasPipeline(Pipeline):
    def transform(self, X):
        transformed_array = super().transform(X)
        return pd.DataFrame(transformed_array, columns=self.get_feature_names_out(), index=X.index)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)