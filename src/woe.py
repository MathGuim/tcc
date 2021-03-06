import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

#if credit[['home']].isnull().any() raise Exception
# isinstance(credit['home'], np.double) Logistic else Linear

class WeigthOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder for categorical features.
    """

    def __init__(self, cols=None, laplace=1e-06):
        """Instantiation
        :param [str] cols: list of columns to encode, or None (then all dataset columns will be encoded at fitting time)
        :param str handle_unseen:
            'impute' - default value, impute a -1 category
            'error'  - raise an error if a category unseen at fitting time is found
            'ignore' - skip unseen categories
        :param int min_samples: minimum samples to take category average into account, must be >= 1
        :param int smoothing: coefficient used to balance categorical average (posterior) vs prior,
            the higher this number, the higher the prior is taken into account in the average
        :return: None
        """
        self.cols = cols
        self.laplace = laplace
        self._encoding = {}

    def fit(self, X, y):
        """Encode given columns of X according to y.
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :param pandas.Series y: pandas Series of target values, shape (n_samples,).
        :return: None
        """
        X = pd.DataFrame(X, columns=self.cols)
        classes = y.unique()
        assert len(classes) == 2
        
        for col in self.cols:
            cross = pd.crosstab(X[col], y)
            woe = {}
            for cat in X[col].unique():
                num = (cross[classes[0]].loc[cat] + self.laplace) / (cross[classes[0]].sum() + 2 * self.laplace)
                den = (cross[classes[1]].loc[cat] + self.laplace) / (cross[classes[1]].sum() + 2 * self.laplace)
                woe[cat] = np.log(num / den)
            self._encoding[col] = woe

        return self
    
    
    def transform(self, X):
        """Transform categorical data based on mapping learnt at fitting time.
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :return: encoded DataFrame of shape (n_samples, n_features), initial categorical columns are dropped, and
            replaced with encoded columns. DataFrame passed in argument is unchanged.
        :rtype: pandas.DataFrame
        """
        if not self._encoding:
            raise ValueError('`fit` method must be called before `transform`.')

        X_encoded = pd.DataFrame(X, columns=self.cols)
        assert all(c in X_encoded.columns for c in self.cols)

        for col, mapping in self._encoding.items():
            X_encoded.loc[:, col] = X_encoded[col].map(mapping)
        
        return X_encoded
