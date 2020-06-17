from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#if credit[['home']].isnull().any() raise Exception
# isinstance(credit['home'], np.double) Logistic else Linear

class TargetEncoder:
    """
    Target Encoder for categorical features.
    """

    def __init__(self, cols=None):
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
        self._encoding = {}
    

    def fit(self, X, y):
        """Encode given columns of X according to y.
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :param pandas.Series y: pandas Series of target values, shape (n_samples,).
        :return: None
        """
        X = pd.DataFrame(X, columns=self.cols)
        
        for col in self.cols:
            ohe = OneHotEncoder().fit(X[[col]])
            names = [n.strip('x0_') for n in ohe.get_feature_names()]
            ohe_mat = ohe.transform(X[[col]])
            model = LogisticRegression(fit_intercept=True, penalty='none', max_iter=200).fit(ohe_mat, y)
            coef = {names[i]: -float(model.coef_[:, i] + model.intercept_) for i in range(len(names))}
            self._encoding[col] = coef
            
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

    def fit_transform(self, X, y=None):
        """Encode given columns of X according to y, and transform X based on the learnt mapping.
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :param pandas.Series y: pandas Series of target values, shape (n_samples,).
            Required only for encoders that need it: TargetEncoder, WeightOfEvidenceEncoder
        :return: encoded DataFrame of shape (n_samples, n_features), initial categorical columns are dropped, and
            replaced with encoded columns. DataFrame passed in argument is unchanged.
        :rtype: pandas.DataFrame
    """
        self.fit(X, y)
        return self.transform(X)