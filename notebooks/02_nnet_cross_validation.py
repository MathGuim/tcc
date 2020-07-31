import pandas as pd
import numpy as np
import json 
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)



from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler

from src.nnet import FeedFowardNNet, label_encoder

import torch
import torch.nn as nn
from torch import optim
torch.manual_seed(42)


def emb_sz_rule(n_cat):
    return min(600, round(1.6 * n_cat ** 0.56))


def emb_sz_rule2(n_cat):
    return min(50, (n_cat + 1) // 2)


rng = np.random.RandomState(42)

credit = pd.read_csv("./data/raw/credit_data.csv").rename(columns=str.lower)
cat_names = ["home", "marital", "records", "job"]
credit[cat_names] = credit[cat_names].astype("str")
credit = label_encoder(credit, cols=["status", "home", "marital", "records", "job"])

X, y = credit.drop(columns="status"), credit["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

cat_dims = [int(credit[col].nunique()) for col in cat_names]
emb_szs = [(x, emb_sz_rule(x)) for x in cat_dims]

net = NeuralNetClassifier(
    FeedFowardNNet,
    max_epochs=7,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    batch_size=256,
    module__emb_szs=emb_szs,
    module__cont=[0, 2, 3, 7, 8, 9, 10, 11, 12],
    module__categ=[1, 4, 5, 6],
    module__out_sz=2,
    iterator_train__shuffle=True,
    device="cuda",
)

imputer = Pipeline([("impute", KNNImputer(add_indicator=True))])

scaler = Pipeline([("scale", StandardScaler())])

col_transform = ColumnTransformer(
    [
        ("number", imputer, selector(dtype_include=np.number)),
        ("float", scaler, selector(dtype_include=float)),
    ]
)

clf = Pipeline([("preproc", col_transform), ("net", net)])

nnet_params = {
    "net__module__layers": [[512, 256, 128], [1024, 512, 256, 128]],
    "net__module__nonlin": [nn.ReLU, nn.LeakyReLU],
    "net__module__emb_drop": [0, 0.05, 0.1],
    "net__module__ps": [0, 0.01, 0.05],
    "net__lr": [.05, 1e-2],
    "net__module__emb_szs": [
        [(7, 5), (6, 4), (2, 2), (5, 4)],
        [(7, 6), (6, 5), (2, 3), (5, 5)],
    ],
}

cv = RandomizedSearchCV(
    estimator=clf,
    scoring="accuracy",
    cv=StratifiedKFold(n_splits=3),
    random_state=rng,
    param_distributions=nnet_params,
    n_iter=20,
)
cv.fit(X_train, y_train)
pd.DataFrame(cv.cv_results_).to_csv("models/nnet_cross_validation.csv")
print(" Best params: %s" % cv.best_params_)
print("Best training accuracy: %.3f" % cv.best_score_)

clf.set_params(**cv.best_params_)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

joblib.dump(clf, 'nnet.joblib')
