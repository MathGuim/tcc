import pandas as pd
import numpy as np
import json
import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV,
    GridSearchCV,
    train_test_split,
    ParameterSampler,
)

from scipy.stats import uniform, randint
from src.target import TargetEncoder
from src.woe import WeigthOfEvidenceEncoder

rng = np.random.RandomState(42)

credit = pd.read_csv("./data/raw/credit_data.csv").rename(columns=str.lower)

X, y = credit.drop(columns="status"), credit["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

ohe_encoding = Pipeline(
    [("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder())]
)

effect_encoding = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("target", TargetEncoder(cols=["home", "marital", "records", "job"])),
    ]
)

woe_encoding = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("woe", WeigthOfEvidenceEncoder(cols=["home", "marital", "records", "job"])),
    ]
)

encodings = {
    "ohe": ohe_encoding,
    "likelihood": effect_encoding,
    "woe": woe_encoding,
}

cnt_transf = Pipeline([("impute", KNNImputer(add_indicator=True))])

transformers = {
    key: ColumnTransformer(
        [
            ("cat", enc, selector(dtype_include=object)),
            ("cnt", cnt_transf, selector(dtype_include=np.number)),
        ]
    )
    for key, enc in encodings.items()
}

classifiers = {
    key: Pipeline([("preproc", item), ("model", xgb.XGBClassifier(random_state=rng))])
    for key, item in transformers.items()
}

xgb_params = {
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__n_estimators": randint(10, 50),
    "model__colsample_bytree": uniform(.5, .5),
    "model__subsample": uniform(.5, .5),
}

for key, clf in classifiers.items():
    cv = RandomizedSearchCV(
        estimator=clf,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=3),
        n_jobs=3,
        random_state=rng,
        param_distributions=xgb_params,
        n_iter=50
    )
    cv.fit(X_train, y_train)
    pd.DataFrame(cv.cv_results_).to_csv('models/' + key + '_cross_validation.csv')

    with open("models/xgb_" + key + "_best_param.json", "w") as f:
        params = json.dumps(cv.best_params_)
        f.write(params)

    # Test Set
    clf.set_params(**cv.best_params_)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(key, accuracy_score(y_test, y_pred))
