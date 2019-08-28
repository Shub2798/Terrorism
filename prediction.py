from random import shuffle

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle

scaler = preprocessing.StandardScaler()

n_folds = 5


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


def load_data():
    terror_data = pd.read_csv('/Users/kashish/PycharmProjects/data/new_features.csv', usecols=[1, 2, 3, 4, 5, 7, 9])
    y_data = pd.read_csv('/Users/kashish/PycharmProjects/rnn/train_y.csv', usecols=[1])
    result = pd.concat([terror_data, y_data], axis=1)

    df = result[np.isfinite(result['longitude'])]
    df = shuffle(df)

    test = df.iloc[:2000, :]
    train = df.iloc[2000:, :]

    test_data = test.iloc[:, :7]
    test_values = test.iloc[:, 7:]

    terror_data = train.iloc[:, :7]

    y_data = train.iloc[:, 7:]

    final_data = terror_data.as_matrix()
    test_data = test_data.as_matrix()
    test_values = test_values.as_matrix()

    train_y = y_data.as_matrix()

    return final_data, train_y, test_data, test_values

