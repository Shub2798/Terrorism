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


def interpret_data(x):
    return scaler.inverse_transform(x)



print("hello world")
x_train, y_train, x_test, y_test = load_data()

final_data = scaler.fit_transform(x_train)

x_test1 = scaler.fit_transform(x_test)

train_y = scaler.fit_transform(y_train)

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# this is the start of random forest
##################################################

rf = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=0, max_features='auto')

max_features = np.array(['sqrt', 'log2', 'auto'])

grid = rf

train_y = train_y.ravel()

grid.fit(final_data, train_y)

predicted_train = grid.predict(x_test1)

predicted_train = predicted_train.reshape(2000, 1)

new = scaler.inverse_transform(predicted_train) - y_test

print(np.sum(new ** 2) / 2000)

#####################################################
######################################################


ENet.fit(final_data, train_y)
KRR.fit(final_data, train_y)

predicted_train = ENet.predict(x_test1)
predicted_train1 = KRR.predict(x_test1)

predicted_train = predicted_train.reshape(2000, 1)
predicted_train1 = predicted_train1.reshape(2000, 1)

new = scaler.inverse_transform(predicted_train) - y_test
new1 = scaler.inverse_transform(predicted_train1) - y_test

print(np.sum(new ** 2) / 2000)
print(np.sum(new1 ** 2) / 2000)

new_int = np.around(scaler.inverse_transform(predicted_train), decimals=0)

predicted_train = scaler.inverse_transform(predicted_train)

averaged_models = StackingAveragedModels(base_models=(ENet, rf), meta_model=KRR)
averaged_models.fit(final_data, train_y)
predicted_train = averaged_models.predict(x_test1)

predicted_train = predicted_train.reshape(2000, 1)

new_int = np.around(scaler.inverse_transform(predicted_train), decimals=0)

new = scaler.inverse_transform(predicted_train) - y_test

print(np.sum(new ** 2) / 2000)

df = pd.DataFrame(data=scaler.inverse_transform(predicted_train), columns=['predicted'])
df1 = pd.DataFrame(data=y_test, columns=['Real'])
result = pd.concat([df, df1], axis=1)

result.to_csv("show1.csv")
