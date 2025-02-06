from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

from typing import Optional


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 100,
        subsample: float = 0.3,
        learning_rate: float = 0.1,
        early_stopping_rounds: Optional[int] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,

        bootstrap_type: str = None,
        bagging_temperature: float = None,
        rsm: float = None,
        quantization_type: str = None,
        nbins: int = None
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_set = eval_set
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)


        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature

    
        self.rsm  = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.selected_features_list = []


        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z) 


    def partial_fit(self, X, y):
        m = self.base_model_class(**self.base_model_params)
        m.fit(X, y)
        return m

    def quantize_features(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()

        if np.any(np.isnan(X)):
            col_median = np.nanmedian(X, axis=0)
            for i in range(X.shape[1]):
                X[np.isnan(X[:, i]), i] = col_median[i]

        if self.quantization_type == 'Uniform':
            bins = np.linspace(np.min(X), np.max(X), self.nbins + 1)
            X_quantized = np.digitize(X, bins) - 1
            return X_quantized

        elif self.quantization_type == 'Quantile':
            discretizer = KBinsDiscretizer(n_bins=self.nbins, encode='ordinal', strategy='quantile')
            X_quantized = discretizer.fit_transform(X)
            return X_quantized

        return X 

    def sample_with_bagging(self, X):
        n_samples = X.shape[0]
        if self.bootstrap_type == 'Bernoulli':
            n_samples_to_draw = int(n_samples * self.subsample)
            indices = np.random.choice(n_samples, size=n_samples_to_draw, replace=True)
            return indices
        elif self.bootstrap_type == 'Bayesian':
            weights = (-np.log(np.random.uniform(0, 1, n_samples))) ** self.bagging_temperature
            weights /= weights.sum()
            indices = np.random.choice(n_samples, size=n_samples, replace=True, p=weights)
            return indices
        else:
            return np.arange(n_samples)     

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=True):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(self.eval_set[1].shape[0]) if self.eval_set is not None else None
        
        best_valid_loss = float('inf')
        rounds_without_improvement = 0

        for i in range(self.n_estimators):
            qu = y_train.copy()
            qu[qu == 0] = -1

            shift = -self.loss_derivative(qu, train_predictions)
            ind = self.sample_with_bagging(X_train)

            if self.bootstrap_type is not None:
                x_boot, shift_boot = X_train[ind], shift[ind]
            else:
                x_boot, shift_boot = X_train, shift

            n_features = x_boot.shape[1]
            if self.rsm is not None:
                n_features_to_select = int(n_features * self.rsm)
                selected_features = np.random.choice(n_features, n_features_to_select, replace=False)
            else:
                selected_features = np.arange(n_features)

            x_boot_selected = x_boot[:, selected_features]
            self.selected_features_list.append(selected_features)

            if self.quantization_type is not None:
                x_boot_quantized = self.quantize_features(x_boot_selected)
            else:
                x_boot_quantized = x_boot_selected


            m = self.partial_fit(x_boot_quantized, shift_boot)
            preds_train = m.predict(x_boot_quantized)
            gamma = self.find_optimal_gamma(y_train, train_predictions, preds_train)

            self.models.append(m)
            self.gammas.append(gamma)
            train_predictions += preds_train * self.learning_rate * gamma

            train_loss = self.loss_fn(qu, train_predictions)
            self.history['train_loss'].append(train_loss)

            if self.eval_set is not None:
                X_val, y_val = self.eval_set
                x_val_selected = X_val[:, selected_features]

                if self.quantization_type is not None:
                    x_val_quantized = self.quantize_features(x_val_selected)
                else:
                    x_val_quantized = x_val_selected

                preds_val = m.predict(x_val_quantized)

                valid_predictions += preds_val * self.learning_rate * gamma
                
                quu = y_val.copy()
                quu[quu == 0] = -1
                valid_loss = self.loss_fn(quu, valid_predictions)
                self.history['valid_loss'].append(valid_loss)

                if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        rounds_without_improvement = 0
                else:
                        rounds_without_improvement += 1

                if self.early_stopping_rounds is not None and rounds_without_improvement == self.early_stopping_rounds:
                    print(f"early stopping at {i + 1}")
                    break

        if plot:
            self.plot_history(X_train, y_train)
        

        return i, valid_loss

    def predict_proba(self, X):
        preds = np.zeros(X.shape[0])

        for m, gamma, selected in zip(self.models, self.gammas, self.selected_features_list):
            X_sel = X[:, selected]

            if self.quantization_type is not None:
                x_sel_quantized = self.quantize_features(X_sel)
            else:
                x_sel_quantized = X_sel

            preds += gamma * m.predict(x_sel_quantized)

        preds =  self.sigmoid(preds)
        p = np.zeros([X.shape[0], 2])

        p[:, 0] = 1 - preds
        p[:, 1] =  preds

        return p

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        k = y.copy()
        k[k == 0] = -1
        losses = [self.loss_fn(k, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)

    def plot_history(self, X, y):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'valid_loss' in self.history:
            plt.plot(self.history['valid_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        plt.legend()
        plt.tight_layout()
        plt.show()

    def feature_importances_(self):
        avg = np.sum(m.feature_importances_ for m in self.models) / len(self.models)
        avg /= np.sum(avg)
        return avg