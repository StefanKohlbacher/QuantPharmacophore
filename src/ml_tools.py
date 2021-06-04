import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA


class PCAPredictor(PCA):

    def __init__(self,
                 predictionModel,
                 explainedVariance=0.8,
                 **kwargs
                 ):
        self.predictionModel = predictionModel
        self.kwargs = kwargs
        self.n_components_adjusted = None
        self._explainedVariance = explainedVariance
        super(PCAPredictor, self).__init__(**kwargs)

    def predict(self, X):
        transformed = self.transform(X)

        if self.n_components is None:
            transformed = transformed[:, :self.n_components_adjusted]
        return self.predictionModel.predict(transformed)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        # fit_transform as implemented by parent PCA
        transformed = super(PCAPredictor, self).fit_transform(X)

        if self.n_components is None:
            cumulativeVarianceExplained = np.cumsum(self.explained_variance_ratio_)
            self.n_components_adjusted = np.sum((cumulativeVarianceExplained < self._explainedVariance)) + 1

            transformed = transformed[:, :self.n_components_adjusted]

        # train on-top ML-model
        self.predictionModel.fit(transformed, y)

        return transformed


def analyse_regression(y_true: np.ndarray, y_pred: np.ndarray, weights=None) -> dict:
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    assert not np.any(np.isnan(y_true)) and not np.any(np.isnan(y_pred)), 'y_true: %s\ny_pred: %s' % (str(y_true), str(y_pred))
    assert not np.any(np.isinf(y_true)) and not np.any(np.isinf(y_pred)), 'y_true: %s\ny_pred: %s' % (str(y_true), str(y_pred))
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    try:
        mse = mean_squared_error(y_true, y_pred, sample_weight=weights, squared=True)
        rmse = mean_squared_error(y_true, y_pred, sample_weight=weights, squared=False)
    except ValueError:
        return {"MSE": None,
                "RMSE": None,
                "Mean error": None,
                "Std error": None,
                "Nr samples": None,
                "Median error": None,
                "Max error": None,
                "Min error": None,
                "1st quartile error": None,
                "3rd quartile error": None,
                "Prediction range": None,
                "True range": None,
                "R2": None,
                }
    diff = np.absolute(y_true - y_pred)
    return {"MSE": mse,
            "RMSE": rmse,
            "Mean error": np.average(diff, weights=weights),
            "Std error": np.std(diff),
            "Nr samples": y_true.shape[0],
            "Median error": np.median(diff),
            "Max error": np.max(diff),
            "Min error": np.min(diff),
            "1st quartile error": np.percentile(diff, 25),
            "3rd quartile error": np.percentile(diff, 75),
            "Prediction range": np.max(y_pred) - np.min(y_pred),
            "True range": np.max(y_true) - np.min(y_true),
            "R2": r2_score(y_true, y_pred)
            }


def aggregateRegressionCrossValidationResults(results):
    """
    Assumes results are in form of a nested dictionary, whereas each entry is obtained by one fold
    of the CV and contains results analysed by the function 'analyse_regression'.
    :param results:
    :return: Mean and standard deviation of metrics for the k-fold cross-validation.
    """
    results = pd.DataFrame.from_dict(results, orient='index')
    aggregated = {metric: {'Mean': None, 'Std': None} for metric in results.columns.values}

    # handle MSE
    aggregated['MSE']['Std'], aggregated['MSE']['Mean'] = results['MSE'].std(), results['MSE'].mean()

    # handle RMSE
    aggregated['RMSE']['Std'], aggregated['RMSE']['Mean'] = results['RMSE'].std(), results['RMSE'].mean()

    # handle Mean error
    aggregated['Mean error']['Std'], aggregated['Mean error']['Mean'] = results['Mean error'].std(), results['Mean error'].mean()

    # handle Std error
    aggregated['Std error']['Std'], aggregated['Std error']['Mean'] = results['Std error'].std(), results['Std error'].mean()

    # handle Median error
    aggregated['Median error']['Std'], aggregated['Median error']['Mean'] = results['Median error'].std(), results['Median error'].mean()

    # handle Max error
    aggregated['Max error']['Std'], aggregated['Max error']['Mean'] = results['Max error'].std(), results['Max error'].mean()

    # handle Min error
    aggregated['Min error']['Std'], aggregated['Min error']['Mean'] = results['Min error'].std(), results['Min error'].mean()

    # handle First quartile error
    aggregated['1st quartile error']['Std'], aggregated['1st quartile error']['Mean'] = results['1st quartile error'].std(), results['1st quartile error'].mean()

    # handle Third quartile error
    aggregated['3rd quartile error']['Std'], aggregated['3rd quartile error']['Mean'] = results['3rd quartile error'].std(), results['3rd quartile error'].mean()

    # handle Nr Samples
    aggregated['Nr samples']['Std'], aggregated['Nr samples']['Mean'] = results['Nr samples'].std(), results['Nr samples'].mean()

    # handle Prediction range
    aggregated['Prediction range']['Std'], aggregated['Prediction range']['Mean'] = results['Prediction range'].std(), results['Prediction range'].mean()

    # handle True range
    aggregated['True range']['Std'], aggregated['True range']['Mean'] = results['True range'].std(), results['True range'].mean()

    # handle R2
    aggregated['R2']['Std'], aggregated['R2']['Mean'] = results['R2'].std(), results['R2'].mean()

    return aggregated
