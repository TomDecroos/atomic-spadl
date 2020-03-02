import numpy as np
import pandas as pd
import tqdm

import scipy.stats as stats
import sklearn.cluster as cluster

import cvxpy as cp


class Gauss:
    def fit(self, X, w):
        self.mean = np.average(X, weights=w, axis=0)
        self.cov = np.cov(X.T, aweights=w)
        np.fill_diagonal(self.cov, self.cov.diagonal() + 1e-6)
        return self

    def pdf(self, X):
        return stats.multivariate_normal.pdf(
            X, mean=self.mean, cov=self.cov  # , allow_singular=True
        )

    def _n_parameters(self):
        return len(self.mean) + len(self.cov.flatten())


class VonMises:
    def fit(self, alpha, w):
        sin = np.average(np.sin(alpha), weights=w, axis=0)
        cos = np.average(np.cos(alpha), weights=w, axis=0)

        self.loc = np.arctan2(sin, cos)
        self.R = np.sqrt(sin ** 2 + cos ** 2)  # mean resultant length
        self.kappa = (
            self.R * (2 - self.R ** 2) / (1 - self.R ** 2)
        )  # approximation for kappa
        return self

    def pdf(self, alpha):
        return stats.vonmises.pdf(alpha, kappa=self.kappa, loc=self.loc).flatten()

    def _n_parameters(self):
        return 2


class Categorical:
    def fit(self, X, w, tol=1e-4):
        self.value = stats.mode(X[w > tol]).mode[0]
        return self

    def pdf(self, X):
        n, _d = X.shape
        p = np.zeros(n) + 1
        for i in range(len(self.value)):
            p *= X[:, i] == self.value[i]
        return p

    def _n_parameters(self):
        return 1


class CategoricalModel:
    def __init__(self, tol=1e-6):
        self.tol = tol

    def fit(self, X, weights=None):
        if weights:
            X = X[weights > self.tol]
        self.categories = set(X)
        return self

    def predict_proba(self, X, weights=None):
        p = pd.DataFrame()
        if weights is None:
            weights = np.zeros(len(X)) + 1
        for c in self.categories:
            p[str(c)] = ((X == c) & (weights > self.tol)).apply(float)
        return p


class MixtureModel:
    def __init__(self, k, distribution=Gauss, max_iter=10, tol=1e-6):
        self.k = k
        self.n_components = k
        self.distribution = distribution
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, weights=None, verbose=False):

        # handle sparsity
        if weights is None:
            weights = np.zeros(len(X)) + 1
        pos_weights_idx = weights > self.tol
        X = X[pos_weights_idx]
        weights = weights[pos_weights_idx]

        responsibilities = self.init_responsibilities(X, weights)

        # learn models on initial weights
        self.priors = responsibilities.sum(axis=0) / responsibilities.sum()
        # print(self.priors)
        # check if less clusters found than given components
        if any(self.priors < self.tol):
            raise NotEnoughClustersException

        self.submodels = list(
            self.distribution().fit(X, weights * resp) for resp in responsibilities.T
        )

        iterations = (
            range(self.max_iter) if not verbose else tqdm.tqdm(range(self.max_iter))
        )
        self.loglikelihood = -np.inf
        for self._n_iter in iterations:
            # Expectation
            for i in range(self.k):
                responsibilities[:, i] = self.priors[i] * self.submodels[i].pdf(X)

            # enough improvement or not?
            new_loglikelihood = (weights * np.log(responsibilities.sum(axis=1))).sum()

            if new_loglikelihood > self.loglikelihood + self.tol:
                self.loglikelihood = new_loglikelihood
            else:
                break

            # normalize responsibilities such that each data point occurs with P=1
            responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]

            # Maximalization
            self.priors = responsibilities.sum(axis=0) / responsibilities.sum()
            for i in range(self.k):
                self.submodels[i].fit(X, weights * responsibilities[:, i])

        self.weight_total = weights.sum()

        return self

    def init_responsibilities(self, X, weights):
        # initialize weights with KMeans
        n_samples, _ = X.shape
        labels = cluster.KMeans(n_clusters=self.k, n_init=10).fit_predict(
            X, sample_weight=weights
        )
        resp = np.zeros((n_samples, self.k))
        resp[np.arange(n_samples), labels] = 1
        return resp

    def predict_proba(self, X, weights=None):
        p = np.zeros((len(X), self.k))

        # handle sparsity
        if weights is None:
            weights = np.zeros(len(X)) + 1
        pos_weights_idx = weights > self.tol
        X = X[pos_weights_idx]
        weights = weights[pos_weights_idx]

        pdfs = np.vstack([m.pdf(X) for m in self.submodels]).T
        resp = self.priors * pdfs
        probs = resp / resp.sum(axis=1)[:, np.newaxis]

        p[pos_weights_idx, :] = (weights * probs.T).T
        return p

    def params(self):
        return list(m.__dict__ for m in self.submodels)

    def _n_parameters(self):
        return sum(m._n_parameters() for m in self.submodels)


class NotEnoughClustersException(Exception):
    pass


def learn_mixture(df, columns, weights_df, weights_column, k, distribution=Gauss):
    try:
        model = MixtureModel(k=k, distribution=distribution)
        a = df[columns].values
        if weights_df is not None and weights_column is not None:
            w = weights_df[weights_column].values
            model.fit(a, w)
            model.name = weights_column
        else:
            model.fit(a)
        return model
    except NotEnoughClustersException:
        return None


def ilp_select_models_bic(models, verbose=False):
    x = cp.Variable(len(models), boolean=True)
    c = np.array(list(m.loglikelihood for m in models))
    n_parameters = np.array(list(m._n_parameters() for m in models))
    dataweights = {}
    for m in models:
        if m.name not in dataweights:
            dataweights[m.name] = m.weight_total
    n_data = sum(dataweights.values())

    objective = cp.Minimize(
        np.log(n_data) * cp.sum(n_parameters * x) - 2 * cp.sum(c * x)
    )

    constraints = []
    for name in set(m.name for m in models):
        name_idx = np.array(list(int(m.name == name) for m in models))
        constraints += [name_idx * x == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)
    idx, = np.where(x.value > 0.3)
    return list(models[i] for i in idx)


def select_models_solo_bic(models):
    for m in models:
        m.solo_bic = np.log(m.weight_total) * m._n_parameters() - 2 * m.loglikelihood

    ms = []
    for name in set(m.name for m in models):
        bestm = min([m for m in models if m.name == name], key=lambda m: m.solo_bic)
        ms.append(bestm)
    return ms


def probabilities(models, df, columns, weights_df):
    weights = []
    for model in models:
        probs = model.predict_proba(df[columns].values, weights_df[model.name].values)
        nextlevel_columns = list(f"{model.name}_{i}" for i in range(model.k))
        weights.append(pd.DataFrame(probs, columns=nextlevel_columns))
    return pd.concat(weights, axis=1)

