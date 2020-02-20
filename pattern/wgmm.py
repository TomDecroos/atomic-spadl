import sklearn
import sklearn.mixture as mix
from sklearn.mixture._gaussian_mixture import (
    _estimate_gaussian_parameters,
    _compute_precision_cholesky,
    )
import numpy as np
import numpy.linalg as linalg

import sklearn.cluster as cluster


import pandas as pd
import numpy as np

import cvxpy as cp

class WeightedGMM(mix.GaussianMixture):
    
    def fit(self,X,weights=None,*k,**kw):
        if weights is None:
            weights = np.zeros(len(X)) + 1
        pos_weight_idx = weights > 1e-10
        self.Xweights = weights[pos_weight_idx]
        return super().fit(X[pos_weight_idx],*k,**kw)
        
    def _m_step(self, X, log_resp):
            """M step.
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
            log_resp : array-like, shape (n_samples, n_components)
                Logarithm of the posterior probabilities (or responsibilities) of
                the point of each sample in X.
            """
            resp = (self.Xweights*np.exp(log_resp).T).T
            self.weights_, self.means_, self.covariances_ = (
                _estimate_gaussian_parameters(X,resp, self.reg_covar,
                                              self.covariance_type))
            
            self.weights_ /= np.sum(self.Xweights)
                
            self.precisions_cholesky_ = _compute_precision_cholesky(
                self.covariances_, self.covariance_type)
            
    def _initialize_parameters(self, X, random_state):
            """Initialize the model parameters.
            Parameters
            ----------
            X : array-like, shape  (n_samples, n_features)
            random_state : RandomState
                A random number generator instance.
            """
            n_samples, _ = X.shape

            if self.init_params == 'kmeans':
                resp = np.zeros((n_samples, self.n_components))
                clus = cluster.KMeans(n_clusters=self.n_components, n_init=20,
                                       random_state=random_state)
                label = clus.fit(X,sample_weight=self.Xweights).labels_
                resp[np.arange(n_samples), label] = 1
            elif self.init_params == 'random':
                resp = random_state.rand(n_samples, self.n_components)
                resp /= resp.sum(axis=1)[:, np.newaxis]
            else:
                raise ValueError("Unimplemented initialization method '%s'"
                                 % self.init_params)

            self._initialize(X, resp)
            
    def score_samples(self,X,weights=None,err=1e-10):
        if weights is None:
            weights = np.zeros(len(X)) + 1
        pos_weights_idx = weights > err
        
        scores = np.zeros(len(X))
        pos_weights = weights[pos_weights_idx]
        pos_scores = super().score_samples(X[pos_weights_idx])
        scores[pos_weights_idx] = (pos_weights*pos_scores.T).T
        return scores
    
    def predict_proba(self,X,weights=None,err=1e-10):
        if weights is None:
            weights = np.zeros(len(X)) + 1
        pos_weights_idx = weights > err
        
        p = np.zeros((len(X),self.n_components))
        pos_weights = weights[pos_weights_idx]
        pos_probs = super().predict_proba(X[pos_weights_idx])
        p[pos_weights_idx,:] = (pos_weights*pos_probs.T).T
        return p
    
def learn_wgmm(df,columns,weights_df,weights_column,n_components,n_init=1):
    model = WeightedGMM(n_components=n_components,n_init = n_init)
    a = df[columns].values
    w = weights_df[weights_column].values
    model.fit(a,w)
    model.likelihood = model.score_samples(a,w).sum()
    model.name = weights_column
    return model

def probabilities(models):
    weights = []
    for model in models:
        probs = model.predict_proba(model.df[model.columns].values,model.weights)
        columns = list(f"{model.name}_{i}" for i in range(model.n_components))
        weights.append(pd.DataFrame(probs,columns=columns))
    return pd.concat(weights,axis=1)


def ilp_select_models_bic(models,verbose=False):
    x = cp.Variable(len(models),boolean=True)
    c = np.array(list(m.likelihood for m in models))
    n_parameters = np.array(list(m._n_parameters() for m in models))
    dataweights = {}
    for m in models:
        if m.name not in dataweights:
            dataweights[m.name] = m.weights.sum()
    n_data = sum(dataweights.values())

    objective = cp.Minimize(np.log(n_data) * cp.sum(n_parameters*x) - 2 * cp.sum(c*x))

    constraints = []
    for name in set(m.name for m in models):
        name_idx = np.array(list(int(m.name == name) for m in models))
        constraints += [name_idx*x == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)
    idx, = np.where(x.value > 0.3)
    return list(models[i] for i in idx)

def select_models_solo_bic(models):
    for m in models:
        m.solo_bic = np.log(m.weights.sum()) * m._n_parameters() - 2 * m.likelihood

    ms = []
    for name in set(m.name for m in models):
        bestm = min([m for m in models if m.name == name],key=lambda m: m.solo_bic)
        ms.append(bestm)
    return ms