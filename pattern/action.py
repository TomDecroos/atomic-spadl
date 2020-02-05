import sklearn.mixture as mix
import numpy as np
import pandas as pd
import tqdm
import cvxpy as cp

import sklearn.exceptions


def simplify(actions):
    a=actions
    corner_idx = a.type_name.str.contains("corner")
    a["type_name"] = a["type_name"].mask(corner_idx,"corner")

    freekick_idx = a.type_name.str.contains("freekick")
    a["type_name"] = a["type_name"].mask(freekick_idx,"freekick")

    keeper_idx = a.type_name.str.contains("keeper")
    a["type_name"] = a["type_name"].mask(keeper_idx,"keeper_catch")

    a["x"] = a.x.mask(a.type_name=="goalkick",5)
    a["y"] = a.y.mask(a.type_name=="goalkick",32.5)
    return a


class GMMEnsemble:

    def __init__(self,cols,models):
        self.cols = cols
        self.models = models

    def components(self):
        return {k : m.n_components for k,m in self.models.items()}
    
    def total_components(self):
        return sum(self.components().values())

    def fit(self,actions,verbose=False):
        bad_models = []
        for k in self.models:
            type_idx = actions.type_name == k
            a = actions[type_idx][self.cols]
            if len(a) > 1:
                if len(a) < self.models[k].n_components:
                    self.models[k].n_components = 1
                if verbose:
                    print(f"learning {self.models[k].n_components} {k} components from {len(a)} actions")
                self.models[k].fit(a)
            else:
                bad_models.append(k)
        if verbose:
            print("bad models:", bad_models)
        for k in bad_models:
            del self.models[k]
    
    def _columns(self):
        return [k + str(i+1) for k,i in self._components()]
    
    def _components(self):
        return [(k,i)
                for k,n_c in self.components().items()
                for i in range(n_c)]
    
    def _global_to_local(self,idx):
        return list(self._components())[idx]


    
    def predict_proba(self,actions):
        components = self._columns()
        probs = np.zeros((len(actions),len(components)))
        i = 0
        for k,model in self.models.items():
            type_idx = actions.type_name == k
            if any(type_idx):
                a = actions[type_idx][self.cols]
                type_probs = model.predict_proba(a)
                probs[type_idx,i:i+model.n_components] = type_probs
            i = i + model.n_components
        return pd.DataFrame(data=probs,columns=components)



def greedy_gmme(actions,cols,n,verbose=False):
    type_names = set(actions.type_name)
    dfs = {k : actions[actions.type_name == k][cols] for k in type_names}

    base = GMMEnsemble(cols, {k: mix.GaussianMixture(1) for k in type_names})
    cand = GMMEnsemble(cols, {k: mix.GaussianMixture(2) for k in type_names})
    base.fit(actions)
    cand.fit(actions)
    base_bic = {k : m.score_samples(dfs[k]).sum() for k,m in base.models.items()}
    cand_bic = {k : m.score_samples(dfs[k]).sum() for k,m in cand.models.items()}

    while base.total_components() < n:
        bic_delta, k = max((cand_bic[k] - base_bic[k],k) for k in cand.models)
        if bic_delta > 0:
            base.models[k], base_bic[k] = cand.models[k], cand_bic[k]
            if verbose:
                print(bic_delta, k, base.total_components())
        n_c = cand.models[k].n_components
        cand.models[k] = mix.GaussianMixture(n_c+1)
        if len(dfs[k]) > n_c +1:
            cand.models[k].fit(dfs[k])
            cand_bic[k] = cand.models[k].score_samples(dfs[k]).sum()
    return base
 

def candidates(actions,cols,maxn,verbose=False):
    type_names = set(actions.type_name)
    dfs = {k : actions[actions.type_name == k][cols] for k in type_names}

    candidates = []
    for type_name in type_names:
        a = dfs[type_name]
        n_components = range(1,min(len(a),maxn)+1)
        if verbose:
            n_components= tqdm.tqdm(n_components,desc=f"Training {type_name} components")
        for i in n_components:
            try:
                model = mix.GaussianMixture(i)
                model.fit(a)
                score = model.score_samples(a).sum()
                candidates.append((type_name,model,score))
            except sklearn.exceptions.ConvergenceWarning:
                break
    return candidates


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

def ilp_select_models_likelihood(models,max_components,verbose=False):
    x = cp.Variable(len(models),boolean=True)
    c = np.array(list(m.likelihood for m in models))
    n_components = np.array(list(m.n_components for m in models))

    objective = cp.Maximize(cp.sum(c*x))

    constraints = []
    constraints += [n_components*x <= max_components]
    for name in set(m.name for m in models):
        name_idx = np.array(list(int(m.name == name) for m in models))
        constraints += [name_idx*x == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)
    idx, = np.where(x.value > 0.3)
    return list(models[i] for i in idx)

def ilp_select_candidates(candidates,verbose=False,objective="likelihood",n_samples=1,max_components=999):

    x = cp.Variable(len(candidates),boolean=True)
    c = np.array(list(score for t,m,score in candidates))
    n_components = np.array(list(m for t,m,s in candidates))

    objectivemap = {
        "likelihood": cp.Maximize(cp.sum(c*x)),
        "aic": cp.Minimize(-2*cp.sum(c*x) + 2 * cp.sum(n_components*x)),
        "bic": cp.Minimize(-2*cp.sum(c*x) + np.log(n_samples) * cp.sum(n_components*x)),
    }

    if objective=="likelihood":
        constraints = [n_components*x <= max_components]
    for ty in set(t for t,m,score in candidates):
        ty_idx = np.array(list(int(t == ty) for t,m,s in candidates))
        constraints += [ty_idx*x == 1]
    
    #constraints += [0 <= x, x <= 1]

    prob = cp.Problem(objectivemap[objective], constraints)
    prob.solve(verbose=verbose)
    idx, = np.where(x.value > 0.3)
    return idx


def uncouple_probabilities(probs,left_to_right):
    uc_probs = pd.DataFrame()
    for t in probs.columns:
        uc_probs[t] = probs[t].mask(~left_to_right,0)
    for t in probs.columns:
        uc_probs[t + "_opp"] = probs[t].mask(left_to_right,0)
    return uc_probs