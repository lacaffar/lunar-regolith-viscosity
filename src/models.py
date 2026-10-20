"""Our models. All share one interface:

    m.fit(train_df)                  train_df: rows of train_lp.csv (wt% oxides, T_K, log_eta)
    mean, std = m.predict(wt, T_K)   wt: DataFrame of wt% with feo/fe2o3 split (11 oxides)

Models
  RidgePhys   ridge on physics features: x, NBO/T, and their products with 1/T (Arrhenian-like)
  GBM         histogram gradient boosting on x, NBO/T, T, 1/T
  VFTNet      MLP: composition -> (A, B, C, log sigma); log eta = A + B / (T - C).
              Temperature enters only through VFT. B kept positive with softplus.
  Ensemble    K independently initialised VFTNets (Lakshminarayanan et al. 2017):
              mean = average of member means, variance = mean member variance + spread.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import chem

torch.set_num_threads(4)


def features(wt, T_K=None):
    """Mole fractions (11) + NBO/T from a wt% table with feo/fe2o3 split."""
    w = wt.copy()
    for c in chem.OXIDES:
        if c not in w:
            w[c] = 0.0
    x = chem.wt_to_mol(w[chem.OXIDES], chem.OXIDES)
    f = x.copy()
    f["nbo_t"] = chem.nbo_t(x).values
    return f.reset_index(drop=True)


def train_xy(df):
    wt = df[chem.OXIDES]
    return features(wt), df["T_K"].to_numpy(float), df["log_eta"].to_numpy(float)


# ----------------------------------------------------------------- ridge
class RidgePhys:
    name = "Ridge (NBO/T)"

    def _design(self, F, T):
        invT = 1e3 / T
        cols = [F.values, F.values * invT[:, None], invT[:, None], (invT ** 2)[:, None],
                (F["nbo_t"].values * invT ** 2)[:, None]]
        return np.hstack(cols)

    def fit(self, df):
        F, T, y = train_xy(df)
        X = self._design(F, T)
        self.sc = StandardScaler().fit(X)
        self.m = RidgeCV(alphas=np.logspace(-4, 2, 13)).fit(self.sc.transform(X), y)
        self.sigma = np.std(y - self.m.predict(self.sc.transform(X)))
        return self

    def predict(self, wt, T_K):
        F = features(wt)
        X = self._design(F, np.asarray(T_K, float))
        mu = self.m.predict(self.sc.transform(X))
        return mu, np.full_like(mu, self.sigma)


# ----------------------------------------------------------------- GBM
class GBM:
    name = "Gradient boosting"

    def _design(self, F, T):
        X = F.copy()
        X["T"] = T
        X["invT"] = 1e4 / T
        return X

    def fit(self, df):
        F, T, y = train_xy(df)
        self.m = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, max_leaf_nodes=31,
                                               min_samples_leaf=10, l2_regularization=1e-3,
                                               random_state=0).fit(self._design(F, T), y)
        self.sigma = np.std(y - self.m.predict(self._design(F, T)))
        return self

    def predict(self, wt, T_K):
        F = features(wt)
        mu = self.m.predict(self._design(F, np.asarray(T_K, float)))
        return mu, np.full_like(mu, self.sigma)


# ----------------------------------------------------------------- VFT net
class _Net(torch.nn.Module):
    def __init__(self, n_in, hidden=(128, 128), dropout=0.05):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(dropout)]
            d = h
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(d, 4)
        with torch.no_grad():  # start near a typical silicate VFT curve
            self.head.bias.copy_(torch.tensor([-4.0, 0.0, 0.0, -1.5]))

    def forward(self, x):
        z = self.head(self.body(x))
        A = z[:, 0]
        B = 4000.0 * torch.nn.functional.softplus(z[:, 1] + 0.5)   # K, positive
        C = 1200.0 * torch.sigmoid(z[:, 2] - 0.2)                   # K, in (0, 1200)
        log_sigma = z[:, 3].clamp(-3.5, 1.0)                        # per-composition noise
        return A, B, C, log_sigma


class VFTNet:
    name = "VFT-MLP"

    def __init__(self, seed=0, epochs=400, lr=2e-3, batch=512, hidden=(128, 128), weight_decay=1e-4):
        self.seed, self.epochs, self.lr, self.batch = seed, epochs, lr, batch
        self.hidden, self.wd = hidden, weight_decay

    def _x(self, F):
        return torch.tensor(self.sc.transform(F.values), dtype=torch.float32)

    def fit(self, df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        F, T, y = train_xy(df)
        self.sc = StandardScaler().fit(F.values)
        X = self._x(F)
        T_t = torch.tensor(T, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        self.net = _Net(X.shape[1], self.hidden)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        n = len(y)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                A, B, C, ls = self.net(X[idx])
                gap = T_t[idx] - C
                mu = A + B / gap.clamp(min=20.0)
                # Gaussian NLL with a 0.03 log-unit noise floor (instrument precision)
                var = torch.exp(2 * ls) + 0.03 ** 2
                nll = 0.5 * (torch.log(var) + (y_t[idx] - mu) ** 2 / var)
                # keep every training temperature above the Vogel temperature
                pen = torch.relu(60.0 - gap) / 60.0
                loss = nll.mean() + pen.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def vft_params(self, wt):
        self.net.eval()
        A, B, C, ls = self.net(self._x(features(wt)))
        return A.numpy(), B.numpy(), C.numpy(), np.sqrt(np.exp(2 * ls.numpy()) + 0.03 ** 2)

    def predict(self, wt, T_K):
        A, B, C, s = self.vft_params(wt)
        T = np.asarray(T_K, float)
        return A + B / np.maximum(T - C, 20.0), s


class Ensemble:
    name = "Deep ensemble (VFT-MLP x10)"

    def __init__(self, k=10, **kw):
        self.members = [VFTNet(seed=i, **kw) for i in range(k)]

    def fit(self, df):
        for m in self.members:
            m.fit(df)
        return self

    def member_predictions(self, wt, T_K):
        out = [m.predict(wt, T_K) for m in self.members]
        return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])

    def predict(self, wt, T_K):
        mus, sigs = self.member_predictions(wt, T_K)
        mu = mus.mean(0)
        var = (sigs ** 2).mean(0) + mus.var(0)
        # post-hoc calibration factor from grouped CV (1.0 = raw ensemble)
        return mu, getattr(self, "sigma_scale", 1.0) * np.sqrt(var)

    def vft_params(self, wt):
        return [m.vft_params(wt) for m in self.members]
