import torch
import torch.nn.functional as F

class KMeans(object):
    def __init__(
        self,
        n_clusters=8,
        *,
        n_init=10,
        max_iter=2000,
        tol=1e-4,
        copy_x=True,
    ) -> None:

        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.copy_x = copy_x

    def fit(self, X, y=None):

        N, D = X.shape

        self.labels_ = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        self.inertia_ = torch.zeros(self.n_clusters, device=X.device)
        self.cluster_centers_ = X[torch.randperm(N)[:self.n_clusters]]

        self.cluster_centers_ = torch.empty((0, D), device=X.device)
        for i in range(self.n_clusters):
            if i == 0:
                self.cluster_centers_ = X[torch.randperm(N)[:1]]
            else:
                dist = torch.cdist(X, self.cluster_centers_)
                dist = dist.sum(dim=1)
                self.cluster_centers_ = torch.cat([self.cluster_centers_, X[torch.multinomial(dist, 1)]], dim=0)
        
        for i in range(self.max_iter):
            self.labels_ = self._predict(X)
            self.cluster_centers_ = self._update_centers(X)
            self.inertia_ = self._update_inertia(X)
            if self._is_converged():
                break
        
        return self

    def _predict(self, X):
        dist = torch.cdist(X, self.cluster_centers_)
        return torch.argmin(dist, dim=1)

    def _update_centers(self, X):
        centers = torch.zeros(self.n_clusters, X.shape[1], device=X.device)
        for i in range(self.n_clusters):
            if (self.labels_==i).sum() != 0:
                centers[i] = X[self.labels_ == i].mean(dim=0)
            else:
                dist = torch.cdist(X, self.cluster_centers_).sum(dim=-1)
                centers[i] = X[torch.multinomial(dist, 1)]
        return centers

    def _update_inertia(self, X):
        inertia = torch.zeros(self.n_clusters, device=X.device)
        for i in range(self.n_clusters):
            inertia[i] = ((X[self.labels_ == i] - self.cluster_centers_[i])**2).sum()
        return inertia

    def _is_converged(self):
        return self.inertia_.sum() < self.tol