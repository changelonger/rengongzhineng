from typing import Callable, Optional, Tuple
import numpy as np
np.random.seed(42)

class CustomKMeans:
    def __init__(self, 
                 n_clusters: int = 3, 
                 max_iter: int = 300, 
                 distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = None) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        assert distance_metric is not None
        self.distance_metric = distance_metric
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = np.array([self.distance_metric(x, centroids) for x in X])
        labels = np.argmin(distances, axis=1)
        return labels

    def compute_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            points_in_cluster = X[labels == i]
            centroids[i] = np.mean(points_in_cluster, axis=0) if len(points_in_cluster) > 0 else self.cluster_centers_[i]
        return centroids

    def fit(self, X: np.ndarray, tol: float = 1e-6) -> 'CustomKMeans':

        self.cluster_centers_ = self.initialize_centroids(X)

        for iteration in range(self.max_iter):
            self.labels_ = self.assign_clusters(X, self.cluster_centers_)
            new_centroids = self.compute_centroids(X, self.labels_)
            if np.allclose(self.cluster_centers_, new_centroids, atol=tol):
                break
            self.cluster_centers_ = new_centroids

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.assign_clusters(X, self.cluster_centers_)


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    # 计算每行的平方差并求和，最后开平方
    squared_diff = (x2 - x1) ** 2         # 利用广播自动对齐维度
    sum_squared = np.sum(squared_diff, axis=1)
    distances = np.sqrt(sum_squared)
    return distances


def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    #TODO   
    return np.sum(np.abs(x1 - x2), axis=1)


def chebyshev_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    #TODO
    return np.max(np.abs(x1 - x2), axis=1)

def main() -> None:
    X = np.random.rand(100, 2)

    kmeans_euclidean = CustomKMeans(n_clusters=3, distance_metric=euclidean_distance)
    kmeans_euclidean.fit(X)
    
    kmeans_manhattan = CustomKMeans(n_clusters=3, distance_metric=manhattan_distance)
    kmeans_manhattan.fit(X)
    
    kmeans_chebyshev = CustomKMeans(n_clusters=3, distance_metric=chebyshev_distance)
    kmeans_chebyshev.fit(X)
    
    x = np.random.rand(3, 2)
    euclidean_predict = kmeans_euclidean.predict(x)
    manhattan_predict = kmeans_manhattan.predict(x)
    chebyshev_predict = kmeans_chebyshev.predict(x)
    print(f'{euclidean_predict=},{manhattan_predict=},{chebyshev_predict=}')

if __name__ == '__main__':
    main()