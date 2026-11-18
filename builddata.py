from sklearn.datasets import make_blobs, make_classification

def generate_data(dim: int,distribution: str = 'gaussian', n_samples: int = 100):
    if distribution == 'gaussian':
        data = make_blobs(n_samples=n_samples, n_features=dim, random_state=42)
        return data[0],data[1]
    elif distribution == 'class':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=dim,
            n_informative=dim,      
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            random_state=42)
        return X, y
    else:
        raise ValueError("Unsupported distribution")



