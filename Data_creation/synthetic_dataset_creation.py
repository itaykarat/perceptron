from sklearn import datasets


class synthetic_data:
    def __init__(self):
        self.dataset = self.create_synthetic_data_set()

    def create_synthetic_data_set(self):
        X, y = datasets.make_blobs(n_samples=150, n_features=2,
                                   centers=2, cluster_std=1.05,
                                   random_state=2)

        data = (X, y)

        return data
