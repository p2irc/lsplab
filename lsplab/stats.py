from sklearn import decomposition


class pca(object):
    def __init__(self):
        pass

    def train(self, features, new_dim):
        self.model = decomposition.PCA(n_components=new_dim)
        self.model.fit(features)

    def transform(self, x):
        if x.ndim == 1:
            return self.model.transform(x.reshape(1, -1))[0]
        else:
            return self.model.transform(x)
