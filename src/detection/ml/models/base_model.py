class BasePredictiveModel:
    def fit(self, X, Y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
