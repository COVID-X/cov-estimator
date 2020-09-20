import pickle

from cov_estimator.nodes.abstract_estimator import AbstractEstimator


class SklearnEstimator(AbstractEstimator):
    """
    Adapter class to work with any type of sklearn estimators, as long as they contains a 'predict' method
    """

    def __init__(self, model, name):
        super(SklearnEstimator, self).__init__(model, name)
        self.model = model
        self.name = name
        self.dependencies = []

    def __call__(self, *args):
        self.dependencies = args
        return self

    def predict(self, data):
        # first, check if you are a terminal node, meaning there is some
        # data you are required to proccess.

        if type(data.get(self.name, False)) != bool:
            return self.model.predict(data.get(self.name))

        # in case the estimator is not terminal,
        # it should contains only 1 dependency
        prev_result = self.dependencies.pop().predict(data)

        return self.model.predict(prev_result)

    def __copy__(self):
        model = SklearnEstimator(self.model, self.name)
        model._set_model_path(self.path)
        return model

    def __getstate__(self):
        copy_node = self.__copy__()
        with open(self.path, "wb") as f:
            pickle.dump(copy_node.model, f)
        copy_node.model = None
        return copy_node.__dict__

    def __setstate__(self, state):
        super(SklearnEstimator, self).__setstate__(state)
        with open(state["model_path"], "rb") as f:
            self.model = pickle.load(f)
        self.name = state["name"]
        self.dependencies = state["dependencies"]
