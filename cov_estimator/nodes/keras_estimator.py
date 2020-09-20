import os

import numpy as np
import tensorflow as tf

from cov_estimator.nodes.abstract_estimator import AbstractEstimator


class KerasEstimator(AbstractEstimator):
    def __init__(self, model, name):
        super(KerasEstimator, self).__init__(model, name)
        self.model = model
        self.name = name
        self.dependencies = []

    def __call__(self, *args):
        self.dependencies = args
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        This wrapper method expects a data object
        which should be as expected for the model
        Params:
            - data: numpy array
        Returns:
            model result (most likely ... a numpy array)
        """
        # first, check if you are a terminal node, meaning there is some
        # data you are required to proccess.

        if type(data.get(self.name, False)) != bool:
            return self.model.predict(data.get(self.name))

        # in case the estimator is not terminal,
        # it should contains only 1 dependency
        prev_result = self.dependencies.pop().predict(data)

        return self.model.predict(prev_result)

    def __copy__(self):
        model = KerasEstimator(self.model, self.name)
        model._set_model_path(self.model_path)
        return model

    def __getstate__(self):
        copy_node = self.__copy__()
        copy_node.model.save(copy_node.model_path)
        copy_node.model = None
        return copy_node.__dict__

    def __setstate__(self, state):
        super(KerasEstimator, self).__setstate__(state)
        self.model = tf.keras.models.load_model(state["model_path"])
        self.name = state["name"]
        self.path = state["model_path"]
        self.dependencies = state["dependencies"]
