import tensorflow as tf
from sklearn.base import BaseEstimator

from cov_estimator.nodes.keras_estimator import KerasEstimator
from cov_estimator.nodes.sklearn_estimator import SklearnEstimator
from cov_estimator.nodes.transformation_node import TransformNode


class Estimator:
    def __new__(cls, model, name):
        if issubclass(model.__class__, tf.keras.Model):
            return KerasEstimator(model, name)

        if issubclass(model.__class__, BaseEstimator):
            return SklearnEstimator(model, name)

        if callable(model):
            # then it's a callable object
            return TransformNode(model, name)

        raise NotImplementedError("Unknown {} estimator".format(type(model)))
