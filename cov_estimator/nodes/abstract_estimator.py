import os


class AbstractEstimator:
    """
    An estimator is an abstraction to some complex model, either Keras
    or Scikit-Learn models are allowed at the moment.
    """

    def __init__(self, *args):
        # this is part of the tree construction
        self.dependencies = []

    def _set_model_path(self, path: str):
        """
        Set the base path for this specific model
        """
        for node in self.dependencies:
            node._set_model_path(path)
        self.model_path = os.path.join(path, self.name)

    def __setstate__(self, state):
        self.model_path = state["model_path"]
