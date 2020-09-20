import os
import pickle
import shutil


class Pipeline:
    """
    Given an estimator this class contains the logic necessary for
    serialization and deserailization of it.
    """

    def __init__(self, estimator, path: str):
        self.estimator = estimator
        self.path = path

    def __call__(self, data):
        return self.estimator.predict(data)

    def __setstate__(self, state):
        self.path = state["path"]
        self.estimator = state["estimator"]

    @staticmethod
    def save(pipeline):
        try:
            os.makedirs(pipeline.path)
        except FileExistsError:
            # the path already exists, we shoudl wipe the folder
            shutil.rmtree(pipeline.path)
            os.makedirs(pipeline.path)

        pipeline.estimator._set_model_path(pipeline.path)

        with open(os.path.join(pipeline.path, "pipeline.pkl"), "wb") as f:

            pickle.dump(pipeline, f)

    @staticmethod
    def load(path: str):
        pipeline = None
        try:
            with open(os.path.join(path, "pipeline.pkl"), "rb") as f:
                pipeline = pickle.load(f)

        except Exception as e:
            print(str(e))

        return pipeline
