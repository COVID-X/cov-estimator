"""
We define an artifact as a complete model which is represented as a DAG 
with one output node (which can output a matrix or any type of desied output)
"""
import os
import pickle

import numpy as np
import tensorflow as tf

# from model import infer_prediction


class Artifact(object):
    """
    This is the base class for any artifact representation of a model estimator.
    The aim of this class is to encapsulate all necessary information for an estimator
    from the NN models, the preprocessing, eventualy it will include the comparison metrics.
    Important:
        - remeber, each subclass of Artifact must be pickable! this is a must, no exceptions.
    """

    def __init__(self):
        super().__init__()

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, img: np.ndarray) -> np.ndarray:
        raise NotImplemented


class StepNNArtifact(Artifact):
    def __init__(self, artifact_name, artifact_path: dict, **kwargs):
        """
        builds the step neural network artifact based on the arguments;
        Params:
            - artifact_name: this is simplya  name for string representation.
            - artifact_path: dictionary with os based path to the location from where the artifact must load and be save.
            - img_shape: 2-tuple with size of the img as (width, height)
        """
        super().__init__()
        self.artifact_name = artifact_name
        self.path = artifact_path
        # necessary for preprocessing the data
        self.img_shape = kwargs["img_shape"]

        # either loads the prevoius logger, or create a new one
        if not kwargs.get("logger", None):
            try:
                import logging

                self.logger = logging.getLogger(self.artifact_name)
            except ImportError:
                self.logger = None
        else:
            self.logger = kwargs["logger"]
        # load models

        self.pneumonia_model = self.load_model(
            os.path.join(self.path["artifact"], self.path["pneumonia"])
        )
        self.covid_model = self.load_model(
            os.path.join(self.path["artifact"], self.path["covid"])
        )

    def load_model(self, path):
        """
        Loads network from path, should be in h5 format.
        Obs:
            - please, make sure to specified a compiled model
        """
        try:
            model = tf.keras.models.load_model(path)
            model.compile()
            return model
        except Exception as e:
            if self.logger:
                self.logger.error("could not load model {}. \n{}".format(path, str(e)))
            else:
                # if no logger, then raise exception
                raise e

    def preprocess(self, data: dict) -> np.ndarray:
        """
        Process img to it's equivalent from trainig conditions.
        """

        if self.logger != None:
            self.logger.info("about to load img as tensor")
        img = self.load_img(data["img_path"], self.img_shape)
        data["img"] = img
        return data

    def predict(self, data: dict) -> dict:
        """
        returns the inference result form network forward pass.
        Params:
            - img: numpy array of size (length, width, 3)
        """
        # norm_img = process_img(img)
        # first we load img from local path
        img_path = data["img_path"]

        img = self.load_img(img_path, self.img_shape)

        pneumonia_model, covid_model = self.pneumonia_model, self.covid_model

        # as the step model does a 2 steps inference

        probability_of_pneumonia = float(
            pneumonia_model.predict(np.expand_dims(img, axis=0))
        )
        # the model has found pneumonia signs which coudl either be cause by covid-19 or others
        covid_probablity = float(covid_model.predict(np.expand_dims(img, axis=0)))
        # this is the real trick, think of it as a tree .... the real probability of getting to the leaf is the
        # multiplication of all prev steps actually hapening.
        final_covid_prob = float(covid_probablity * probability_of_pneumonia)

        # covid_prob, neumonia_prob, no_findings
        # probability_vec = [final_covid_prob, probability_of_pneumonia - final_covid_prob, 1 - probability_of_pneumonia]

        result = {
            "normal": 1 - probability_of_pneumonia,
            # this is because the 'probability_of_pneumonia' var includes covid19 prob
            "pneumonia": probability_of_pneumonia - final_covid_prob,
            "covid-19": final_covid_prob,
        }

        predictions = {
            "class_prob_dist": result,
            "metadata": "shoudl include something else",
        }

        return predictions

    def load_img(self, img_path: str, img_shape: tuple):
        """
        Loads img from local file to memory
        Params:
            - img_path (str): path to the img file
            - img_shape: 2-tuple representing (width, height) of the output img

        Return : numpy representation fo img
        If it fails, just logs the error and return None
        """
        try:
            img = self.process_path(img_path, img_shape).numpy()[:, :, 0]
            # img are store as 2D images but models expects a 3D tensor
            # it means we just need to copy the same image for the 3 channels
            img = np.repeat(img[..., np.newaxis], 3, -1)
            return img
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "There was an error while laoding img: \n{}".format(str(e))
                )
            return None

    def process_path(self, file_path: str, image_shape: [...]):
        """
        Reads image from a file into a tensor format.
        """
        if self.logger:
            self.logger.info("PROCESSING: " + str(file_path))
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img, image_shape)
        return img

    def decode_img(self, img, image_shape):
        """
        Decodes the string representation image into a tensor format.
        """
        # convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_image(img)

        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, image_shape)

    def __getstate__(self):
        """
        Overrites the pickle protocol to store the model
        """
        # first we save the networks in h5 format under the artifact path
        try:
            os.mkdirs(self.path)
        except Exception:
            # already exist
            pass
        # we save both models
        self.pneumonia_model.save(self.path["pneumonia"])
        self.covid_model.save(self.path["covid"])
        # flush their reference for serialization.
        self.pneumonia_model = None
        self.covid_model = None

        return self.__dict__

    def __setstate__(self, state):
        """
        restore state.
        loads the fucking net.
        """

        # load eveything else
        self.artifact_name = state["artifact_name"]
        self.logger = state.get("logger")
        self.pneumonia_model = self.load_model(state["path"]["pneumonia"])
        self.covid_model = self.load_model(state["path"]["covid"])
        self.img_shape = state["img_shape"]
        self.path = state["path"]


if __name__ == "__main__":
    data = {"img_path": "person20_bacteria_67.jpeg"}
    # data = pd.DataFrame({'data': [data]})

    art = StepNNArtifact(
        "test",
        {
            "artifact": "ensamble_model",
            "pneumonia": "xray_model.h5",
            "covid": "covid_model.h5",
        },
        img_shape=(180, 180),
    )

    # norm_data = art.preprocess(data)
    pred = art.predict(data)

    print(pred)

    with open("artifact.pkl", "wb") as f:
        pickle.dump(art, f)

    with open("artifact.pkl", "rb") as f:
        art = pickle.load(f)

    pred = art.predict(data)
    print("done")
