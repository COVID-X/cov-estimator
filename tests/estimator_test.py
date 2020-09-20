import os
import sys
import shutil

test_module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, test_module_path + "/../")

import logging

import numpy as np
import tensorflow as tf

from cov_estimator.nodes import Estimator
from cov_estimator import Pipeline

logger = logging
IMG_PATH = "tests/data/kjr-21-e24-g004-l-b.jpg"

COVID_MODEL_PATH = "tests/data/ensamble_model/covid_model.h5"
PNEUMONIA_MODEL_PATH = "tests/data/ensamble_model/xray_model.h5"

PIPELINE_PATH = "pipline-test"


def load_img(img_path: str, img_shape: tuple):
    """
    Loads img from local file to memory
    Params:
        - img_path (str): path to the img file
        - img_shape: 2-tuple representing (width, height) of the output img

    Return : numpy representation fo img
    If it fails, just logs the error and return None
    """
    try:
        img = process_path(img_path, img_shape).numpy()[:, :, 0]
        # img are store as 2D images but models expects a 3D tensor
        # it means we just need to copy the same image for the 3 channels
        img = np.repeat(img[..., np.newaxis], 3, -1)
        return img
    except Exception as e:
        if logger:
            logger.error("There was an error while laoding img: \n{}".format(str(e)))
        return None


def process_path(file_path: str, image_shape: [...]):
    """
    Reads image from a file into a tensor format.
    """
    if logger:
        logger.info("PROCESSING: " + str(file_path))
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_shape)
    return img


def decode_img(img, image_shape):
    """
    Decodes the string representation image into a tensor format.
    """
    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_image(img)

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, image_shape)


def given_an_img():
    img = load_img(IMG_PATH, (180, 180))

    img = np.expand_dims(img, axis=0)
    return img


def given_an_estimator():
    est = Estimator(
        (
            lambda data: {
                "pneumonia": data["pneumonia"],
                "covid": data["pneumonia"] * data["covid"],
                "normal": 1 - data["pneumonia"],
            }
        ),
        "mult_1",
    )

    covid = tf.keras.models.load_model(COVID_MODEL_PATH)
    covid_est = Estimator(covid, "covid")

    pneumonia = tf.keras.models.load_model(PNEUMONIA_MODEL_PATH)
    pneumonia_est = Estimator(pneumonia, "pneumonia")

    est = est(pneumonia_est, covid_est)
    return est


def given_actual_result():
    data = {
        "pneumonia": np.array([0.9999976]),
        "covid": np.array([0.18703818]),
        "normal": np.array([2.3841858e-06]),
    }
    return data


def given_data():
    data = {
        "covid": given_an_img(),
        "pneumonia": given_an_img(),
    }
    return data


# def test_pipeline_result():
#     estimator = given_an_estimator()
#     model = Pipeline(estimator, PIPELINE_PATH)
#     data = given_data()

#     result = model(data)
#     assert then_result_match(result, given_actual_result())


def test_pipeline_serialization():
    estimator = given_an_estimator()
    model = Pipeline(estimator, PIPELINE_PATH)

    Pipeline.save(model)
    seralized_model = Pipeline.load(PIPELINE_PATH)
    assert then_model_equals_serialized(model, seralized_model)
    # remove test serialization
    try:
        shutil.rmtree(PIPELINE_PATH)
    except Exception as e:
        pass


def then_result_match(output, expected):
    for k in output.keys():
        if not np.allclose(output[k], expected[k], rtol=0.001):
            return False
    return True


def then_model_equals_serialized(model, serialized_model):

    real_output = model(given_data())
    serialized_result = serialized_model(given_data())

    for k in real_output.keys():
        if not np.allclose(real_output[k], serialized_result[k], rtol=0.0001):
            return False

    return True


if __name__ == "__main__":
    test_pipeline_result()

    test_pipeline_serialization()
