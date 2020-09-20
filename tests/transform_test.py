import os
import sys

test_module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, test_module_path + "/../")

import pickle

from cov_estimator.nodes.transformation_node import TransformNode


def given_a_node():
    t = TransformNode(lambda data: data["x"] + data["y"], "example_node")
    return t


def test_node_serialization():

    node = given_a_node()
    serialization = pickle.dumps(node)
    serialized_node = pickle.loads(serialization)
    assert then_results_match(node, serialized_node)


def then_results_match(real_node, serialized_node):
    data = {"x": 2, "y": 3}

    real_result = real_node.predict(data)
    serialized_result = serialized_node.predict(data)
    return real_result == serialized_result


if __name__ == "__main__":

    test_node_serialization()
