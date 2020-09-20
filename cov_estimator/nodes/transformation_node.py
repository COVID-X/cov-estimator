import itertools

import lambdaJSON

from cov_estimator.nodes.abstract_estimator import AbstractEstimator


class TransformNode(AbstractEstimator):
    """
    This node holds simple arithmetic tranformation which are provided by the user.
    Note, the transofmration function will be serialized so the user is responsible for
    keeping dependecies available both in the serialization env as well as the deserialization
    env.

    NOTICE:
        the callable transformation shoudl expect a single param (dictionary) which contains
        the result of each dependency as values and the key is the estimators name.
        Eg: f: funcion
            e1: estimator 1, f depends on it
            e2: estimator 2, f depends on it too

            then, f should expects a d argument containing
            d['e1'] = results_1
            d['e2'] = result_2
    """

    def __init__(self, f, name):
        super(TransformNode, self).__init__(f, name)
        self.transformation_function = f
        self.name = name

    def __call__(self, *args):
        """
        This method expects the estimators from which to
        it depends in order to apply the transformation f
        """
        self.dependencies = args
        return self

    def predict(self, data):
        """
        Calculate the result of this node. Basedn on the data which is the input to the
        hole pipeline.
        Params:
            - data: tf.Tensor holding the input to each root node.
        """

        prev_results = data
        for estiamtor in self.dependencies:
            prev_results[estiamtor.name] = estiamtor.predict(data)

        return self.transformation_function(prev_results)

    def __copy__(self):
        return TransformNode(self.transformation_function, self.name)

    def __getstate__(self):
        copy_node = self.__copy__()
        serialized_fun = lambdaJSON.serialize(copy_node.transformation_function)
        copy_node.transformation_function = serialized_fun
        copy_node = copy_node(self.dependencies)
        return copy_node.__dict__

    def __setstate__(self, state):

        self.transformation_function = lambdaJSON.deserialize(
            state["transformation_function"]
        )
        self.name = state["name"]
        self.dependencies = list(itertools.chain.from_iterable(state["dependencies"]))
