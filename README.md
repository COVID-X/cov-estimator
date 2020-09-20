# cov-estimator
### Simple estimator module for serialization

This is a simple library to build adapters to construct complex 
estimators using sklearn models as well as keras.

### Basic architecture:
```

    from cov_estimator.nodes import Estimator
    from cov_estimator import Pipeline

    img = "path/to/img"

    img = load_img(img, (180, 180))

    img = np.expand_dims(img, axis=0)

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

    covid = tf.keras.models.load_model("path/to/model_1")
    covid_est = Estimator(covid, "covid")

    pneumonia = tf.keras.models.load_model("path/to/model_2")
    pneumonia_est = Estimator(pneumonia, "pneumonia")

    est = est(pneumonia_est, covid_est)
    data = {"covid": img, "pneumonia": img}
    model = Pipeline(data, 'Path/to/save')
    result = model(result)

    print("result: {}".format(result)
```

### Whats happening?
Basically what the library does is create an abstract tree of dependecies which evaluate each node.
For it to properly work, you need to pass a dictionary containing as key each nodes which expects some input.
You can use, lambda funcions to apply some important transformation. it will always get a single dictionary as input, 
the dictionary will contain the outputs from each correspondant node.
**Obs!** 
Each node inside in the graph must have a unique name, it's the developer's responsibility to


## Stack:
- Python 3.8
- Virtualenv
- tensorflow > 2.0 
- numpy > 1.6
- sklearn