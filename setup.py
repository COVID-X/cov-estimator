import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cov_estimator", # Replace with your own username
    version="0.0.1",
    author="Levensworth",
    author_email="santiago.bassani96@gmail.com",
    description="A small ML serialization software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/COVID-X/cov-estimator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
