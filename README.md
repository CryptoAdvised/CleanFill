# ClearFill
ClearFill is a python library you can use to fill a matrix using various predictions techniques. This is useful in the context of collaborative filtering. It can be used to predict items rating in the context of recommendation engine. This code will fill NaN (Not A Number) with some predicted value according to the desired method of prediction. 

It's a simple data transformation tool.

# How it works
ClearFill take in a numpy array matrix containing NaN and fill them with estimated value. For a demonstration simply look at test.py

# Available prediction methode
- Linear regression
- Nearest value
- Slope One (Fastest)
- Weighted Slope One
- Bipolar Slope One

# Depedencies
You'll need numpy and scipy installed in your venv to run this library.
